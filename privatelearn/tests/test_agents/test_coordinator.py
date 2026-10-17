"""Tests for the coordinator agent (T8), against scripted hospitals on the real bus."""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from agents.coordinator import CANONICAL, CoordinatorAgent
from agents.runtime import Agent, MessageBus
from model import logistic
from model.aggregation import FEDPROX_MU
from model.privacy import make_pair_secrets, mask_update
from shared import schemas

SECRETS = make_pair_secrets([0, 1, 2], seed="coord-tests")


def make_data(seed, shift=0.0, n=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)) + shift
    y = (X[:, 1] + 0.5 * X[:, 5] + rng.normal(scale=0.5, size=n) > shift).astype(float)
    return X, y


class FakeHospital(Agent):
    """Speaks the T7 protocol on synthetic local data; faults = {round: kind}."""

    def __init__(self, hid, X, y, masked=False, faults=None, columns=CANONICAL, dp_eps=0.0):
        super().__init__(hid)
        self.X, self.y, self.masked = X, y, masked
        self.faults = faults or {}
        self.columns = columns
        self.dp_eps = dp_eps
        self.trained = []
        self.broadcasts = []

    def step(self, messages):
        out = []
        for m in messages:
            p = m.payload
            if m.kind == "schema_request":
                out.append(self.send(m.sender, "schema", {"hospital": self.id, "label": "outcome", "columns": [
                    {"canonical": c, "source": [c], "unit": "x", "transform": "rename"} for c in self.columns]}))
            elif m.kind == "probe":
                if self.faults.get(p["round"]) == "offline":
                    continue
                loss = logistic.loss(np.array(p["w"]), self.X, self.y)
                if self.faults.get(p["round"]) == "outlier":
                    loss += 5.0
                if self.faults.get(p["round"]) == "spread":
                    loss += {"h0": 0.0, "h1": 0.3, "h2": 0.6}[self.id]
                out.append(self.send(m.sender, "probe_result", {
                    "round": p["round"], "hospital": self.id, "loss": loss, "auc": 0.5, "n": len(self.y)}))
            elif m.kind == "train":
                if self.id not in p["participants"]:
                    continue
                self.trained.append((p["round"], p["strategy"], p["mu"]))
                if self.faults.get(p["round"]) == "silent":
                    continue
                w = np.array(p["w"])
                delta = p["weights"][self.id] * (logistic.train(w, self.X, self.y) - w)
                idx = int(self.id[1])
                if self.masked:
                    parts = sorted(int(h[1]) for h in p["participants"])
                    delta = mask_update(delta, idx, parts, p["round"], SECRETS)
                out.append(self.send(m.sender, "update", {
                    "v": "1.0", "round": p["round"], "hospital": self.id,
                    "delta_w": [float(v) for v in delta], "loss": 0.5, "n": len(self.y),
                    "privacy": {"masked": self.masked, "mask_seed": f"r{p['round']}", "dp_eps": self.dp_eps,
                                "dp_mechanism": "gaussian", "dp_sigma": 0.0}}))
            elif m.kind == "broadcast":
                self.broadcasts.append(p)
        return out


DATA = [make_data(1, 0.0), make_data(2, 0.3), make_data(3, -0.3)]
XT, YT = make_data(9)


def evaluate(w, features=None):
    if features:
        return logistic.auc(YT, logistic.predict_proba(w, XT[:, [CANONICAL.index(f) for f in features]]))
    return logistic.auc(YT, logistic.predict_proba(w, XT))


def run(masked=False, faults=None, rounds=4, dp_eps=0.0, **kw):
    faults = faults or {}
    bus = MessageBus()
    hs = [bus.register(FakeHospital(f"h{i}", *DATA[i], masked=masked, faults=faults.get(f"h{i}"),
                                    dp_eps=dp_eps)) for i in range(3)]
    coord = bus.register(CoordinatorAgent(["h0", "h1", "h2"], evaluate, [0.0] * 6, [1.0] * 6,
                                          rounds=rounds, names={"h0": "A", "h1": "B", "h2": "C"}, **kw))
    bus.run(max_ticks=60)
    return coord, hs, bus


def manual_fedavg(rounds=4):
    w = np.zeros(7)
    for _ in range(rounds):
        w = w + np.mean([logistic.train(w, X, y) - w for X, y in DATA], axis=0)
    return w


class TestRounds(unittest.TestCase):
    def test_runs_exactly_the_requested_rounds(self):
        coord, hs, _ = run()
        self.assertTrue(coord.done)
        self.assertEqual([h["round"] for h in coord.history], [1, 2, 3, 4])
        self.assertEqual(len(coord.auc_series), 5)
        self.assertTrue(coord.history[-1]["broadcast"]["converged"])
        for h in hs:
            self.assertEqual([b["round"] for b in h.broadcasts], [1, 2, 3, 4])

    def test_unmasked_result_is_fedavg(self):
        coord, _, _ = run(tau=10.0)  # agreeing losses → FedAvg every round
        self.assertEqual({h["strategy"] for h in coord.history}, {"fedavg"})
        np.testing.assert_allclose(coord.w, manual_fedavg(), atol=1e-12)

    def test_masked_result_identical_to_unmasked(self):
        plain, _, _ = run(tau=10.0)
        masked, _, bus = run(masked=True, tau=10.0)
        np.testing.assert_allclose(masked.w, plain.w, atol=1e-9)
        # the coordinator never received an update close to a real one
        real = DATA[0]
        w0 = np.zeros(7)
        first = [m.payload["delta_w"] for m in bus.log if m.kind == "update" and m.payload["round"] == 1][0]
        self.assertGreater(np.linalg.norm(np.array(first) - (logistic.train(w0, *real) - w0) / 3), 0.5)

    def test_every_message_and_trace_row_is_contract_valid(self):
        coord, _, bus = run(masked=True, faults={"h2": {2: "offline"}, "h1": {3: "outlier"}})
        for m in bus.log:
            if m.kind == "broadcast":
                schemas.validate_broadcast(m.payload)
        for r in coord.history:
            schemas.validate_broadcast(r["broadcast"])
            for u in r["updates"]:
                schemas.validate_trace_update(u)


class TestDecisions(unittest.TestCase):
    def test_history_records_the_clear_text_losses_behind_each_decision(self):
        coord, _, _ = run(faults={"h2": {2: "offline"}})
        self.assertEqual(set(coord.history[0]["probe_losses"]), {"h0", "h1", "h2"})
        self.assertEqual(set(coord.history[1]["probe_losses"]), {"h0", "h1"})
        self.assertAlmostEqual(coord.history[0]["probe_losses"]["h0"], np.log(2), places=12)

    def test_offline_hospital_excluded_and_weights_renormalised(self):
        coord, hs, _ = run(faults={"h2": {2: "offline"}})
        r2 = coord.history[1]
        self.assertEqual(r2["excluded"], ["h2"])
        self.assertIn("C offline", r2["decision"])
        self.assertNotIn(2, [t[0] for t in hs[2].trained])
        row = [u for u in r2["updates"] if u["hospital"] == "h2"][0]
        self.assertEqual((row["sent"], row["loss"], row["n"]), (False, None, 120))

    def test_outlier_trimmed_before_training(self):
        coord, hs, _ = run(masked=True, faults={"h1": {3: "outlier"}})
        r3 = coord.history[2]
        self.assertEqual(r3["strategy"], "trimmed_mean")
        self.assertEqual(r3["excluded"], ["h1"])
        self.assertIn("B loss", r3["decision"])
        self.assertNotIn(3, [t[0] for t in hs[1].trained])  # never asked to train → masks still cancel
        row = [u for u in r3["updates"] if u["hospital"] == "h1"][0]
        self.assertTrue(row["rejected"])
        self.assertTrue(r3["applied"])

    def test_trimmed_round_equals_fedavg_of_the_rest(self):
        coord, _, _ = run(masked=True, faults={"h1": {1: "outlier"}}, rounds=1, tau=10.0)
        w0 = np.zeros(7)
        expected = np.mean([logistic.train(w0, *DATA[i]) - w0 for i in (0, 2)], axis=0)
        np.testing.assert_allclose(coord.w, expected, atol=1e-9)

    def test_spread_losses_select_fedprox_and_send_mu(self):
        coord, hs, _ = run(faults={h: {1: "spread"} for h in ("h0", "h1", "h2")}, rounds=1, tau=0.05)
        self.assertEqual(coord.history[0]["strategy"], "fedprox")
        self.assertEqual(hs[0].trained[0], (1, "fedprox", FEDPROX_MU))

    def test_silent_after_masking_discards_round(self):
        coord, _, _ = run(masked=True, faults={"h0": {2: "silent"}}, tau=10.0)
        r2 = coord.history[1]
        self.assertFalse(r2["applied"])
        self.assertIn("round discarded", r2["decision"])
        np.testing.assert_allclose(r2["broadcast"]["w"], coord.history[0]["broadcast"]["w"])
        self.assertTrue(coord.history[2]["applied"])

    def test_eps_total_is_per_hospital_cumulative(self):
        coord, _, _ = run(dp_eps=1.5, faults={"h2": {2: "offline"}})
        self.assertEqual([r["eps_total"] for r in coord.history], [1.5, 3.0, 4.5, 6.0])
        self.assertEqual(coord.eps_spent, {"h0": 6.0, "h1": 6.0, "h2": 4.5})


class TestSchemaNegotiation(unittest.TestCase):
    def test_agreed_schema_and_report(self):
        coord, _, bus = run(rounds=1)
        agreed = [m.payload for m in bus.log if m.kind == "schema_agreed"]
        self.assertEqual(len(agreed), 3)  # one copy per hospital
        self.assertEqual(agreed[0]["features"], CANONICAL)
        self.assertEqual(set(coord.schema_report["per_hospital"]), {"h0", "h1", "h2"})

    def test_contract_keeps_only_common_features(self):
        # negotiation agrees a contract instead of rejecting an incomplete hospital:
        # a feature is usable only if every hospital proposed it.
        X5, y = DATA[0][0][:, :5], DATA[0][1]
        bus = MessageBus()
        bus.register(FakeHospital("h0", X5, y, columns=CANONICAL[:5]))
        coord = CoordinatorAgent(["h0"], evaluate, [0.0] * 6, [1.0] * 6, rounds=1)
        bus.register(coord)
        bus.run(max_ticks=10)
        self.assertEqual(coord.features, CANONICAL[:5])
        self.assertEqual(coord.n_weights, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
