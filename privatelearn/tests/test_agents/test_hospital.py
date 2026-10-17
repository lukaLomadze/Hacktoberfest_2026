"""Tests for the hospital agents (T7). Uses the committed T1 data."""

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from agents.hospital import (
    CANONICAL,
    HospitalAgent,
    SchemaAgent,
    load_local_csv,
)
from agents.runtime import Agent, Message, MessageBus, sanitize
from model import logistic
from model.privacy import make_pair_secrets, mask_update, unmask_sum
from pipeline import prep_data

DATA = ROOT / "data"
MANIFEST = json.loads((DATA / "hospitals" / "manifest.json").read_text(encoding="utf-8"))
MEAN, STD = MANIFEST["feature_mean"], MANIFEST["feature_std"]


def hospital(h, **kw):
    return HospitalAgent.from_csv(f"h{h}", DATA / "hospitals" / f"hospital_{h}.csv",
                                  feature_mean=MEAN, feature_std=STD, **kw)


def msg(kind, payload, to="h0", sender="coord"):
    return Message(sender, to, kind, payload)


def agreed(h):
    a = h.step([msg("schema_agreed", {"features": CANONICAL}, to=h.id)])
    assert a == []
    return h


def train_msg(round_, w, participants=("h0", "h1", "h2"), strategy="fedavg", mu=0.0):
    n = {p: 120 for p in participants}
    total = sum(n.values())
    return {"round": round_, "w": list(w), "participants": list(participants),
            "weights": {p: n[p] / total for p in participants}, "strategy": strategy, "mu": mu}


class TestSchemaAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = prep_data.load_raw(DATA / "raw" / "diabetes.csv")
        cls.clean, _ = prep_data.impute(raw)

    def test_maps_every_hospital_to_canonical_within_1e6(self):
        order = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Insulin", "Age"]
        for h in range(3):
            cols, rows = load_local_csv(DATA / "hospitals" / f"hospital_{h}.csv")
            X, labels = SchemaAgent(cols).to_canonical(rows)
            ids = MANIFEST["row_ids"][f"h{h}"]
            expected = np.column_stack([self.clean[c][ids] for c in order])
            np.testing.assert_allclose(X, expected, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(labels, self.clean["Outcome"][ids])

    def test_recognises_each_local_convention(self):
        h1 = SchemaAgent(load_local_csv(DATA / "hospitals" / "hospital_1.csv")[0]).mapping
        self.assertEqual(h1["glucose"]["transform"], "x * 18.0182")
        self.assertEqual(h1["blood_pressure"]["unit"], "kPa")
        self.assertEqual(h1["bmi"]["source"], ["weight_kg", "height_m"])
        self.assertEqual(h1["age"]["transform"], "x / 12")
        h2 = SchemaAgent(load_local_csv(DATA / "hospitals" / "hospital_2.csv")[0]).mapping
        self.assertEqual(h2["pregnancies"]["source"], ["num_pregnancies"])
        self.assertEqual(h2["insulin"]["transform"], "x / 6.945")

    def test_missing_or_ambiguous_columns_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "cannot map"):
            SchemaAgent(["Age", "BMI", "Glucose", "Insulin", "Pregnancies", "Outcome"])
        with self.assertRaisesRegex(ValueError, "both map"):
            SchemaAgent(["age", "age_months", "bmi", "glucose", "bp_sys", "insulin", "pregnancies", "outcome"])
        with self.assertRaisesRegex(ValueError, "label"):
            SchemaAgent(["Age", "BMI", "Glucose", "BloodPressure", "Insulin", "Pregnancies"])

    def test_schema_message_has_names_only(self):
        h = hospital(1)
        (out,) = h.step([msg("schema_request", {}, to="h1")])
        self.assertEqual(out.kind, "schema")
        sanitize(out.kind, out.payload)  # passes the bus privacy checks
        text = json.dumps(out.payload)
        for r in h._rows[:5]:  # no patient value appears in the description
            for v in r.values():
                if v not in (0.0, 1.0, 12.0):
                    self.assertNotIn(repr(v), text)


class TestHospitalRound(unittest.TestCase):
    def test_probe_reports_local_loss_and_auc(self):
        h = agreed(hospital(0))
        (out,) = h.step([msg("probe", {"round": 1, "w": [0.0] * 7})])
        self.assertEqual(out.kind, "probe_result")
        self.assertAlmostEqual(out.payload["loss"], np.log(2), places=9)  # w = 0 → p = 0.5
        self.assertEqual(out.payload["n"], 120)
        self.assertEqual(set(out.payload), {"round", "hospital", "loss", "auc", "n"})

    def test_update_is_contract_valid_weighted_delta(self):
        h = agreed(hospital(2))
        w = np.zeros(7)
        (out,) = h.step([msg("train", train_msg(1, w), to="h2")])
        self.assertEqual(out.kind, "update")
        sanitize("update", out.payload)
        expected = logistic.train(w, h._X, h._labels) - w
        np.testing.assert_allclose(out.payload["delta_w"], expected / 3, atol=1e-12)
        self.assertFalse(out.payload["privacy"]["masked"])

    def test_single_hospital_round_equals_local_sgd(self):
        h = agreed(hospital(0))
        w = np.full(7, 0.1)
        (out,) = h.step([msg("train", train_msg(1, w, participants=("h0",)))])
        np.testing.assert_allclose(w + np.array(out.payload["delta_w"]), logistic.train(w, h._X, h._labels))

    def test_fedprox_passes_mu_and_anchor_to_training(self):
        seen = {}

        def spy(w0, X, y, **kw):
            seen.update(kw)
            return logistic.train(w0, X, y, **kw)

        h = agreed(hospital(0, train_fn=spy))
        h.step([msg("train", train_msg(1, np.zeros(7), strategy="fedprox", mu=0.1))])
        self.assertEqual(seen["mu"], 0.1)
        np.testing.assert_allclose(seen["w_anchor"], np.zeros(7))
        seen.clear()
        h.step([msg("train", train_msg(2, np.zeros(7), strategy="fedavg", mu=0.1))])
        self.assertNotIn("mu", seen)

    def test_non_participant_and_offline_stay_silent(self):
        h = agreed(hospital(1, faults={2: "offline"}))
        self.assertEqual(h.step([msg("train", train_msg(1, np.zeros(7), participants=("h0", "h2")), to="h1")]), [])
        self.assertEqual(h.step([msg("probe", {"round": 2, "w": [0.0] * 7}, to="h1")]), [])
        self.assertEqual(len(h.step([msg("probe", {"round": 3, "w": [0.0] * 7}, to="h1")])), 1)

    def test_flip_labels_fault_raises_probe_loss(self):
        w = logistic.train(np.zeros(7), *agreed(hospital(2))._data(1), epochs=50, lr=0.5)
        honest = agreed(hospital(2)).step([msg("probe", {"round": 3, "w": list(w)}, to="h2")])[0]
        bad = agreed(hospital(2, faults={3: "flip_labels"})).step([msg("probe", {"round": 3, "w": list(w)}, to="h2")])[0]
        self.assertGreater(bad.payload["loss"], honest.payload["loss"] + 0.3)

    def test_broadcast_updates_global_model(self):
        h = agreed(hospital(0))
        b = {"v": "1.0", "round": 1, "w": [0.1] * 7, "strategy": "fedavg", "eps_total": 0.0,
             "auc": 0.8, "losses": [0.6], "converged": False}
        self.assertEqual(h.step([msg("broadcast", b)]), [])
        self.assertEqual(h.w_global, [0.1] * 7)
        self.assertEqual(h.activity[-1]["agent"], "challenger")

    def test_requires_agreed_schema_and_known_kinds(self):
        h = hospital(0)
        with self.assertRaises(RuntimeError):
            h.step([msg("probe", {"round": 1, "w": [0.0] * 7})])
        with self.assertRaises(ValueError):
            h.step([msg("gossip", {})])


class TestMaskedRoundOnTheBus(unittest.TestCase):
    """Three hospital agents + a scripted coordinator on the real bus, with masking."""

    def test_masked_updates_sum_to_fedavg_step(self):
        secrets = make_pair_secrets([0, 1, 2], seed="test-handshake")

        def protector(delta, idx, participants, round_):
            own = {k: v for k, v in secrets.items() if idx in k}
            vec = mask_update(delta, idx, participants, round_, own)
            return vec, {"masked": True, "mask_seed": f"r{round_}", "dp_eps": 0.0,
                         "dp_mechanism": "gaussian", "dp_sigma": 0.0}

        class Coord(Agent):
            def __init__(self):
                super().__init__("coord")
                self.tick = 0
                self.updates = []

            def step(self, messages):
                self.updates += [m.payload for m in messages if m.kind == "update"]
                self.tick += 1
                if self.tick == 1:
                    return [self.send("*", "schema_agreed", {"features": CANONICAL})]
                if self.tick == 2:
                    return [self.send("*", "train", train_msg(1, [0.0] * 7))]
                return []

        bus = MessageBus()
        hs = [bus.register(hospital(i, protector=protector)) for i in range(3)]
        coord = bus.register(Coord())
        bus.run(max_ticks=10)

        self.assertEqual(len(coord.updates), 3)
        for u in coord.updates:
            self.assertTrue(u["privacy"]["masked"])
        total = unmask_sum([u["delta_w"] for u in coord.updates])
        fedavg = np.mean([logistic.train(np.zeros(7), h._X, h._labels) for h in hs], axis=0)
        np.testing.assert_allclose(total, fedavg, atol=1e-9)
        # an individual masked update is far from that hospital's real share
        real0 = logistic.train(np.zeros(7), hs[0]._X, hs[0]._labels) / 3
        self.assertGreater(float(np.linalg.norm(np.array(coord.updates[0]["delta_w"]) - real0)), 0.5)
        # nothing on the bus carried raw rows (the bus would have raised), and the log has only these kinds
        self.assertEqual({m.kind for m in bus.log}, {"schema_agreed", "train", "update"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
