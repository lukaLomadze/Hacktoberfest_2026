"""Tests for the end-to-end run (T9): the acceptance criteria from issue #13."""

import json
import pathlib
import sys
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agents.decider import LLMDecider, RuleDecider
from model.aggregation import select_strategy
from pima_decider import PimaDecider
from pipeline import run_loop
from shared import io, schemas

# Fixed hospital randomness so the runs below repeat; a real run draws it fresh (secret_seed=None).
# 42 reproduces the masks and noise the run used before they became secret, so the acceptance
# numbers below are checked on the same draw as before. Other draws move the AUCs by ~0.01.
TEST_SECRET = 42


def without_runtime(t):
    return json.loads(json.dumps(t))


def assert_same_trace(test, a, b, tol=1e-9, path="trace"):
    """Equal structure and strings; floats equal within tol (numpy builds differ in the last bits)."""
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b), path)
        for k in a:
            assert_same_trace(test, a[k], b[k], tol, f"{path}.{k}")
    elif isinstance(a, list):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_trace(test, x, y, tol, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        test.assertAlmostEqual(a, b, delta=tol, msg=path)
    else:
        test.assertEqual(a, b, path)


class TenEpochDecider(PimaDecider):
    """The Planner picks 10 epochs instead of the default 4 every round."""

    def plan(self, context, fallback):
        d = super().plan(context, fallback)
        return {**d, "epochs": 10}


class TestSigmaFollowsThePlan(unittest.TestCase):
    """X1: DP σ is calibrated to the epochs/lr the Planner chose, not the constants."""

    def test_sigma_scales_with_epochs_times_lr_at_the_same_epsilon(self):
        base = run_loop.plan_sigma(120, 4, 0.1)
        self.assertAlmostEqual(run_loop.plan_sigma(120, 10, 0.1) / base, 2.5, places=4)
        self.assertAlmostEqual(run_loop.plan_sigma(120, 4, 0.2) / base, 2.0, places=4)

    def test_sigma_refuses_a_plan_it_cannot_calibrate_to(self):
        # NaN would bisect to σ ≈ 1e-6: an update with no noise reported as ε-DP
        for epochs, lr in ((4, float("nan")), (4, float("inf")), (4, 0.0), (0, 0.1), (4.5, 0.1)):
            with self.assertRaises(ValueError):
                run_loop.plan_sigma(120, epochs, lr)

    def test_a_ten_epoch_plan_records_a_larger_sigma_at_the_same_epsilon(self):
        default = run_loop.run(decider=PimaDecider(), secret_seed=TEST_SECRET)
        longer = run_loop.run(decider=TenEpochDecider(), secret_seed=TEST_SECRET)
        for a, b in zip(default["rounds"], longer["rounds"]):
            for u, v in zip(a["updates"], b["updates"]):
                if not (u["sent"] and not u.get("rejected")):
                    self.assertEqual((u["dp_sigma"], v["dp_sigma"]), (0.0, 0.0))  # nothing was noised
                    continue
                self.assertEqual(u["dp_eps"], v["dp_eps"])  # same ε …
                self.assertAlmostEqual(v["dp_sigma"] / u["dp_sigma"], 2.5, places=4)  # … more noise
        self.assertEqual([r["eps_total"] for r in longer["rounds"]], [1.5, 3.0, 4.5, 6.0])


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start = time.perf_counter()
        cls.trace = run_loop.run(decider=PimaDecider(), secret_seed=TEST_SECRET)
        cls.seconds = time.perf_counter() - start

    def test_exactly_four_rounds_and_valid_trace(self):
        self.assertEqual(len(self.trace["rounds"]), 4)
        schemas.validate_trace(self.trace)
        json.dumps(self.trace, allow_nan=False)  # strict JSON, no NaN/inf

    def test_federated_within_001_of_centralized(self):
        m = self.trace["meta"]
        self.assertLessEqual(abs(m["auc_federated"] - m["auc_centralized"]), 0.01)
        self.assertEqual(m["auc_federated"], self.trace["metrics"]["auc_series"][-1])

    def test_privacy_cost_is_measured_and_small(self):
        m = self.trace["meta"]
        self.assertLess(abs(m["auc_federated"] - m["auc_federated_no_privacy"]), 0.01)
        plain = run_loop.run(private=False, decider=PimaDecider(), secret_seed=TEST_SECRET)
        self.assertFalse(any(u["masked"] for r in plain["rounds"] for u in r["updates"]))

    def test_weakest_hospital_gains_from_federation(self):
        m = self.trace["meta"]
        self.assertGreater(m["auc_federated"], min(m["auc_solo_by_hospital"].values()) + 0.03)

    def test_runs_under_60_seconds(self):
        self.assertLess(self.seconds, 60.0)

    def test_honesty_note_present(self):
        self.assertIn("SIMULATED", self.trace["meta"]["honesty"])

    def test_scenario_offline_then_outlier_excluded(self):
        r = self.trace["rounds"]
        self.assertEqual(r[1]["excluded"], ["h2"])
        self.assertIn("offline", r[1]["decision"])
        self.assertEqual(r[2]["strategy"], "trimmed_mean")
        self.assertEqual(r[2]["excluded"], ["h2"])
        self.assertEqual(r[0]["excluded"], [])
        self.assertEqual(r[3]["excluded"], [])

    def test_every_sent_update_was_masked_with_dp(self):
        for r in self.trace["rounds"]:
            for u in r["updates"]:
                if u["sent"] and not u.get("rejected"):
                    self.assertTrue(u["masked"])
                    self.assertEqual(u["dp_eps"], run_loop.EPS_PER_ROUND)

    def test_every_update_records_the_sigma_of_its_plan(self):
        # the default plan (4 epochs @ lr 0.1): σ = meta.dp_sigma / participants
        for r in self.trace["rounds"]:
            parts = sum(1 for u in r["updates"] if u["sent"] and not u.get("rejected"))
            for u in r["updates"]:
                expected = self.trace["meta"]["dp_sigma"] / parts if u["sent"] and not u.get("rejected") else 0.0
                self.assertAlmostEqual(u["dp_sigma"], expected, places=9)

    def test_epsilon_accounting(self):
        self.assertEqual([r["eps_total"] for r in self.trace["rounds"]], [1.5, 3.0, 4.5, 6.0])

    def test_all_hospitals_end_with_the_identical_model(self):
        hashes = self.trace["meta"]["model_hashes"]
        self.assertEqual(set(hashes), {"h0", "h1", "h2"})
        self.assertEqual(len(set(hashes.values())), 1)

    def test_attack_recovers_unprotected_and_fails_protected(self):
        a = self.trace["attack"]
        self.assertTrue(a["without_privacy"]["recovered"])
        self.assertLess(a["without_privacy"]["feature_error"], 0.2)
        self.assertFalse(a["with_privacy"]["recovered"])
        self.assertGreater(a["with_privacy"]["feature_error"], 1.0)

    def test_no_patient_values_in_trace(self):
        text = json.dumps(self.trace)
        for key in ('"rows"', '"X"', '"records"', '"patients"'):
            self.assertNotIn(key, text)
        self.assertEqual(self.trace["patient_bytes_transferred"], 0)

    def test_deterministic(self):
        self.assertEqual(without_runtime(run_loop.run(decider=PimaDecider(), secret_seed=TEST_SECRET)), without_runtime(self.trace))

    def test_committed_trace_matches_a_fresh_run(self):
        committed = ROOT / "trace.json"
        self.assertTrue(committed.exists(), "run `python src/pipeline/run_loop.py` and commit trace.json")
        saved = json.loads(committed.read_text(encoding="utf-8"))
        provider = saved.get("meta", {}).get("decision_maker", {}).get("provider", "rule")
        if provider != "rule":
            self.skipTest(f"committed trace was generated with {provider}; LLM choices are not byte-reproducible")
        assert_same_trace(self, saved, without_runtime(self.trace))


def _spy_on_protection(test):
    """Record the pair secrets and noise seeds the hospitals protect their updates with."""
    seen = []
    real = run_loop.prepare_update

    def spy(delta, client_id, participants, round_, secrets, noise_seed, **kw):
        seen.append((tuple(sorted(secrets.items())), noise_seed))
        return real(delta, client_id, participants, round_, secrets, noise_seed=noise_seed, **kw)

    run_loop.prepare_update = spy
    test.addCleanup(setattr, run_loop, "prepare_update", real)
    return seen


class TestHospitalSecretsArePrivate(unittest.TestCase):
    """The masks and DP noise must not be recomputable from anything in the trace."""

    def test_masks_and_noise_do_not_come_from_the_public_seed(self):
        seen = _spy_on_protection(self)
        trace = run_loop.run(decider=PimaDecider())
        public = run_loop.make_pair_secrets([0, 1, 2], seed=f"handshake-{trace['meta']['seed']}")
        self.assertTrue(seen)
        for secrets, noise_seed in seen:
            for pair, value in secrets:
                self.assertNotEqual(value, public[pair])
            self.assertNotIn(f"dp-{trace['meta']['seed']}-", noise_seed)
            self.assertNotIn(noise_seed, json.dumps(trace))

    def test_every_run_draws_fresh_secrets(self):
        seen = _spy_on_protection(self)
        run_loop.run(decider=PimaDecider())
        first = list(seen)
        seen.clear()
        run_loop.run(decider=PimaDecider())
        self.assertNotEqual(first[0], seen[0])


class ExcludeEveryone(PimaDecider):
    def strategy(self, losses, names, fallback, **kw):
        return {"strategy": "trimmed_mean", "flagged": list(range(len(names))),
                "reasoning": "exclude all", "made_by": "llm"}


class AlwaysFailingLLM(LLMDecider):
    """An LLM decider whose every call fails (no network, timeout, bad JSON)."""

    def _ask(self, system, user):
        raise TimeoutError("the model did not answer")


class TestRunsNeverStall(unittest.TestCase):
    def test_a_decision_excluding_every_hospital_falls_back_to_the_rule(self):
        trace = run_loop.run(decider=ExcludeEveryone(), secret_seed=TEST_SECRET)
        self.assertEqual(len(trace["rounds"]), 4)
        for r in trace["rounds"]:
            self.assertTrue(r["applied"])
            self.assertEqual(r["decided_by"], "rule")

    def test_the_llm_decider_refuses_to_exclude_every_hospital(self):
        d = LLMDecider(api_key="test")
        d._ask = lambda system, user: {"strategy": "trimmed_mean", "excluded": ["A", "B"], "reasoning": "x"}
        out = d.strategy([0.6, 0.7], ["A", "B"], lambda: select_strategy([0.6, 0.7], names=["A", "B"]))
        self.assertEqual(out["made_by"], "rule")
        self.assertEqual(out["flagged"], [])
        self.assertEqual(d.fallbacks, 1)

    def test_a_round_nobody_answers_is_reported_not_silently_dropped(self):
        manifest = io.load_json(ROOT / "data" / "hospitals" / "manifest.json")
        manifest["faults"] = {h: {"2": "offline"} for h in ("h0", "h1", "h2")}
        with self.assertRaisesRegex(RuntimeError, "round 2: no hospital responded"):
            run_loop.run(decider=PimaDecider(), manifest=manifest, secret_seed=TEST_SECRET)


class TestOfflineFallback(unittest.TestCase):
    """Without a working LLM the manifest's schema map lets the run finish (made_by=rule)."""

    def check(self, trace):
        self.assertEqual(len(trace["rounds"]), 4)
        self.assertEqual(trace["meta"]["features"],
                         ["pregnancies", "glucose", "blood_pressure", "bmi", "insulin", "age"])
        for h in trace["meta"]["decision_maker"]["schema"].values():
            self.assertEqual(h["made_by"], "rule")
        self.assertGreater(trace["meta"]["auc_federated"], 0.8)

    def test_runs_with_no_llm(self):
        self.check(run_loop.run(decider=RuleDecider(), secret_seed=TEST_SECRET))

    def test_runs_when_every_llm_call_fails(self):
        d = AlwaysFailingLLM(api_key="test")
        self.check(run_loop.run(decider=d, secret_seed=TEST_SECRET))
        self.assertGreater(d.fallbacks, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
