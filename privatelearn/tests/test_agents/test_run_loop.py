"""Tests for the end-to-end run (T9): the acceptance criteria from issue #13."""

import json
import pathlib
import sys
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_loop
from shared import schemas


def without_runtime(t):
    return json.loads(json.dumps(t))


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start = time.perf_counter()
        cls.trace = run_loop.run()
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
        plain = run_loop.run(private=False)
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
        self.assertEqual(without_runtime(run_loop.run()), without_runtime(self.trace))

    def test_committed_trace_matches_a_fresh_run(self):
        committed = ROOT / "trace.json"
        self.assertTrue(committed.exists(), "run `python src/pipeline/run_loop.py` and commit trace.json")
        saved = json.loads(committed.read_text(encoding="utf-8"))
        provider = saved.get("meta", {}).get("decision_maker", {}).get("provider", "rule")
        if provider != "rule":
            self.skipTest(f"committed trace was generated with {provider}; LLM choices are not byte-reproducible")
        self.assertEqual(without_runtime(saved), without_runtime(self.trace))


if __name__ == "__main__":
    unittest.main(verbosity=2)
