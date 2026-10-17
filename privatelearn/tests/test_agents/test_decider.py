"""Tests for the LLM decision layer: deterministic rules and safe fallbacks.

No network is used: LLMDecider is constructed with no key, so every call must
fall back to the rule and never raise.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.decider import LLMDecider, RuleDecider, build_decider
from agents.hospital import column_profile
from model.aggregation import select_strategy

FEATURES = ["pregnancies", "glucose", "blood_pressure", "bmi", "insulin", "age"]


class TestRuleDecider(unittest.TestCase):
    def test_strategy_matches_select_strategy(self):
        losses, names = [0.30, 0.32, 0.71], ["Hospital A", "Hospital B", "Hospital C"]
        out = RuleDecider().strategy(losses, names, lambda: select_strategy(losses, names=names))
        ref = select_strategy(losses, names=names)
        self.assertEqual(out["strategy"], ref.strategy)
        self.assertEqual(out["flagged"], list(ref.flagged))
        self.assertEqual(out["made_by"], "rule")

    def test_describe_uses_fallback(self):
        fb = {"label": "outcome", "reasoning": "r", "made_by": "rule",
              "columns": [{"source": ["a"], "semantic": "age", "unit": "u", "factor": 1.0}]}
        out = RuleDecider().describe(["a", "b"], lambda: fb)
        self.assertEqual(out["label"], "outcome")
        self.assertEqual(out["made_by"], "rule")

    def test_plan_uses_fallback(self):
        out = RuleDecider().plan({}, lambda: {"epochs": 4, "lr": 0.1, "mu": 0.0, "why": "contract"})
        self.assertEqual((out["epochs"], out["lr"], out["mu"]), (4, 0.1, 0.0))
        self.assertEqual(out["made_by"], "rule")


class TestLLMFallback(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("DEEPSEEK_API_KEY", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["DEEPSEEK_API_KEY"] = self._old

    def test_strategy_falls_back_without_key(self):
        losses, names = [0.30, 0.32, 0.71], ["A", "B", "C"]
        d = LLMDecider(api_key="")
        out = d.strategy(losses, names, lambda: select_strategy(losses, names=names))
        self.assertEqual(out["made_by"], "rule")
        self.assertEqual(out["strategy"], select_strategy(losses, names=names).strategy)
        self.assertGreaterEqual(d.fallbacks, 1)
        self.assertTrue(any("DEEPSEEK_API_KEY" in e for e in d.errors))

    def test_describe_falls_back_without_key(self):
        d = LLMDecider(api_key="")
        fb = {"label": "y", "columns": [{"source": ["age"], "semantic": "age", "unit": "u", "factor": 1.0}]}
        out = d.describe(["age"], lambda: fb)
        self.assertEqual(out["made_by"], "rule")
        self.assertEqual(out["label"], "y")

    def test_plan_falls_back_without_key(self):
        d = LLMDecider(api_key="")
        out = d.plan({}, lambda: {"epochs": 4, "lr": 0.1, "mu": 0.0, "why": "x"})
        self.assertEqual(out["made_by"], "rule")
        self.assertEqual(out["epochs"], 4)

    def test_info_reports_provider(self):
        self.assertEqual(LLMDecider(api_key="").info()["provider"], "deepseek")


class TestBuild(unittest.TestCase):
    def test_default_is_rule(self):
        self.assertIsInstance(build_decider(False), RuleDecider)

    def test_llm_flag_builds_llm(self):
        self.assertIsInstance(build_decider(True), LLMDecider)


class TestGuards(unittest.TestCase):
    """A mocked model reply: no network, and the guardrails must reject bad plans."""

    @staticmethod
    def _decider(reply):
        d = LLMDecider(api_key="test")
        d._ask = lambda system, user: reply  # type: ignore[method-assign]
        return d

    def test_low_lr_is_accepted(self):
        # no domain bounds: the agent decides; the trace records what it chose
        d = self._decider({"epochs": 3, "lr": 0.01, "mu": 0.0, "reasoning": "conservative"})
        out = d.plan({}, lambda: {"epochs": 4, "lr": 0.1, "mu": 0.0, "why": "default"})
        self.assertEqual(out["made_by"], "llm")
        self.assertEqual((out["epochs"], out["lr"]), (3, 0.01))

    def test_nonpositive_hyperparameters_fall_back(self):
        # only structural sanity is enforced, never a domain range
        d = self._decider({"epochs": 0, "lr": 0.1, "mu": 0.0, "reasoning": "bad"})
        out = d.plan({}, lambda: {"epochs": 4, "lr": 0.1, "mu": 0.0, "why": "default"})
        self.assertEqual(out["made_by"], "rule")

    def test_valid_plan_is_used(self):
        d = self._decider({"epochs": 5, "lr": 0.12, "mu": 0.0, "reasoning": "slightly more steps"})
        out = d.plan({}, lambda: {"epochs": 9, "lr": 0.9, "mu": 0.0, "why": "default"})
        self.assertEqual(out["made_by"], "llm")
        self.assertEqual((out["epochs"], out["lr"]), (5, 0.12))

    def test_describe_uses_llm(self):
        cols = [{"source": [c], "semantic": c, "unit": "u", "factor": 1.0} for c in FEATURES]
        d = self._decider({"label": "outcome", "columns": cols, "reasoning": "direct names"})
        out = d.describe(list(FEATURES) + ["outcome"], lambda: {"label": "x", "columns": []})
        self.assertEqual(out["made_by"], "llm")
        self.assertEqual(out["label"], "outcome")
        self.assertEqual([c["semantic"] for c in out["columns"]], FEATURES)


class TestLocalProfile(unittest.TestCase):
    def test_profile_is_stats_only(self):
        rows = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
        p = column_profile(["a"], rows)
        self.assertEqual(set(p["a"]), {"min", "p05", "median", "mean", "p95", "max"})
        self.assertEqual((p["a"]["min"], p["a"]["median"], p["a"]["max"]), (1.0, 2.0, 3.0))
        self.assertEqual(p["a"]["mean"], 2.0)

    def test_profile_empty_rows(self):
        self.assertEqual(column_profile(["a"], []), {})


class TestReconcile(unittest.TestCase):
    """The contract must reconcile different column names and units across hospitals."""

    def _decider(self, reply):
        d = LLMDecider(api_key="test")
        d._ask = lambda system, user: reply
        return d

    def test_describe_arbitrary_columns(self):
        # a bank-like column, not medical: the model names it from name + stats
        reply = {"label": "default", "reasoning": "income in USD",
                 "columns": [{"source": ["annual_income"], "semantic": "income",
                              "unit": "USD", "factor": 1.0},
                             {"source": ["credit_score"], "semantic": "credit_score",
                              "unit": "count", "factor": 1.0}]}
        d = self._decider(reply)
        out = d.describe(["annual_income", "credit_score", "default"],
                         lambda: {"label": "x", "columns": []})
        self.assertEqual(out["made_by"], "llm")
        self.assertEqual([c["semantic"] for c in out["columns"]], ["income", "credit_score"])

    def test_contract_reconciles_synonyms_and_units(self):
        # "bp" (kPa) and "blood_pressure" (mmHg) are the same measurement, differently named/unit-ed
        proposals = {
            "h0": {"label": "default", "columns": [
                {"source": ["BloodPressure"], "semantic": "blood_pressure", "unit": "mmHg", "factor": 1.0},
                {"source": ["Age"], "semantic": "age", "unit": "years", "factor": 1.0}]},
            "h1": {"label": "default", "columns": [
                {"source": ["bp_kpa"], "semantic": "bp", "unit": "kPa", "factor": 7.50062},
                {"source": ["age_months"], "semantic": "age", "unit": "months", "factor": 1.0 / 12.0}]},
        }
        reply = {"features": ["blood_pressure", "age"], "model": "logistic",
                 "columns": {
                     "h0": {"blood_pressure": {"source": ["BloodPressure"], "factor": 1.0},
                            "age": {"source": ["Age"], "factor": 1.0}},
                     "h1": {"blood_pressure": {"source": ["bp_kpa"], "factor": 7.50062},
                            "age": {"source": ["age_months"], "factor": 1.0 / 12.0}}},
                 "reasoning": "bp and blood_pressure are the same measurement"}
        d = self._decider(reply)
        out = d.contract(proposals, lambda: {})
        self.assertEqual(out["made_by"], "llm")
        self.assertEqual(out["features"], ["blood_pressure", "age"])
        self.assertEqual(out["columns"]["h1"]["blood_pressure"]["factor"], 7.50062)
        self.assertEqual(out["columns"]["h1"]["age"]["factor"], 1.0 / 12.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
