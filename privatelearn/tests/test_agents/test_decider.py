"""Tests for the LLM decision layer: deterministic rules and safe fallbacks.

No network is used: LLMDecider is constructed with no key, so every call must
fall back to the rule and never raise.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.decider import CANONICAL, LLMDecider, RuleDecider, build_decider
from agents.hospital import column_profile
from model.aggregation import select_strategy


class TestRuleDecider(unittest.TestCase):
    def test_strategy_matches_select_strategy(self):
        losses, names = [0.30, 0.32, 0.71], ["Hospital A", "Hospital B", "Hospital C"]
        out = RuleDecider().strategy(losses, names, lambda: select_strategy(losses, names=names))
        ref = select_strategy(losses, names=names)
        self.assertEqual(out["strategy"], ref.strategy)
        self.assertEqual(out["flagged"], list(ref.flagged))
        self.assertEqual(out["made_by"], "rule")

    def test_schema_uses_fallback(self):
        out = RuleDecider().schema(["a", "b"], lambda: ({"age": {"canonical": "age"}}, "outcome"))
        self.assertEqual(out["mapping"], {"age": {"canonical": "age"}})
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

    def test_schema_falls_back_without_key(self):
        d = LLMDecider(api_key="")
        out = d.schema(["age"], lambda: ({"age": {"canonical": "age"}}, "y"))
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

    def test_too_small_lr_falls_back(self):
        # observed in a real run: "conservative" lr 0.01 left the model near-random
        d = self._decider({"epochs": 3, "lr": 0.01, "mu": 0.0, "reasoning": "conservative"})
        out = d.plan({}, lambda: {"epochs": 4, "lr": 0.1, "mu": 0.0, "why": "default"})
        self.assertEqual(out["made_by"], "rule")
        self.assertEqual((out["epochs"], out["lr"]), (4, 0.1))

    def test_too_few_epochs_falls_back(self):
        d = self._decider({"epochs": 1, "lr": 0.1, "mu": 0.0, "reasoning": "fast"})
        out = d.plan({}, lambda: {"epochs": 4, "lr": 0.1, "mu": 0.0, "why": "default"})
        self.assertEqual(out["made_by"], "rule")

    def test_valid_plan_is_used(self):
        d = self._decider({"epochs": 5, "lr": 0.12, "mu": 0.0, "reasoning": "slightly more steps"})
        out = d.plan({}, lambda: {"epochs": 9, "lr": 0.9, "mu": 0.0, "why": "default"})
        self.assertEqual(out["made_by"], "llm")
        self.assertEqual((out["epochs"], out["lr"]), (5, 0.12))

    def test_schema_uses_llm_mapping(self):
        mapping = {c: {"source": [c], "factor": 1.0, "unit": "u", "transform": "id"} for c in CANONICAL}
        d = self._decider({"label": "outcome", "mapping": mapping, "reasoning": "direct names"})
        out = d.schema(list(CANONICAL) + ["outcome"], lambda: ({}, "fallback"))
        self.assertEqual(out["made_by"], "llm")
        self.assertEqual(out["label"], "outcome")
        self.assertEqual(set(out["mapping"]), set(CANONICAL))


class TestLocalProfile(unittest.TestCase):
    def test_profile_is_stats_only(self):
        rows = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
        p = column_profile(["a"], rows)
        self.assertEqual(set(p["a"]), {"min", "p05", "median", "mean", "p95", "max"})
        self.assertEqual((p["a"]["min"], p["a"]["median"], p["a"]["max"]), (1.0, 2.0, 3.0))
        self.assertEqual(p["a"]["mean"], 2.0)

    def test_profile_empty_rows(self):
        self.assertEqual(column_profile(["a"], []), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
