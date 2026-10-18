"""End-to-end: five real OS processes talking HTTP, with the scripted agents and with the
DeepSeek agent loop against a fake DeepSeek API (no key, no cost).

Runs and models go to a temporary folder (FEDHEART_RUNS_DIR / FEDHEART_MODELS_DIR), so the
tests never overwrite a real run or models/heart_model.json.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

TMP = tempfile.mkdtemp(prefix="fedheart-test-")
os.environ["FEDHEART_RUNS_DIR"] = str(pathlib.Path(TMP) / "runs")
os.environ["FEDHEART_MODELS_DIR"] = str(pathlib.Path(TMP) / "models")

import run_simulation  # noqa: E402
from fake_deepseek import FakeDeepSeek  # noqa: E402
from fedheart import config  # noqa: E402

# test_core may have imported config before the variables above were set
config.RUNS_DIR = pathlib.Path(os.environ["FEDHEART_RUNS_DIR"])
config.MODELS_DIR = pathlib.Path(os.environ["FEDHEART_MODELS_DIR"])
from fedheart.inference import HeartModel  # noqa: E402


def latest_run(suffix):
    runs = sorted(pathlib.Path(os.environ["FEDHEART_RUNS_DIR"]).glob(f"*-{suffix}"))
    return runs[-1]


class EndToEnd(unittest.TestCase):
    def check_run(self, run_dir, made_by):
        items = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))["items"]
        types = [e["type"] for e in items]
        self.assertEqual(items[-1], {**items[-1], "type": "job.status", "status": "succeeded"})
        self.assertIn("model.published", types)
        self.assertEqual([e["seq"] for e in items], list(range(1, len(items) + 1)))
        msgs = [e for e in items if e["type"] == "agent.message"]
        self.assertTrue(any(e["made_by"] == made_by for e in msgs))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        fin = summary["finished"]
        self.assertTrue(fin["every_hospital_holds_same_model"])
        self.assertGreater(fin["test_pooled"]["auc"], 0.75)
        # each hospital process stored the identical model it received
        hashes = {json.loads(p.read_text())["sha256"] for p in (run_dir / "sites").glob("*/received_model.json")}
        self.assertEqual(hashes, {fin["model_sha256"]})
        model = HeartModel.load(pathlib.Path(os.environ["FEDHEART_MODELS_DIR"]) / "heart_model.json")
        p = model.predict_proba({"age": 63, "sex": 1, "cp": 4, "chol": None})
        self.assertTrue(0.0 < p < 1.0)
        bench = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
        self.assertIn("centralized", bench)
        return items, summary

    def test_scripted_agents(self):
        run_simulation.main(["--agents", "rule", "--no-mirror", "--max-rounds", "8", "--base-port", "8301",
                             "--noise-seed", "1", "--outage", "switzerland:2"])
        items, summary = self.check_run(latest_run("rule"), "rule")
        r2 = next(r for r in summary["rounds"] if r["round"] == 2)
        self.assertIn("switzerland", r2["offline"])  # a simulated outage is handled, not fatal

    def test_deepseek_agent_loop_against_fake_api(self):
        with FakeDeepSeek() as fake:
            os.environ["DEEPSEEK_BASE_URL"] = f"http://127.0.0.1:{fake.port}"
            os.environ["DEEPSEEK_API_KEY"] = "sk-test-not-a-real-key"
            try:
                run_simulation.main(["--agents", "deepseek", "--no-mirror", "--max-rounds", "8", "--base-port",
                                     "8311", "--noise-seed", "1"])
            finally:
                os.environ.pop("DEEPSEEK_BASE_URL")
                os.environ.pop("DEEPSEEK_API_KEY")
            self.assertEqual(fake.errors, [], "an agent's tool call was rejected")
            reqs = [r for r in fake.requests if r["path"] == "/chat/completions"]
            self.assertGreater(len(reqs), 20)
            self.assertEqual({r["body"]["model"] for r in reqs}, {config.DEFAULT_MODEL})
            efforts = {r["body"]["reasoning_effort"] for r in reqs}
            self.assertEqual(efforts, {"high", "low"})  # coordinator high, hospitals low
            body = reqs[0]["body"]
            self.assertNotIn("tool_choice", body)  # a forced tool choice is a 400 in thinking mode
            self.assertEqual(body["tools"][0]["type"], "function")
            # append-only history (reasoning_content kept): every request extends the previous
            # one of the same agent, which is also what DeepSeek's context cache needs
            by_agent = {}
            for r in reqs:
                key = r["body"]["messages"][0]["content"][:80]
                prev = by_agent.get(key)
                if prev is not None:
                    self.assertEqual(r["body"]["messages"][:len(prev)], prev)
                by_agent[key] = r["body"]["messages"]
            self.assertEqual(len(by_agent), 5)  # four hospital agents + the coordinator
        items, summary = self.check_run(latest_run("deepseek"), "llm")
        usage = summary["usage"]["coordinator"]
        self.assertGreater(usage["cache_read_tokens"], 0)
        self.assertGreater(usage["estimated_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
