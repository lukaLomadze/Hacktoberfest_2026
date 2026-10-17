"""LLM decision layer for PrivateLearn's agents.

Historically the agents decided with fixed rules: an alias table for schemas, fixed
hyperparameters, and a rule-based aggregation selector. This module lets the same
agents ask a DeepSeek chat model to make the decision and explain it, with a
deterministic fallback so tests, CI and offline runs never need a key.

Where the LLM is used (decision points only):
  Schema agent    map the hospital's local column NAMES to the canonical schema
  Planner agent   choose local training hyperparameters for this round
  Coordinator     choose the aggregation strategy from the probe losses

Where it is never used: gradients, aggregation, masking, DP, AUC — an LLM doing
arithmetic is unauditable. It decides; the deterministic code computes.

Privacy: the LLM sees only column names and aggregate numbers (losses). It is
never sent a patient row. The prompts below are built from metadata alone.

Replay model: the LLM runs at TRACE-GENERATION time. Its choice and reasoning are
written into trace.json, and the player replays them without calling the model.
So a recorded demo stays byte-stable, offline and free.

Provider: DeepSeek, OpenAI-compatible chat completions. Set DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import json
import os
import urllib.request

from model.aggregation import STRATEGIES, select_strategy

CANONICAL_UNITS = {
    "pregnancies": "count",
    "glucose": "mg/dL",
    "blood_pressure": "mmHg",
    "bmi": "kg/m^2",
    "insulin": "uU/mL",
    "age": "years",
}
CANONICAL = list(CANONICAL_UNITS)

SYSTEM_STRATEGY = (
    "You are the Coordinator agent of a federated learning run over three hospitals. "
    "You see only each hospital's probe loss on the current global model, never any data. "
    "Choose the aggregation strategy for this round. Reply with a single JSON object with "
    'keys "strategy" (one of fedavg, fedprox, trimmed_mean), "excluded" (hospital labels to '
    'exclude, only for trimmed_mean) and "reasoning" (one sentence).'
)
SYSTEM_SCHEMA = (
    "You are the Schema agent of one hospital in a federated learning federation. You see only "
    "this hospital's local column NAMES and units, never patient values. Map them to the "
    "federation's canonical schema. Reply with a single JSON object with keys "
    '"label" (the outcome column name) and "mapping" (for every canonical feature: '
    '{"source": [column names], "factor": number to multiply local values by to reach the '
    'canonical unit, "unit": string, "transform": short description}) and "reasoning" '
    "(one sentence). BMI may be derived from weight_kg and height_m with factor null."
)
SYSTEM_PLAN = (
    "You are the Planner agent of one hospital in a federated learning run. You see the round "
    'number, the aggregation strategy and the last loss. Reply with a single JSON object with '
    'keys "epochs" (integer 1-20), "lr" (float 0.001-1.0), "mu" (float >= 0, only for fedprox) '
    'and "reasoning" (one sentence).'
)


class RuleDecider:
    """Deterministic: exactly the rules the project used before. made_by='rule'."""

    made_by = "rule"

    def schema(self, columns, fallback):
        mapping, label = fallback()
        return {"mapping": mapping, "label": label,
                "reasoning": "matched the known column aliases", "made_by": "rule"}

    def plan(self, context, fallback):
        d = fallback()
        return {**d, "reasoning": d.get("why", ""), "made_by": "rule"}

    def strategy(self, losses, names, fallback, **kw):
        d = fallback()
        return {"strategy": d.strategy, "flagged": list(d.flagged),
                "reasoning": d.reason, "made_by": "rule"}

    def info(self):
        return {"provider": "rule", "model": "deterministic", "llm_calls": 0, "fallbacks": 0}


class LLMDecider:
    """DeepSeek-backed decisions, with a deterministic fallback on any failure."""

    made_by = "llm"

    def __init__(self, api_key=None, model="deepseek-chat",
                 endpoint="https://api.deepseek.com/chat/completions", timeout=30):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.calls = 0
        self.fallbacks = 0
        self.errors = []
        self._round = None

    # -- transport ---------------------------------------------------------

    def _ask(self, system, user):
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        return json.loads(text)

    def _fallback(self, exc, fb):
        self.fallbacks += 1
        self.errors.append(f"{type(exc).__name__}: {exc}"[:200])
        return fb()

    # -- decisions ---------------------------------------------------------

    def strategy(self, losses, names, fallback, **kw):
        try:
            ctx = {
                "round": kw.get("round"),
                "probe_losses": {n: round(float(v), 3) for n, v in zip(names, losses)},
                "candidates": list(STRATEGIES),
                "guidance": "fedavg when they agree; trimmed_mean when exactly one is an outlier "
                            "(exclude it); fedprox when they disagree broadly.",
            }
            out = self._ask(SYSTEM_STRATEGY, json.dumps(ctx))
            strat = str(out.get("strategy", "")).strip()
            if strat not in STRATEGIES:
                raise ValueError(f"bad strategy {strat!r}")
            excluded = out.get("excluded") or []
            flagged = [names.index(n) for n in excluded if n in names]
            if strat == "trimmed_mean" and not flagged:
                raise ValueError("trimmed_mean chosen with no excluded hospital")
            if strat != "trimmed_mean":
                flagged = []
            self.calls += 1
            return {"strategy": strat, "flagged": flagged,
                    "reasoning": str(out.get("reasoning", "")).strip(), "made_by": "llm"}
        except Exception as exc:  # noqa: BLE001 - any failure must fall back, never abort a run
            d = self._fallback(exc, fallback)
            return {"strategy": d.strategy, "flagged": list(d.flagged),
                    "reasoning": d.reason, "made_by": "rule"}

    def schema(self, columns, fallback):
        try:
            ctx = {"local_columns": list(columns), "canonical": CANONICAL_UNITS,
                   "note": "local values may need a unit conversion; BMI may be weight/height^2"}
            out = self._ask(SYSTEM_SCHEMA, json.dumps(ctx))
            mapping = {}
            for c in CANONICAL:
                e = out["mapping"][c]
                source = e["source"] if isinstance(e["source"], list) else [e["source"]]
                if not source or any(s not in columns for s in source):
                    raise ValueError(f"{c}: source {source} not among {list(columns)}")
                factor = e.get("factor")
                if factor is not None:
                    factor = float(factor)
                mapping[c] = {"canonical": c, "source": [str(s) for s in source],
                              "unit": str(e.get("unit", CANONICAL_UNITS[c])),
                              "transform": str(e.get("transform", "rename")), "factor": factor}
            label = str(out["label"])
            if label not in columns:
                raise ValueError(f"label {label!r} not among {list(columns)}")
            self.calls += 1
            return {"mapping": mapping, "label": label,
                    "reasoning": str(out.get("reasoning", "")).strip(), "made_by": "llm"}
        except Exception as exc:  # noqa: BLE001
            mapping, label = self._fallback(exc, fallback)
            return {"mapping": mapping, "label": label,
                    "reasoning": "matched the known column aliases", "made_by": "rule"}

    def plan(self, context, fallback):
        try:
            out = self._ask(SYSTEM_PLAN, json.dumps(context))
            epochs = int(out["epochs"])
            lr = float(out["lr"])
            mu = float(out.get("mu", 0.0) or 0.0)
            if not 1 <= epochs <= 20:
                raise ValueError(f"epochs {epochs} out of range")
            if not 0.001 <= lr <= 1.0:
                raise ValueError(f"lr {lr} out of range")
            if mu < 0:
                raise ValueError(f"mu {mu} negative")
            self.calls += 1
            return {"epochs": epochs, "lr": lr, "mu": mu,
                    "why": str(out.get("reasoning", "")).strip(), "made_by": "llm"}
        except Exception as exc:  # noqa: BLE001
            d = self._fallback(exc, fallback)
            return {**d, "made_by": "rule"}

    def info(self):
        return {"provider": "deepseek", "model": self.model,
                "llm_calls": self.calls, "fallbacks": self.fallbacks,
                "recent_errors": self.errors[-3:]}


def build_decider(use_llm: bool, model: str = "deepseek-chat"):
    """RuleDecider unless the caller explicitly asks for the LLM and a key is present."""
    if use_llm:
        return LLMDecider(model=model)
    return RuleDecider()
