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
import urllib.error
import urllib.request

from model.aggregation import STRATEGIES, select_strategy

SYSTEM_STRATEGY = (
    "You are the Coordinator agent of a federated learning run over three hospitals. "
    "You see only each hospital's probe loss on the current global model, never any data. "
    "Choose the aggregation strategy for this round. Reply with a single JSON object with "
    'keys "strategy" (one of fedavg, fedprox, trimmed_mean), "excluded" (hospital labels to '
    'exclude, only for trimmed_mean) and "reasoning" (one sentence).'
)
SYSTEM_DESCRIBE = (
    "You are the Schema agent of one hospital in a federated learning federation. You see this "
    "hospital's local column names and, for each, summary statistics (min/p05/median/mean/p95/max). "
    "Never see or return patient values. For every column give: a semantic name that captures what "
    "it measures (a common, normalised name such as 'age', 'income', 'blood_pressure'), the unit "
    "the values are in, and a factor that converts the local values to a standard unit. Use 1.0 "
    "when the values are already in the standard unit or are only renamed; use null ONLY when the "
    "value is derived from two or more source columns (e.g. BMI from weight and height), never for "
    "a single column. Use the statistics to disambiguate units (a value around 1.7 is metres, "
    "around 70 is inches). Reply with a single JSON object "
    'with keys "label" (the outcome/target column name) and "columns" (a list of '
    '{"source": [names], "semantic": name, "unit": unit, "factor": number-or-null}) and '
    '"reasoning" (one sentence).'
)
SYSTEM_PLAN = (
    "You are the Planner agent of one hospital in a federated learning run. You see the round, "
    "the aggregation strategy and the previous mean loss. Choose local training hyperparameters "
    "that actually move the model; the validated default is 4 epochs at lr 0.1, so deviate only "
    "with a reason. Reply with a single JSON object with keys "
    '"epochs" (positive integer), "lr" (positive float), "mu" (float >= 0, used only for fedprox) '
    'and "reasoning" (one sentence).'
)
SUPPORTED_MODELS = ("logistic",)
SYSTEM_CONTRACT = (
    "You are the Coordinator agent of a federation. Each hospital described its local columns with "
    "a semantic name, unit and conversion factor. Reconcile them into one shared schema: group "
    "columns that measure the same thing across hospitals (their names and units differ) into "
    "features, and keep only the feature set EVERY hospital can provide. For each hospital, map each "
    "feature to its local source column(s) and the factor that converts them to the shared unit "
    "(use 1.0 for a rename with no unit change; use null only for a feature derived from two or "
    "more source columns). "
    'When the input includes "shared_features", those are the dataset\'s canonical feature names: '
    "use exactly those names for the shared features, mapping each hospital's described columns "
    "onto the matching canonical name (hospitals may name and unit the same thing differently). "
    'Reply with a single JSON object: {"features": [names], "model": a supported model, "columns": '
    '{hospital: {feature: {"source": [names], "factor": number-or-null}}}, "reasoning": one sentence}.'
)


class RuleDecider:
    """Deterministic: exactly the rules the project used before. made_by='rule'."""

    made_by = "rule"

    def describe(self, columns, fallback, profile=None):
        return fallback()

    def plan(self, context, fallback):
        d = fallback()
        return {**d, "reasoning": d.get("why", ""), "made_by": "rule"}

    def strategy(self, losses, names, fallback, **kw):
        d = fallback()
        return {"strategy": d.strategy, "flagged": list(d.flagged),
                "reasoning": d.reason, "made_by": "rule"}

    def contract(self, proposals, fallback, feature_names=None):
        """Agree the federation contract from every hospital's proposed feature names."""
        return fallback()

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
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        payload = None
        # Try JSON mode first; some providers reject response_format with a 400,
        # so retry once without it before giving up and falling back to rules.
        for json_mode in (True, False):
            body = {"model": self.model, "messages": messages,
                    "temperature": 0, "max_tokens": 500}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if json_mode and exc.code in (400, 404, 422):
                    continue
                raise
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

    def describe(self, columns, fallback, profile=None):
        try:
            out = self._ask(SYSTEM_DESCRIBE, json.dumps(
                {"local_columns": list(columns), "profile": profile}))
            label = str(out["label"])
            if label not in columns:
                raise ValueError(f"label {label!r} not among {list(columns)}")
            cols = []
            for e in out["columns"]:
                source = e["source"] if isinstance(e["source"], list) else [e["source"]]
                if not source or any(s not in columns for s in source):
                    raise ValueError(f"source {source} not among {list(columns)}")
                factor = e.get("factor")
                if factor is not None:
                    factor = float(factor)
                elif len(source) == 1:
                    # a null factor on a single source means "rename-only / no conversion"
                    factor = 1.0
                cols.append({"source": [str(s) for s in source],
                             "semantic": str(e["semantic"]),
                             "unit": str(e.get("unit", "")), "factor": factor})
            if not cols:
                raise ValueError("no columns described")
            self.calls += 1
            return {"label": label, "columns": cols,
                    "reasoning": str(out.get("reasoning", "")).strip(), "made_by": "llm"}
        except Exception as exc:  # noqa: BLE001
            d = self._fallback(exc, fallback)
            d["made_by"] = "rule"
            return d

    def plan(self, context, fallback):
        try:
            out = self._ask(SYSTEM_PLAN, json.dumps(context))
            epochs = int(out["epochs"])
            lr = float(out["lr"])
            mu = float(out.get("mu", 0.0) or 0.0)
            # Only structural sanity is enforced. No domain bounds: the agent decides the
            # hyperparameters and the trace records them; the machinery does not overrule it.
            if epochs <= 0:
                raise ValueError(f"epochs {epochs} must be positive")
            if lr <= 0:
                raise ValueError(f"lr {lr} must be positive")
            if mu < 0:
                raise ValueError(f"mu {mu} negative")
            self.calls += 1
            return {"epochs": epochs, "lr": lr, "mu": mu,
                    "why": str(out.get("reasoning", "")).strip(), "made_by": "llm"}
        except Exception as exc:  # noqa: BLE001
            d = self._fallback(exc, fallback)
            return {**d, "made_by": "rule"}

    def contract(self, proposals, fallback, feature_names=None):
        try:
            payload = {"hospitals": proposals, "supported_models": list(SUPPORTED_MODELS)}
            if feature_names:
                payload["shared_features"] = list(feature_names)
            out = self._ask(SYSTEM_CONTRACT, json.dumps(payload))
            feats = [str(f) for f in out["features"]]
            model = str(out.get("model", "logistic"))
            if model not in SUPPORTED_MODELS:
                raise ValueError(f"unsupported model {model!r}")
            cols = out["columns"]
            if not isinstance(cols, dict) or set(cols) != set(proposals):
                raise ValueError("columns must map every hospital")
            for h, m in cols.items():
                if not set(feats) <= set(m):
                    raise ValueError(f"hospital {h} is missing a feature")
            self.calls += 1
            return {"features": feats, "model": model, "columns": cols,
                    "reasoning": str(out.get("reasoning", "")).strip(), "made_by": "llm"}
        except Exception as exc:  # noqa: BLE001
            d = self._fallback(exc, fallback)
            d["made_by"] = "rule"
            return d

    def info(self):
        return {"provider": "deepseek", "model": self.model,
                "llm_calls": self.calls, "fallbacks": self.fallbacks,
                "recent_errors": self.errors[-3:]}


def build_decider(use_llm: bool, model: str = "deepseek-chat"):
    """RuleDecider unless the caller explicitly asks for the LLM and a key is present."""
    if use_llm:
        return LLMDecider(model=model)
    return RuleDecider()
