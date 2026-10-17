"""Hospital agents for PrivateLearn (T7).

One `HospitalAgent` is one hospital on the bus (id "h0", "h1", "h2"). Inside it a
small team of agents works on the hospital's own rows, which never leave the
object — only schema descriptions, losses/AUCs and (protected) weight deltas do:

- SchemaAgent     describes the local columns and maps them to the canonical
                  6-feature schema (names + unit conversions, CONTRACTS.md §4).
- PlannerAgent    schedules local training (epochs, lr, FedProx μ) per round.
- TrainerAgent    trains from the global model and emits delta_w + loss + n.
- ChallengerAgent scores a model on local data (loss + AUC) — used both to probe
                  the global model before a round and to check every broadcast.

Protocol with the coordinator (kinds are free-form except update/broadcast):

    coord → hospital  schema_request {}
    hospital → coord  schema         {hospital, label, columns:[{canonical, source, unit, transform}]}
    coord → hospital  schema_agreed  {features:[...], feature_mean?, feature_std?}
    coord → hospital  probe          {round, w}
    hospital → coord  probe_result   {round, hospital, loss, auc, n}
    coord → hospital  train          {round, w, participants, weights, strategy, mu}
    hospital → coord  update         CONTRACTS.md §3.1 (delta_w = weight·Δw, protected)
    coord → hospital  broadcast      CONTRACTS.md §3.2

The coordinator fixes `participants` from the clear-text probe losses BEFORE any
masking, so pairwise masks cancel over exactly that set. Each participant sends
`weights[hospital]·Δw` (weights = n_i/Σn), so the sum of updates is the FedAvg step.

Privacy is injected: `protector(delta, hospital_index, participants, round)` returns
(vector_to_send, privacy_dict). The default sends the vector unprotected and says
so; the end-to-end run (T9) plugs in model.privacy's clip → DP noise → masking.
Training is injected too (`train_fn`, default model.logistic.train), so DP training
with per-example clipping can be swapped in.
"""

from __future__ import annotations

import csv
import re

import numpy as np

from agents.runtime import Agent
from model import logistic

# Generic, domain-neutral target-column detection for the offline (no-LLM) fallback.
GENERIC_LABELS = {"outcome", "label", "target", "y", "class", "default", "response"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_local_csv(path):
    """Read a hospital CSV → (header, rows as dicts of floats). Stays inside the hospital."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [{k: float(v) for k, v in r.items()} for r in reader]
        return list(reader.fieldnames), rows


def column_profile(columns, rows):
    """Summary statistics per column, computed on the hospital's own rows.

    A column named "height" could be metres or inches; only the distribution can
    tell. This profile is what the schema agent may send to the model — statistics,
    never an individual row. (For stricter privacy, make these DP statistics.)
    """
    n = len(rows)
    prof = {}
    if not n:
        return prof
    for c in columns:
        vals = sorted(float(r[c]) for r in rows)

        def q(p):
            i = min(n - 1, max(0, int(round(p * (n - 1)))))
            return round(vals[i], 4)

        prof[c] = {"min": round(vals[0], 4), "p05": q(0.05), "median": q(0.5),
                   "mean": round(sum(vals) / n, 4), "p95": q(0.95),
                   "max": round(vals[-1], 4)}
    return prof


class SchemaAgent:
    """Describes a hospital's local columns so the federation can agree a schema.

    With an LLM, every column gets a *semantic* name, its unit and a factor to a
    standard unit — decided from the column name and its summary statistics, so it
    works for any table (medical, financial, …), not just the demo's six columns.
    Without an LLM it falls back to the offline alias table.
    """

    def __init__(self, columns, decider=None, profile=None):
        self.columns = list(columns)
        self.decider = decider
        self.profile = profile or {}
        d = decider.describe(self.columns, self._infer, self.profile) if decider is not None else None
        if d is None:
            d = self._infer()
        self.label = d["label"]
        self.described = d["columns"]
        self.reasoning = d.get("reasoning", "")
        self.made_by = d.get("made_by", "rule")

    def _infer(self):
        """Offline fallback: no domain knowledge. Each column keeps its normalised name; the
        LLM path is what actually infers semantics, units and conversions."""
        label = None
        cols = []
        for c in self.columns:
            if _norm(c) in GENERIC_LABELS:
                label = c
            else:
                cols.append({"source": [c], "semantic": _norm(c), "unit": "unknown", "factor": 1.0})
        if label is None:
            label = self.columns[-1]
        return {"label": label, "columns": cols,
                "reasoning": "offline: normalised column names, no unit inference", "made_by": "rule"}

    def describe(self, hospital: str) -> dict:
        """The schema message: names, units and conversions only — never values."""
        return {"hospital": hospital, "label": self.label, "columns": self.described}

    def to_canonical(self, rows, mapping, order):
        """Local rows → (X in the agreed `order`, labels) using the agreed per-hospital mapping."""
        X = np.empty((len(rows), len(order)))
        for j, f in enumerate(order):
            m = mapping[f]
            if m.get("factor") is None:  # derived feature, e.g. BMI from weight/height
                w, h = m["source"]
                X[:, j] = [r[w] / r[h] ** 2 for r in rows]
            else:
                X[:, j] = [r[m["source"][0]] * m["factor"] for r in rows]
        labels = np.array([r[self.label] for r in rows], dtype=float)
        return X, labels


class PlannerAgent:
    """Schedules local training. Contract defaults: 4 full-batch epochs at lr 0.1."""

    def __init__(self, epochs: int = 4, lr: float = 0.1, decider=None):
        self.epochs = epochs
        self.lr = lr
        self.decider = decider

    def plan(self, strategy: str, mu: float, context=None) -> dict:
        mu = float(mu) if strategy == "fedprox" else 0.0
        why = f"{self.epochs} epochs @ lr {self.lr}" + (f", FedProx μ={mu:g}" if mu else "")
        base = {"epochs": self.epochs, "lr": self.lr, "mu": mu, "why": why}
        if self.decider is None:
            return {**base, "made_by": "rule"}
        return self.decider.plan({"strategy": strategy, **(context or {})}, lambda: base)


class TrainerAgent:
    """Trains from the global model on local rows; returns Δw and the local loss."""

    def __init__(self, train_fn=None):
        self.train_fn = train_fn or logistic.train

    def train(self, w_global, X, labels, plan):
        w_global = np.asarray(w_global, dtype=float)
        kwargs = {"epochs": plan["epochs"], "lr": plan["lr"]}
        if plan["mu"]:
            kwargs.update(mu=plan["mu"], w_anchor=w_global)
        w_new = self.train_fn(w_global, X, labels, **kwargs)
        return w_new - w_global, logistic.loss(w_new, X, labels)


class ChallengerAgent:
    """Scores a model against this hospital's own rows."""

    def score(self, w, X, labels) -> dict:
        w = np.asarray(w, dtype=float)
        p = logistic.predict_proba(w, X)
        return {"loss": logistic.loss(w, X, labels), "auc": logistic.auc(labels, p)}


def no_protection(delta, hospital_index, participants, round_):
    """Default protector: send the update as is, and report that honestly."""
    return np.asarray(delta, dtype=float), {
        "masked": False, "mask_seed": "", "dp_eps": 0.0, "dp_mechanism": "gaussian", "dp_sigma": 0.0}


class HospitalAgent(Agent):
    """A hospital on the bus. Raw rows are private attributes and never sent."""

    def __init__(self, hospital_id: str, columns, rows, feature_mean=None, feature_std=None,
                 protector=None, train_fn=None, faults=None, planner=None, decider=None):
        super().__init__(hospital_id)
        self.index = int(hospital_id.lstrip("h"))
        self._rows = rows  # local patient rows: never put on the bus
        self.schema = SchemaAgent(columns, decider, column_profile(columns, rows))
        self.planner = planner or PlannerAgent(decider=decider)
        self.trainer = TrainerAgent(train_fn)
        self.challenger = ChallengerAgent()
        self.protector = protector or no_protection
        self.faults = dict(faults or {})  # {round: "offline" | "flip_labels" | "age_in_months"}
        self._mean = None if feature_mean is None else np.asarray(feature_mean, dtype=float)
        self._std = None if feature_std is None else np.asarray(feature_std, dtype=float)
        self._X = None
        self._labels = None
        self._order = None
        self.w_global = None
        self.activity = []  # [{round, agent, action}] for the trace; no patient values

    @classmethod
    def from_csv(cls, hospital_id: str, path, **kwargs):
        columns, rows = load_local_csv(path)
        return cls(hospital_id, columns, rows, **kwargs)

    @property
    def n(self) -> int:
        return len(self._rows)

    def _log(self, round_, agent, action):
        self.activity.append({"round": round_, "agent": agent, "action": action})

    def _data(self, round_):
        """This round's (X, labels). Simulated faults corrupt them the way real exports do."""
        if self._X is None:
            raise RuntimeError(f"{self.id}: no agreed schema yet")
        X, labels = self._X, self._labels
        fault = self.faults.get(round_)
        if fault == "flip_labels":
            labels = 1.0 - labels
        elif fault == "age_in_months":
            # the export silently switches the age column to months, same column name
            j = self._order.index("age")
            X = X.copy()
            X[:, j] = (X[:, j] * self._std[j] + self._mean[j]) * 12.0
            X[:, j] = (X[:, j] - self._mean[j]) / self._std[j]
        return X, labels

    # ---- message handlers -------------------------------------------------------

    def _on_schema_request(self, msg):
        self._log(0, "schema", f"described {len(self.schema.columns)} local columns")
        return [self.send(msg.sender, "schema", self.schema.describe(self.id))]

    def _on_schema_agreed(self, msg):
        p = msg.payload
        order = list(p.get("features") or [])
        mine = (p.get("columns") or {}).get(self.id)
        if mine is None:
            # the coordinator did not send a per-hospital mapping (older messages): use this
            # hospital's own description keyed by semantic name
            mine = {c["semantic"]: {"source": c["source"], "factor": c["factor"]}
                    for c in self.schema.described}
        if not order or any(f not in mine for f in order):
            raise ValueError(f"{self.id}: agreed features {order} are not provided by this hospital")
        if "feature_mean" in p:
            self._mean = np.asarray(p["feature_mean"], dtype=float)
            self._std = np.asarray(p["feature_std"], dtype=float)
        if self._mean is None or self._std is None:
            raise ValueError(f"{self.id}: no normalisation statistics (feature_mean/feature_std)")
        X, labels = self.schema.to_canonical(self._rows, mine, order)
        self._order = list(order)
        self._X = (X - self._mean) / self._std
        self._labels = labels
        self._log(0, "schema", f"mapped to the agreed {len(order)}-feature schema")
        return []

    def _on_probe(self, msg):
        r = msg.payload["round"]
        if self.faults.get(r) == "offline":
            self._log(r, "planner", "offline this round")
            return []
        X, labels = self._data(r)
        s = self.challenger.score(msg.payload["w"], X, labels)
        self._log(r, "challenger", f"global model on local data: loss {s['loss']:.3f}, AUC {s['auc']:.3f}")
        return [self.send(msg.sender, "probe_result",
                          {"round": r, "hospital": self.id, "loss": s["loss"], "auc": s["auc"], "n": self.n})]

    def _on_train(self, msg):
        p = msg.payload
        r = p["round"]
        if self.faults.get(r) == "offline":
            return []  # an offline hospital never sees the train message (already logged at probe)
        if self.id not in p["participants"]:
            self._log(r, "planner", "not a participant this round")
            return []
        X, labels = self._data(r)
        plan = self.planner.plan(p.get("strategy", "fedavg"), p.get("mu", 0.0),
                                 context={"round": r, "last_loss": p.get("last_loss")})
        self._log(r, "planner", f"{plan['epochs']} ep @ lr {plan['lr']:g}"
                                 + (f", mu={plan['mu']:g}" if plan["mu"] else "")
                                 + f" — {plan['why']}")
        delta, local_loss = self.trainer.train(p["w"], X, labels, plan)
        weight = float(p["weights"][self.id])
        participants = sorted(int(h.lstrip("h")) for h in p["participants"])
        vec, privacy = self.protector(weight * delta, self.index, participants, r)
        self._log(r, "trainer", f"trained on {self.n} local rows, loss {local_loss:.3f}; "
                                f"sent {'masked ' if privacy['masked'] else ''}Δw")
        update = {"v": "1.0", "round": r, "hospital": self.id,
                  "delta_w": [float(v) for v in vec], "loss": float(local_loss), "n": self.n,
                  "privacy": privacy}
        return [self.send(msg.sender, "update", update)]

    def _on_broadcast(self, msg):
        p = msg.payload
        if self.faults.get(p["round"]) == "offline":
            return []  # offline: misses this broadcast, catches up from the next one
        self.w_global = list(p["w"])
        if self._X is not None:
            s = self.challenger.score(p["w"], self._X, self._labels)
            self._log(p["round"], "challenger", f"new global model: local AUC {s['auc']:.3f}")
        return []

    HANDLERS = {
        "schema_request": _on_schema_request,
        "schema_agreed": _on_schema_agreed,
        "probe": _on_probe,
        "train": _on_train,
        "broadcast": _on_broadcast,
    }

    def step(self, messages):
        out = []
        for m in messages:
            handler = self.HANDLERS.get(m.kind)
            if handler is None:
                raise ValueError(f"{self.id}: unexpected message kind {m.kind!r} from {m.sender}")
            out.extend(handler(self, m))
        return out
