"""Hospital agents for PrivateLearn (T7).

One `HospitalAgent` is one participant on the bus. Its id is any opaque string ("h0",
"node_9f3a", …) and is never parsed for meaning. Inside it a
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
    coord → hospital  schema_agreed  {features:[...], participants?, feature_mean?, feature_std?}
    coord → hospital  probe          {round, w}
    hospital → coord  probe_result   {round, hospital, loss, auc, n}
    coord → hospital  train          {round, w, participants, weights, strategy, mu}
    hospital → coord  update         CONTRACTS.md §3.1 (delta_w = weight·Δw, protected)
    coord → hospital  broadcast      CONTRACTS.md §3.2

The coordinator fixes `participants` from the clear-text probe losses BEFORE any
masking, so pairwise masks cancel over exactly that set. Each participant sends
`weights[hospital]·Δw` (weights = n_i/Σn), so the sum of updates is the FedAvg step.

Privacy is injected: `protector(delta, hospital_index, participants, round, plan=plan)`
returns (vector_to_send, privacy_dict). `plan` is the Planner's plan this update was
trained with ({epochs, lr, mu, ...}): a record's influence on Δw grows with epochs·lr, so
DP noise must be calibrated to the plan actually run, not to a constant. Mask indices are
positions in the sorted list of the federation's participants (`schema_agreed.participants`),
so they are the same on every participant and stable across rounds even when someone is
offline; this round's `participants` are passed as those positions. The default sends the vector unprotected and says
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
    Without an LLM it uses the manifest-provided local→canonical mapping when one is
    supplied (data-driven, works for any dataset), else normalised names only.
    """

    def __init__(self, columns, decider=None, profile=None, schema_map=None):
        self.columns = list(columns)
        self.decider = decider
        self.profile = profile or {}
        self.schema_map = schema_map  # local→canonical mapping from the manifest (offline fallback)
        d = decider.describe(self.columns, self._infer, self.profile) if decider is not None else None
        if d is None:
            d = self._infer()
        self.label = d["label"]
        self.described = d["columns"]
        self.reasoning = d.get("reasoning", "")
        self.made_by = d.get("made_by", "rule")

    def _infer(self):
        """Offline fallback: no domain knowledge. When the manifest supplies this client's
        local→canonical mapping, use it (data-driven, works for any dataset); otherwise each
        column keeps its normalised name and the LLM path is what actually infers semantics,
        units and conversions."""
        if self.schema_map is not None:
            sm = self.schema_map
            cols = []
            for f, m in sm["columns"].items():
                entry = {"source": list(m["source"]), "semantic": f,
                         "unit": m.get("unit", "unknown"), "factor": m.get("factor", 1.0)}
                if m.get("derive") is not None:
                    entry["derive"] = m["derive"]
                cols.append(entry)
            return {"label": sm["label"], "columns": cols,
                    "reasoning": "manifest-provided local schema", "made_by": "rule"}

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
        """Local rows → (X in the agreed `order`, labels) using the agreed per-hospital mapping.

        A mapping entry with `factor is None` is either a derived feature (two or more
        source columns; `derive` names the derivation, default "bmi" = a/b²) or — when an
        LLM used null to mean "no conversion / rename-only" — a single source column used
        as-is. A single source cannot be derived, so treat it as a direct copy (factor 1).
        """
        X = np.empty((len(rows), len(order)))
        for j, f in enumerate(order):
            m = mapping[f]
            if m.get("factor") is None:
                src = list(m["source"])
                if len(src) == 1:  # null factor, single source → rename-only, no conversion
                    X[:, j] = [r[src[0]] for r in rows]
                    continue
                derive = m.get("derive", "bmi")
                if derive == "bmi":
                    w, h = src
                    X[:, j] = [r[w] / r[h] ** 2 for r in rows]
                elif derive == "ratio":
                    a, b = src
                    X[:, j] = [r[a] / r[b] for r in rows]
                elif derive == "product":
                    a, b = src
                    X[:, j] = [r[a] * r[b] for r in rows]
                elif derive == "sum":
                    X[:, j] = [sum(r[s] for s in src) for r in rows]
                else:
                    raise ValueError(f"unknown derive {derive!r} for feature {f!r}")
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


def no_protection(delta, hospital_index, participants, round_, plan=None):
    """Default protector: send the update as is, and report that honestly."""
    return np.asarray(delta, dtype=float), {
        "masked": False, "mask_seed": "", "dp_eps": 0.0, "dp_mechanism": "gaussian", "dp_sigma": 0.0}


class HospitalAgent(Agent):
    """A hospital on the bus. Raw rows are private attributes and never sent."""

    def __init__(self, hospital_id: str, columns, rows, feature_mean=None, feature_std=None,
                 protector=None, train_fn=None, faults=None, planner=None, decider=None,
                 schema_map=None):
        super().__init__(hospital_id)
        self.roster = None  # every participant id of this federation, sorted (from schema_agreed)
        self._rows = rows  # local patient rows: never put on the bus
        self.schema = SchemaAgent(columns, decider, column_profile(columns, rows), schema_map=schema_map)
        self.planner = planner or PlannerAgent(decider=decider)
        self.trainer = TrainerAgent(train_fn)
        self.challenger = ChallengerAgent()
        self.protector = protector or no_protection
        # {round: "offline" | "flip_labels" | "age_in_months" | {"unit_swap": {feature, factor}}}
        # Round keys arrive as strings when faults are read from the manifest JSON.
        self.faults = {int(k): v for k, v in (faults or {}).items()}
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
    def index(self) -> int:
        """This participant's mask index: its position in the sorted participant list."""
        return self.mask_index(self.id)

    def mask_index(self, participant_id: str, roster=None) -> int:
        roster = roster or self.roster
        if roster is None:
            raise RuntimeError(f"{self.id}: participant list unknown before schema_agreed")
        if participant_id not in roster:
            raise ValueError(f"{self.id}: {participant_id!r} is not a participant of this federation")
        return roster.index(participant_id)

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
            # legacy alias: the export silently switches the age column to months, same column name
            X = self._unit_swap(X, "age", 12.0)
        elif isinstance(fault, dict) and "unit_swap" in fault:
            us = fault["unit_swap"]
            X = self._unit_swap(X, us["feature"], float(us["factor"]))
        return X, labels

    def _unit_swap(self, X, feature: str, factor: float):
        """A silent unit change: the export reports `feature` scaled by `factor`, same name.

        Returns a copy — the fault must affect only this round, never the hospital's stored rows.
        """
        j = self._order.index(feature)
        out = X.copy()
        out[:, j] = (out[:, j] * self._std[j] + self._mean[j]) * factor
        out[:, j] = (out[:, j] - self._mean[j]) / self._std[j]
        return out

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
        roster = p.get("participants")
        if roster is not None:
            roster = sorted(str(h) for h in roster)
            if self.id not in roster:
                raise ValueError(f"{self.id}: not in the agreed participant list {roster}")
            self.roster = roster
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
        # a coordinator that never sent the full list: this round's participants are all we
        # know, and they are the same on every participant, so the masks still cancel
        roster = self.roster or sorted(str(h) for h in p["participants"])
        participants = sorted(self.mask_index(h, roster) for h in p["participants"])
        vec, privacy = self.protector(weight * delta, self.mask_index(self.id, roster), participants, r,
                                       plan=plan)
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
