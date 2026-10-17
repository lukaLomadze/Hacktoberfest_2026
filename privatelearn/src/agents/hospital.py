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

CANONICAL = ["pregnancies", "glucose", "blood_pressure", "bmi", "insulin", "age"]
CANONICAL_UNITS = {"pregnancies": "count", "glucose": "mg/dL", "blood_pressure": "mmHg",
                   "bmi": "kg/m^2", "insulin": "uU/mL", "age": "years"}
LABEL_ALIASES = {"outcome", "label", "diabetic", "diabetes"}

# Unit factors (CONTRACTS.md §4): canonical = local × factor.
MGDL_PER_MMOLL = 18.0182
MMHG_PER_KPA = 7.50062
PMOLL_PER_UUML = 6.945


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Recognisers: normalised local column name → (canonical, unit, transform, factor).
# The Schema agent matches columns against these, so a new hospital with these
# naming conventions maps without code changes.
ALIASES = {
    "pregnancies": ("pregnancies", "count", "rename", 1.0),
    "numpregnancies": ("pregnancies", "count", "rename", 1.0),
    "glucose": ("glucose", "mg/dL", "rename", 1.0),
    "glucosemgdl": ("glucose", "mg/dL", "rename", 1.0),
    "glucosemmoll": ("glucose", "mmol/L", "x * 18.0182", MGDL_PER_MMOLL),
    "bloodpressure": ("blood_pressure", "mmHg", "rename", 1.0),
    "bpsys": ("blood_pressure", "mmHg", "rename", 1.0),
    "bpmmhg": ("blood_pressure", "mmHg", "rename", 1.0),
    "bpkpa": ("blood_pressure", "kPa", "x * 7.50062", MMHG_PER_KPA),
    "bmi": ("bmi", "kg/m^2", "rename", 1.0),
    "insulin": ("insulin", "uU/mL", "rename", 1.0),
    "insulinpmoll": ("insulin", "pmol/L", "x / 6.945", 1.0 / PMOLL_PER_UUML),
    "age": ("age", "years", "rename", 1.0),
    "ageyears": ("age", "years", "rename", 1.0),
    "agemonths": ("age", "months", "x / 12", 1.0 / 12.0),
}
HEIGHT_ALIASES = {"heightm"}
WEIGHT_ALIASES = {"weightkg"}


def load_local_csv(path):
    """Read a hospital CSV → (header, rows as dicts of floats). Stays inside the hospital."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [{k: float(v) for k, v in r.items()} for r in reader]
        return list(reader.fieldnames), rows


class SchemaAgent:
    """Maps a hospital's local columns to the canonical schema (names + units)."""

    def __init__(self, columns):
        self.columns = list(columns)
        self.mapping, self.label = self._infer()

    def _infer(self):
        by_norm = {_norm(c): c for c in self.columns}
        mapping = {}
        label = None
        for n, col in by_norm.items():
            if n in LABEL_ALIASES:
                label = col
            elif n in ALIASES:
                canonical, unit, transform, factor = ALIASES[n]
                if canonical in mapping:
                    raise ValueError(f"columns {mapping[canonical]['source']} and {[col]} both map to {canonical}")
                mapping[canonical] = {"canonical": canonical, "source": [col], "unit": unit,
                                      "transform": transform, "factor": factor}
        if "bmi" not in mapping:
            h = [by_norm[n] for n in by_norm if n in HEIGHT_ALIASES]
            w = [by_norm[n] for n in by_norm if n in WEIGHT_ALIASES]
            if h and w:
                mapping["bmi"] = {"canonical": "bmi", "source": [w[0], h[0]], "unit": "kg, m",
                                  "transform": "weight_kg / height_m^2", "factor": None}
        missing = [c for c in CANONICAL if c not in mapping]
        if missing:
            raise ValueError(f"schema: cannot map canonical feature(s) {missing} from columns {self.columns}")
        if label is None:
            raise ValueError(f"schema: no label column among {self.columns}")
        return mapping, label

    def describe(self, hospital: str) -> dict:
        """The schema message: names, units and conversions only — never values."""
        return {
            "hospital": hospital,
            "label": self.label,
            "columns": [{k: self.mapping[c][k] for k in ("canonical", "source", "unit", "transform")}
                        for c in CANONICAL],
        }

    def to_canonical(self, rows, order=CANONICAL):
        """Local rows → (X canonical units in `order`, labels)."""
        X = np.empty((len(rows), len(order)))
        for j, c in enumerate(order):
            m = self.mapping[c]
            if c == "bmi" and m["factor"] is None:
                wcol, hcol = m["source"]
                X[:, j] = [r[wcol] / r[hcol] ** 2 for r in rows]
            else:
                X[:, j] = [r[m["source"][0]] * m["factor"] for r in rows]
        labels = np.array([r[self.label] for r in rows], dtype=float)
        return X, labels


class PlannerAgent:
    """Schedules local training. Contract defaults: 4 full-batch epochs at lr 0.1."""

    def __init__(self, epochs: int = 4, lr: float = 0.1):
        self.epochs = epochs
        self.lr = lr

    def plan(self, strategy: str, mu: float) -> dict:
        mu = float(mu) if strategy == "fedprox" else 0.0
        why = f"{self.epochs} epochs @ lr {self.lr}" + (f", FedProx μ={mu:g}" if mu else "")
        return {"epochs": self.epochs, "lr": self.lr, "mu": mu, "why": why}


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
                 protector=None, train_fn=None, faults=None, planner=None):
        super().__init__(hospital_id)
        self.index = int(hospital_id.lstrip("h"))
        self._rows = rows  # local patient rows: never put on the bus
        self.schema = SchemaAgent(columns)
        self.planner = planner or PlannerAgent()
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
        order = p.get("features", CANONICAL)
        if sorted(order) != sorted(CANONICAL):
            raise ValueError(f"{self.id}: agreed features {order} are not the canonical six")
        if "feature_mean" in p:
            self._mean = np.asarray(p["feature_mean"], dtype=float)
            self._std = np.asarray(p["feature_std"], dtype=float)
        if self._mean is None or self._std is None:
            raise ValueError(f"{self.id}: no normalisation statistics (feature_mean/feature_std)")
        X, labels = self.schema.to_canonical(self._rows, order)
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
        if self.id not in p["participants"] or self.faults.get(r) == "offline":
            self._log(r, "planner", "not a participant this round")
            return []
        X, labels = self._data(r)
        plan = self.planner.plan(p.get("strategy", "fedavg"), p.get("mu", 0.0))
        self._log(r, "planner", plan["why"])
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
