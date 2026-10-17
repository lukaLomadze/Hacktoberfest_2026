"""Coordinator agent for PrivateLearn (T8).

Runs the federated rounds against HospitalAgents (T7) over the T6 bus:

  1. Schema negotiation — collects every hospital's `schema` (names/units only),
     checks all six canonical features are covered, and sends one `schema_agreed`.
  2. Each round:
     a. `probe {round, w}` — hospitals score the global model on local data and
        reply with a clear-text loss/AUC. No reply = offline this round.
     b. Decide BEFORE anything is masked: model.aggregation.select_strategy on the
        probe losses picks FedAvg / FedProx / trimmed-mean and flags outliers.
        participants = responders − flagged; weights = n_i / Σn over participants.
     c. `train {round, w, participants, weights, strategy, mu}`.
     d. Sum the participants' (masked, n-weighted) updates. Pairwise masks cancel
        because the participant set was fixed in (b); the sum IS the aggregation
        step: FedAvg, FedProx's server step, or trimmed-mean with the outliers
        dropped. If a participant goes silent after (c) the masks cannot cancel, so
        the round is discarded — never applied half-masked.
     e. `broadcast` (CONTRACTS.md §3.2) with the new w, global AUC, ε spent.

On the synchronous bus a reply to a message sent at coordinator step k arrives at
step k+2, so missing replies are detected deterministically at that deadline.

The coordinator never sees a raw row or an individual unmasked update (when the
hospitals use model.privacy masking). `history` records each round in the trace
shape (CONTRACTS.md §5) for the end-to-end run (T9).
"""

from __future__ import annotations

import numpy as np

from agents.runtime import BROADCAST, Agent
from model.aggregation import FEDPROX_MU, select_strategy
from model.privacy import unmask_sum

CANONICAL = ["pregnancies", "glucose", "blood_pressure", "bmi", "insulin", "age"]
REPLY_DELAY = 2  # coordinator steps between sending a request and receiving replies


class CoordinatorAgent(Agent):
    def __init__(self, hospitals, eval_fn, feature_mean, feature_std, rounds: int = 4,
                 names=None, tau: float = 0.1, outlier_sigma: float = 2.0, mu: float = FEDPROX_MU,
                 w0=None, converge_tol: float = 1e-3, agent_id: str = "coord"):
        super().__init__(agent_id)
        self.hospitals = list(hospitals)
        self.names = dict(names or {h: h for h in self.hospitals})
        self.eval_fn = eval_fn  # w -> global AUC on the evaluation set the coordinator holds
        self.feature_mean = [float(v) for v in feature_mean]
        self.feature_std = [float(v) for v in feature_std]
        self.rounds = rounds
        self.tau = tau
        self.outlier_sigma = outlier_sigma
        self.mu = mu
        self.converge_tol = converge_tol
        self.w = np.zeros(7) if w0 is None else np.asarray(w0, dtype=float).copy()

        self.round = 0
        self.phase = "start"  # start → schema → probe → train → (probe …) → done
        self.steps = 0
        self.deadline = None
        self.inbox = []
        self.schemas = {}
        self.schema_report = None
        self.eps_spent = {h: 0.0 for h in self.hospitals}
        self.history = []
        self.auc_series = []
        self.loss_series = []
        self.done = False
        self._round = None
        self.n_seen = {}

    # ---------------------------------------------------------------- helpers

    def _wait(self, phase):
        self.phase = phase
        self.deadline = self.steps + REPLY_DELAY
        self.inbox = []

    def _names(self, ids):
        return ", ".join(self.names[h] for h in ids)

    def _eps_total(self):
        return round(max(self.eps_spent.values()), 6) if self.eps_spent else 0.0

    # ---------------------------------------------------------------- phases

    def _negotiate_schema(self):
        missing = [h for h in self.hospitals if h not in self.schemas]
        if missing:
            raise RuntimeError(f"schema negotiation: no schema from {missing}")
        sources = {}
        for h, s in self.schemas.items():
            covered = {c["canonical"] for c in s["columns"]}
            if covered != set(CANONICAL):
                raise ValueError(f"schema from {h} covers {sorted(covered)}, not the canonical six")
            sources[h] = {c["canonical"]: {"source": c["source"], "unit": c["unit"],
                                           "transform": c["transform"]} for c in s["columns"]}
        self.schema_report = {"features": CANONICAL, "per_hospital": sources}
        return {"features": CANONICAL, "feature_mean": self.feature_mean,
                "feature_std": self.feature_std}

    def _start_round(self):
        self.round += 1
        auc_before = float(self.eval_fn(self.w))
        if not self.auc_series:
            self.auc_series.append(auc_before)
        self._round = {"round": self.round, "auc_before": auc_before}
        self._wait("probe")
        return [self.send(BROADCAST, "probe", {"round": self.round, "w": [float(v) for v in self.w]})]

    def _decide(self):
        r = self._round
        probes = {m.payload["hospital"]: m.payload for m in self.inbox
                  if m.kind == "probe_result" and m.payload["round"] == self.round}
        responders = [h for h in self.hospitals if h in probes]
        offline = [h for h in self.hospitals if h not in probes]
        r["probes"] = probes
        r["offline"] = offline
        for h in responders:
            self.n_seen[h] = int(probes[h]["n"])
        if not responders:
            raise RuntimeError(f"round {self.round}: no hospital responded")

        losses = [probes[h]["loss"] for h in responders]
        dec = select_strategy(losses, tau=self.tau, outlier_sigma=self.outlier_sigma,
                              names=[self.names[h] for h in responders])
        flagged = [responders[i] for i in dec.flagged]
        participants = [h for h in responders if h not in flagged]
        total_n = sum(probes[h]["n"] for h in participants)
        weights = {h: probes[h]["n"] / total_n for h in participants}

        notes = []
        if offline:
            notes.append(f"{self._names(offline)} offline")
        notes.append(dec.reason)
        r.update(strategy=dec.strategy, decision="; ".join(notes), flagged=flagged,
                 participants=participants, flag_reason=dec.reason)
        self._wait("train")
        return [self.send(BROADCAST, "train", {
            "round": self.round, "w": [float(v) for v in self.w], "participants": participants,
            "weights": weights, "strategy": dec.strategy,
            "mu": self.mu if dec.strategy == "fedprox" else 0.0})]

    def _aggregate(self):
        r = self._round
        updates = {m.payload["hospital"]: m.payload for m in self.inbox
                   if m.kind == "update" and m.payload["round"] == self.round}
        stray = [h for h in updates if h not in r["participants"]]
        missing = [h for h in r["participants"] if h not in updates]
        if stray:
            raise ValueError(f"round {self.round}: updates from non-participants {stray}")

        if missing:
            # masks only cancel over the full participant set: never apply a broken sum
            applied = False
            r["decision"] += f"; {self._names(missing)} went silent after masking → round discarded"
        else:
            step = unmask_sum([updates[h]["delta_w"] for h in r["participants"]])
            self.w = self.w + step
            applied = True
            for h in r["participants"]:
                self.eps_spent[h] += float(updates[h]["privacy"]["dp_eps"])

        auc_after = float(self.eval_fn(self.w))
        part_losses = [float(updates[h]["loss"]) for h in r["participants"] if h in updates]
        mean_loss = float(np.mean(part_losses)) if part_losses else float("nan")
        prev = self.loss_series[-1] if self.loss_series else None
        converged = bool(self.round >= self.rounds or
                         (prev is not None and applied and part_losses
                          and abs(prev - mean_loss) < self.converge_tol))
        if part_losses:
            self.loss_series.append(mean_loss)
        self.auc_series.append(auc_after)

        broadcast = {"v": "1.0", "round": self.round, "w": [float(v) for v in self.w],
                     "strategy": r["strategy"], "eps_total": self._eps_total(), "auc": auc_after,
                     "losses": part_losses, "converged": converged}
        self.history.append(self._trace_round(r, updates, broadcast, applied))
        out = [self.send(BROADCAST, "broadcast", broadcast)]
        if self.round >= self.rounds:
            self.phase, self.done = "done", True
        else:
            out += self._start_round()
        return out

    def _trace_round(self, r, updates, broadcast, applied):
        def n_of(h):  # last n this hospital reported; an offline hospital is shown with it
            return int(self.n_seen.get(h, max(self.n_seen.values(), default=1)))

        rows = []
        for h in self.hospitals:
            if h in r["offline"]:
                rows.append({"hospital": h, "loss": None, "n": n_of(h), "sent": False,
                             "masked": False, "dp_eps": 0.0, "reason": "offline"})
            elif h in r["flagged"]:
                p = r["probes"][h]
                rows.append({"hospital": h, "loss": float(p["loss"]), "n": int(p["n"]), "sent": True,
                             "rejected": True, "reason": f"probe loss {p['loss']:.3f} is an outlier",
                             "masked": False, "dp_eps": 0.0})
            elif h in updates:
                u = updates[h]
                rows.append({"hospital": h, "loss": float(u["loss"]), "n": int(u["n"]), "sent": True,
                             "masked": bool(u["privacy"]["masked"]), "dp_eps": float(u["privacy"]["dp_eps"])})
            else:  # a participant that went silent after masking
                rows.append({"hospital": h, "loss": None, "n": n_of(h), "sent": False,
                             "masked": False, "dp_eps": 0.0, "reason": "silent after masking"})
        excluded = [h for h in self.hospitals if h in r["offline"] or h in r["flagged"]
                    or (h in r["participants"] and h not in updates)]
        return {"round": r["round"], "strategy": r["strategy"], "eps_total": broadcast["eps_total"],
                "auc_before": r["auc_before"], "auc_after": broadcast["auc"], "decision": r["decision"],
                "applied": applied, "excluded": excluded, "updates": rows, "broadcast": broadcast}

    # ---------------------------------------------------------------- bus entry point

    def step(self, messages):
        self.steps += 1
        self.inbox.extend(messages)
        if self.phase == "start":
            self._wait("schema")
            return [self.send(BROADCAST, "schema_request", {})]
        if self.phase == "done" or self.steps < (self.deadline or 0):
            return []
        if self.phase == "schema":
            for m in self.inbox:
                if m.kind == "schema":
                    self.schemas[m.payload["hospital"]] = m.payload
            agreed = self._negotiate_schema()
            return [self.send(BROADCAST, "schema_agreed", agreed)] + self._start_round()
        if self.phase == "probe":
            return self._decide()
        if self.phase == "train":
            return self._aggregate()
        raise RuntimeError(f"unknown phase {self.phase!r}")
