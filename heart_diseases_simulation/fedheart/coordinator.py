"""The coordinator: its own OS process, no data, one Claude agent that runs the job.

    python -m fedheart.coordinator --run-dir runs/<id> --hospitals cleveland=http://127.0.0.1:8101,...

The process first does the mechanical part (wait for the four hospitals, relay their
Diffie-Hellman public keys). Then its agent — one Claude conversation for the whole job —
decides everything else through tools:

    collect_data_descriptions  every hospital agent describes its data (in parallel)
    propose_contract           features + quality rules; hospitals review; on agreement the
                               federated mean/std is computed by secure aggregation
    evaluate_candidate_models  every hospital cross-validates candidate models locally
    select_model               fix the model family and initialise it
    run_round                  one federated round with the strategy the agent chose
    finish_training            the agent decides training is done: pick the checkpoint,
                               evaluate it on every hospital's test split, save and publish

The tools do the arithmetic (unmasking sums, applying the update, pooling metrics). The
coordinator never sees a patient row or one hospital's individual update.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from . import config, events, features, llm, metrics, models, policies, privacy
from .transport import RemoteError, Traffic, post

SYSTEM = """You are the coordinator agent of a federated-learning job. Four hospitals — {sites} — \
want ONE model that predicts angiographic heart disease (num > 0) well for patients at every one of \
them. None of them may share patient data, and you must never ask for it.

You never see a patient row, a prediction or one hospital's individual update. You see what the \
hospitals' own agents tell you, aggregate statistics, pooled metrics, and the SUM of the hospitals' \
masked updates. Every hospital is run by its own agent that can disagree with you, decline a round, \
or go offline.

How the job goes (each step is a tool; the tools do the arithmetic, you make the decisions)
1. collect_data_descriptions — each hospital agent describes its data and data-quality problems.
2. propose_contract — choose the features, their encoding, missing-value handling and quality rules \
for broken values. Every hospital reviews it; revise when they object with good reasons. A feature \
that is missing at most sites is noise for those sites; an indicator of missingness that is constant \
at one site can act as a proxy for that site (and its disease rate) rather than for the patient. When \
all accept, the federated mean/std is computed by secure aggregation.
3. evaluate_candidate_models — every hospital cross-validates the candidate models on its own \
training split (no privacy noise in that check). Then select_model. Remember that training will \
run with differential-privacy noise on every parameter: more parameters means more noise for the \
same privacy, and small hospitals overfit easily.
4. run_round, repeatedly. Each round you choose the aggregation rule:
   - fedavg: weighted average of the hospitals' updates (weights = share of training rows).
   - fedprox: the same average, but local training is pulled towards the current global model with \
strength mu (useful when hospitals pull in different directions, e.g. very different disease rates).
   - trimmed_mean: exclude named hospitals from this round, decided BEFORE training because \
masked updates can only be summed over a fixed set (use it for a hospital whose results look broken).
   - server_lr scales the aggregated step (1.0 = plain averaging).
   Each round report gives pooled validation metrics over all four hospitals, every hospital's own \
validation loss/AUC, the privacy each hospital has spent, and the best checkpoints so far.
5. finish_training — YOU decide when training is enough. There is no fixed number of rounds: \
minimum {min_rounds}, hard cap {max_rounds}. Every round costs each participating hospital part of \
its privacy budget (epsilon {epsilon} total each, delta {delta}), so stopping once the validation \
metrics stop improving protects patients; stopping too early leaves accuracy on the table. Small \
validation sets are noisy: judge trends over several rounds, the pooled loss (a proper scoring rule) \
as well as AUC, and the worst-off hospital, not one lucky number. You may finish from the best \
checkpoint rather than the last one. finish_training evaluates the chosen model once on every \
hospital's untouched test split, saves it and publishes it.

Write reasoning for the hospitals and the public log: concrete, one to three sentences, citing the \
numbers you used."""

KICKOFF = """The four hospital processes are online and have exchanged keys. Federation status:
{status}

Run the job now, step by step, until you have called finish_training."""


class Federation:
    def __init__(self, run_dir: pathlib.Path, urls: dict, args, log: events.EventLog):
        self.run_dir = run_dir
        self.urls = urls  # site id -> base url
        self.args = args
        self.log = log
        self.traffic = Traffic()
        self.pool_exec = cf.ThreadPoolExecutor(max_workers=max(4, len(urls)))
        self.nodes = {}
        self.active = list(urls)  # sites still in the job
        self.descriptions = None
        self.proposals = 0
        self.contract = None
        self.norm = None
        self.input_names = []
        self.cv = None
        self.model_spec = None
        self.model = None
        self.params = None
        self.round = 0
        self.checkpoints = {}
        self.val = {}  # round -> {"sites": {...}, "pooled": {...}}
        self.rounds = []
        self.participation = {s: 0 for s in urls}
        self.eps_spent = {s: 0.0 for s in urls}
        self.finished = None
        self.started = time.time()

    # ------------------------------------------------------------------ plumbing

    def node(self, site):
        return config.SITE_BY_ID[site]["node_id"]

    def call(self, site, path, payload=None, timeout=900.0):
        return post(self.urls[site], path, payload, timeout=timeout, traffic=self.traffic, peer=site)

    def call_all(self, sites, path, payload_for, timeout=900.0):
        """Call many hospitals in parallel → ({site: response}, {site: error message})."""
        futs = {self.pool_exec.submit(self.call, s, path, payload_for(s) if callable(payload_for) else payload_for,
                                      timeout): s for s in sites}
        ok, failed = {}, {}
        for f in cf.as_completed(futs):
            s = futs[f]
            try:
                ok[s] = f.result()
            except RemoteError as e:
                failed[s] = e.message
        return ok, failed

    def relay_agent_log(self, site, resp, round_):
        site_info = config.SITE_BY_ID[site]
        for m in resp.get("agent_log", []):
            self.log.emit("agent.message", round_, **{"from": {"node_id": site_info["node_id"], "agent": m["agent"],
                                                                "company_name": site_info["name"]}},
                          to="coord", text=m["text"], made_by=m.get("made_by", "rule"), data=m.get("data"))

    def say(self, text, round_=None, data=None, made_by=None):
        self.log.emit("agent.message", self.round if round_ is None else round_,
                      **{"from": {"node_id": "coord", "agent": "coordinator", "company_name": "Coordinator"}},
                      to="*", text=text, made_by=made_by or self.made_by, data=data)

    def state(self, site, state, detail="", round_=None):
        self.log.emit("participant.state", self.round if round_ is None else round_,
                      membership_id=f"mem_{site}", node_id=self.node(site), state=state, detail=detail)

    def packet(self, frm, to, kind, masked=False, rejected=False, round_=None):
        self.log.emit("packet", self.round if round_ is None else round_, **{"from": frm}, to=to, kind=kind,
                      masked=masked, rejected=rejected)

    made_by = "llm"

    # ------------------------------------------------------------------ mechanical start

    def connect(self, wait_s=120):
        self.log.emit("job.status", 0, status="running", error=None)
        self.log.update_job(status="running", step="waiting_for_nodes", started_at=events.now_iso())
        self.log.emit("job.step", 0, step="waiting_for_nodes")
        deadline = time.time() + wait_s
        for site in self.urls:
            while True:
                try:
                    self.nodes[site] = self.call(site, "/hello", timeout=5)
                    break
                except RemoteError:
                    if time.time() > deadline:
                        raise RuntimeError(f"hospital {site} did not come online") from None
                    time.sleep(0.5)
            n = self.nodes[site]
            self.state(site, "online", f"{n['rows']} local rows · agent {n['agent']['model']}", 0)

    def key_exchange(self):
        self.log.update_job(step="key_exchange")
        self.log.emit("job.step", 0, step="key_exchange")
        for site in self.urls:
            self.packet(self.node(site), "coord", "key", round_=0)
        roster = [{"node_id": n["node_id"], "dh_public": n["dh_public"]} for n in self.nodes.values()]
        privacy_cfg = {"epsilon": self.args.epsilon, "delta": self.args.delta, "max_rounds": self.args.max_rounds,
                       "clip": self.args.clip}
        ok, failed = self.call_all(self.urls, "/roster", {"job_id": f"{events.JOB_ID}-{self.run_dir.name}",
                                                          "participants": roster, "privacy": privacy_cfg})
        if failed:
            raise RuntimeError(f"key exchange failed: {failed}")
        # each pair derived its secret independently; matching fingerprints prove they agree
        for a in ok:
            for b in ok:
                if a < b:
                    fa = ok[a]["pair_key_fingerprints"][self.node(b)]
                    fb = ok[b]["pair_key_fingerprints"][self.node(a)]
                    if fa != fb:
                        raise RuntimeError(f"pair key mismatch between {a} and {b}")
        for site in self.urls:
            self.packet("coord", self.node(site), "key", round_=0)
            self.relay_agent_log(site, ok[site], 0)
        self.say(f"relayed {len(roster)} Diffie-Hellman public keys; all {len(roster) * (len(roster) - 1) // 2} "
                 "hospital pairs derived matching secrets (fingerprints checked). I cannot compute them.", 0,
                 made_by="rule")

    def status_text(self):
        lines = []
        for site in self.urls:
            n = self.nodes[site]
            lines.append(f"- {site}: {n['name']} ({n['place']}), {n['rows']} patients in total, agent "
                         f"{n['agent']['model']}")
        lines.append(f"Each hospital keeps 60% of its patients for training, 20% for validation and 20% for a "
                     f"final test you can use once (in finish_training).")
        lines.append(f"Privacy: epsilon {self.args.epsilon} per hospital for the whole job (delta {self.args.delta}),"
                     f" per-example clip {self.args.clip}; rounds: min {self.args.min_rounds}, "
                     f"max {self.args.max_rounds}.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ agent tools

    def tool_collect(self, a):
        if self.descriptions is not None:
            return {"already_collected": True, "descriptions": self.descriptions}
        self.log.update_job(step="schema_negotiation")
        self.log.emit("job.step", 0, step="schema_negotiation")
        self.say("asked every hospital to describe its data: " + str(a.get("note", "")), 0)
        for s in self.active:
            self.packet("coord", self.node(s), "schema", round_=0)
        ok, failed = self.call_all(self.active, "/describe", {"note": str(a.get("note", ""))})
        if failed:
            raise llm.ToolError(f"hospitals failed to describe their data: {failed}")
        self.descriptions = {}
        for s in self.active:
            self.packet(self.node(s), "coord", "schema", round_=0)
            self.relay_agent_log(s, ok[s], 0)
            self.descriptions[s] = {"rows_train": ok[s]["rows_train"], **ok[s]["description"]}
        return {"descriptions": self.descriptions}

    def tool_propose(self, a):
        if self.descriptions is None:
            raise llm.ToolError("call collect_data_descriptions first")
        if self.contract is not None:
            raise llm.ToolError("the contract is already agreed")
        contract = features.validate_contract({"features": a.get("features"), "quality_rules": a.get("quality_rules")})
        self.proposals += 1
        reasoning = str(a.get("reasoning", ""))
        names = ", ".join(f["name"] + ("+missing flag" if f["missing"] != "impute_mean" else "")
                          for f in contract["features"])
        self.say(f"proposal {self.proposals}: {len(contract['features'])} features ({names}); "
                 f"{len(contract['quality_rules'])} quality rules — {reasoning}", 0, data=contract)
        ok, failed = self.call_all(self.active, "/review", {"contract": contract, "reasoning": reasoning})
        if failed:
            raise llm.ToolError(f"review failed at {failed}")
        reviews = {}
        for s in self.active:
            self.relay_agent_log(s, ok[s], 0)
            reviews[s] = {"accept": ok[s]["accept"], "objections": ok[s]["objections"],
                          "reasoning": ok[s]["reasoning"]}
        objecting = [s for s, r in reviews.items() if not r["accept"]]
        if objecting and not a.get("drop_objecting_hospitals"):
            return {"accepted": False, "reviews": reviews,
                    "hint": "revise the proposal and call propose_contract again, or set drop_objecting_hospitals "
                            "to continue without the hospitals that object"}
        if objecting:
            if len(self.active) - len(objecting) < 2:
                raise llm.ToolError("fewer than two hospitals would remain")
            for s in objecting:
                self.active.remove(s)
                self.state(s, "excluded", "left the job: did not accept the contract", 0)
            self.say(f"continuing without {', '.join(objecting)}, which did not accept the contract", 0)
        self.contract = contract
        norm = self._secure_stats()
        return {"accepted": True, "reviews": reviews, "participants": self.active,
                "federated_statistics": norm, "model_inputs": self.input_names}

    def _secure_stats(self):
        self.log.update_job(step="normalization")
        self.log.emit("job.step", 0, step="normalization")
        ok, failed = self.call_all(self.active, "/stats", {"contract": self.contract,
                                                           "participants": [self.node(s) for s in self.active]})
        if failed:
            raise llm.ToolError(f"secure statistics failed: {failed}")
        for s in self.active:
            self.packet(self.node(s), "coord", "stats", masked=True, round_=0)
            self.relay_agent_log(s, ok[s], 0)
        totals = privacy.unmask_sum([privacy.from_wire(ok[s]["masked_stats"]) for s in self.active])
        self.norm = features.finalize_norm(self.contract, totals)
        okn, failed = self.call_all(self.active, "/norm", {"contract": self.contract, "norm": self.norm})
        if failed:
            raise llm.ToolError(f"hospitals could not encode their data: {failed}")
        for s in self.active:
            self.packet("coord", self.node(s), "broadcast", round_=0)
            self.relay_agent_log(s, okn[s], 0)
        self.input_names = features.encoded_names(self.contract)
        self.say(f"contract agreed by {len(self.active)} hospitals; federated mean/std of "
                 f"{self.norm['rows']} training patients computed from masked sums (pooled disease rate "
                 f"{self.norm['label_positive_rate']:.2f}); model input has {len(self.input_names)} values", 0,
                 data={"features": [f["name"] for f in self.contract["features"]]})
        return {"rows": self.norm["rows"], "label_positive_rate": self.norm["label_positive_rate"],
                "features": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                             for k, v in self.norm["features"].items()}}

    def tool_cv(self, a):
        if self.contract is None:
            raise llm.ToolError("agree the contract first")
        cands = a.get("candidates") or []
        if not 1 <= len(cands) <= 6:
            raise llm.ToolError("give 1 to 6 candidate models")
        specs = [models.validate_spec(c) for c in cands]
        ok, failed = self.call_all(self.active, "/cv", {"candidates": specs})
        if failed:
            raise llm.ToolError(f"cross-validation failed at {failed}")
        out = []
        total = sum(ok[s]["rows_train"] for s in ok)
        for i, spec in enumerate(specs):
            per = {s: {"cv_auc": ok[s]["results"][i]["cv_auc"], "cv_loss": ok[s]["results"][i]["cv_loss"],
                       "train_rows": ok[s]["rows_train"],
                       "train_negatives": ok[s]["rows_train"] - ok[s]["positives_train"]} for s in ok}
            aucs = [(ok[s]["rows_train"], per[s]["cv_auc"]) for s in ok if per[s]["cv_auc"] is not None]
            out.append({"model": spec, "parameters": models.make(spec, len(self.input_names)).n_params,
                        "weighted_cv_auc": round(sum(n * v for n, v in aucs) / sum(n for n, _ in aucs), 4),
                        "weighted_cv_loss": round(sum(ok[s]["rows_train"] * per[s]["cv_loss"] for s in ok) / total, 4),
                        "per_hospital": per})
        for s in self.active:
            self.relay_agent_log(s, ok[s], 0)
        self.cv = out
        return {"candidates": out, "note": "local CV without DP noise; each hospital trained on its own rows only"}

    def tool_select(self, a):
        if self.contract is None:
            raise llm.ToolError("agree the contract first")
        if self.model_spec is not None:
            raise llm.ToolError("the model is already selected")
        spec = models.validate_spec(a)
        self.model_spec = spec
        self.model = models.make(spec, len(self.input_names))
        self.params = self.model.init(self.args.seed)
        self.checkpoints[0] = self.params.copy()
        detail = f"{spec['type']}" + (f" ({spec['hidden']} hidden units)" if spec.get("hidden") else "") + \
                 f", l2 {spec['l2']:g}, {self.model.n_params} parameters"
        self.say(f"selected {detail} — {a.get('reasoning', '')}", 0, data={"model": spec})
        self.log.update_job(step="training")
        self.log.emit("job.step", 0, step="training")
        self.val[0] = self._evaluate(0)
        return {"model": spec, "parameters": self.model.n_params,
                "initial_validation": self._val_summary(0)}

    def _evaluate(self, round_):
        payload = {"round": round_, "params": [float(v) for v in self.params], "model": self.model_spec,
                   "split": "val"}
        for s in self.active:
            self.packet("coord", self.node(s), "probe", round_=round_)
        ok, failed = self.call_all(self.active, "/evaluate", payload)
        for s in ok:
            self.relay_agent_log(s, ok[s], round_)
        for s in failed:
            self.state(s, "offline", f"no answer to the validation probe ({failed[s][:120]})", round_)
        reports = {s: ok[s]["report"] for s in ok}
        return {"sites": reports, "pooled": metrics.pool(reports), "offline": sorted(failed)}

    def _val_summary(self, r):
        v = self.val[r]
        return {"pooled": {k: v["pooled"].get(k) for k in ("n", "loss", "auc", "accuracy", "balanced_accuracy",
                                                           "worst_site_auc")},
                "per_hospital": {s: {"loss": rep["loss"], "auc": rep["auc"], "n": rep["n"],
                                     "positives": rep["positives"]} for s, rep in v["sites"].items()},
                "offline": v["offline"]}

    def _best(self):
        scored = [(r, v["pooled"]) for r, v in self.val.items()
                  if v["pooled"] and len(v["sites"]) == len(self.active)]
        if not scored:
            return {}
        by_loss = min(scored, key=lambda t: t[1]["loss"])
        by_auc = max(scored, key=lambda t: (t[1]["auc"] or 0))
        since = self.round - by_loss[0]
        return {"best_loss_round": by_loss[0], "best_loss": by_loss[1]["loss"], "best_auc_round": by_auc[0],
                "best_auc": by_auc[1]["auc"], "rounds_since_best_loss": since}

    def tool_round(self, a):
        if self.model is None:
            raise llm.ToolError("select a model first")
        if self.finished:
            raise llm.ToolError("training is finished")
        if self.round >= self.args.max_rounds:
            raise llm.ToolError(f"the hard cap of {self.args.max_rounds} rounds is reached: call finish_training")
        strategy = str(a.get("strategy", ""))
        if strategy not in ("fedavg", "fedprox", "trimmed_mean"):
            raise llm.ToolError("strategy must be fedavg, fedprox or trimmed_mean")
        exclude = [str(s) for s in (a.get("exclude") or [])]
        if any(s not in self.active for s in exclude):
            raise llm.ToolError(f"exclude must name active hospitals: {self.active}")
        if strategy == "trimmed_mean" and not exclude:
            raise llm.ToolError("trimmed_mean needs at least one hospital in exclude")
        if strategy != "trimmed_mean" and exclude:
            raise llm.ToolError("exclude is only used with trimmed_mean")
        mu = float(a.get("mu") or 0.0)
        if strategy == "fedprox" and not 0.001 <= mu <= 1.0:
            raise llm.ToolError("fedprox needs mu between 0.001 and 1.0")
        if strategy != "fedprox":
            mu = 0.0
        server_lr = float(a.get("server_lr") or 1.0)
        if not 0.1 <= server_lr <= 2.0:
            raise llm.ToolError("server_lr must be between 0.1 and 2.0")
        reasoning = str(a.get("reasoning", ""))

        self.round += 1
        r = self.round
        before = self.val[max(self.val)]["pooled"]
        candidates = [s for s in self.active if s not in exclude]
        for s in exclude:
            self.state(s, "excluded", "excluded by the coordinator before masking", r)

        # phase 1: every candidate's agent decides whether and how to train (commit)
        for s in candidates:
            self.state(s, "planning", "agent choosing its local plan", r)
        global_ctx = {k: before.get(k) for k in ("loss", "auc", "worst_site_auc")}
        ok, failed = self.call_all(candidates, "/plan", {"round": r, "strategy": strategy, "mu": mu,
                                                         "global": global_ctx, "note": reasoning})
        offline = sorted(failed)
        for s in offline:
            self.state(s, "offline", f"no answer to the plan request ({failed[s][:120]})", r)
        declined, plans = [], {}
        for s in candidates:
            if s in ok:
                self.relay_agent_log(s, ok[s], r)
                if ok[s]["participate"]:
                    plans[s] = {"epochs": ok[s]["epochs"], "lr": ok[s]["lr"]}
                else:
                    declined.append(s)
                    self.state(s, "waiting", "declined this round", r)
        participants = sorted(plans)
        self.log.emit("round.decision", r, strategy=strategy, participants=[self.node(s) for s in participants],
                      excluded=[self.node(s) for s in exclude], offline=[self.node(s) for s in offline],
                      declined=[self.node(s) for s in declined], reasoning=reasoning, decided_by=self.made_by)
        report = {"round": r, "strategy": strategy, "mu": mu, "server_lr": server_lr, "participants": participants,
                  "excluded": exclude, "offline": offline, "declined": declined, "plans": plans}
        applied = False
        if len(participants) < 2:
            report["applied"] = False
            report["note"] = "fewer than two hospitals committed, so nothing was trained this round"
        else:
            n_train = {s: self.descriptions[s]["rows_train"] for s in participants}
            total = sum(n_train.values())
            weights = {s: n_train[s] / total for s in participants}
            for s in participants:
                self.state(s, "training", f"{plans[s]['epochs']} DP-GD steps @ lr {plans[s]['lr']:g}", r)
            payload = lambda s: {"round": r, "params": [float(v) for v in self.params], "model": self.model_spec,
                                 "participants": [self.node(p) for p in participants], "weight": weights[s],
                                 "mu": mu}
            tok, tfailed = self.call_all(participants, "/train", payload)
            for s in participants:
                if s in tok:
                    self.relay_agent_log(s, tok[s], r)
                    self.state(s, "sent", "masked Δw sent", r)
                    self.packet(self.node(s), "coord", "update", masked=True, round_=r)
                    self.participation[s] += 1
                    self.eps_spent[s] = tok[s]["privacy"]["epsilon_spent"]
                else:
                    self.state(s, "offline", f"went silent after committing ({tfailed[s][:120]})", r)
                    self.packet(self.node(s), "coord", "update", masked=True, rejected=True, round_=r)
            report["train_loss"] = {s: [tok[s]["train_loss_before"], tok[s]["train_loss_after"]] for s in tok}
            report["noise_sigma"] = {s: tok[s]["privacy"]["sigma"] for s in tok}
            if tfailed:
                report["applied"] = False
                report["note"] = (f"{sorted(tfailed)} failed after committing; the masked sum cannot be unmasked "
                                  "without them, so this round was discarded (model unchanged)")
                self.say(report["note"], r)
            else:
                step = privacy.unmask_sum([privacy.from_wire(tok[s]["masked_update"]) for s in participants])
                self.params = self.params + server_lr * step
                report["update_norm"] = round(float(np.linalg.norm(server_lr * step)), 5)
                applied = True
                report["applied"] = True
        if applied:
            for s in self.active:
                self.packet("coord", self.node(s), "broadcast", round_=r)
            self.val[r] = self._evaluate(r)
        else:
            self.val[r] = self.val[max(k for k in self.val if k < r)]
        self.checkpoints[r] = self.params.copy()
        after = self.val[r]["pooled"]
        report["validation"] = self._val_summary(r)
        report["privacy_epsilon_spent"] = {s: round(v, 4) for s, v in self.eps_spent.items() if s in self.active}
        report["epsilon_budget"] = self.args.epsilon
        report["rounds_used"] = r
        report["rounds_left_before_cap"] = self.args.max_rounds - r
        report["best_so_far"] = self._best()
        self.rounds.append(report)
        auc_series = [self.val[k]["pooled"].get("auc") for k in sorted(self.val)]
        loss_series = [self.val[k]["pooled"].get("loss") for k in sorted(self.val)]
        eps_total = max(self.eps_spent.values()) if self.eps_spent else 0.0
        self.log.emit("round.completed", r, auc_before=before.get("auc"), auc_after=after.get("auc"),
                      loss=after.get("loss"), eps_total=round(eps_total, 4), applied=applied)
        self.log.update_job(round=r, metrics={"auc_series": auc_series, "loss_series": loss_series,
                                              "eps_total": round(eps_total, 4)})
        for s in participants:
            self.state(s, "waiting", "", r)
        return report

    def tool_finish(self, a):
        if self.model is None:
            raise llm.ToolError("select a model first")
        if self.finished:
            return {"already_finished": True, **self.finished}
        if self.round < self.args.min_rounds:
            raise llm.ToolError(f"run at least {self.args.min_rounds} rounds first (done: {self.round})")
        cp = a.get("checkpoint", "best_loss")
        best = self._best()
        if not best and cp in ("best_loss", "best_auc"):
            cp = "last"
        if cp == "best_loss":
            r = best["best_loss_round"]
        elif cp == "best_auc":
            r = best["best_auc_round"]
        elif cp == "last":
            r = self.round
        else:
            try:
                r = int(cp)
            except (TypeError, ValueError):
                raise llm.ToolError("checkpoint must be best_loss, best_auc, last or a round number") from None
        if r not in self.checkpoints or r == 0:
            raise llm.ToolError(f"no trained checkpoint for round {r}")
        reasoning = str(a.get("reasoning", ""))
        self.say(f"training is enough after {self.round} rounds; publishing the round-{r} checkpoint — {reasoning}",
                 self.round, data={"checkpoint_round": r})
        self.log.update_job(step="evaluating")
        self.log.emit("job.step", self.round, step="evaluating")
        params = self.checkpoints[r]
        ok, failed = self.call_all(self.active, "/finalize", {"params": [float(v) for v in params],
                                                              "model": self.model_spec, "round": r})
        for s in ok:
            self.relay_agent_log(s, ok[s], self.round)
        reports = {s: ok[s]["report"] for s in ok}
        pooled = metrics.pool(reports)
        digest = hashlib.sha256(json.dumps([round(float(v), 10) for v in params]).encode()).hexdigest()
        hashes = {s: ok[s]["model_sha256"] for s in ok}
        artifact = self._model_artifact(params, r, pooled, reports, digest, reasoning)
        events.write_json_atomic(self.run_dir / "model.json", artifact)
        self.finished = {"checkpoint_round": r, "rounds_run": self.round,
                         "test_pooled": {k: pooled.get(k) for k in ("n", "auc", "loss", "accuracy", "sensitivity",
                                                                     "specificity", "worst_site_auc")},
                         "test_per_hospital": {s: {"auc": rep["auc"], "accuracy": rep["accuracy"], "n": rep["n"]}
                                               for s, rep in reports.items()},
                         "model_sha256": digest, "every_hospital_holds_same_model": set(hashes.values()) == {digest},
                         "failed": failed, "saved_to": str(self.run_dir / "model.json")}
        return self.finished

    def _model_artifact(self, params, r, pooled, reports, digest, reasoning):
        return {"format": "fedheart-model/1", "created_at": events.now_iso(), "run_id": self.run_dir.name,
                "model": self.model_spec, "params": [float(v) for v in params], "sha256": digest,
                "inputs": self.input_names, "contract": self.contract,
                "normalization": {k: {kk: vv for kk, vv in v.items() if kk in ("mean", "std")}
                                  for k, v in self.norm["features"].items()},
                "label": config.LABEL,
                "training": {"checkpoint_round": r, "rounds_run": self.round, "participants": self.active,
                             "strategies": [x["strategy"] for x in self.rounds],
                             "privacy": {"epsilon_budget": self.args.epsilon, "delta": self.args.delta,
                                         "clip": self.args.clip, "epsilon_spent": self.eps_spent,
                                         "max_rounds": self.args.max_rounds},
                             "stop_reasoning": reasoning, "decided_by": self.made_by},
                "test_metrics": {"pooled": pooled,
                                 "per_hospital": {s: {k: v for k, v in rep.items() if k not in ("hist_pos", "hist_neg")}
                                                  for s, rep in reports.items()}},
                "validation_history": [{"round": k, **{kk: self.val[k]["pooled"].get(kk) for kk in ("loss", "auc")}}
                                       for k in sorted(self.val)]}


def build_tools(fed: Federation) -> llm.ToolBox:
    tb = llm.ToolBox()
    feat_item = {"type": "object", "properties": {
        "name": {"type": "string", "enum": config.FEATURE_NAMES},
        "encoding": {"type": "string", "enum": list(features.ENCODINGS)},
        "missing": {"type": "string", "enum": list(features.MISSING)}}, "required": ["name", "encoding", "missing"]}
    rule_item = {"type": "object", "properties": {
        "column": {"type": "string", "enum": config.FEATURE_NAMES},
        "rule": {"type": "string", "enum": list(features.RULES)},
        "min": {"type": "number"}, "max": {"type": "number"}}, "required": ["column", "rule"]}
    model_props = {"type": {"type": "string", "enum": list(config.SUPPORTED_MODELS)},
                   "l2": {"type": "number", "minimum": 0, "maximum": 10},
                   "hidden": {"type": "integer", "minimum": 2, "maximum": 64, "description": "mlp only"}}
    tb.add("collect_data_descriptions", "Ask every hospital's agent to describe its local data (in parallel).",
           {"note": {"type": "string", "description": "what you want them to pay attention to"}}, ["note"],
           fed.tool_collect)
    tb.add("propose_contract",
           "Propose the shared feature contract. Every hospital agent reviews it. If all accept, the federated "
           "statistics are computed by secure aggregation and returned.",
           {"features": {"type": "array", "items": feat_item}, "quality_rules": {"type": "array", "items": rule_item},
            "reasoning": {"type": "string"},
            "drop_objecting_hospitals": {"type": "boolean",
                                         "description": "continue without hospitals that object (they leave the job)"}},
           ["features", "quality_rules", "reasoning"], fed.tool_propose)
    tb.add("evaluate_candidate_models",
           "Every hospital cross-validates each candidate (5-fold, its own training split, no DP noise).",
           {"candidates": {"type": "array", "items": {"type": "object", "properties": model_props,
                                                      "required": ["type", "l2"]}}},
           ["candidates"], fed.tool_cv)
    tb.add("select_model", "Fix the model for training and initialise it. Returns the initial validation metrics.",
           {**model_props, "reasoning": {"type": "string"}}, ["type", "l2", "reasoning"], fed.tool_select)
    tb.add("run_round",
           "Run one federated round with the aggregation rule you choose. Returns the round report.",
           {"strategy": {"type": "string", "enum": ["fedavg", "fedprox", "trimmed_mean"]},
            "exclude": {"type": "array", "items": {"type": "string"}, "description": "trimmed_mean only: site ids"},
            "mu": {"type": "number", "description": "fedprox only, 0.001-1.0"},
            "server_lr": {"type": "number", "description": "0.1-2.0, default 1.0"},
            "reasoning": {"type": "string"}},
           ["strategy", "reasoning"], fed.tool_round)
    tb.add("finish_training",
           "Declare training finished: evaluate the chosen checkpoint on every hospital's test split, save and "
           "publish it.",
           {"checkpoint": {"type": "string", "description": "best_loss, best_auc, last, or a round number"},
            "reasoning": {"type": "string"}},
           ["checkpoint", "reasoning"], fed.tool_finish)
    return tb


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Run the coordinator process")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--hospitals", required=True, help="site=url,site=url,...")
    ap.add_argument("--agents", choices=["claude", "rule"], default="claude")
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=config.MAX_ROUNDS)
    ap.add_argument("--min-rounds", type=int, default=config.MIN_ROUNDS)
    ap.add_argument("--epsilon", type=float, default=config.EPSILON_TOTAL)
    ap.add_argument("--delta", type=float, default=config.DELTA)
    ap.add_argument("--clip", type=float, default=config.CLIP)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--mirror", default="", help="also write the hub mock bundle here")
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    urls = dict(item.split("=", 1) for item in args.hospitals.split(","))
    sites = [s for s in urls]
    job = {"id": events.JOB_ID, "group_id": events.GROUP_ID, "kind": "train", "status": "queued",
           "step": "waiting_for_nodes", "round": 0, "rounds": None, "parent_version_id": None,
           "participants": [{"membership_id": f"mem_{s}", "node_id": config.SITE_BY_ID[s]["node_id"],
                             "company_name": config.SITE_BY_ID[s]["name"], "state": "waiting",
                             "color": config.SITE_BY_ID[s]["color"]} for s in sites],
           "metrics": {"auc_series": [], "loss_series": [], "eps_total": 0}, "started_at": None,
           "finished_at": None, "error": None}
    group = {"id": events.GROUP_ID, "name": "Heart disease (real agent run)", "status": "training",
             "description": "Four UCI cohorts as four hospitals; every decision made by the agents live",
             "task_spec": {"question": "Does this patient have angiographic heart disease?",
                           "task_type": "binary_classification",
                           "label": {"name": "heart_disease", "classes": ["no", "yes"], "positive": "yes"},
                           "input_schema": [{"name": c["name"], "type": "number", "unit": c["unit"],
                                             "description": c["meaning"], "required": False} for c in config.COLUMNS],
                           "model": {"type": "chosen by the coordinator agent"},
                           "training": {"rounds": None, "max_rounds": args.max_rounds, "min_participants": 2,
                                        "aggregation": "chosen per round by the coordinator agent"},
                           "privacy": {"secure_aggregation": True,
                                       "dp": {"enabled": args.epsilon > 0, "epsilon_total": args.epsilon,
                                              "delta": args.delta}},
                           "metric": "auc"},
             "pricing": None, "member_count": len(sites), "current_version": None, "created_at": events.now_iso(),
             "updated_at": events.now_iso()}
    source = {"run_id": run_dir.name, "agents": args.agents,
              "coordinator_model": args.model if args.agents == "claude" else "scripted"}
    log = events.EventLog(run_dir, job, group, pathlib.Path(args.mirror) if args.mirror else None, source)
    fed = Federation(run_dir, urls, args, log)
    fed.made_by = "llm" if args.agents == "claude" else "rule"

    try:
        fed.connect()
        fed.key_exchange()
        tb = build_tools(fed)
        transcript = run_dir / "transcripts" / "coordinator.jsonl"
        system = SYSTEM.format(sites=", ".join(f"{s} ({config.SITE_BY_ID[s]['name']})" for s in sites),
                               min_rounds=args.min_rounds, max_rounds=args.max_rounds, epsilon=args.epsilon,
                               delta=args.delta)
        if args.agents == "claude":
            agent = llm.ClaudeAgent("coordinator", system, tb, args.model, args.effort, transcript)
        else:
            agent = llm.ScriptedAgent("coordinator", tb, policies.coordinator_policy, transcript)
        print(f"[coordinator] agent {'' if args.agents == 'claude' else 'scripted '}{args.model if args.agents == 'claude' else ''} "
              "is running the job", flush=True)
        agent.run(KICKOFF.format(status=fed.status_text()), done=lambda: fed.finished is not None,
                  max_turns=4 * args.max_rounds + 40, context={"fed": fed})
        usage = {"coordinator": agent.usage.to_dict()}
        for s in sites:
            try:
                usage[s] = post(urls[s], "/usage", {})["usage"]
            except RemoteError:
                usage[s] = None
        events.write_json_atomic(run_dir / "usage.json", usage)
        fin = fed.finished
        version = {"id": "mv_heart", "group_id": events.GROUP_ID, "version": 1, "parent_version_id": None,
                   "job_id": events.JOB_ID, "model_type": fed.model_spec["type"],
                   "features": [f["name"] for f in fed.contract["features"]],
                   "metrics": {"auc": fin["test_pooled"]["auc"],
                               "auc_by_site": {s: v["auc"] for s, v in fin["test_per_hospital"].items()}},
                   "contributors": [{"company_id": s, "rows": fed.descriptions[s]["rows_train"],
                                     "rounds": fed.participation[s]} for s in fed.active],
                   "sha256": fin["model_sha256"], "created_at": events.now_iso()}
        log.set_version(version)
        log.emit("job.step", fed.round, step="published")
        log.emit("model.published", fed.round, version_id="mv_heart", version=1,
                 metrics={"auc": fin["test_pooled"]["auc"], "checkpoint_round": fin["checkpoint_round"],
                          "auc_by_site": version["metrics"]["auc_by_site"]})
        for s in fed.active:
            fed.state(s, "done", "holds the published model", fed.round)
        group.update(status="trained", current_version={"id": "mv_heart", "version": 1}, updated_at=events.now_iso())
        log.update_job(status="succeeded", step="published", rounds=fed.round, finished_at=events.now_iso())
        log.emit("job.status", fed.round, status="succeeded", error=None)
        events.write_json_atomic(run_dir / "summary.json", {
            "finished": fin, "rounds": fed.rounds, "contract": fed.contract, "cv": fed.cv, "model": fed.model_spec,
            "descriptions": fed.descriptions, "traffic": fed.traffic.by_peer, "usage": usage,
            "runtime_s": round(time.time() - fed.started, 1)})
        print(f"[coordinator] finished after {fed.round} rounds; pooled test AUC {fin['test_pooled']['auc']}",
              flush=True)
    except Exception as e:
        log.update_job(status="failed", error=f"{type(e).__name__}: {e}", finished_at=events.now_iso())
        log.emit("job.status", fed.round, status="failed", error=f"{type(e).__name__}: {e}")
        print(f"[coordinator] FAILED: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        fed.pool_exec.shutdown(wait=False)


if __name__ == "__main__":
    main()
