"""One hospital: its own OS process, its own data file, its own DeepSeek agent.

    python -m fedheart.hospital --site cleveland --port 8101 --run-dir runs/<id>

The process loads ONLY its site's file and keeps the rows in memory. It answers the
coordinator over HTTP with aggregates (statistics, metrics, histograms) and masked,
noised model updates. Its agent — a DeepSeek conversation that lasts the whole job — makes
the hospital's decisions through four tools:

    inspect_local_data       summary statistics of the local training split (never rows)
    submit_data_description  what each column means here and which values are broken
    review_contract          accept or object to the coordinator's feature contract
    submit_training_plan     participate this round or not; local epochs and learning rate

Everything numeric (CV, DP-GD training, masking, evaluation) is code, not the agent.
Endpoints: /hello /roster /describe /review /stats /norm /cv /evaluate /plan /train
/finalize /shutdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import secrets
import sys
import threading

import numpy as np

from . import config, data, features, llm, metrics, models, policies, privacy
from .transport import RemoteError, serve

SYSTEM = """You are the autonomous agent of {name} ({place}), one of four hospitals in a \
federated-learning job that trains ONE heart-disease risk model together with the other three \
hospitals and a coordinator agent. You act for your hospital and its patients.

What you can and cannot see
- Your hospital's patient rows never leave this hospital's process: not to the coordinator, not to \
the other hospitals, and not to you. You only see summary statistics through inspect_local_data, \
computed on your local TRAINING split (your validation and test splits stay untouched).
- Everything you write in notes, summary, objections or reasoning is sent to the coordinator and \
shown on the job's public log. Never write individual-level values there.

Your local export (UCI "processed" format, '?' = missing). Data dictionary:
{dictionary}
Label: {label}.

Your responsibilities during the job
1. Describe your data when asked: what each column means here, which recorded values are not \
physiologically possible and must be treated as missing, which columns are mostly missing, and \
anything unusual about your patient population. Use quality rules for broken values: \
zero_means_missing, or valid_range with min/max.
2. Review the coordinator's proposed feature contract for your hospital: accept it only if your \
hospital can provide the features meaningfully; otherwise object with concrete reasons (the \
coordinator may revise and ask again).
3. Every training round, decide your local plan: participate or not, local epochs (1-10) and \
learning rate (0.01-2.0).

How local training works (so you can reason about the trade-offs)
- You train from the current global model with DP-GD: full-batch gradient descent on your training \
split, every patient's gradient clipped to L2 norm {clip}, Gaussian noise added at every step.
- Your hospital's record-level privacy budget for the whole job is epsilon={epsilon} (delta={delta}), \
split evenly over at most {max_rounds} rounds. Every round you take part costs the same privacy no \
matter how many epochs you choose, so more epochs per round means more noise per step \
(sigma grows with sqrt(epochs)) and more drift away from the other hospitals. Declining a round \
spends no privacy, but your patients then do not shape that round's model.
- Features are z-scored with federated statistics; learning rates between about 0.05 and 1.0 are \
typical for this model size.
- Your update is weighted by your share of the training rows, masked with secrets shared only with the \
other hospitals, and summed; the coordinator never sees your individual update.

Style: decide, act through the tool, keep reasoning to one to three sentences."""


def _dictionary():
    lines = []
    for c in config.COLUMNS:
        lines.append(f"- {c['name']}: {c['meaning']} ({c['unit']})")
    return "\n".join(lines)


class Hospital:
    def __init__(self, site_id: str, run_dir: pathlib.Path, agents: str, model: str, effort: str,
                 seed: int, outage=(), noise_seed=None):
        self.site = config.SITE_BY_ID[site_id]
        self.node_id = self.site["node_id"]
        self.run_dir = run_dir
        self.agents_mode = agents
        self.llm_model = model
        self.effort = effort
        self.outage = set(outage)
        self.lock = threading.Lock()  # one request at a time: the agent is one conversation
        self.noise_rng = (np.random.default_rng(noise_seed) if noise_seed is not None
                          else np.random.default_rng(secrets.randbits(128)))

        table, y = data.load_site(site_id)  # the only place the file is read
        splits = data.stratified_split(y, data.site_seed("split", seed, site_id))
        self._split = {k: (data.subset(table, idx), y[idx]) for k, idx in splits.items()}
        self.n_total = int(len(y))

        self.dh_private, self.dh_public = privacy.dh_keypair()
        self.job = None
        self.keys = {}
        self.roster = []
        self.contract = None
        self.pending_contract = None
        self.norm = None
        self._X = {}
        self.input_names = []
        self.rho_total = None
        self.rho_round = None
        self.rho_spent = 0.0
        self.plans = {}
        # mask purposes already used: a second answer under the same purpose would let the
        # coordinator subtract two masked vectors (the masks cancel) and see an individual
        # update or partial sum, so every purpose is answered at most once
        self.used_purposes = set()
        self.history = []  # this hospital's own round history (its agent's memory aid)
        self.last_eval = {}
        self.agent = None
        self.phase = None
        self.result = None

    # ------------------------------------------------------------------ agent + tools

    def _make_agent(self):
        p = self.job["privacy"]
        system = SYSTEM.format(name=self.site["name"], place=self.site["place"], dictionary=_dictionary(),
                               label=config.LABEL["meaning"], clip=p["clip"], epsilon=p["epsilon"],
                               delta=p["delta"], max_rounds=p["max_rounds"])
        tb = llm.ToolBox()
        tb.add("inspect_local_data",
               "Summary statistics of this hospital's local training split. view='overview' gives every "
               "column's missing count, zero count, distinct values, min/median/max and the label rate; "
               "view='column' gives quantiles and value counts for one column; view='contract_impact' shows "
               "what the contract under review would mean for this hospital's rows.",
               {"view": {"type": "string", "enum": ["overview", "column", "contract_impact"]},
                "column": {"type": "string", "description": "column name, for view='column'"}},
               ["view"], self._tool_inspect)
        rule_schema = {"type": "object", "properties": {
            "column": {"type": "string"}, "rule": {"type": "string", "enum": list(features.RULES)},
            "min": {"type": "number"}, "max": {"type": "number"}}, "required": ["column", "rule"]}
        tb.add("submit_data_description",
               "Answer the coordinator's request to describe this hospital's data.",
               {"columns": {"type": "array", "items": {"type": "object", "properties": {
                   "name": {"type": "string"}, "usable": {"type": "boolean"},
                   "note": {"type": "string"}}, "required": ["name", "usable", "note"]}},
                "quality_rules": {"type": "array", "items": rule_schema},
                "summary": {"type": "string", "description": "two to four sentences for the coordinator"}},
               ["columns", "quality_rules", "summary"], self._tool_describe)
        tb.add("review_contract",
               "Accept or object to the feature contract the coordinator proposed.",
               {"accept": {"type": "boolean"},
                "objections": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"}},
               ["accept", "objections", "reasoning"], self._tool_review)
        tb.add("submit_training_plan",
               "Decide this hospital's part in the current round.",
               {"participate": {"type": "boolean"},
                "epochs": {"type": "integer", "minimum": 1, "maximum": 10},
                "lr": {"type": "number", "minimum": 0.01, "maximum": 2.0},
                "reasoning": {"type": "string"}},
               ["participate", "epochs", "lr", "reasoning"], self._tool_plan)
        transcript = self.run_dir / "transcripts" / f"{self.site['id']}.jsonl"
        if self.agents_mode == "deepseek":
            self.agent = llm.DeepSeekAgent(f"hospital:{self.site['id']}", system, tb, self.llm_model, self.effort,
                                         transcript)
        else:
            self.agent = llm.ScriptedAgent(f"hospital:{self.site['id']}", tb, policies.hospital_policy, transcript)

    def _need_phase(self, *phases):
        if self.phase not in phases:
            raise llm.ToolError(f"that tool is not for this step (current step: {self.phase})")

    def _tool_inspect(self, a):
        t, y = self._split["train"]
        view = a.get("view")
        if view == "overview":
            return features.overview(t, y)
        if view == "column":
            col = a.get("column")
            if col not in config.COLUMN_BY_NAME:
                raise llm.ToolError(f"unknown column {col!r}")
            return {"column": col, **config.COLUMN_BY_NAME[col], **features.column_summary(t[col], detail=True)}
        if view == "contract_impact":
            if self.pending_contract is None:
                raise llm.ToolError("no contract is under review right now")
            return features.contract_impact(t, self.pending_contract)
        raise llm.ToolError("view must be overview, column or contract_impact")

    def _tool_describe(self, a):
        self._need_phase("describe")
        rules = features.validate_contract({"features": [{"name": "age"}], "quality_rules": a.get("quality_rules")})
        cols = []
        for c in a.get("columns") or []:
            if c.get("name") not in config.COLUMN_BY_NAME:
                raise llm.ToolError(f"unknown column {c.get('name')!r}")
            cols.append({"name": c["name"], "usable": bool(c.get("usable")), "note": str(c.get("note", ""))[:300]})
        self.result = {"columns": cols, "quality_rules": rules["quality_rules"],
                       "summary": str(a.get("summary", ""))[:1200]}
        return {"ok": True, "sent_to_coordinator": True}

    def _tool_review(self, a):
        self._need_phase("review")
        self.result = {"accept": bool(a.get("accept")),
                       "objections": [str(o)[:400] for o in (a.get("objections") or [])][:8],
                       "reasoning": str(a.get("reasoning", ""))[:800]}
        return {"ok": True}

    def _tool_plan(self, a):
        self._need_phase("plan")
        epochs, lr = int(a.get("epochs", 0)), float(a.get("lr", 0))
        if not 1 <= epochs <= 10:
            raise llm.ToolError("epochs must be an integer from 1 to 10")
        if not (0.01 <= lr <= 2.0) or not np.isfinite(lr):
            raise llm.ToolError("lr must be between 0.01 and 2.0")
        participate = bool(a.get("participate"))
        if participate and self.rho_spent + self.rho_round > self.rho_total * (1 + 1e-9):
            raise llm.ToolError("your privacy budget is exhausted: you can only decline")
        self.result = {"participate": participate, "epochs": epochs, "lr": lr,
                       "reasoning": str(a.get("reasoning", ""))[:600]}
        return {"ok": True, "committed": participate}

    def _ask(self, phase, text, context):
        """Run the agent until it has called this step's submit tool."""
        self.phase, self.result = phase, None
        try:
            self.agent.run(text, done=lambda: self.result is not None, context=context)
        finally:
            self.phase = None
        return self.result

    def _msg(self, agent, text, data_=None):
        return {"agent": agent, "text": text, "made_by": self.agent.made_by if self.agent else "rule",
                "data": data_}

    # ------------------------------------------------------------------ endpoints

    def hello(self, _):
        return {"site": self.site["id"], "node_id": self.node_id, "name": self.site["name"],
                "place": self.site["place"], "rows": self.n_total, "dh_public": format(self.dh_public, "x"),
                "agent": {"provider": "deepseek" if self.agents_mode == "deepseek" else "rule",
                          "model": self.llm_model if self.agents_mode == "deepseek" else "scripted"}}

    def roster_(self, p):
        self.job = {"job_id": str(p["job_id"]), "privacy": p["privacy"]}
        self.roster = sorted(e["node_id"] for e in p["participants"])
        if self.node_id not in self.roster:
            raise RemoteError(409, "this hospital is not in the roster")
        self.keys = {e["node_id"]: privacy.pair_key(self.dh_private, int(e["dh_public"], 16), self.node_id,
                                                    e["node_id"])
                     for e in p["participants"] if e["node_id"] != self.node_id}
        pr = p["privacy"]
        self.rho_total = privacy.rho_from_epsilon(pr["epsilon"], pr["delta"]) if pr["epsilon"] > 0 else float("inf")
        self.rho_round = self.rho_total / pr["max_rounds"] if pr["epsilon"] > 0 else 0.0
        self._make_agent()
        # a fingerprint of the pair keys lets two hospitals check they derived the same secret
        fp = {peer: hashlib.sha256(k).hexdigest()[:12] for peer, k in self.keys.items()}
        return {"ok": True, "pair_key_fingerprints": fp,
                "agent_log": [self._msg("schema", f"derived {len(self.keys)} pairwise secrets by Diffie-Hellman; "
                                                  "the coordinator only relayed public keys")]}

    def describe(self, p):
        t, y = self._split["train"]
        text = ("STEP: describe your data.\nThe coordinator asks every hospital to describe its local export so "
                "the federation can agree which features to use. Inspect your data, then call "
                "submit_data_description.\nCoordinator's note: " + str(p.get("note", "")))
        res = self._ask("describe", text, {"step": "describe", "overview": features.overview(t, y)})
        return {"site": self.site["id"], "rows_train": int(len(y)), "description": res,
                "agent_log": [self._msg("schema", res["summary"], {"quality_rules": res["quality_rules"]})]}

    def review(self, p):
        contract = features.validate_contract(p["contract"])
        self.pending_contract = contract
        text = ("STEP: review the proposed feature contract.\n" + json.dumps(contract) +
                "\nCoordinator's reasoning: " + str(p.get("reasoning", "")) +
                "\nCheck what it means for your hospital (inspect_local_data view='contract_impact'), then call "
                "review_contract.")
        res = self._ask("review", text, {"step": "review", "contract": contract,
                                         "impact": features.contract_impact(self._split["train"][0], contract)})
        self.pending_contract = None
        verdict = "accepted the contract" if res["accept"] else "objected: " + "; ".join(res["objections"])
        return {"site": self.site["id"], **res,
                "agent_log": [self._msg("schema", f"{verdict} — {res['reasoning']}")]}

    def stats(self, p):
        contract = features.validate_contract(p["contract"])
        t, y = self._split["train"]
        vec = features.local_stats(t, y, contract)
        participants = sorted(p.get("participants") or self.roster)
        if self.node_id not in participants or any(x not in self.roster for x in participants):
            raise RemoteError(409, "invalid participant list for the statistics")
        purpose = f"{self.job['job_id']}|stats"
        if purpose in self.used_purposes:
            raise RemoteError(409, "the masked statistics were already sent for this job")
        masked = privacy.mask_vector(vec, self.node_id, participants, self.keys, purpose)
        self.used_purposes.add(purpose)
        return {"site": self.site["id"], "length": len(vec), "masked_stats": privacy.to_wire(masked),
                "agent_log": [self._msg("schema", f"sent {len(vec)} masked feature sums for the federated "
                                                  "mean/std")]}

    def norm_(self, p):
        self.contract = features.validate_contract(p["contract"])
        self.norm = p["norm"]
        for part, (t, _) in self._split.items():
            self._X[part] = features.encode(t, self.contract, self.norm)
        self.input_names = features.encoded_names(self.contract)
        return {"site": self.site["id"], "input_dim": len(self.input_names), "inputs": self.input_names,
                "agent_log": [self._msg("schema", f"encoded local rows to the agreed {len(self.input_names)} inputs")]}

    def cv(self, p):
        """k-fold cross-validation of each candidate on the local training split (no DP)."""
        X, y = self._X["train"], self._split["train"][1]
        rng =np.random.default_rng(data.site_seed("cvfolds", self.site["id"]))
        idx_pos, idx_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        k = 5
        fold_of = np.empty(len(y), dtype=int)
        fold_of[idx_pos] = np.arange(len(idx_pos)) % k
        fold_of[idx_neg] = np.arange(len(idx_neg)) % k
        out = []
        for spec in p["candidates"]:
            spec = models.validate_spec(spec)
            losses, aucs = [], []
            for f in range(k):
                tr, te = fold_of != f, fold_of == f
                m = models.make(spec, X.shape[1])
                w = models.fit(m, X[tr], y[tr], seed=data.site_seed("init", f))
                pr = m.predict_proba(w, X[te])
                losses.append(metrics.log_loss(y[te], pr))
                a = metrics.auc(y[te], pr)
                if a is not None:
                    aucs.append(a)
            out.append({"model": spec, "cv_loss": round(float(np.mean(losses)), 4),
                        "cv_auc": round(float(np.mean(aucs)), 4) if aucs else None, "folds": k})
        return {"site": self.site["id"], "rows_train": int(len(y)), "positives_train": int(y.sum()), "results": out,
                "agent_log": [self._msg("challenger", "local 5-fold CV: " + ", ".join(
                    f"{r['model']['type']} AUC {r['cv_auc']}" for r in out))]}

    def evaluate(self, p):
        r = int(p.get("round", 0))
        if r in self.outage:
            raise RemoteError(503, "hospital offline (simulated outage)")
        split = p.get("split", "val")
        if split != "val":
            raise RemoteError(403, "only the validation split can be evaluated during training")
        spec = models.validate_spec(p["model"])
        m = models.make(spec, len(self.input_names))
        w = np.asarray(p["params"], dtype=float)
        rep = metrics.site_report(self._split["val"][1], m.predict_proba(w, self._X["val"]))
        self.last_eval = {"round": r, "loss": rep["loss"], "auc": rep["auc"]}
        if self.history and self.history[-1]["round"] == r and "val_after" not in self.history[-1]:
            self.history[-1]["val_after"] = {"loss": rep["loss"], "auc": rep["auc"]}
        auc_txt = "n/a" if rep["auc"] is None else f"{rep['auc']:.3f}"
        return {"site": self.site["id"], "report": rep,
                "agent_log": [self._msg("challenger", f"global model on local validation: loss {rep['loss']:.3f}, "
                                                      f"AUC {auc_txt}")]}

    def plan(self, p):
        r = int(p["round"])
        if r in self.outage:
            raise RemoteError(503, "hospital offline (simulated outage)")
        eps_spent = privacy.epsilon_from_rho(self.rho_spent, self.job["privacy"]["delta"])
        ctx = {"round": r, "strategy": p.get("strategy"), "fedprox_mu": p.get("mu", 0.0),
               "coordinator_note": p.get("note", ""), "global_validation_before_round": p.get("global"),
               "your_validation_on_current_model": self.last_eval,
               "your_recent_rounds": self.history[-3:],
               "privacy": {"epsilon_spent": round(eps_spent, 4), "epsilon_budget": self.job["privacy"]["epsilon"],
                           "rounds_you_can_still_afford": (int(round((self.rho_total - self.rho_spent) / self.rho_round))
                                                           if self.rho_round else None)},
               "rows_train": int(len(self._split["train"][1]))}
        text = ("STEP: plan your part in training round " + str(r) + ".\n" + json.dumps(ctx) +
                "\nCall submit_training_plan.")
        res = self._ask("plan", text, {**ctx, "step": "plan"})
        self.plans[r] = res
        if res["participate"]:
            txt = f"{res['epochs']} ep @ lr {res['lr']:g} — {res['reasoning']}"
        else:
            txt = f"declined this round — {res['reasoning']}"
        return {"site": self.site["id"], "round": r, **res, "agent_log": [self._msg("planner", txt)]}

    def train(self, p):
        r = int(p["round"])
        plan = self.plans.get(r)
        if not plan or not plan["participate"]:
            raise RemoteError(409, f"no committed plan for round {r}")
        participants = sorted(p["participants"])
        if self.node_id not in participants:
            raise RemoteError(409, "not a participant this round")
        purpose = f"{self.job['job_id']}|round-{r}"
        if purpose in self.used_purposes:
            raise RemoteError(409, f"already trained and sent a masked update for round {r}")
        if self.rho_spent + self.rho_round > self.rho_total * (1 + 1e-9):
            raise RemoteError(409, "privacy budget exhausted")
        spec = models.validate_spec(p["model"])
        X, y = self._X["train"], self._split["train"][1]
        m = models.make(spec, X.shape[1])
        w0 = np.asarray(p["params"], dtype=float)
        n, clip = len(y), float(self.job["privacy"]["clip"])
        sigma = privacy.sigma_for(n, clip, plan["epochs"], self.rho_round) if self.rho_round else 0.0
        loss_before = m.loss(w0, X, y)
        w1, info = models.dp_train(m, w0, X, y, plan["epochs"], plan["lr"], clip, sigma, self.noise_rng,
                                   mu=float(p.get("mu", 0.0)), anchor=w0)
        loss_after = m.loss(w1, X, y)
        self.rho_spent += self.rho_round
        weight = float(p["weight"])
        masked = privacy.mask_vector(weight * (w1 - w0), self.node_id, participants, self.keys, purpose)
        self.used_purposes.add(purpose)
        eps_spent = privacy.epsilon_from_rho(self.rho_spent, self.job["privacy"]["delta"])
        self.history.append({"round": r, "plan": {"epochs": plan["epochs"], "lr": plan["lr"]},
                             "train_loss": [round(loss_before, 4), round(loss_after, 4)],
                             "noise_sigma": round(sigma, 4)})
        return {"site": self.site["id"], "round": r, "n": n, "masked_update": privacy.to_wire(masked),
                "train_loss_before": round(loss_before, 6), "train_loss_after": round(loss_after, 6),
                "privacy": {"sigma": round(sigma, 6), "clip": clip, "epsilon_spent": round(eps_spent, 6),
                            "rho_spent": self.rho_spent, "clipped_fraction": info["clipped_fraction"]},
                "agent_log": [self._msg("trainer", f"trained {plan['epochs']} DP-GD steps on {n} local rows "
                                                   f"(σ={sigma:.3f}), loss {loss_before:.3f}→{loss_after:.3f}; "
                                                   f"sent masked Δw · ε spent {eps_spent:.2f}")]}

    def finalize(self, p):
        spec = models.validate_spec(p["model"])
        m = models.make(spec, len(self.input_names))
        w = np.asarray(p["params"], dtype=float)
        rep = metrics.site_report(self._split["test"][1], m.predict_proba(w, self._X["test"]))
        digest = hashlib.sha256(json.dumps([round(float(v), 10) for v in w]).encode()).hexdigest()
        local = self.run_dir / "sites" / self.site["id"]
        local.mkdir(parents=True, exist_ok=True)
        (local / "received_model.json").write_text(json.dumps(
            {"model": spec, "params": [float(v) for v in w], "sha256": digest}), encoding="utf-8")
        auc_txt = "n/a" if rep["auc"] is None else f"{rep['auc']:.3f}"
        return {"site": self.site["id"], "report": rep, "model_sha256": digest,
                "epsilon_spent": round(privacy.epsilon_from_rho(self.rho_spent, self.job["privacy"]["delta"]), 6),
                "agent_log": [self._msg("challenger", f"final model on local TEST split: AUC {auc_txt}, "
                                                      f"accuracy {rep['accuracy']:.3f}; stored model "
                                                      f"{digest[:12]}")]}

    def usage(self, _):
        return {"site": self.site["id"], "usage": self.agent.usage.to_dict() if self.agent else None}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Run one hospital process")
    ap.add_argument("--site", required=True, choices=[s["id"] for s in config.SITES])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--agents", choices=["deepseek", "rule"], default="deepseek")
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--effort", default="low", choices=config.EFFORTS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outage", default="", help="comma-separated rounds in which this hospital is offline")
    ap.add_argument("--noise-seed", type=int, default=None, help="tests only: reproducible DP noise")
    args = ap.parse_args(argv)

    outage = [int(x) for x in args.outage.split(",") if x.strip()]
    h = Hospital(args.site, pathlib.Path(args.run_dir), args.agents, args.model, args.effort, args.seed, outage,
                 args.noise_seed)

    stop = threading.Event()

    def locked(fn):
        def wrapped(payload):
            with h.lock:
                return fn(payload)
        return wrapped

    routes = {"/hello": h.hello, "/roster": h.roster_, "/describe": h.describe, "/review": h.review,
              "/stats": h.stats, "/norm": h.norm_, "/cv": h.cv, "/evaluate": h.evaluate, "/plan": h.plan,
              "/train": h.train, "/finalize": h.finalize, "/usage": h.usage}
    handlers = {k: (v if k == "/hello" else locked(v)) for k, v in routes.items()}
    handlers["/shutdown"] = lambda _: (stop.set(), {"ok": True})[1]
    server = serve(args.port, handlers)
    print(f"[{args.site}] listening on 127.0.0.1:{args.port} · {h.n_total} local rows · agent "
          f"{args.model if args.agents == 'deepseek' else 'scripted'}", flush=True)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    stop.wait()
    server.shutdown()


if __name__ == "__main__":
    main()
