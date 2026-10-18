"""Scripted stand-ins for the DeepSeek agents (`--agents rule`).

They drive exactly the same tools as the DeepSeek agents and see only the tool results, so
the whole multi-process pipeline can be tested in CI without an API key. Every event of
such a run says made_by="rule". They are deliberately simple, fixed rules — the DeepSeek
agents are what the demo is about.
"""

from __future__ import annotations

# columns where a recorded 0 is physiologically impossible (the hospital rule's domain list)
IMPOSSIBLE_ZERO = ("trestbps", "chol", "thalach")


def hospital_policy(agent, ctx):
    step = ctx.get("step")
    if step == "describe":
        ov = agent.call("inspect_local_data", {"view": "overview"})
        cols, rules = [], []
        for name, s in ov["columns"].items():
            if name in IMPOSSIBLE_ZERO and s["zeros"] > 0:
                rules.append({"column": name, "rule": "zero_means_missing"})
            usable = s["missing_rate"] < 0.5 and not (name in IMPOSSIBLE_ZERO and s["zeros"] == ov["rows"])
            cols.append({"name": name, "usable": usable,
                         "note": f"{s['missing_rate']:.0%} missing, {s['zeros']} zeros"})
        agent.call("submit_data_description", {
            "columns": cols, "quality_rules": rules,
            "summary": f"{ov['rows']} training rows, disease rate {ov['label_positive_rate']:.2f}; "
                       f"rule-based description (no LLM)."})
    elif step == "review":
        agent.call("review_contract", {"accept": True, "objections": [],
                                       "reasoning": "rule: accept every valid contract"})
    elif step == "plan":
        afford = (ctx.get("privacy") or {}).get("rounds_you_can_still_afford")
        agent.call("submit_training_plan", {"participate": afford is None or afford > 0, "epochs": 5, "lr": 0.5,
                                            "reasoning": "rule: fixed plan, 5 epochs at lr 0.5"})
    else:
        raise ValueError(f"unknown step {step!r}")


def coordinator_policy(agent, ctx, patience: int = 5):
    desc = agent.call("collect_data_descriptions", {"note": "describe columns and broken values"})["descriptions"]
    sites = list(desc)
    usable = {}
    rules = {}
    for d in desc.values():
        for c in d["columns"]:
            usable[c["name"]] = usable.get(c["name"], 0) + (1 if c["usable"] else 0)
        for r in d["quality_rules"]:
            rules[(r["column"], r["rule"])] = r
    from . import config

    feats = [{"name": c["name"], "encoding": c["kind"], "missing": "impute_mean"}
             for c in config.COLUMNS if usable.get(c["name"], 0) >= len(sites) - 1]
    res = agent.call("propose_contract", {"features": feats, "quality_rules": list(rules.values()),
                                          "reasoning": "rule: columns usable at all but at most one hospital",
                                          "drop_objecting_hospitals": True})
    cv = agent.call("evaluate_candidate_models", {"candidates": [{"type": "logistic", "l2": 0.01},
                                                                 {"type": "mlp", "hidden": 8, "l2": 0.01}]})
    best = max(cv["candidates"], key=lambda c: c["weighted_cv_auc"])
    if best["model"]["type"] != "logistic":
        logit = next(c for c in cv["candidates"] if c["model"]["type"] == "logistic")
        if best["weighted_cv_auc"] - logit["weighted_cv_auc"] < 0.005:
            best = logit
    agent.call("select_model", {**best["model"], "reasoning": "rule: best weighted CV AUC (ties → logistic)"})
    while True:
        rep = agent.call("run_round", {"strategy": "fedavg", "reasoning": "rule: always FedAvg"})
        b = rep["best_so_far"]
        if rep["rounds_left_before_cap"] <= 0 or (rep["rounds_used"] >= 3 and b.get("rounds_since_best_loss", 0) >= patience):
            break
    agent.call("finish_training", {"checkpoint": "best_loss",
                                   "reasoning": f"rule: no pooled validation-loss improvement for {patience} rounds"})
    return res
