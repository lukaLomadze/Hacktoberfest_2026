"""Offline benchmark for the write-up: how good is the federated model, really?

This is the EXPERIMENTER's view, not part of the federation: it reads all four files to
train the comparisons that federation exists to avoid —
  solo         each hospital alone (its own training split, no privacy noise)
  centralized  all four training splits pooled in one place (illegal in practice; the ceiling)
  federated    the published model
— and scores every model on the same pooled test set (the four hospitals' test splits),
using the published model's contract and federated normalisation.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from . import config, data, features, metrics, models


def run(model_path, seed: int = config.SEED) -> dict:
    art = json.loads(pathlib.Path(model_path).read_text(encoding="utf-8"))
    contract, norm = art["contract"], {"features": art["normalization"]}
    sites = art["training"]["participants"]
    enc = {}
    for s in sites:
        table, y, splits = data.load_split_site(s, seed)
        enc[s] = {part: (features.encode(data.subset(table, idx), contract, norm), y[idx])
                  for part, idx in splits.items()}
    d = len(art["inputs"])
    spec = art["model"]

    def score(m, w):
        reps = {s: metrics.site_report(enc[s]["test"][1], m.predict_proba(w, enc[s]["test"][0])) for s in sites}
        pooled = metrics.pool(reps)
        return {"pooled_auc": pooled["auc"], "pooled_loss": pooled["loss"], "pooled_accuracy": pooled["accuracy"],
                "auc_by_site": {s: r["auc"] for s, r in reps.items()}}

    out = {"note": "experimenter benchmark on the pooled test set; solo and centralized use raw data from every "
                   "hospital and exist only for comparison", "model": spec, "rows": {}}
    fed = models.make(spec, d)
    out["federated"] = score(fed, np.asarray(art["params"]))
    X = np.vstack([enc[s]["train"][0] for s in sites])
    y = np.concatenate([enc[s]["train"][1] for s in sites])
    m = models.make(spec, d)
    out["centralized"] = score(m, models.fit(m, X, y, epochs=600))
    out["solo"] = {s: score(m, models.fit(m, enc[s]["train"][0], enc[s]["train"][1], epochs=600)) for s in sites}
    out["rows"] = {s: {p: int(len(enc[s][p][1])) for p in ("train", "val", "test")} for s in sites}
    return out


def table(b: dict) -> str:
    sites = list(b["solo"])
    head = "| model | pooled test AUC | " + " | ".join(f"{s} AUC" for s in sites) + " |"
    sep = "|" + "---|" * (len(sites) + 2)

    def row(name, r):
        cells = " | ".join("—" if r["auc_by_site"][s] is None else f"{r['auc_by_site'][s]:.3f}" for s in sites)
        return f"| {name} | **{r['pooled_auc']:.3f}** | {cells} |"

    lines = [head, sep, row("federated (published)", b["federated"])]
    lines += [row(f"{s} alone", b["solo"][s]) for s in sites]
    lines.append(row("all data pooled (not allowed)", b["centralized"]))
    return "\n".join(lines)
