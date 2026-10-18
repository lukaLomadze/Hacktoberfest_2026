"""Metrics, including pooled metrics computed from aggregates only.

The coordinator never sees a prediction or a label. Each hospital reports, for its own
split: n, log-loss, a confusion matrix at 0.5 and two histograms of predicted risk (one
per true class). Summing the histograms across sites gives the pooled ROC AUC of all
four hospitals together, exact up to the bin width.
"""

from __future__ import annotations

import math

import numpy as np

from . import config


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, scores):
    """ROC AUC by the rank (Mann-Whitney) formula; None when only one class is present."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def histograms(y, p, bins: int = config.HIST_BINS):
    edges = np.linspace(0.0, 1.0, bins + 1)
    y = np.asarray(y, dtype=bool)
    hp, _ = np.histogram(np.asarray(p)[y], bins=edges)
    hn, _ = np.histogram(np.asarray(p)[~y], bins=edges)
    return hp.astype(int).tolist(), hn.astype(int).tolist()


def auc_from_histograms(hist_pos, hist_neg):
    """Pooled AUC from summed per-class histograms (ties within a bin count one half)."""
    hp = np.asarray(hist_pos, dtype=float)
    hn = np.asarray(hist_neg, dtype=float)
    P, N = hp.sum(), hn.sum()
    if P == 0 or N == 0:
        return None
    neg_below = np.concatenate([[0.0], np.cumsum(hn)[:-1]])
    return float((hp * (neg_below + 0.5 * hn)).sum() / (P * N))


def site_report(y, p) -> dict:
    """Everything a hospital shares about one split: aggregates, never rows."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pred = p >= 0.5
    hp, hn = histograms(y, p)
    a = auc(y, p)
    return {"n": int(len(y)), "positives": int(y.sum()), "loss": round(log_loss(y, p), 6),
            "auc": None if a is None else round(a, 6),
            "accuracy": round(float((pred == (y > 0.5)).mean()), 6),
            "confusion": {"tp": int((pred & (y > 0.5)).sum()), "fp": int((pred & (y < 0.5)).sum()),
                          "tn": int((~pred & (y < 0.5)).sum()), "fn": int((~pred & (y > 0.5)).sum())},
            "hist_pos": hp, "hist_neg": hn}


def pool(reports: dict) -> dict:
    """Combine site_report()s → pooled n, loss, AUC, accuracy, sensitivity, specificity."""
    reps = [r for r in reports.values() if r]
    if not reps:
        return {}
    n = sum(r["n"] for r in reps)
    loss = sum(r["n"] * r["loss"] for r in reps) / n
    hp = np.sum([r["hist_pos"] for r in reps], axis=0)
    hn = np.sum([r["hist_neg"] for r in reps], axis=0)
    c = {k: sum(r["confusion"][k] for r in reps) for k in ("tp", "fp", "tn", "fn")}
    a = auc_from_histograms(hp, hn)
    sens = c["tp"] / max(1, c["tp"] + c["fn"])
    spec = c["tn"] / max(1, c["tn"] + c["fp"])
    site_aucs = [r["auc"] for r in reps if r["auc"] is not None]
    return {"n": n, "loss": round(loss, 6), "auc": None if a is None else round(a, 6),
            "accuracy": round((c["tp"] + c["tn"]) / n, 6), "sensitivity": round(sens, 6),
            "specificity": round(spec, 6), "balanced_accuracy": round((sens + spec) / 2, 6),
            "worst_site_auc": round(min(site_aucs), 6) if site_aucs else None,
            "mean_site_auc": round(sum(site_aucs) / len(site_aucs), 6) if site_aucs else None}


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))
