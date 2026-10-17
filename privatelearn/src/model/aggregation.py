"""Aggregation strategies and the adaptive selector for PrivateLearn.

The coordinator receives one update per hospital: delta_w (R^7), loss, and n.
It never sees raw data. It picks a strategy per round and aggregates:

- FedAvg        : weighted average by sample count.
- trimmed_mean  : drop the k largest-norm updates (robust to one bad/poisoned client).
- FedProx       : cap each update's norm at mu (proximal projection), then FedAvg —
                  keeps any single client from dragging the global model.

`select_strategy` is the adaptive part: it reads the loss distribution and the
update norms and returns (strategy, reason). The reason string feeds the demo's
decision log, so a judge can see WHY a strategy was chosen. numpy only.
"""

from __future__ import annotations

import numpy as np

STRATEGIES = ("fedavg", "fedprox", "trimmed_mean")


def _as_deltas(deltas):
    d = np.asarray(deltas, dtype=float)
    if d.ndim != 2:
        raise ValueError("deltas must be a 2-D array of shape (clients, weights)")
    return d


def fedavg(deltas, ns):
    """Weighted average of updates, weighted by each client's sample count."""
    d = _as_deltas(deltas)
    ns = np.asarray(ns, dtype=float)
    if len(ns) != len(d):
        raise ValueError("ns must have one entry per client")
    if ns.sum() <= 0:
        raise ValueError("sum of ns must be positive")
    return np.sum(d * ns[:, None], axis=0) / ns.sum()


def trimmed_mean(deltas, ns, k: int = 1):
    """Average after dropping the k updates with the largest L2 norm."""
    d = _as_deltas(deltas)
    if len(d) <= k:
        raise ValueError(f"cannot trim k={k} from only {len(d)} clients")
    keep = np.argsort(np.linalg.norm(d, axis=1))[: max(1, len(d) - k)]
    return fedavg(d[keep], np.asarray(ns)[keep])


def fedprox(deltas, ns, mu: float):
    """Clip each update's norm to mu (proximal projection), then FedAvg."""
    d = _as_deltas(deltas).copy()
    for i in range(len(d)):
        n = float(np.linalg.norm(d[i]))
        if n > mu:
            d[i] *= mu / n
    return fedavg(d, ns)


def select_strategy(losses, deltas, tau: float = 0.1, outlier_sigma: float = 2.0, k: int = 1):
    """Choose the aggregation rule for this round and explain why."""
    losses = np.asarray(losses, dtype=float)
    d = _as_deltas(deltas)
    if len(losses) != len(d):
        raise ValueError("losses and deltas must have the same length")

    # 1. Robust outlier detection: median + MAD, so a single huge loss cannot
    #    inflate the threshold past itself (mean±2σ fails exactly there).
    med = float(np.median(losses))
    mad = float(np.median(np.abs(losses - med)))
    if mad > 0 and np.any(losses > med + outlier_sigma * 1.4826 * mad):
        bad = int(np.argmax(losses))
        return "trimmed_mean", (
            f"client {bad} loss {losses[bad]:.3f} is >{outlier_sigma:.0f}σ above median {med:.3f} → trimmed-mean"
        )

    # 2. Heterogeneity: losses spread out, or update norms vary wildly -> regularize.
    sd = float(losses.std())
    norms = np.array([float(np.linalg.norm(row)) for row in d])
    norm_spread = float(norms.std() / (norms.mean() + 1e-9))
    if sd > tau or norm_spread > 0.5:
        return "fedprox", f"loss std {sd:.3f} > τ={tau} or norm spread {norm_spread:.2f} → FedProx"

    # 3. Otherwise the updates agree -> plain FedAvg.
    return "fedavg", f"loss std {sd:.3f} < τ={tau}, updates agree → FedAvg"


def aggregate(deltas, ns, strategy: str, mu: float = 1.0, k: int = 1):
    if strategy == "fedavg":
        return fedavg(deltas, ns)
    if strategy == "fedprox":
        return fedprox(deltas, ns, mu)
    if strategy == "trimmed_mean":
        return trimmed_mean(deltas, ns, k)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
