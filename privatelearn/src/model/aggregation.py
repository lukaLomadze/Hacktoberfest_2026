"""Aggregation strategies and the adaptive selector for PrivateLearn.

The coordinator receives one update per hospital: delta_w (R^7), loss, and n.
It never sees raw data. It picks a strategy per round and aggregates:

- FedAvg        : average of the deltas weighted by sample count n.
- FedProx       : the same n-weighted average on the server; the difference is in
                  LOCAL training, which adds the proximal term mu/2·||w − w_global||²
                  (model.logistic.train(..., mu=FEDPROX_MU, w_anchor=w_global)) so
                  heterogeneous hospitals cannot drift far from the global model.
- trimmed_mean  : drop the flagged client(s), then FedAvg over the rest.

Works with secure aggregation (model.privacy): under masking the coordinator only
sees the SUM of the updates, so per-client norms are meaningless. The selector
therefore decides from the clear-text losses alone, and trimming means excluding a
participant BEFORE the masked round (pairwise masks only cancel over a fixed
participant set). Passing `deltas` to `select_strategy` adds an update-norm check
for the unmasked path only.

`select_strategy` returns (strategy, reason, flagged); the reason string feeds the
demo's decision log. numpy only.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

STRATEGIES = ("fedavg", "fedprox", "trimmed_mean")
FEDPROX_MU = 0.1  # proximal strength used in local training when strategy == "fedprox"
MAD_TO_SIGMA = 1.4826  # MAD × this ≈ σ for normally distributed values


class Decision(NamedTuple):
    strategy: str
    reason: str
    flagged: tuple  # client indices to exclude (only non-empty for trimmed_mean)


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


def fedprox(deltas, ns):
    """Server side of FedProx: the n-weighted average (the proximal term is local)."""
    return fedavg(deltas, ns)


def trimmed_mean(deltas, ns, drop=None, k: int = 1):
    """FedAvg after excluding clients.

    drop: indices to exclude (from select_strategy's `flagged`). If None, the k
    largest-norm updates are dropped — only meaningful for UNmasked updates.
    """
    d = _as_deltas(deltas)
    ns = np.asarray(ns, dtype=float)
    if drop is None:
        if len(d) <= k:
            raise ValueError(f"cannot trim k={k} from only {len(d)} clients")
        keep = np.sort(np.argsort(np.linalg.norm(d, axis=1))[: len(d) - k])
    else:
        drop = {int(i) for i in drop}
        if any(i < 0 or i >= len(d) for i in drop):
            raise ValueError(f"drop indices {sorted(drop)} out of range for {len(d)} clients")
        keep = np.array([i for i in range(len(d)) if i not in drop], dtype=int)
        if keep.size == 0:
            raise ValueError("cannot drop every client")
    return fedavg(d[keep], ns[keep])


def select_strategy(losses, deltas=None, tau: float = 0.1, outlier_sigma: float = 2.0,
                    names=None, min_gap: float = 0.15) -> Decision:
    """Choose the aggregation rule for this round and explain why.

    1. trimmed_mean if some loss is > outlier_sigma robust-σ above the median
       (median + MAD, so one huge loss cannot inflate its own threshold) AND at
       least `min_gap` nats above it — with only three hospitals the MAD can be
       ~0, and honest non-IID hospitals differ by a few hundredths. The flagged
       clients are returned so the SAME clients get excluded.
    2. fedprox if the losses are spread out (std > tau) — or, unmasked only,
       the update norms vary wildly.
    3. fedavg otherwise.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty 1-D sequence")
    names = list(names) if names is not None else [f"client {i}" for i in range(len(losses))]
    if len(names) != len(losses):
        raise ValueError("names must have one entry per client")

    med = float(np.median(losses))
    scale = max(MAD_TO_SIGMA * float(np.median(np.abs(losses - med))), 1e-3)
    threshold = med + max(outlier_sigma * scale, min_gap)
    flagged = tuple(int(i) for i in np.flatnonzero(losses > threshold))
    if flagged and len(flagged) < len(losses):
        who = ", ".join(f"{names[i]} loss {losses[i]:.3f}" for i in flagged)
        return Decision("trimmed_mean",
                        f"{who} is >{outlier_sigma:g}σ above median {med:.3f} → trimmed-mean, excluded",
                        flagged)

    sd = float(losses.std())
    if sd > tau:
        return Decision("fedprox", f"loss std {sd:.3f} > τ={tau:g}, hospitals disagree → FedProx", ())
    if deltas is not None:
        d = _as_deltas(deltas)
        if len(d) != len(losses):
            raise ValueError("losses and deltas must have the same length")
        norms = np.linalg.norm(d, axis=1)
        spread = float(norms.std() / (norms.mean() + 1e-9))
        if spread > 0.5:
            return Decision("fedprox", f"update-norm spread {spread:.2f} > 0.5 → FedProx", ())
    return Decision("fedavg", f"loss std {sd:.3f} ≤ τ={tau:g}, hospitals agree → FedAvg", ())


def aggregate(deltas, ns, strategy: str, drop=None):
    if strategy == "fedavg":
        return fedavg(deltas, ns)
    if strategy == "fedprox":
        return fedprox(deltas, ns)
    if strategy == "trimmed_mean":
        return trimmed_mean(deltas, ns, drop=drop)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
