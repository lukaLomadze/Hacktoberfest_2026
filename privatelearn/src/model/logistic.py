"""Logistic regression for PrivateLearn. numpy only — no torch.

Contract (docs/CONTRACTS.md §2): weights w ∈ R^7 (bias + 6 features),
p = σ(w·[1;x]), mean BCE loss, gradient (p−y)·[1;x], local training =
4 full-batch epochs at lr 0.1, and the update sent to the coordinator is
delta_w = w_new − w_prev (plus loss and n), never raw data.
"""

from __future__ import annotations

import numpy as np

N_FEATURES = 6
N_WEIGHTS = 7  # bias + 6 features


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_bias(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"X must be (n, {N_FEATURES}), got {X.shape}")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(add_bias(X) @ np.asarray(w, dtype=float))


def loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(predict_proba(w, X), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = predict_proba(w, X)
    Xb = add_bias(X)
    return (Xb.T @ (p - np.asarray(y, dtype=float))) / len(y)


def train_step(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    mu: float = 0.0,
    w_anchor: np.ndarray | None = None,
) -> np.ndarray:
    """One gradient step. mu > 0 adds the FedProx proximal term mu/2·||w − w_anchor||²."""
    g = gradient(w, X, y)
    if mu:
        if w_anchor is None:
            raise ValueError("mu > 0 needs w_anchor (the current global model)")
        g = g + mu * (w - np.asarray(w_anchor, dtype=float))
    return w - lr * g


def train(
    w0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 4,
    lr: float = 0.1,
    mu: float = 0.0,
    w_anchor: np.ndarray | None = None,
) -> np.ndarray:
    """Full-batch gradient descent from w0. Returns the trained weights.

    For FedProx pass mu > 0 and w_anchor = the global model the round started from.
    """
    w = np.asarray(w0, dtype=float).copy()
    for _ in range(epochs):
        w = train_step(w, X, y, lr, mu, w_anchor)
    return w


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via Mann–Whitney U (rank formulation, handles ties)."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos = s[y]
    neg = s[~y]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    rank = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # tied scores share the average of their 1-based ranks
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        rank[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos_ranks = float(rank[y].sum())
    return (sum_pos_ranks - pos.size * (pos.size + 1.0) / 2.0) / (pos.size * neg.size)


def accuracy(y_true: np.ndarray, w: np.ndarray, X: np.ndarray, threshold: float = 0.5) -> float:
    p = predict_proba(w, X)
    return float(np.mean((p >= threshold) == np.asarray(y_true, dtype=bool)))
