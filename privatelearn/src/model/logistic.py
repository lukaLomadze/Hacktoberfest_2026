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


def train_step(w: np.ndarray, X: np.ndarray, y: np.ndarray, lr: float = 0.1) -> np.ndarray:
    return w - lr * gradient(w, X, y)


def train(
    w0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 4,
    lr: float = 0.1,
) -> np.ndarray:
    """Full-batch gradient descent from w0. Returns the trained weights."""
    w = np.asarray(w0, dtype=float).copy()
    for _ in range(epochs):
        w = train_step(w, X, y, lr)
    return w


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via Mann–Whitney U (rank formulation, handles ties)."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos = s[y]
    neg = s[~y]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    rank = np.argsort(np.argsort(s)) + 1  # average rank (ties get the same rank)
    sum_pos_ranks = float(rank[y].sum())
    return (sum_pos_ranks - pos.size * (pos.size + 1.0) / 2.0) / (pos.size * neg.size)


def accuracy(y_true: np.ndarray, w: np.ndarray, X: np.ndarray, threshold: float = 0.5) -> float:
    p = predict_proba(w, X)
    return float(np.mean((p >= threshold) == np.asarray(y_true, dtype=bool)))
