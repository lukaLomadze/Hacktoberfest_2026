"""The two model families the coordinator can choose from, as flat parameter vectors.

Both expose the same interface so masking, DP noise and aggregation work on any model:
a model is a spec ({"type", "l2", "hidden"}) plus a 1-D float vector of parameters.

  logistic  p = σ(b + w·x)                       d + 1 parameters, convex
  mlp       p = σ(b2 + w2·tanh(W1·x + b1))       d·h + 2h + 1 parameters, non-convex

Local training is DP-GD: full-batch gradient descent where every patient's gradient is
clipped to L2 norm `clip` and Gaussian noise N(0, σ²) is added to the mean gradient at
every step. That makes each step a Gaussian mechanism with record-level sensitivity
2·clip/n whatever the model is (convex or not), so privacy.py can account it exactly.
numpy only.
"""

from __future__ import annotations

import numpy as np

from . import config, metrics


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def validate_spec(spec) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("model spec must be an object like {'type': 'logistic', 'l2': 0.01}")
    kind = str(spec.get("type", ""))
    if kind not in config.SUPPORTED_MODELS:
        raise ValueError(f"model type must be one of {config.SUPPORTED_MODELS}")
    l2 = float(spec.get("l2", 0.0) or 0.0)
    if not (0.0 <= l2 <= 10.0):
        raise ValueError("l2 must be between 0 and 10")
    out = {"type": kind, "l2": l2}
    if kind == "mlp":
        hidden = int(spec.get("hidden") or 16)
        if not (2 <= hidden <= 64):
            raise ValueError("mlp hidden units must be between 2 and 64")
        out["hidden"] = hidden
    return out


class Model:
    def __init__(self, spec: dict, d: int):
        self.spec = validate_spec(spec)
        self.d = int(d)
        self.l2 = self.spec["l2"]

    # subclasses: n_params, init(seed), logits(params, X), per_example_grads(params, X, y)

    def predict_proba(self, params, X):
        return sigmoid(self.logits(np.asarray(params, dtype=float), X))

    def loss(self, params, X, y) -> float:
        return metrics.log_loss(y, self.predict_proba(params, X))

    def reg_grad(self, params):
        """Gradient of l2/2·||weights||² (biases are not penalised). Data-independent."""
        return self.l2 * params * self.penalty_mask()


class Logistic(Model):
    @property
    def n_params(self) -> int:
        return self.d + 1

    def penalty_mask(self):
        m = np.ones(self.n_params)
        m[0] = 0.0
        return m

    def init(self, seed: int):
        return np.zeros(self.n_params)

    def logits(self, params, X):
        return params[0] + X @ params[1:]

    def per_example_grads(self, params, X, y):
        r = self.predict_proba(params, X) - y
        return np.hstack([r[:, None], r[:, None] * X])


class MLP(Model):
    def __init__(self, spec, d):
        super().__init__(spec, d)
        self.h = self.spec["hidden"]

    @property
    def n_params(self) -> int:
        return self.d * self.h + 2 * self.h + 1

    def _unpack(self, params):
        d, h = self.d, self.h
        W1 = params[: d * h].reshape(d, h)
        b1 = params[d * h: d * h + h]
        w2 = params[d * h + h: d * h + 2 * h]
        b2 = params[-1]
        return W1, b1, w2, b2

    def penalty_mask(self):
        m = np.zeros(self.n_params)
        m[: self.d * self.h] = 1.0
        m[self.d * self.h + self.h: self.d * self.h + 2 * self.h] = 1.0
        return m

    def init(self, seed: int):
        rng = np.random.default_rng(seed)
        p = np.zeros(self.n_params)
        p[: self.d * self.h] = rng.normal(0.0, 1.0 / np.sqrt(self.d), self.d * self.h)
        p[self.d * self.h + self.h: self.d * self.h + 2 * self.h] = rng.normal(0.0, 1.0 / np.sqrt(self.h), self.h)
        return p

    def logits(self, params, X):
        W1, b1, w2, b2 = self._unpack(params)
        return np.tanh(X @ W1 + b1) @ w2 + b2

    def per_example_grads(self, params, X, y):
        W1, b1, w2, b2 = self._unpack(params)
        hid = np.tanh(X @ W1 + b1)  # (n, h)
        r = sigmoid(hid @ w2 + b2) - y  # (n,)
        da = (r[:, None] * w2[None, :]) * (1.0 - hid ** 2)  # (n, h)
        gW1 = (X[:, :, None] * da[:, None, :]).reshape(len(y), -1)
        return np.hstack([gW1, da, r[:, None] * hid, r[:, None]])


def make(spec: dict, d: int) -> Model:
    spec = validate_spec(spec)
    return Logistic(spec, d) if spec["type"] == "logistic" else MLP(spec, d)


def dp_train(model: Model, params0, X, y, epochs: int, lr: float, clip: float, sigma: float,
             rng: np.random.Generator, mu: float = 0.0, anchor=None):
    """DP-GD from params0. σ = 0 turns the noise off (clipping stays). → (params, info)."""
    w = np.asarray(params0, dtype=float).copy()
    anchor = w.copy() if anchor is None else np.asarray(anchor, dtype=float)
    n = len(y)
    clipped = 0
    for _ in range(epochs):
        g = model.per_example_grads(w, X, y)
        norms = np.linalg.norm(g, axis=1)
        clipped += int((norms > clip).sum())
        g = g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))[:, None]
        step = g.sum(axis=0) / n
        if sigma > 0:
            step = step + rng.normal(0.0, sigma, size=step.shape)
        step = step + model.reg_grad(w)
        if mu:
            step = step + mu * (w - anchor)
        w = w - lr * step
    return w, {"clipped_fraction": round(clipped / max(1, epochs * n), 3)}


def fit(model: Model, X, y, epochs: int = 400, lr: float = None, seed: int = 0):
    """Plain (non-private) full-batch GD to convergence — for local CV and offline baselines."""
    lr = lr if lr is not None else (0.5 if model.spec["type"] == "logistic" else 0.3)
    w = model.init(seed)
    for _ in range(epochs):
        g = model.per_example_grads(w, X, y).mean(axis=0) + model.reg_grad(w)
        w = w - lr * g
    return w
