"""Gradient-inversion attack demo for PrivateLearn.

For logistic regression on a SINGLE sample, the gradient is g = (p − y)·[1; x],
so the sample is exactly recoverable from the update: x = g[1:] / g[0]. This is
the "deep leakage from gradients" idea in its simplest, provably-exact form.

The demo runs the same attack twice on the same sample:
- without_privacy: recover x from the raw update → error ≈ 0 (recovered=True).
- with_privacy: recover x from a masked + DP-noised update → huge error
  (recovered=False).

The returned dict matches the `attack` field of trace.json exactly.
"""

from __future__ import annotations

import numpy as np

from model.logistic import gradient
from model.privacy import add_dp_noise, clip_norm, mask_update


def reconstruct_single_sample(delta_w):
    """x = g[1:] / g[0], the exact inversion for a single-sample linear model."""
    d = np.asarray(delta_w, dtype=float)
    if abs(d[0]) < 1e-12:
        return None
    return d[1:] / d[0]


def relative_error(x_true, x_rec):
    if x_rec is None:
        return float("inf")
    x = np.asarray(x_true, dtype=float)
    return float(np.linalg.norm(np.asarray(x_rec, dtype=float) - x) / (np.linalg.norm(x) + 1e-9))


def run_attack(w, x, y, client_id=0, other_ids=(0, 1, 2), C=1.0, sigma=0.1, base_seed=42):
    """Run gradient inversion on a single (x, y) sample, with and without privacy."""
    w = np.asarray(w, dtype=float)
    x_true = np.asarray(x, dtype=float)

    # the update a hospital would send (true gradient, before any protection)
    delta = gradient(w, x_true.reshape(1, -1), np.array([float(y)]))

    rec_plain = reconstruct_single_sample(delta)
    err_plain = relative_error(x_true, rec_plain)

    protected = mask_update(add_dp_noise(clip_norm(delta, C), sigma), client_id, other_ids, base_seed)
    rec_protected = reconstruct_single_sample(protected)
    err_protected = relative_error(x_true, rec_protected)

    return {
        "target": f"h{client_id}",
        "without_privacy": {"recovered": bool(err_plain < 0.2), "feature_error": round(err_plain, 4)},
        "with_privacy": {"recovered": bool(err_protected < 0.2), "feature_error": round(err_protected, 4)},
    }
