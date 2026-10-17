"""Gradient-inversion attack demo for PrivateLearn.

For logistic regression on a SINGLE sample, the gradient is g = (p − y)·[1; x],
so the sample is exactly recoverable from the update: x = g[1:] / g[0] (any
scaling, e.g. delta_w = −lr·g or norm clipping, cancels in the ratio). This is the
"deep leakage from gradients" idea in its simplest, provably exact form.

Honest framing: our hospitals send delta_w after 4 epochs over 120 rows, not a
one-row gradient. The demo is the WORST CASE — a batch of one — and shows that
even then masking + DP leave nothing to invert.

The demo runs the same attack twice on the same sample:
- without_privacy: invert the raw update → error ≈ 0 (recovered=True).
- with_privacy: invert the clipped + DP-noised + masked update the coordinator
  actually sees → error ≫ 1 (recovered=False).

The returned dict matches the `attack` field of trace.json exactly.
"""

from __future__ import annotations

import numpy as np

from model.logistic import gradient
from model.privacy import DP_DELTA, make_pair_secrets, prepare_update

RECOVERY_THRESHOLD = 0.2  # relative feature error below which a row counts as recovered


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


def run_attack(w, x, y, client_id=0, participants=(0, 1, 2), round_=1, secrets=None,
               C=1.0, sigma=1.0, seed=42, dp_delta=DP_DELTA):
    """Run gradient inversion on a single (x, y) sample, with and without privacy.

    `secrets` are the hospitals' pair secrets (generated from `seed` if omitted);
    the attacker never has them, only the masked vector.
    """
    w = np.asarray(w, dtype=float)
    x_true = np.asarray(x, dtype=float)
    if secrets is None:
        secrets = make_pair_secrets(participants, seed)

    # the update a hospital would send (true gradient, before any protection)
    delta = gradient(w, x_true.reshape(1, -1), np.array([float(y)]))

    err_plain = relative_error(x_true, reconstruct_single_sample(delta))

    protected, _ = prepare_update(delta, client_id, participants, round_, secrets,
                                  noise_seed=seed, C=C, sigma=sigma, dp_delta=dp_delta)
    err_protected = relative_error(x_true, reconstruct_single_sample(protected))

    def result(err):
        return {"recovered": bool(err < RECOVERY_THRESHOLD),
                "feature_error": round(err, 4) if np.isfinite(err) else 1e9}

    return {"target": f"h{client_id}", "without_privacy": result(err_plain), "with_privacy": result(err_protected)}
