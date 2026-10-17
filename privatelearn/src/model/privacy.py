"""Secure aggregation (pairwise masking) and differential privacy for PrivateLearn.

Each hospital: clips its update to L2 norm C, adds Gaussian DP noise, then masks
it with pairwise masks. For every pair (i, j) the two hospitals hold the SAME
mask; client i adds +mask and client j adds -mask. When the coordinator sums all
masked updates, the masks cancel exactly and only the true sum survives — so the
coordinator sees a noise-like vector per client, and never an individual update.

Everything is deterministic given a base seed, so the demo and tests reproduce.
"""

from __future__ import annotations

import math

import numpy as np


def clip_norm(update, C: float):
    """Clip an update to L2 norm <= C (bounds the sensitivity for DP)."""
    u = np.asarray(update, dtype=float).copy()
    n = float(np.linalg.norm(u))
    if n > C:
        u = u * (C / n)
    return u


def add_dp_noise(update, sigma: float, rng=None):
    """Add Gaussian noise of std `sigma` to every coordinate."""
    rng = rng if rng is not None else np.random.default_rng(0)
    return np.asarray(update, dtype=float) + rng.normal(0.0, sigma, size=np.asarray(update).shape)


def epsilon(sigma: float, sensitivity: float, dp_delta: float = 1e-5) -> float:
    """(eps, dp_delta)-DP Gaussian mechanism epsilon, for L2 sensitivity `sensitivity`."""
    if sigma <= 0:
        return float("inf")
    return math.sqrt(2.0 * math.log(1.25 / dp_delta)) * sensitivity / sigma


def _pair_seed(i: int, j: int, base_seed: int) -> int:
    """Deterministic seed for the unordered pair {i, j}."""
    a, b = (i, j) if i < j else (j, i)
    return (base_seed * 2654435761 + a * 374761393 + b * 668265263) & 0xFFFFFFFF


def mask_update(update, client_id: int, other_ids, base_seed: int = 42, mask_scale: float = 1.0):
    """Add pairwise masks: for pair (i,j), i adds +r and j adds -r (same r)."""
    masked = np.asarray(update, dtype=float).copy()
    for j in other_ids:
        if j == client_id:
            continue
        rng = np.random.default_rng(_pair_seed(client_id, j, base_seed))
        mask = rng.normal(0.0, mask_scale, size=masked.shape)
        masked = masked + (mask if client_id < j else -mask)
    return masked


def unmask_sum(masked_updates):
    """Coordinator sums masked updates; the pairwise masks cancel exactly."""
    return np.sum(np.asarray(masked_updates, dtype=float), axis=0)


def prepare_update(update, client_id, other_ids, C: float = 1.0, sigma: float = 0.1,
                   base_seed: int = 42, dp_delta: float = 1e-5):
    """Hospital side: clip → DP noise → mask. Returns (masked_update, eps)."""
    u = clip_norm(update, C)
    u = add_dp_noise(u, sigma)
    masked = mask_update(u, client_id, other_ids, base_seed)
    return masked, epsilon(sigma, C, dp_delta)
