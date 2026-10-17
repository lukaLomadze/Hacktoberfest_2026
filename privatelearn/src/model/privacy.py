"""Secure aggregation (pairwise masking) and differential privacy for PrivateLearn.

Each hospital, every round: clip its update to L2 norm C → add Gaussian DP noise →
add pairwise masks. For every pair (i, j) of this round's participants both
hospitals derive the SAME mask from a secret only the two of them hold; i adds it,
j subtracts it. Summing every participant's masked update cancels the masks exactly,
so the coordinator learns only the sum — never an individual update.

Three properties the demo relies on, each tested:
- the pair secrets never reach the coordinator, so it cannot recompute the masks;
- masks depend on the round, so differencing two rounds reveals nothing;
- the participant set is fixed BEFORE masking (a hospital that drops out after
  masking would break cancellation), so offline/excluded hospitals are left out
  of `participants` up front.

DP accounting uses the analytic Gaussian mechanism (Balle & Wang, 2018), which is
valid for any ε (the classic sqrt(2 ln(1.25/δ))·Δ/σ bound only holds for ε < 1).
Per-round ε adds up across rounds (basic composition) — a stated-ε demonstration,
not an audited accountant.

Everything is deterministic given its seeds, so the demo and tests reproduce.
"""

from __future__ import annotations

import hashlib
import math
from itertools import combinations

import numpy as np

DP_DELTA = 1e-5


# ---------------------------------------------------------------- clipping + noise

def clip_norm(update, C: float):
    """Clip an update to L2 norm <= C (bounds the sensitivity for DP)."""
    u = np.asarray(update, dtype=float).copy()
    n = float(np.linalg.norm(u))
    if n > C:
        u = u * (C / n)
    return u


def add_dp_noise(update, sigma: float, rng):
    """Add Gaussian noise of std `sigma` to every coordinate.

    `rng` is required: a fixed default would add the same, subtractable vector
    for every hospital and round, which is not differential privacy.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator, seeded per hospital and round")
    u = np.asarray(update, dtype=float)
    return u + rng.normal(0.0, sigma, size=u.shape)


# ---------------------------------------------------------------- seeds + secrets

def derive_seed(*parts) -> int:
    """Deterministic 64-bit seed from any printable parts (SHA-256 based)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _pair(i: int, j: int) -> tuple:
    return (i, j) if i < j else (j, i)


def make_pair_secrets(ids, seed) -> dict:
    """One shared secret per unordered pair of hospitals.

    Stands in for a pairwise key agreement (e.g. Diffie–Hellman) run directly
    between hospitals; the coordinator never receives these values.
    """
    return {_pair(i, j): hashlib.sha256(f"pair-secret|{seed}|{min(i, j)}|{max(i, j)}".encode()).hexdigest()
            for i, j in combinations(sorted(ids), 2)}


def pair_mask(secret: str, i: int, j: int, round_: int, shape, scale: float = 1.0):
    """The mask shared by pair {i, j} in this round."""
    a, b = _pair(i, j)
    rng = np.random.default_rng(derive_seed("mask", secret, a, b, round_))
    return rng.normal(0.0, scale, size=shape)


# ---------------------------------------------------------------- secure aggregation

def mask_update(update, client_id: int, participants, round_: int, secrets: dict,
                mask_scale: float = 1.0):
    """Add pairwise masks: for pair (i, j), the lower id adds +m and the higher -m.

    `participants` must be this round's final participant set (including
    client_id); `secrets` needs the entries for client_id's pairs only.
    """
    participants = sorted(set(participants))
    if client_id not in participants:
        raise ValueError(f"client {client_id} is not in this round's participants {participants}")
    u = np.asarray(update, dtype=float)
    masked = u.copy()
    for j in participants:
        if j == client_id:
            continue
        key = _pair(client_id, j)
        if key not in secrets:
            raise ValueError(f"missing pair secret for {key}")
        m = pair_mask(secrets[key], client_id, j, round_, u.shape, mask_scale)
        masked = masked + (m if client_id < j else -m)
    return masked


def unmask_sum(masked_updates):
    """Coordinator sums masked updates; the pairwise masks cancel exactly."""
    return np.sum(np.asarray(masked_updates, dtype=float), axis=0)


# ---------------------------------------------------------------- DP accounting

def _log_phi(x: float) -> float:
    """log of the standard normal CDF, stable far into the lower tail."""
    if x > -30.0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))
    # Mills-ratio asymptotic: Φ(x) ≈ φ(x)/|x| · (1 − 1/x²)
    return -0.5 * x * x - math.log(-x) - 0.5 * math.log(2.0 * math.pi) + math.log1p(-1.0 / (x * x))


def _delta_of(eps: float, sigma: float, sensitivity: float) -> float:
    """δ achieved by the Gaussian mechanism at ε (Balle & Wang 2018, Thm 8)."""
    a = sensitivity / (2.0 * sigma)
    b = eps * sigma / sensitivity
    return math.exp(_log_phi(a - b)) - math.exp(eps + _log_phi(-a - b))


def epsilon(sigma: float, sensitivity: float, dp_delta: float = DP_DELTA) -> float:
    """Smallest ε such that Gaussian noise `sigma` on L2-sensitivity `sensitivity`
    is (ε, dp_delta)-DP. inf for no noise."""
    if sigma <= 0:
        return float("inf")
    if _delta_of(0.0, sigma, sensitivity) <= dp_delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while _delta_of(hi, sigma, sensitivity) > dp_delta:
        hi *= 2.0
        if hi > 1e6:
            return float("inf")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _delta_of(mid, sigma, sensitivity) > dp_delta:
            lo = mid
        else:
            hi = mid
    return hi


def sigma_for_epsilon(eps: float, sensitivity: float, dp_delta: float = DP_DELTA) -> float:
    """Noise std needed for (eps, dp_delta)-DP at this sensitivity."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    lo, hi = 1e-6, 1.0
    while epsilon(hi, sensitivity, dp_delta) > eps:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if epsilon(mid, sensitivity, dp_delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------- record-level DP training

def record_sensitivity(n: int, lr: float = 0.1, epochs: int = 4, example_clip: float = 1.0) -> float:
    """L2 bound on how much ONE patient's row can change a hospital's delta_w.

    Full-batch GD on the mean loss with per-example gradients clipped to
    `example_clip`: replacing one row moves each step's mean gradient by at most
    2·example_clip/n, and GD on a convex smooth loss is non-expansive at small lr,
    so over `epochs` steps the change is ≤ epochs·lr·2·example_clip/n.
    This is the patient-level sensitivity DP is about (not the whole hospital's).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 2.0 * epochs * lr * example_clip / n


def train_dp(w0, X, y, epochs: int = 4, lr: float = 0.1, example_clip: float = 1.0,
             mu: float = 0.0, w_anchor=None):
    """Local training with per-example gradient clipping (bounds record_sensitivity).

    Same full-batch GD as model.logistic.train (including the optional FedProx
    proximal term), but each row's gradient is clipped to `example_clip` first.
    """
    from model.logistic import add_bias, sigmoid

    w = np.asarray(w0, dtype=float).copy()
    Xb = add_bias(X)
    y = np.asarray(y, dtype=float)
    for _ in range(epochs):
        per_ex = (sigmoid(Xb @ w) - y)[:, None] * Xb
        norms = np.linalg.norm(per_ex, axis=1, keepdims=True)
        per_ex = per_ex * np.minimum(1.0, example_clip / np.maximum(norms, 1e-12))
        g = per_ex.mean(axis=0)
        if mu:
            if w_anchor is None:
                raise ValueError("mu > 0 needs w_anchor (the current global model)")
            g = g + mu * (w - np.asarray(w_anchor, dtype=float))
        w = w - lr * g
    return w


# ---------------------------------------------------------------- hospital side

def noise_rng(noise_seed, client_id: int, round_: int) -> np.random.Generator:
    """Per-hospital, per-round DP noise stream (noise_seed stays at the hospital)."""
    return np.random.default_rng(derive_seed("dp", noise_seed, client_id, round_))


def prepare_update(update, client_id: int, participants, round_: int, secrets: dict,
                   noise_seed, C: float = 1.0, sigma: float = 1.0, dp_delta: float = DP_DELTA,
                   mask_scale: float = 1.0):
    """Hospital side: clip → DP noise → mask. Returns (masked_update, eps_this_round)."""
    u = clip_norm(update, C)
    u = add_dp_noise(u, sigma, noise_rng(noise_seed, client_id, round_))
    masked = mask_update(u, client_id, participants, round_, secrets, mask_scale)
    return masked, epsilon(sigma, C, dp_delta)
