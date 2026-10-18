"""Secure aggregation and differential-privacy accounting.

Secure aggregation (Bonawitz et al. 2017, without the dropout-recovery part):
  1. Every hospital makes a Diffie-Hellman key pair (RFC 3526 2048-bit group) and sends
     only its PUBLIC key to the coordinator, which relays the roster to everyone.
  2. Each pair of hospitals derives the same 256-bit secret g^(ab) on its own; the
     coordinator relayed the public keys but cannot compute the secret.
  3. A vector is sent as fixed-point integers mod 2^64 plus, for every peer, a mask
     expanded from the pair secret with SHAKE-256 (a cryptographic PRG). The lower id
     adds the mask, the higher id subtracts it, so the masks cancel in the sum and every
     single message is uniformly random to the coordinator.
  The participant set must be fixed before masking: if one participant goes silent
  afterwards, the sum cannot be unmasked and that round is discarded (never half-applied).
  Honest limit: public keys are not authenticated here; production would sign them.

Differential privacy (record level, per hospital):
  Local training is DP-GD (models.dp_train). One step with per-example clip C, n rows and
  noise σ is a Gaussian mechanism with sensitivity 2C/n (replace one patient), i.e.
  ρ = (2C/n)² / (2σ²) zero-concentrated DP. ρ adds up over steps and rounds, and
  ε(δ) = ρ + 2·sqrt(ρ·ln(1/δ)) (Bun & Steinke 2016). Each hospital gets a total ε budget
  for the job; every round may spend budget/max_rounds, so a coordinator that stops early
  leaves privacy budget unspent. Masking and noise together: the coordinator learns only
  a noisy sum.
"""

from __future__ import annotations

import hashlib
import math
import secrets as _secrets

import numpy as np

# RFC 3526, group 14 (2048-bit MODP), generator 2
DH_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
DH_G = 2

FIXED_SCALE = 2 ** 24  # fixed-point resolution 6e-8
FIXED_LIMIT = 2 ** 38  # |value| bound so a sum of a few sites cannot wrap


# ---------------------------------------------------------------- key agreement

def dh_keypair():
    private = _secrets.randbelow(2 ** 256) + 2
    return private, pow(DH_G, private, DH_P)


def pair_key(my_private: int, their_public: int, id_a: str, id_b: str) -> bytes:
    if not (2 <= their_public <= DH_P - 2):
        raise ValueError("invalid Diffie-Hellman public key")
    shared = pow(their_public, my_private, DH_P)
    lo, hi = sorted([id_a, id_b])
    return hashlib.sha256(b"fedheart-pair|" + lo.encode() + b"|" + hi.encode() + b"|" +
                          shared.to_bytes(256, "big")).digest()


# ---------------------------------------------------------------- masking

def encode_fixed(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("cannot mask a non-finite value")
    if np.any(np.abs(x) >= FIXED_LIMIT):
        raise ValueError("value too large for fixed-point masking")
    return np.round(x * FIXED_SCALE).astype(np.int64).astype(np.uint64)


def decode_fixed(u) -> np.ndarray:
    return np.asarray(u, dtype=np.uint64).astype(np.int64).astype(float) / FIXED_SCALE


def mask_stream(key: bytes, purpose: str, length: int) -> np.ndarray:
    raw = hashlib.shake_256(key + b"|" + purpose.encode()).digest(8 * length)
    return np.frombuffer(raw, dtype="<u8").astype(np.uint64)


def mask_vector(x, my_id: str, participants, keys: dict, purpose: str) -> np.ndarray:
    """x (floats) → masked uint64 vector. `keys` maps peer id → pair key; `purpose` must
    be unique per use (e.g. "round-7"), so no mask is ever reused."""
    participants = sorted(set(participants))
    if my_id not in participants:
        raise ValueError(f"{my_id} is not among the participants {participants}")
    out = encode_fixed(x)
    for peer in participants:
        if peer == my_id:
            continue
        if peer not in keys:
            raise ValueError(f"no pair key with {peer}")
        m = mask_stream(keys[peer], purpose, len(out))
        out = out + m if my_id < peer else out - m  # uint64 arithmetic wraps mod 2^64
    return out


def unmask_sum(vectors) -> np.ndarray:
    """Sum masked vectors mod 2^64 → the float sum (masks cancel)."""
    stack = np.array([np.asarray(v, dtype=np.uint64) for v in vectors], dtype=np.uint64)
    return decode_fixed(stack.sum(axis=0, dtype=np.uint64))


def to_wire(u) -> list:
    return [int(v) for v in np.asarray(u, dtype=np.uint64)]


def from_wire(values) -> np.ndarray:
    return np.array([int(v) for v in values], dtype=np.uint64)


# ---------------------------------------------------------------- zCDP accounting

def rho_from_epsilon(eps: float, delta: float) -> float:
    """The zCDP ρ whose (ε, δ) conversion equals eps."""
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    L = math.log(1.0 / delta)
    return (math.sqrt(L + eps) - math.sqrt(L)) ** 2


def epsilon_from_rho(rho: float, delta: float) -> float:
    if rho <= 0:
        return 0.0
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def step_rho(n: int, clip: float, sigma: float) -> float:
    return (2.0 * clip / n) ** 2 / (2.0 * sigma ** 2)


def sigma_for(n: int, clip: float, epochs: int, rho_round: float) -> float:
    """Per-step noise σ so that `epochs` DP-GD steps cost exactly rho_round."""
    return (2.0 * clip / n) * math.sqrt(epochs / (2.0 * rho_round))
