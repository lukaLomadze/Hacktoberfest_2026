"""Loading one hospital's own export and splitting it. Runs INSIDE the hospital process.

A site's file is the UCI "processed" export: 14 comma-separated values per patient, '?'
for a missing value. Nothing here is ever sent anywhere; the hospital process keeps the
arrays private and only shares aggregates (see hospital.py).
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np

from . import config


def site_seed(*parts) -> int:
    """A deterministic 64-bit seed from printable parts (same on every OS)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def load_site(site_id: str, data_dir=None) -> tuple[dict, np.ndarray]:
    """→ ({column: float array, NaN = missing}, y) for one site; y = 1 when num > 0."""
    site = config.SITE_BY_ID[site_id]
    path = (data_dir or config.DATA_DIR) / site["file"]
    with open(path, newline="", encoding="ascii") as fh:
        rows = [r for r in csv.reader(fh) if r and any(v.strip() for v in r)]
    width = len(config.FEATURE_NAMES) + 1
    bad = [i for i, r in enumerate(rows) if len(r) != width]
    if bad:
        raise ValueError(f"{path.name}: rows {bad[:5]} do not have {width} values")

    def val(v):
        v = v.strip()
        return float("nan") if v in ("?", "") else float(v)

    table = {c: np.array([val(r[j]) for r in rows]) for j, c in enumerate(config.FEATURE_NAMES)}
    num = np.array([val(r[-1]) for r in rows])
    if np.isnan(num).any():
        raise ValueError(f"{path.name}: missing label")
    return table, (num > 0).astype(float)


def stratified_split(y: np.ndarray, seed: int, fractions=config.SPLIT) -> dict:
    """{'train','val','test'} → sorted row indices, stratified by label, reproducible."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for cls in (0.0, 1.0):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(fractions[1] * n))
        n_test = int(round(fractions[2] * n))
        parts["test"].extend(idx[:n_test])
        parts["val"].extend(idx[n_test:n_test + n_val])
        parts["train"].extend(idx[n_test + n_val:])
    return {k: np.array(sorted(v), dtype=int) for k, v in parts.items()}


def subset(table: dict, idx) -> dict:
    return {c: v[idx] for c, v in table.items()}


def load_split_site(site_id: str, seed: int = config.SEED, data_dir=None):
    """→ (table, y, splits) — the same split the hospital process uses for this seed."""
    table, y = load_site(site_id, data_dir)
    return table, y, stratified_split(y, site_seed("split", seed, site_id))
