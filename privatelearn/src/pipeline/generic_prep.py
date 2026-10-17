"""Generic, dataset-agnostic data preparation for PrivateLearn.

This is the "swap the hospitals for banks" engine: it turns ANY tabular CSV with a
custom target column into the exact same artifacts `prep_data.py` produces for the
Pima demo — per-client local CSVs (each with its own column names and units),
a canonical holdout, and a `manifest.json` that the runtime reads.

The runtime (`run_loop.py`) has no dataset knowledge; it reads feature order, the
label, normalisation statistics, the split, and (for the offline fallback) the
local→canonical schema mapping straight from the manifest. The whole point of this
module is that the dataset is described as DATA (a `spec` dict), never as code.

Spec format
-----------
{
  "dataset": str,                # human name, recorded in the manifest + trace meta
  "source": str,                 # provenance (URL / citation) — documentation only
  "raw_target": str,             # column in the raw CSV that is the y (0/1 outcome)
  "target_values": dict|None,    # optional {raw_value: 0/1} to encode a categorical y;
                                 # when None the values must already parse as 0.0/1.0
  "drop_columns": [str],         # raw columns to ignore (e.g. an id column)
  "features": [                  # the CANONICAL schema, in model order
      {"raw": str, "name": str, "unit": str, "impute_zero": bool},
      ...
  ],
  "split_col": str,              # canonical feature to sort the non-IID split by
  "n_clients": int,
  "n_per_client": int,
  "holdout_fraction": float,
  "clients": [                   # one local schema per client, deliberately different
      {"id": str, "target": str,   # local target column name
       "columns": {canonical: {"source": [local...], "factor": float|None,
                               "unit": str, "derive": str|None}, ...}},
      ...
  ],
  "node_labels": {id: str},      # display labels for the trace (e.g. "Bank A")
  "faults": {...},               # optional demo scenario, same shape as run_loop.FAULTS
  "honesty": str,                # optional honesty note carried into the trace meta
  "holdout_target": str,         # optional canonical name for the holdout target column
}

Conversion convention (identical to `prep_data.py` / CONTRACTS.md §4):
    canonical = local * factor         (so the local CSV stores canonical / factor)

Deterministic: the same raw file + spec + seed produce byte-identical outputs.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shared import io  # noqa: E402

DECIMALS = 9  # local-unit values are rounded here; conversion error stays < 1e-6


def _r(x) -> float:
    return round(float(x), DECIMALS)


def _read_raw(path, drop):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [f for f in reader.fieldnames if f not in drop]
        rows = [{f: r[f] for f in fieldnames} for r in reader]
    return fieldnames, rows


def _encode_target(raw, raw_target, target_values):
    values = [r[raw_target] for r in raw]
    if target_values is not None:
        return np.array([float(target_values[v]) for v in values]), dict(target_values)
    out = np.array([float(v) for v in values])
    uniq = sorted({float(v) for v in values})
    if uniq != [0.0, 1.0]:
        raise ValueError(
            f"target {raw_target!r} has values {uniq}; pass target_values to encode them to 0/1"
        )
    return out, None


def _canonical_matrix(raw, features):
    """(n, F) matrix in canonical units, built from the raw columns."""
    n = len(raw)
    X = np.empty((n, len(features)))
    for j, f in enumerate(features):
        X[:, j] = [float(r[f["raw"]]) for r in raw]
    return X


def _impute(X, features):
    """Replace the 0 sentinel with the median of non-zero values (Pima-style)."""
    medians = {}
    for j, f in enumerate(features):
        if f.get("impute_zero"):
            col = X[:, j]
            nz = col[col != 0]
            med = float(np.median(nz)) if nz.size else 0.0
            medians[f["name"]] = med
            col[col == 0] = med
    return medians


def _split(X, y, split_col_idx, n_clients, n_per_client, holdout_fraction, seed):
    n = len(y)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=n_clients * n_per_client, replace=False)
    chosen = chosen[np.lexsort((chosen, X[chosen, split_col_idx]))]
    blocks = [np.sort(chosen[i * n_per_client:(i + 1) * n_per_client]) for i in range(n_clients)]
    rest = np.setdiff1d(np.arange(n), chosen)
    n_holdout = int(round(holdout_fraction * n))
    holdout = np.sort(rng.choice(rest, size=n_holdout, replace=False))
    return blocks, holdout


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def prepare_dataset(raw_path, spec, out_dir, seed=42):
    """Write per-client CSVs + holdout + manifest for one dataset spec. Returns the manifest."""
    raw_path = pathlib.Path(raw_path)
    out_dir = pathlib.Path(out_dir)

    raw_target = spec["raw_target"]
    drop = list(spec.get("drop_columns") or [])
    features = list(spec["features"])
    clients = list(spec["clients"])
    n_clients = spec["n_clients"]
    n_per_client = spec["n_per_client"]
    holdout_fraction = spec["holdout_fraction"]
    split_col = spec["split_col"]
    holdout_target = spec.get("holdout_target", raw_target)

    if len(clients) != n_clients:
        raise ValueError(f"spec has {len(clients)} clients but n_clients={n_clients}")

    _, raw = _read_raw(raw_path, drop)
    y, target_values = _encode_target(raw, raw_target, spec.get("target_values"))
    X = _canonical_matrix(raw, features)
    medians = _impute(X, features)
    split_col_idx = next(i for i, f in enumerate(features) if f["name"] == split_col)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    blocks, holdout = _split(X, y, split_col_idx, n_clients, n_per_client, holdout_fraction, seed)

    # --- per-client local CSVs --------------------------------------------------
    files = {}
    schema = {}
    for ci, client in enumerate(clients):
        cid = client["id"]
        idx = blocks[ci]
        header = []
        col_meta = []  # (feature name, mapping) in canonical order
        for f in features:
            m = client["columns"][f["name"]]
            header.append(m["source"][0])
            col_meta.append((f["name"], m))
        header.append(client["target"])

        rows = []
        for i in idx:
            row = []
            for j, (fname, m) in enumerate(col_meta):
                factor = m.get("factor")
                if factor is None:
                    raise ValueError(
                        f"client {cid} feature {fname!r}: derived features need runtime "
                        "derivation; give a numeric factor instead")
                row.append(_r(X[i, j] / factor))
            row.append(int(y[i]))
            rows.append(row)
        _write_csv(out_dir / "hospitals" / f"hospital_{ci}.csv", header, rows)
        files[cid] = f"hospitals/hospital_{ci}.csv"
        schema[cid] = {
            "label": client["target"],
            "columns": {f["name"]: {k: client["columns"][f["name"]].get(k)
                                    for k in ("source", "unit", "factor", "derive")}
                        for f in features},
        }

    # --- holdout (canonical names/units) ---------------------------------------
    holdout_features = [f["name"] for f in features]
    test_rows = [[_r(X[i, j]) for j in range(len(features))] + [int(y[i])] for i in holdout]
    _write_csv(out_dir / "holdout" / "test.csv", holdout_features + [holdout_target], test_rows)

    # --- manifest --------------------------------------------------------------
    def balance(idx):
        pos = int(y[idx].sum())
        return {"n": int(len(idx)), "positive": pos, "negative": int(len(idx) - pos),
                "split_min": _r(float(X[idx, split_col_idx].min())),
                "split_max": _r(float(X[idx, split_col_idx].max()))}

    manifest = {
        "v": "1.0",
        "dataset": spec["dataset"],
        "source": spec.get("source", ""),
        "source_rows": int(len(raw)),
        "seed": seed,
        "n_hospitals": n_clients,
        "n_per_hospital": n_per_client,
        "split": f"seeded sample of {n_clients * n_per_client}, sorted by {split_col} ascending "
                 f"(ties by row id), contiguous blocks",
        "split_col": split_col,
        "canonical_features": holdout_features,
        "canonical_units": {f["name"]: f["unit"] for f in features},
        "label": {c["id"]: c["target"] for c in clients} | {"holdout": holdout_target},
        "target_encoding": target_values,
        "imputation": {"zeros_replaced_in": [f["name"] for f in features if f.get("impute_zero")],
                       "with": "median of non-zero values, full dataset", "medians": medians},
        "feature_mean": [float(v) for v in mean],
        "feature_std": [float(v) for v in std],
        "per_hospital_class_balance": {c["id"]: balance(blocks[i]) for i, c in enumerate(clients)},
        "row_ids": {c["id"]: [int(i) for i in blocks[ci]] for ci, c in enumerate(clients)},
        "holdout": {"file": "holdout/test.csv", "fraction": holdout_fraction,
                    "balance": balance(holdout), "row_ids": [int(i) for i in holdout],
                    "feature_columns": holdout_features, "label": holdout_target},
        "files": files,
        "schema": schema,
        "node_labels": spec.get("node_labels", {c["id"]: c["id"] for c in clients}),
        "note": "Values are median-imputed, NOT z-scored. Map local→canonical (Schema agent, "
                "via manifest 'schema'), then z-score with feature_mean/feature_std in "
                "canonical_features order.",
    }
    if spec.get("faults"):
        manifest["faults"] = spec["faults"]
    if spec.get("honesty"):
        manifest["honesty"] = spec["honesty"]
    if spec.get("rounds"):
        manifest["rounds"] = spec["rounds"]
    io.save_json(out_dir / "hospitals" / "manifest.json", manifest)
    return manifest
