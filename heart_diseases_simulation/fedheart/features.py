"""The feature contract: which columns the federation uses and how each is encoded.

The coordinator agent proposes a contract and every hospital agent must accept it. A
contract is plain JSON:

    {"features": [{"name": "age", "encoding": "numeric", "missing": "impute_mean"}, ...],
     "quality_rules": [{"column": "chol", "rule": "zero_means_missing"}, ...]}

  encoding  numeric      z-scored with the FEDERATED mean/std (secure-aggregated sums)
            binary       0/1, z-scored the same way
            categorical  one-hot over the documented codes (missing → all zeros)
  missing   impute_mean                  missing → the federated mean (z = 0)
            impute_mean_with_indicator   same, plus a 0/1 "<name>__missing" column
  rules     zero_means_missing           a recorded 0 is a missing value (e.g. chol = 0)
            valid_range {min, max}       values outside are treated as missing

Everything that touches values (profiles, sums, encoding) runs inside a hospital process.
"""

from __future__ import annotations

import math

import numpy as np

from . import config

ENCODINGS = ("numeric", "binary", "categorical")
MISSING = ("impute_mean", "impute_mean_with_indicator")
RULES = ("zero_means_missing", "valid_range")


# ---------------------------------------------------------------- contract validation

def validate_contract(contract) -> dict:
    """→ a normalised copy, or ValueError with a message an agent can act on."""
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object with 'features' and 'quality_rules'")
    feats = contract.get("features")
    if not isinstance(feats, list) or not feats:
        raise ValueError("contract.features must be a non-empty list")
    out_feats, seen = [], set()
    for f in feats:
        if not isinstance(f, dict):
            raise ValueError(f"feature entry {f!r} must be an object")
        name = str(f.get("name", ""))
        if name not in config.COLUMN_BY_NAME:
            raise ValueError(f"unknown column {name!r}; columns are {config.FEATURE_NAMES}")
        if name in seen:
            raise ValueError(f"column {name!r} listed twice")
        seen.add(name)
        enc = str(f.get("encoding") or config.COLUMN_BY_NAME[name]["kind"])
        if enc not in ENCODINGS:
            raise ValueError(f"{name}: encoding must be one of {ENCODINGS}")
        if enc == "categorical" and "codes" not in config.COLUMN_BY_NAME[name]:
            raise ValueError(f"{name} has no documented category codes; encode it as numeric")
        miss = str(f.get("missing") or "impute_mean")
        if miss not in MISSING:
            raise ValueError(f"{name}: missing must be one of {MISSING}")
        out_feats.append({"name": name, "encoding": enc, "missing": miss})
    rules = []
    for r in contract.get("quality_rules") or []:
        if not isinstance(r, dict):
            raise ValueError(f"quality rule {r!r} must be an object")
        col, rule = str(r.get("column", "")), str(r.get("rule", ""))
        if col not in config.COLUMN_BY_NAME:
            raise ValueError(f"quality rule on unknown column {col!r}")
        if rule not in RULES:
            raise ValueError(f"quality rule must be one of {RULES}")
        entry = {"column": col, "rule": rule}
        meta = config.COLUMN_BY_NAME[col]
        if rule == "zero_means_missing" and (meta["kind"] == "binary" or 0 in meta.get("codes", [])):
            raise ValueError(f"0 is a valid code for {col} ({meta['unit']}); zero_means_missing would "
                             "delete real values")
        if rule == "valid_range":
            lo, hi = r.get("min"), r.get("max")
            if lo is None or hi is None or not float(lo) < float(hi):
                raise ValueError(f"valid_range on {col} needs numeric min < max")
            entry.update(min=float(lo), max=float(hi))
        rules.append(entry)
    return {"features": out_feats, "quality_rules": rules}


def apply_rules(table: dict, rules) -> dict:
    """A copy of the table with the quality rules applied (bad values → NaN)."""
    out = {c: v.astype(float).copy() for c, v in table.items()}
    for r in rules:
        col = out[r["column"]]
        if r["rule"] == "zero_means_missing":
            col[col == 0] = np.nan
        elif r["rule"] == "valid_range":
            with np.errstate(invalid="ignore"):
                col[(col < r["min"]) | (col > r["max"])] = np.nan
    return out


# ---------------------------------------------------------------- local profiles (for the hospital's own agent)

def _q(v, p):
    return round(float(np.quantile(v, p)), 3)


def column_summary(values: np.ndarray, detail: bool = False) -> dict:
    present = values[~np.isnan(values)]
    s = {"missing": int(np.isnan(values).sum()), "missing_rate": round(float(np.isnan(values).mean()), 3),
         "zeros": int((present == 0).sum()), "distinct": int(len(np.unique(present)))}
    if present.size:
        s.update(min=round(float(present.min()), 3), median=_q(present, 0.5), max=round(float(present.max()), 3))
        if detail:
            s.update(p05=_q(present, 0.05), p25=_q(present, 0.25), p75=_q(present, 0.75), p95=_q(present, 0.95),
                     mean=round(float(present.mean()), 3))
            if s["distinct"] <= 10:
                vals, counts = np.unique(present, return_counts=True)
                s["value_counts"] = {f"{v:g}": int(c) for v, c in zip(vals, counts)}
    return s


def overview(table: dict, y: np.ndarray) -> dict:
    return {"rows": int(len(y)), "label_positive_rate": round(float(y.mean()), 3),
            "columns": {c: column_summary(table[c]) for c in config.FEATURE_NAMES}}


def contract_impact(table: dict, contract: dict) -> dict:
    """What this contract would mean for THIS site's rows (after its quality rules)."""
    t = apply_rules(table, contract["quality_rules"])
    n = len(next(iter(t.values())))
    per, any_missing = {}, np.zeros(n, dtype=bool)
    for f in contract["features"]:
        v = t[f["name"]]
        miss = np.isnan(v)
        any_missing |= miss
        e = {"missing_rate_after_rules": round(float(miss.mean()), 3)}
        if f["encoding"] == "categorical":
            codes = config.COLUMN_BY_NAME[f["name"]]["codes"]
            present = v[~miss]
            e["undocumented_codes"] = sorted({f"{x:g}" for x in present if x not in codes})
        per[f["name"]] = e
    return {"rows": n, "rows_with_any_missing_feature": int(any_missing.sum()),
            "fully_missing_features": [k for k, e in per.items() if e["missing_rate_after_rules"] >= 0.999],
            "features": per}


# ---------------------------------------------------------------- federated statistics

def stats_layout(contract: dict) -> list:
    """The order of the secure-aggregated statistics vector, the same at every site."""
    layout = [("_rows", "n"), ("_label", "pos")]
    for f in contract["features"]:
        layout += [(f["name"], "count"), (f["name"], "sum"), (f["name"], "sumsq")]
        if f["encoding"] == "categorical":
            layout += [(f["name"], f"code={c:g}") for c in config.COLUMN_BY_NAME[f["name"]]["codes"]]
    return layout


def local_stats(table: dict, y: np.ndarray, contract: dict) -> np.ndarray:
    """This site's sums in stats_layout order (masked before they leave the site)."""
    t = apply_rules(table, contract["quality_rules"])
    out = []
    for name, key in stats_layout(contract):
        if name == "_rows":
            out.append(float(len(y)))
        elif name == "_label":
            out.append(float(y.sum()))
        else:
            v = t[name]
            present = v[~np.isnan(v)]
            if key == "count":
                out.append(float(present.size))
            elif key == "sum":
                out.append(float(present.sum()))
            elif key == "sumsq":
                out.append(float((present ** 2).sum()))
            else:
                out.append(float((present == float(key.split("=")[1])).sum()))
    return np.array(out)


def finalize_norm(contract: dict, totals: np.ndarray) -> dict:
    """Federated mean/std/missing-rate from the summed statistics (all sites together)."""
    layout = stats_layout(contract)
    val = {k: float(v) for k, v in zip(layout, totals)}
    n_rows = val[("_rows", "n")]
    norm = {"rows": int(round(n_rows)), "label_positive_rate": round(val[("_label", "pos")] / max(n_rows, 1), 4),
            "features": {}}
    for f in contract["features"]:
        name = f["name"]
        count = val[(name, "count")]
        entry = {"missing_rate": round(1.0 - count / max(n_rows, 1), 4)}
        if count > 0:
            mean = val[(name, "sum")] / count
            var = max(val[(name, "sumsq")] / count - mean ** 2, 0.0)
            entry.update(mean=mean, std=math.sqrt(var) if var > 1e-12 else 1.0)
        else:  # no site has this column: it contributes nothing (z = 0 everywhere)
            entry.update(mean=0.0, std=1.0)
        if f["encoding"] == "categorical":
            codes = config.COLUMN_BY_NAME[name]["codes"]
            entry["code_rates"] = {f"{c:g}": round(val[(name, f"code={c:g}")] / max(count, 1), 4) for c in codes}
        norm["features"][name] = entry
    return norm


# ---------------------------------------------------------------- encoding

def encoded_names(contract: dict) -> list:
    names = []
    for f in contract["features"]:
        if f["encoding"] == "categorical":
            names += [f"{f['name']}={c:g}" for c in config.COLUMN_BY_NAME[f["name"]]["codes"]]
        else:
            names.append(f["name"])
        if f["missing"] == "impute_mean_with_indicator":
            names.append(f"{f['name']}__missing")
    return names


def encode(table: dict, contract: dict, norm: dict) -> np.ndarray:
    """Rows → the model's input matrix, in encoded_names order."""
    t = apply_rules(table, contract["quality_rules"])
    cols = []
    for f in contract["features"]:
        v = t[f["name"]]
        miss = np.isnan(v)
        if f["encoding"] == "categorical":
            for c in config.COLUMN_BY_NAME[f["name"]]["codes"]:
                cols.append(np.where(miss, 0.0, (v == c).astype(float)))
        else:
            st = norm["features"][f["name"]]
            cols.append(np.where(miss, 0.0, (np.nan_to_num(v) - st["mean"]) / st["std"]))
        if f["missing"] == "impute_mean_with_indicator":
            cols.append(miss.astype(float))
    return np.column_stack(cols) if cols else np.zeros((len(next(iter(table.values()))), 0))


def encode_record(record: dict, contract: dict, norm: dict) -> np.ndarray:
    """One patient (dict column → value or None) → a 1-row input matrix (for predict.py)."""
    table = {}
    for c in config.FEATURE_NAMES:
        v = record.get(c)
        table[c] = np.array([float("nan") if v is None or v == "" else float(v)])
    return encode(table, contract, norm)
