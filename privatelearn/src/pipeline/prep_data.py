"""T1 — data prep and non-IID split (CONTRACTS.md §1 and §4).

Reads data/raw/diabetes.csv (Pima Indians Diabetes, 768 rows) and writes:

- data/hospitals/hospital_{0,1,2}.csv — 120 rows each in that hospital's LOCAL
  column names and units, with median-imputed but NOT z-scored values. Mapping to
  canonical is the Schema agent's job (T7).
- data/holdout/test.csv — 20% of the full dataset (154 rows), disjoint from every
  hospital, in canonical names/units. Used only to evaluate the global model.
- data/hospitals/manifest.json — seed, sizes, canonical feature order, full-dataset
  medians/mean/std (canonical units), per-hospital class balance and row ids.

Deterministic: same raw file + seed 42 → byte-identical outputs.

    python src/pipeline/prep_data.py
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shared import io  # noqa: E402

SEED = 42
N_HOSPITALS = 3
N_PER_HOSPITAL = 120
HOLDOUT_FRACTION = 0.2

# Canonical feature order (CONTRACTS.md §1). Weight w[1+i] belongs to CANONICAL[i].
CANONICAL = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Insulin", "Age"]
CANONICAL_SNAKE = {
    "Pregnancies": "pregnancies", "Glucose": "glucose", "BloodPressure": "blood_pressure",
    "BMI": "bmi", "Insulin": "insulin", "Age": "age",
}
LABEL = "Outcome"
IMPUTE_ZEROS = ["Glucose", "BloodPressure", "BMI", "Insulin"]

# Unit factors (CONTRACTS.md §4): canonical = local * factor (or local / divisor).
MGDL_PER_MMOLL = 18.0182
MMHG_PER_KPA = 7.50062
PMOLL_PER_UUML = 6.945
MONTHS_PER_YEAR = 12

DECIMALS = 9  # local-unit values are rounded here; conversion error stays < 1e-6


def load_raw(path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if len(rows) != 768:
        raise ValueError(f"{path}: expected 768 Pima rows, got {len(rows)}")
    cols = CANONICAL + [LABEL]
    missing = [c for c in cols if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def impute(raw: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Replace 0 (= missing in Pima) with the median of the non-zero values."""
    out = {c: v.copy() for c, v in raw.items()}
    medians = {}
    for c in IMPUTE_ZEROS:
        med = float(np.median(raw[c][raw[c] != 0]))
        medians[c] = med
        out[c][out[c] == 0] = med
    return out, medians


def split_indices(raw: dict[str, np.ndarray], seed: int = SEED):
    """360 seeded rows sorted by Age (ties by row id) → 3 contiguous blocks; holdout from the rest."""
    n = len(raw[LABEL])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=N_HOSPITALS * N_PER_HOSPITAL, replace=False)
    chosen = chosen[np.lexsort((chosen, raw["Age"][chosen]))]  # primary key: Age, then id
    blocks = [np.sort(chosen[i * N_PER_HOSPITAL:(i + 1) * N_PER_HOSPITAL]) for i in range(N_HOSPITALS)]
    rest = np.setdiff1d(np.arange(n), chosen)
    n_holdout = int(round(HOLDOUT_FRACTION * n))
    holdout = np.sort(rng.choice(rest, size=n_holdout, replace=False))
    return blocks, holdout


def _r(x) -> float:
    return round(float(x), DECIMALS)


def local_rows(hospital: int, clean: dict[str, np.ndarray], idx: np.ndarray, seed: int = SEED):
    """Rows for one hospital in its local schema (CONTRACTS.md §4)."""
    g = {c: clean[c][idx] for c in CANONICAL + [LABEL]}
    if hospital == 0:
        header = ["Age", "BMI", "Glucose", "BloodPressure", "Insulin", "Pregnancies", "Outcome"]
        cols = [g["Age"], g["BMI"], g["Glucose"], g["BloodPressure"], g["Insulin"],
                g["Pregnancies"], g[LABEL]]
        kinds = ["int", "f", "f", "f", "f", "int", "int"]
    elif hospital == 1:
        # Heights are synthetic (Pima has none); weight is derived so weight/height² == BMI.
        rng = np.random.default_rng(seed + 1)
        height = np.round(np.clip(rng.normal(1.62, 0.07, size=len(idx)), 1.45, 1.85), 3)
        weight = g["BMI"] * height ** 2
        header = ["age_months", "glucose_mmol_l", "bp_kpa", "height_m", "weight_kg",
                  "insulin_pmol_l", "pregnancies", "outcome"]
        cols = [g["Age"] * MONTHS_PER_YEAR, g["Glucose"] / MGDL_PER_MMOLL,
                g["BloodPressure"] / MMHG_PER_KPA, height, weight,
                g["Insulin"] * PMOLL_PER_UUML, g["Pregnancies"], g[LABEL]]
        kinds = ["int", "f", "f", "f", "f", "f", "int", "int"]
    elif hospital == 2:
        header = ["age", "bmi", "glucose", "bp_sys", "insulin_pmol_l", "num_pregnancies", "outcome"]
        cols = [g["Age"], g["BMI"], g["Glucose"], g["BloodPressure"],
                g["Insulin"] * PMOLL_PER_UUML, g["Pregnancies"], g[LABEL]]
        kinds = ["int", "f", "f", "f", "f", "int", "int"]
    else:
        raise ValueError(f"unknown hospital {hospital}")
    rows = []
    for i in range(len(idx)):
        rows.append([int(round(c[i])) if k == "int" else _r(c[i]) for c, k in zip(cols, kinds)])
    return header, rows


def _write_csv(path: pathlib.Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def prepare(raw_path, out_dir, seed: int = SEED) -> dict:
    out_dir = pathlib.Path(out_dir)
    raw = load_raw(raw_path)
    clean, medians = impute(raw)
    X = np.column_stack([clean[c] for c in CANONICAL])
    mean = X.mean(axis=0)
    std = X.std(axis=0)  # population std over the full 768 rows
    blocks, holdout = split_indices(raw, seed)

    for h, idx in enumerate(blocks):
        header, rows = local_rows(h, clean, idx, seed)
        _write_csv(out_dir / "hospitals" / f"hospital_{h}.csv", header, rows)

    test_rows = [[_r(clean[c][i]) if c not in ("Age", "Pregnancies") else int(clean[c][i])
                  for c in CANONICAL] + [int(clean[LABEL][i])] for i in holdout]
    _write_csv(out_dir / "holdout" / "test.csv", CANONICAL + [LABEL], test_rows)

    def balance(idx):
        pos = int(clean[LABEL][idx].sum())
        return {"n": int(len(idx)), "positive": pos, "negative": int(len(idx) - pos),
                "age_min": int(clean["Age"][idx].min()), "age_max": int(clean["Age"][idx].max())}

    manifest = {
        "v": "1.0",
        "dataset": "Pima Indians Diabetes (NIDDK)",
        "source_rows": int(len(raw[LABEL])),
        "seed": seed,
        "n_hospitals": N_HOSPITALS,
        "n_per_hospital": N_PER_HOSPITAL,
        "split": "seeded sample of 360, sorted by Age ascending (ties by row id), contiguous blocks",
        "canonical_features": [CANONICAL_SNAKE[c] for c in CANONICAL],
        "canonical_units": {"pregnancies": "count", "glucose": "mg/dL", "blood_pressure": "mmHg",
                            "bmi": "kg/m^2", "insulin": "uU/mL", "age": "years"},
        "label": {"h0": "Outcome", "h1": "outcome", "h2": "outcome", "holdout": "Outcome"},
        "imputation": {"zeros_replaced_in": [CANONICAL_SNAKE[c] for c in IMPUTE_ZEROS],
                       "with": "median of non-zero values, full dataset",
                       "medians": {CANONICAL_SNAKE[c]: m for c, m in medians.items()}},
        "feature_mean": [float(m) for m in mean],
        "feature_std": [float(s) for s in std],
        "per_hospital_class_balance": {f"h{h}": balance(idx) for h, idx in enumerate(blocks)},
        "row_ids": {f"h{h}": [int(i) for i in idx] for h, idx in enumerate(blocks)},
        "holdout": {"file": "holdout/test.csv", "fraction": HOLDOUT_FRACTION,
                    "balance": balance(holdout), "row_ids": [int(i) for i in holdout]},
        "files": {f"h{h}": f"hospitals/hospital_{h}.csv" for h in range(N_HOSPITALS)},
        "note": "Values are median-imputed, NOT z-scored. Map local→canonical (Schema agent), "
                "then z-score with feature_mean/feature_std in canonical_features order.",
    }
    io.save_json(out_dir / "hospitals" / "manifest.json", manifest)
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "diabetes.csv"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    m = prepare(args.raw, args.out, args.seed)
    for h, b in m["per_hospital_class_balance"].items():
        print(f"{h}: n={b['n']} pos={b['positive']} neg={b['negative']} age {b['age_min']}-{b['age_max']}")
    hb = m["holdout"]["balance"]
    print(f"holdout: n={hb['n']} pos={hb['positive']} neg={hb['negative']}")


if __name__ == "__main__":
    main()
