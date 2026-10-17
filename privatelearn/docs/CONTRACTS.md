# PrivateLearn — Contracts (single source of truth)

> Read this before writing any code. It defines the exact shapes every component
> produces and consumes, so parallel agents never drift. Breaking changes go through
> a PR and bump the `v` field to `2.0`.

Version: **1.0**

---

## 0. Headline numbers — MEASURED, not assumed

Measured on the real Pima data, age-band split, logistic regression, held-out 20%:

| | AUC |
|---|---|
| best single hospital alone | ≈ 0.84 |
| federated (FedAvg, 4 rounds) | ≈ 0.88 |
| centralized (illegal) | ≈ 0.88 |

The story is therefore **not** "solo is terrible". It is: federated matches centralized
quality to within noise, **with zero rows shared** — the win is that the best model becomes
legal to build. (The earlier plan assumed 0.72 / 0.87 / 0.88; those were fabricated. The real
dataset is too easy for the "data-starved hospital" premise to bite at 120 rows. T9 re-measures
these; the trace uses whatever T9 produces.)

---

## 1. Data contract

**Dataset:** Pima Indians Diabetes (UCI / NIDDK), 768 rows, binary `Outcome`.

**Canonical feature order (fixed):** `Pregnancies, Glucose, BloodPressure, BMI, Insulin, Age`

**Cleaning (identical for every hospital):**
1. Replace `0` in `Glucose, BloodPressure, BMI, Insulin` with the column median.
2. Z-score every feature using the **full-dataset** mean/std (so the transform is the
   same everywhere, never per-hospital).

**Non-IID split (`seed = 42`):**
1. Sample 360 rows (deterministic, seeded).
2. Sort by `Age` ascending — a **continuous feature, never the label**. Sorting by
   `Outcome` would give each hospital a single class and a model that cannot learn.
3. Contiguous 120-row blocks → `hospital_0` (youngest patients), `hospital_1` (middle),
   `hospital_2` (oldest).

Each hospital keeps a **mix of both classes** but a different age/feature distribution —
the realistic "hospitals serve different populations" that makes FedAvg non-trivial.

Each hospital CSV keeps its **local** column names and units (see §4); only the
Schema agent maps them to canonical. `manifest.json` records:
`{seed, n_per_hospital, feature_mean, feature_std, per_hospital_class_balance, row_ids}`.

---

## 2. Model contract

Logistic regression, binary. Weights `w ∈ R⁷`: `w[0]` bias, `w[1:7]` feature weights
over canonical `x ∈ R⁶`.

- Prediction `p = σ(w · [1;x])`
- Loss (mean BCE) `L = -1/n Σ (y·log p + (1-y)·log(1-p))`
- Gradient `g = 1/n Σ (p-y)·[1;x]`
- Local training: 4 full-batch epochs, `lr = 0.1`
- **Update sent = `delta_w = w_new − w_prev` (R⁷) + `loss` + `n`** (never raw data)
- Metric: AUC via Mann–Whitney ranking; accuracy at threshold 0.5

`numpy` only. No torch.

---

## 3. Messages

### 3.1 Update (hospital → coordinator)

| field | type | meaning |
|---|---|---|
| `v` | string | `"1.0"` |
| `round` | int | current round |
| `hospital` | string | `"h0" | "h1" | "h2"` |
| `delta_w` | number[7] | masked delta if `privacy.masked` |
| `loss` | number | local mean BCE |
| `n` | int | local row count |
| `privacy.masked` | bool | update is masked |
| `privacy.mask_seed` | string | seed of the pairwise mask |
| `privacy.dp_eps` | number | ε of this update |
| `privacy.dp_mechanism` | string | `"gaussian"` |
| `privacy.dp_sigma` | number | noise std |

```json
{"v":"1.0","round":2,"hospital":"h1","delta_w":[0.031,-0.012,0.004,0.021,-0.008,0.017,0.002],
 "loss":0.512,"n":120,
 "privacy":{"masked":true,"mask_seed":"b3f8…","dp_eps":1.5,"dp_mechanism":"gaussian","dp_sigma":0.008}}
```

### 3.2 Broadcast (coordinator → hospitals)

| field | type | meaning |
|---|---|---|
| `v` | string | `"1.0"` |
| `round` | int | round just completed |
| `w` | number[7] | new global weights |
| `strategy` | string | `"fedavg" | "fedprox" | "trimmed_mean"` |
| `eps_total` | number | cumulative ε spent |
| `auc` | number | global AUC after this round |
| `losses` | number[] | per-participant loss |
| `converged` | bool | stop condition met |

---

## 4. Feature schema (canonical + unit conversions)

| # | canonical | type | unit | local → canonical |
|---|---|---|---|---|
| 1 | `age` | float | years | months ÷ 12 |
| 2 | `bmi` | float | kg/m² | derive from weight_kg / height_m² |
| 3 | `glucose` | float | mg/dL | mmol/L × 18.0182 |
| 4 | `blood_pressure` | float | mmHg | kPa × 7.50062 |
| 5 | `insulin` | float | µU/mL | pmol/L ÷ 6.945 |
| 6 | `pregnancies` | int | count | rename only |

**Local schemas (deliberately different, on purpose):**
- `h0` — canonical names: `Age, BMI, Glucose, BloodPressure, Insulin, Pregnancies`
- `h1` — `glucose_mmol_l, bp_kpa, height_m, weight_kg, insulin_pmol_l, pregnancies` (BMI derived)
- `h2` — `age, bmi, glucose, bp_sys, insulin_pmol_l, num_pregnancies`

The Schema agent emits a per-hospital mapping dict; the Coordinator merges three mappings
into **one** common schema (same 6 canonical features, same units).

---

## 5. Trace contract (frontend player input)

The frontend plays **only** `trace.json`. It never computes — it replays. Enriched fields
the player needs are `rounds[i].excluded` and `updates[j].sent/rejected/reason`.

```json
{
  "v":"1.0",
  "meta":{
    "dataset":"Pima Indians Diabetes (NIDDK)",
    "hospitals":3,"rows_per_hospital":120,
    "features":["age","bmi","glucose","blood_pressure","insulin","pregnancies"],
    "weights":7,"rounds":4,
    "auc_solo":0.84,"auc_federated":0.877,"auc_centralized":0.884},
  "nodes":[
    {"id":"h0","label":"Hospital A","color":"#4c6ef5","role":"hospital"},
    {"id":"h1","label":"Hospital B","color":"#40c057","role":"hospital"},
    {"id":"h2","label":"Hospital C","color":"#fa5252","role":"hospital"},
    {"id":"coord","label":"Coordinator","color":"#fab005","role":"coordinator"}],
  "rounds":[
    {"round":1,"strategy":"fedavg","eps_total":1.5,"auc_before":0.78,"auc_after":0.81,
     "decision":"loss variance < τ → FedAvg","excluded":[],
     "updates":[
       {"hospital":"h0","loss":0.51,"n":120,"sent":true,"masked":true,"dp_eps":1.5},
       {"hospital":"h1","loss":0.53,"n":120,"sent":true,"masked":true,"dp_eps":1.5},
       {"hospital":"h2","loss":0.49,"n":120,"sent":true,"masked":true,"dp_eps":1.5}],
     "broadcast":{"v":"1.0","round":1,"w":[0.40,0.85,-0.11,1.00,0.30,-0.03,0.05],
       "strategy":"fedavg","eps_total":1.5,"auc":0.81,"losses":[0.51,0.53,0.49],"converged":false}}
  ],
  "metrics":{"auc_series":[0.78,0.81,0.84,0.86,0.87],"loss_series":[0.60,0.52,0.46,0.41,0.38]},
  "attack":{"target":"h0",
    "without_privacy":{"recovered":true,"feature_error":0.12},
    "with_privacy":{"recovered":false,"feature_error":1.9}},
  "timeline_ms":60000
}
```

Per-round `decision` is a human-readable string for the decision log. `excluded` lists
hospitals whose update was rejected or absent this round.

---

## 6. Validation rules (`src/shared/schemas.py`)

- `validate_update(msg)` — `v=="1.0"`; `delta_w` is 7 floats; `loss` is a finite float;
  `n` is a positive int; `privacy` present with `masked`, `dp_eps`, `dp_mechanism`, `dp_sigma`.
- `validate_broadcast(msg)` — `w` is 7 floats; `strategy` in the allowed set; `eps_total`
  non-negative; `auc` in [0,1]; `converged` is bool.
- `validate_trace(t)` — meta/nodes/rounds/metrics/attack shapes match §5; rounds reference
  only known node ids; `auc_series` length == `rounds+1`.
- On any violation raise `ValueError` naming the field and the problem.

`src/shared/io.py` — `load_json(path)`, `save_json(path, obj, indent=2)`.
