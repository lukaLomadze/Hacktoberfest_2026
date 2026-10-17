# PrivateLearn — Execution Plan

> Agentic federated learning: several hospitals train **one model** together without any
> patient row leaving a building. Each hospital runs a small team of agents; one Coordinator
> negotiates a common feature schema and adaptively aggregates. Everyone ends holding the
> **same** model.

**How to use this file:** the task table (§6) is the source of truth. Create one GitHub issue
per task (T0–T12) with its acceptance criteria in the body. Share §4–§5 (contracts) with every
agent before they write code — they are what keep parallel work from colliding.

---

## 1. The goal in one sentence

Three hospitals, each with too little data to train alone and legally unable to share it,
use agent teams and one coordinator to train a single logistic-regression model that matches
centralized quality — and we prove the data never moved.

## 2. Numbers (deliberately tiny, so everything is instant and explainable)

| | |
|---|---|
| Hospitals | 3 (synthetic partitions) |
| Rows per hospital | 120 |
| Features | 6: age, BMI, glucose, blood_pressure, insulin, pregnancies |
| Model | logistic regression → **7 weights** |
| Rounds | 4 |
| Solo AUC | ≈ 0.84 |
| Federated AUC | ≈ 0.88 |
| Centralized (illegal) AUC | ≈ 0.88 |

These are **measured** on the real Pima data (age-band split, held-out 20%), not assumed —
see `privatelearn/docs/CONTRACTS.md` §0. The story is **not** a big accuracy jump: federated
matches centralized quality to within noise. The win is that the best model becomes **legal**
to build, with zero rows shared. (The original plan claimed 0.72 → 0.87; that was fabricated —
Pima is too easy for 120-row hospitals to underfit. Testing caught it.)

## 3. The 5-minute demo

**0:00–0:15 — Title.** "PrivateLearn — one model, zero shared data."

**0:15–0:50 — The problem.** Three hospital cards, each "120 rows", each plateauing at AUC 0.84.
A red arrow tries to connect them → stamped **"sharing illegal"**. A ghost line above reads
"centralized (illegal) 0.88".
> *"Train alone and the best any of them gets is 0.84. Pool the data — that's 0.88 — but
> patient data can't legally leave the building. The best model is the one nobody is allowed
> to build."*

**0:50–1:40 — How agents solve it.** Inside one hospital, four agent cards light up in order —
Schema → Planner → Trainer → Challenger — then a Coordinator node appears in the centre.
> *"Each hospital runs a small team of agents. A Schema agent describes its columns — named
> differently everywhere. A Trainer fits a logistic regression on local rows. A Challenger
> tries to break it. Only weight deltas and losses leave."*

**1:40–2:15 — The loop.** Packets (7 weight bars + a loss number) fly from hospitals to the
Coordinator, which picks a strategy (FedAvg/FedProx/trimmed-mean), aggregates, broadcasts.
Four round dots fill in.

**2:15–3:00 — Privacy, three layers.** Three shields stack: ① data never leaves · ② secure
aggregation (Coordinator sees only the sum) · ③ differential privacy (ε=1.0).

**3:00–3:45 — The attack fails.** Split screen: a gradient-inversion attacker reconstructs a
recognisable patient row from an *unmasked* update, then the same attack against masked+noised
updates dissolves into static. Stamp: **FAIL**.
> *"Let's prove it. With no protection, it recovers real-looking values. Switch on masking and
> DP — pure noise. The attack fails exactly where it used to succeed."*

**3:45–4:30 — One identical model.** Solo curves sit at 0.84; the federated line climbs to meet
0.88, the illegal centralized number. Below, three identical SHA-256 weight hashes appear over the three hospitals.

**4:30–5:00 — Honesty + close.**
> *"These are 360 simulated patient records. The mechanism — agents, secure aggregation,
> differential privacy, the failing attack — is real. The scale is a simulation."*

### The four "wow" visuals
1. **Moving packets, never rows** — weight bars fly to the Coordinator while a "raw data"
   packet is visibly blocked at each hospital's wall.
2. **The attack that dissolves** — a crisp reconstruction turns to static the moment masking + DP flip on.
3. **Three lines, one crossing** — solo (0.84) below, federated and centralized (0.88) meeting at the top.
4. **Matching fingerprints** — three identical weight hashes appear simultaneously.

---

## 4. Repository layout & ownership

Ownership is **exclusive** — an agent only edits files in its lane. Shared files have one owner.
Branch per task: `<github-username>/<short-task-name>`; PR into `main`, reviewed by someone else.

> If this repo is shared with other hackathon projects, nest everything below in a top-level
> `privatelearn/` directory and keep the paths otherwise identical.

```
privatelearn/
├── README.md                     # C
├── .gitignore                    # A
├── docs/
│   ├── EXECUTION-PLAN.md         # A (this file)
│   ├── CONTRACTS.md              # A (sole owner; the schemas in §5)
│   └── deck.md                   # C (slides + talk track)
├── data/
│   ├── raw/diabetes.csv          # B (download once, commit)
│   ├── hospitals/hospital_{0,1,2}.csv   # B (generated in T1)
│   ├── hospitals/manifest.json   # B
│   └── sample_trace.json         # A (fixture for C, written in T0)
├── src/
│   ├── model/                    # A only (T2–T5)
│   │   ├── logistic.py
│   │   ├── aggregation.py
│   │   ├── privacy.py
│   │   └── attack.py
│   ├── agents/                   # B only (T6–T8)
│   │   ├── runtime.py
│   │   ├── hospital.py
│   │   └── coordinator.py
│   ├── pipeline/                 # B only (T1, T9)
│   │   ├── prep_data.py
│   │   └── run_loop.py
│   └── shared/                   # A only (contract enforcement)
│       ├── schemas.py            # validators for update/broadcast/trace JSON
│       └── io.py
├── web/                          # C only (T10–T11)
│   ├── index.html
│   ├── player.js
│   └── player.css
├── trace.json                    # generated by B (T9), read by C
└── tests/
    ├── test_model/               # A
    └── test_agents/              # B
```

---

## 5. Shared contracts (define once, reuse everywhere)

### 5.1 Dataset & non-IID split
**Pima Indians Diabetes** (UCI / NIDDK, 768 rows). Keep 6 canonical features in this order:
`Pregnancies, Glucose, BloodPressure, BMI, Insulin, Age`. Zero-impute `Glucose/BloodPressure/
BMI/Insulin` with column medians; z-score with full-dataset mean/std.

`seed = 42` → sample 360 rows → sort by `Age` ascending (a feature, never the label — sorting by
Outcome gives single-class hospitals that cannot train) → contiguous 120-row blocks →
hospital_0 youngest, hospital_1 middle, hospital_2 oldest. Each hospital keeps both classes but a
different age distribution. Measured (held-out): solo ≈ 0.84, federated ≈ 0.88, centralized ≈ 0.88.

Each hospital CSV stores its **local** column names/units; only the Schema agent maps to canonical.

### 5.2 Model
Logistic regression, weights `w ∈ R⁷` (bias + 6). `p = σ(w·[1;x])`, mean BCE loss, gradient
`(p−y)·[1;x]`. Local: 4 full-batch epochs, `lr=0.1`. **Update = `delta_w = w_new − w_prev` + `loss` + `n`.**
AUC via Mann–Whitney ranking. `numpy` only.

### 5.3 Update message (hospital → coordinator)
```json
{ "v":"1.0", "round":2, "hospital":"h1",
  "delta_w":[0.031,-0.012,0.004,0.021,-0.008,0.017,0.002],
  "loss":0.512, "n":120,
  "privacy":{"masked":true,"mask_seed":"b3f8…","dp_eps":1.5,"dp_mechanism":"gaussian","dp_sigma":0.008} }
```
No raw-data field exists. With masking, `delta_w` looks random; masks cancel only in the sum.

### 5.4 Broadcast message (coordinator → hospitals)
```json
{ "v":"1.0", "round":2, "w":[0.41,0.88,-0.12,1.02,0.31,-0.04,0.05],
  "strategy":"fedprox", "eps_total":3.0, "auc":0.84,
  "losses":[0.51,0.53,0.49], "converged":false }
```

### 5.5 Canonical feature schema & unit conversions
| # | canonical | unit | local → canonical |
|---|---|---|---|
| 1 | `age` | years | months ÷ 12 |
| 2 | `bmi` | kg/m² | derive from weight_kg / height_m² |
| 3 | `glucose` | mg/dL | mmol/L × 18.0182 |
| 4 | `blood_pressure` | mmHg | kPa × 7.50062 |
| 5 | `insulin` | µU/mL | pmol/L ÷ 6.945 |
| 6 | `pregnancies` | count | rename only |

Hospital locals on purpose: **h0** matches canonical. **h1** uses `glucose_mmol_l`, `bp_kpa`,
`height_m`+`weight_kg`. **h2** uses `age, bmi, glucose, bp_sys, insulin_pmol_l, num_pregnancies`.
The Schema agent emits a per-hospital mapping; the Coordinator merges into **one** common schema.

### 5.6 Trace JSON (single source of truth for the frontend player)
```json
{ "v":"1.0",
  "meta":{"dataset":"Pima Indians Diabetes (NIDDK)","hospitals":3,"rows_per_hospital":120,
    "features":["age","bmi","glucose","blood_pressure","insulin","pregnancies"],
    "weights":7,"rounds":4,"auc_solo":0.84,"auc_federated":0.877,"auc_centralized":0.884},
  "nodes":[{"id":"h0","label":"Hospital A","role":"hospital"},{"id":"h1","label":"Hospital B","role":"hospital"},
           {"id":"h2","label":"Hospital C","role":"hospital"},{"id":"coord","label":"Coordinator","role":"coordinator"}],
  "rounds":[{"round":1,"strategy":"fedavg","eps_total":1.5,"auc_before":0.78,"auc_after":0.81,
    "decision":"loss variance < τ → FedAvg",
    "updates":[{"hospital":"h0","loss":0.51,"n":120,"masked":true,"dp_eps":1.5},
               {"hospital":"h1","loss":0.53,"n":120,"masked":true,"dp_eps":1.5},
               {"hospital":"h2","loss":0.49,"n":120,"masked":true,"dp_eps":1.5}],
    "broadcast":{"w":[0.40,0.85,-0.11,1.00,0.30,-0.03,0.05],"auc":0.81}}],
  "metrics":{"auc_series":[0.84,0.86,0.87,0.875,0.877],"loss_series":[0.60,0.52,0.46,0.41,0.38]},
  "attack":{"target":"h0","without_privacy":{"recovered":true,"feature_error":0.12},
            "with_privacy":{"recovered":false,"feature_error":1.9}},
  "timeline_ms":60000 }
```
`src/shared/schemas.py` validates every message and `trace.json` against these shapes.

---

## 6. Task list (T0–T12)

Lanes: **A** model+math · **B** agents+schema · **C** frontend+demo. `⟶` = critical path.

| ID | Title (owner) | Deps | Hrs | Deliverable | Acceptance criteria |
|---|---|---|---|---|---|
| **T0** ⟶ | Repo scaffold + `CONTRACTS.md` (A) | — | 2 | `.gitignore`, `docs/CONTRACTS.md`, `src/shared/schemas.py`+`io.py`, `data/sample_trace.json` | Every §5 schema documented with examples; validators import and run; sample trace validates; ownership locked. |
| **T1** | Data prep + non-IID split (B) | T0 | 2 | 3 hospital CSVs, `manifest.json`, `pipeline/prep_data.py` | Deterministic (seed 42) → 3×120 rows; skew matches §5.1; hospitals hold **local** names/units. |
| **T2** ⟶ | Logistic-regression core (A) | T0 | 3 | `model/logistic.py`, tests | `forward/grad/loss` match hand fixture to 1e-6; loss decreases over 4 epochs; `auc` 0.5 random / 1.0 separable; numpy only. |
| **T3** ⟶ | Aggregation + adaptive selector (A) | T2 | 3 | `model/aggregation.py` | FedAvg, FedProx, trimmed-mean implemented; selector: FedAvg when loss std < τ, trimmed-mean on a >2σ outlier, FedProx on high heterogeneity; 1-client FedAvg == local SGD. |
| **T4** ⟶ | Secure aggregation + DP (A) | T3 | 3 | `model/privacy.py` | Pairwise masks cancel in sum to 1e-9; coordinator sees only masked vectors; DP noise matches requested ε and is reported per update + cumulative. |
| **T5** | Gradient-inversion attack demo (A) | T4 | 2 | `model/attack.py` + `trace.attack` | Recovers row from unmasked delta (err < 0.2); fails on masked/DP (err > 5× signal). |
| **T6** | Agent runtime + message bus (B) | T0 | 3 | `agents/runtime.py` | Base `Agent` with `step(messages)`; messages are §5.3/§5.4 dicts; headless, unit-tested. |
| **T7** | Hospital agents (B) | T6 | 3 | `agents/hospital.py`, tests | Schema agent maps h0/h1/h2 → canonical (1e-6); Planner schedules; Trainer emits `delta_w+loss+n`; Challenger reports local AUC; local data only. |
| **T8** ⟶ | Coordinator agent (B) | T6,T3 | 3 | `agents/coordinator.py` | Negotiates one common schema from three Schema agents; sums masked updates; applies §5.4 broadcast; tracks `eps_total` + convergence. |
| **T9** ⟶ | End-to-end run → `trace.json` (B) | T7,T8,T4 | 2 | `pipeline/run_loop.py`, `trace.json` | Exactly 4 rounds; trace validates; final AUC ≈ 0.88 and within 0.01 of centralized; runs < 60 s; honesty note present. |
| **T10** | Canvas player (C) | T0 | 5 | `web/index.html`, `player.js`, `player.css` | Renders 4 nodes + animated packets from `sample_trace.json`; scrubber seeks with correct state; vanilla JS, no framework. |
| **T11** | Chart + decision log + privacy toggles (C) | T10 | 3 | extend `web/` | `auc_series`/`loss_series` redraw on scrub; decision log shows strategy + rationale; toggles overlay masking/DP + attack section; reads only `trace.json`. |
| **T12** | Deck + recorded demo (C) | T9,T11 | 3 | `docs/deck.md`, recording, README | Deck ≤ 10 slides incl. honesty slide; recording shows replay + 0.84/0.88/0.88 + failed attack; README reproduces + replays in one command. |

**Critical path (19 h): `T0 → T2 → T3 → T4 → T8 → T9 → T12`.**
Off-path (`T1,T5,T6,T7,T10,T11`) run in parallel; C builds against `sample_trace.json` until T9
emits the real `trace.json`.

---

## 7. 24-hour schedule

| Hours | A — model+math | B — agents+schema | C — frontend+demo |
|---|---|---|---|
| 0–3 | T0 scaffold + contracts | T1 data prep | read contracts; static mock |
| 3–6 | T2 model core | T6 runtime | T10 canvas player (vs sample trace) |
| 6–9 | T3 aggregation | T7 hospital agents | T10 → T11 charts |
| 9–12 | T4 privacy (masking+DP) | T8 coordinator | T11 decision log + toggles |
| 12–14 | T5 attack demo | T8 finish | T11 finish; deck outline |
| 14–17 | buffer | T9 → `trace.json` | T12 deck draft |
| 17–20 | integration review | hand off trace to C | T12 record demo |
| 20–24 | QA, rehearsal, submit | QA, rehearsal, submit | QA, rehearsal, submit |

---

## 8. Judge Q&A (prepared answers)

**"This is just FedAvg."** — FedAvg is the aggregation math; our contribution is the agentic
control loop around it: a Coordinator that negotiates a common schema across mismatched
columns/units, then *chooses* FedAvg vs FedProx vs trimmed-mean per round. FedAvg is one of
three strategies we select, not the system.

**"Why do you need agents?"** — Every hospital is heterogeneous (different names, units,
missing features), so each needs a Schema agent to map its data and a Challenger to probe the
shared model against held-out data. Agents turn a brittle fixed script into a system that
*reasons* about schema mismatch and convergence.

**"Gradients leak."** — Correct, raw deltas can be inverted — which is exactly why we show the
inversion attack and its failure under masking + DP. Locality removes the worst exposure;
masking hides individual updates from the Coordinator; DP breaks exact inversion.

**"Why not centralise?"** — Because pooling patient records is illegal in this setting. We
get the same model (federated ≈ centralized, both ≈ 0.88) without a single row leaving —
nothing is traded away.

**"Did adaptive actually win?"** — We ran all three strategies on the same trace and kept the
decision log. On this dataset adaptive and FedAvg land within noise (both ≈ 0.88); the honest
claim is the reproducible decision log and robustness to the dropout/poisoning cases, not a big
accuracy margin. The trace is committed, so it's reproducible, not asserted.

**"Where does the LLM sit?"** — Only in the agents' *reasoning*: schema mapping, round planning,
aggregation choice, convergence calls. The training math, masking, DP noise and replay are
deterministic code; the LLM never touches weights or data.

---

## 9. Honesty & limitations

**Real:** the loop, local logistic regression from scratch, real updates/losses, real FedAvg/
FedProx/trimmed-mean, real pairwise masking, real DP noise, real attack-and-failure, and the
measured 0.84/0.88/0.88 comparison.

**Simulated:** 360 *synthetic* patient records; three synthetic partitions (not live systems);
the stage demo is a **replay** of a pre-recorded trace, not live networking.

**Not implemented (name it):** no EMR integration or certification; DP accounting is a stated-ε
demonstration, not audited RDP; secure aggregation is the honest pairwise-masking protocol, not
production MPC; limited Byzantine testing.

**Say it plainly:** *"We're not claiming to run a hospital. We claim a small agentic team can
negotiate a common schema and adaptively aggregate without any row leaving its site — and we
built the smallest honest version to prove the loop, not the infrastructure."*

### Plain-words privacy explanation (read aloud)
1. **Locality** — raw data never leaves the hospital; only *numbers* (updates) cross the boundary.
2. **Secure aggregation** — each hospital adds random masks that cancel in the sum, so the
   Coordinator sees only the total, never an individual update.
3. **Differential privacy** — calibrated noise with a stated ε guarantees no individual's data
   meaningfully changes what an observer can learn, at a small measured cost in accuracy.
   Stacked, each layer covers the others' limits — that's what the attack demo shows.

---

## 10. Freeze & rehearsal rules

**Code freeze (T−8h):** no new features, no schema/contract/format changes, no dependency
upgrades. Record `trace.json` once at freeze and treat it as immutable. Bug fixes only, and only
for demo-breaking failures, with verbal sign-off from all three.

**Rehearsal (T−8h → 0):** full timed run-through every 2 hours in the recording environment;
rehearse the *failure narration* too (script what you say if an animation skips). One person
owns the scrubber, one the script, one the clock.

**Final 2h:** cold-start test on a clean clone (trace must load with zero live dependencies);
record two takes, ship the second; no edits after the final take except cutting silence; one
person owns the deck so it doesn't compete with the demo for the same hands.
