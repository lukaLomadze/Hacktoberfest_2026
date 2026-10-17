# PrivateLearn — deck + talk track

10 slides, 5 minutes. Every number below is read from the committed `trace.json`
(regenerate with `python demo.py`). Slides are separated by `---`; the
**Say** block under each slide is the talk track.

---

## 1 · PrivateLearn

**One model. Zero shared data.**

Three hospitals train a single diabetes-risk model together —
and not one patient row leaves a building.

**Say (0:00–0:15):** "PrivateLearn: three hospitals build one model together without
sharing a single patient record. Let me show you why that matters, how agents make it
work, and then try to break it."

---

## 2 · The problem

| | AUC on held-out patients |
|---|---|
| Hospital A alone (youngest patients) | 0.840 |
| Hospital B alone | 0.838 |
| **Hospital C alone (oldest patients)** | **0.763** |
| Pool all 360 rows — *illegal* | 0.836 |

Each hospital serves a different population. The one with the oldest patients
is badly served by its own data — and pooling patient records is not allowed.

**Say (0:15–0:50):** "Every hospital sees a different slice of the population. Hospital C
treats the oldest patients and, alone, gets 0.76. The obvious fix — pool the data — gives
0.84, but patient data legally can't leave the building. The best model is the one
nobody is allowed to build."

---

## 3 · How: a team of agents per hospital

```
 Hospital A          Hospital B          Hospital C
 ┌───────────┐       ┌───────────┐       ┌───────────┐
 │ Schema    │       │ Schema    │       │ Schema    │
 │ Planner   │       │ Planner   │       │ Planner   │
 │ Trainer   │       │ Trainer   │       │ Trainer   │
 │ Challenger│       │ Challenger│       │ Challenger│
 └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
       └── 7 numbers ──► Coordinator ◄── 7 numbers ──┘
```

- **Schema** maps local columns and units to one shared schema
- **Planner** schedules local training · **Trainer** trains on local rows only
- **Challenger** scores every model on local data
- **Coordinator** negotiates the schema and picks the aggregation rule each round

**Say (0:50–1:20):** "Each hospital runs four small agents. Only the weight update — seven
numbers — and a loss ever leave. The Coordinator never sees a row."

---

## 4 · Same data, three dialects

| canonical | Hospital A | Hospital B | Hospital C |
|---|---|---|---|
| glucose (mg/dL) | `Glucose` | `glucose_mmol_l` × 18.0182 | `glucose` |
| blood pressure (mmHg) | `BloodPressure` | `bp_kpa` × 7.50062 | `bp_sys` |
| BMI | `BMI` | `weight_kg / height_m²` | `bmi` |
| insulin (µU/mL) | `Insulin` | `insulin_pmol_l` ÷ 6.945 | `insulin_pmol_l` ÷ 6.945 |
| age (years) | `Age` | `age_months` ÷ 12 | `age` |

The Schema agents describe columns (names and units, never values); the Coordinator
merges them into **one** schema. Conversions are exact to 10⁻⁶.

**Say (1:20–1:40):** "Real hospitals don't agree on column names or units. Hospital B
records glucose in mmol per litre and has no BMI column at all — its Schema agent derives
it. The Coordinator merges three dialects into one schema before any training starts."

---

## 5 · The loop — and the decisions

| round | what the Coordinator saw | decision |
|---|---|---|
| 1 | all losses equal | **FedAvg** |
| 2 | Hospital C offline | **FedAvg** over A and B |
| 3 | C's loss 2.49 vs median 0.64 — its export started sending *age in months* | **trimmed-mean**, C excluded |
| 4 | losses agree | **FedAvg** |

Probe → decide → train → aggregate → broadcast. The Coordinator decides from
clear-text losses **before** anything is masked.

**Say (1:40–2:15):** "Every round the Coordinator asks each hospital how the current model
does on its data. In round 2 Hospital C drops out. In round 3 its export silently starts
sending age in months — its loss spikes to 2.5, the Coordinator excludes it for that
round, and the model is protected. Every decision is logged with its reason."

---

## 6 · Privacy, three layers

1. **Locality** — rows never leave the hospital process; only 7 numbers + a loss do.
2. **Secure aggregation** — pairwise masks from secrets *only the hospitals hold*,
   fresh every round. Individual updates look like noise; the masks cancel in the sum
   to 10⁻⁹. The Coordinator sees only the total.
3. **Differential privacy** — per-patient gradient clipping + Gaussian noise:
   **ε = 1.5 per round, δ = 10⁻⁵**, at patient-level sensitivity.

Cost of all three, same scenario with privacy off vs on: **AUC 0.831 vs 0.832** —
no measurable accuracy cost.

**Say (2:15–3:00):** "Three layers. Data never leaves. Updates are masked so the
Coordinator only ever sees the sum. And each update carries calibrated noise so no single
patient changes what anyone can learn. We ran the identical scenario with privacy switched off: 0.831. With all three layers on:
0.832. Privacy cost us nothing measurable."

---

## 7 · Let's attack it

Gradient inversion on a **real** patient of Hospital A
(worst case: an update from a single record).

| | reconstruction error | result |
|---|---|---|
| unprotected update | **0.00** | **RECOVERED** — exact patient row |
| masked + DP update (what the Coordinator receives) | **2.40** | **FAILED** — noise |

**Say (3:00–3:45):** "Now let's try to steal a patient. From an unprotected update, the
attacker recovers the record exactly — zero error. Same attack against what our
Coordinator actually receives: error 2.4, pure noise. The attack fails exactly where it
used to succeed."

---

## 8 · One identical model

| | AUC |
|---|---|
| Hospital C alone | 0.763 |
| **PrivateLearn (federated, masked, DP)** | **0.832** |
| Pooled data (illegal) | 0.836 |

- Within **0.004** of the illegal centralized model.
- Hospital C: **+0.07** for joining.
- All three hospitals hold the byte-identical model:
  SHA-256 `d549c827…69ac7` × 3.
- **0 bytes** of patient data transferred.

**Say (3:45–4:30):** "The result: within four thousandths of the model we're not allowed
to build, the weakest hospital gains seven points, and all three end up with exactly the
same model — same fingerprint. Zero bytes of patient data moved."

---

## 9 · Honesty slide

**Real:** the Pima Indians Diabetes data (360 records), the agents, the schema mapping,
the logistic regression, pairwise masking, DP noise, the attack and its failure, and
every number on these slides — produced by the code and committed in `trace.json`.

**Simulated:** three hospitals are partitions of one public dataset; the demo replays a
recorded run, not live networking; the round-2 outage and round-3 unit error are
scripted faults.

**Not claimed:** no EMR integration or certification; DP is a stated-ε demonstration
(analytic Gaussian, basic composition), not an audited accountant; pairwise masking is
the honest textbook protocol, not production MPC. On this dataset the strongest
hospital alone is already near the pooled model — the win is legality and the weakest
hospital, not a big accuracy jump.

**Say (4:30–4:50):** "To be clear about what's real: the data, the mechanism and every
number are real; the hospitals are simulated partitions and this is a replay. We built
the smallest honest version that proves the loop."

---

## 10 · Reproduce it

```bash
cd privatelearn
python demo.py          # rebuild data + trace.json, then replay in the browser
python -m unittest discover -s tests/test_model
python -m unittest discover -s tests/test_agents
```

numpy only · deterministic (seed 42) · the full run takes under a second.

**Say (4:50–5:00):** "One command rebuilds everything you just saw. Thank you."

---

## Recording checklist (needs a human)

1. `python demo.py --no-run` and set the browser window to **1280×720** (the player fits it
   with no scrolling; `tools/screenshot.py` captures that exact frame).
2. Read the talk track above; press **Space** to play, **←/→** to step a round.
3. Pause on round 3 (Hospital C excluded) and on the attack panel (RECOVERED vs FAILED).
4. Finish on round 4: AUC 0.832, the three reference numbers, the green **0 bytes** counter.
5. Record two takes; keep the second.
