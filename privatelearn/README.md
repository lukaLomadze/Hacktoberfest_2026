# PrivateLearn

**One model, zero shared data.** Three hospitals train a single diabetes-risk model
together; agents negotiate a common schema across mismatched columns and units, a
coordinator adaptively aggregates, and no patient row ever leaves a hospital. Updates are
masked (the coordinator sees only their sum) and carry patient-level differential privacy.

## Run it — one command

```bash
cd privatelearn
python demo.py
```

Rebuilds the hospital data and `trace.json` from the raw dataset, then opens the player at
`http://localhost:8000/web/index.html`. `python demo.py --no-run` replays the committed trace.
Needs Python 3.9+ and numpy.

## What the run shows (from `trace.json`)

| | AUC on 154 held-out patients |
|---|---|
| Hospital A / B / **C** alone | 0.840 / 0.838 / **0.763** |
| **federated** — masked + DP (ε = 1.5/round, δ = 1e-5) | **0.832** |
| same scenario, privacy off | 0.831 |
| all 360 rows pooled (illegal) | 0.836 |

- Within 0.004 of the illegal pooled model; the weakest hospital gains +0.07.
- Round 2: Hospital C goes offline. Round 3: C's export sends age in months, its loss
  spikes, and the coordinator excludes it before masking.
- Gradient inversion recovers a real patient row from an unprotected update (error 0.00)
  and fails on what the coordinator actually receives (error 2.40).
- All three hospitals end with the byte-identical model (same SHA-256).

Deck and 5-minute talk track: [`docs/deck.md`](docs/deck.md). Honest limits are on slide 9.

## How it works

| path | what |
|---|---|
| `src/pipeline/prep_data.py` | T1 — Pima data → three hospitals (age bands) in their own column names/units + a holdout |
| `src/model/logistic.py` | T2 — logistic regression (numpy only), FedProx proximal term |
| `src/model/aggregation.py` | T3 — FedAvg / FedProx / trimmed-mean + the adaptive selector |
| `src/model/privacy.py` | T4 — pairwise masking, analytic-Gaussian DP, per-example clipping |
| `src/model/attack.py` | T5 — gradient-inversion attack |
| `src/agents/runtime.py` | T6 — deterministic message bus with contract + raw-data checks |
| `src/agents/hospital.py` | T7 — Schema / Planner / Trainer / Challenger agents |
| `src/agents/coordinator.py` | T8 — schema negotiation, per-round decisions, masked aggregation |
| `src/pipeline/run_loop.py` | T9 — the end-to-end run → `trace.json` |
| `web/` | T10–T11 — the replay player (vanilla JS); spec in `docs/DESIGN.md` |
| `docs/CONTRACTS.md` | every message and file shape; the source of truth |

## Tests

```bash
python -m unittest discover -s tests/test_model -v
python -m unittest discover -s tests/test_agents -v
```

The test directories are not packages, so run each one (`discover -s tests` finds nothing).

## Regenerate pieces by hand

```bash
python src/pipeline/prep_data.py     # data/hospitals/*.csv, data/holdout/test.csv, manifest.json
python src/pipeline/run_loop.py      # trace.json (deterministic, seed 42)
python -m http.server 8000           # then open http://localhost:8000/web/index.html
```
