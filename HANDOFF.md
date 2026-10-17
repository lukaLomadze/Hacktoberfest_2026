# HANDOFF — state of the PrivateLearn project

Read this first. It is the bridge from the previous session (which built most of this)
to whatever comes next. Rules in `AGENTS.md` still apply.

## What this project is

**PrivateLearn** — *agentic federated learning*. Three simulated hospitals each hold a
different slice of the same patients, named/unit-ed differently. Each hospital runs a
small team of **LLM agents** that describe its columns and train locally; a Coordinator
agent negotiates one shared schema and aggregates. No patient row leaves a hospital —
only column descriptions, losses, and masked+noised weight deltas cross the wire.

The demo is honest about what it is: a logistic-regression model over the Pima diabetes
dataset, replaying a recorded run. The **win** is not a big accuracy jump — it is that the
best model becomes *legal to build* (federated ≈ centralized, zero rows shared).

## Current state (all merged to `main`)

- Tasks T0–T12 done. No open PRs, no open issues.
- The **agents are LLMs** (DeepSeek) at the decision points: schema mapping, training
  plan, aggregation strategy. Their choices + reasoning are recorded in `trace.json`.
- **No hard-coded feature names or counts in the runtime.** The runtime learns feature
  names from the dataset manifest + the agents' negotiation.
- `trace.json` is the **DeepSeek-driven run** (17 LLM decisions, 0 fallbacks).

Measured numbers (the committed trace, held-out 20%):

| | AUC |
|---|---|
| Hospital A / B / **C** alone | 0.840 / 0.838 / **0.763** |
| **federated (masked + DP)** | **0.832** |
| centralized (illegal) | 0.836 |
| federated, privacy off | 0.831 |

The pitch: *"federation lifts the worst-off hospital from 0.763 to 0.832 and matches
centralized to within 0.004 — with zero rows shared. Privacy costs ~0.001."* Do **not**
claim it beats every hospital (A alone is 0.840).

## Architecture (the mental model)

```
MessageBus (deterministic tick loop)  ← privacy boundary: rejects raw-data payloads
 ├─ HospitalAgent h0/h1/h2
 │    ├─ SchemaAgent  — describes local columns (semantic name + unit + factor)
 │    ├─ PlannerAgent — local epochs / lr / mu
 │    ├─ TrainerAgent — trains from the global model, emits Δw + loss
 │    └─ ChallengerAgent — scores a model on local rows
 └─ CoordinatorAgent
      ├─ negotiates the contract (reconciles each hospital's columns → shared features + model)
      ├─ probes hospitals, chooses the aggregation strategy (FedAvg/FedProx/trimmed-mean)
      ├─ sums only masked updates (pairwise masks cancel to 1e-9)
      └─ broadcasts the new model
```

- **Decider** (`src/agents/decider.py`): `LLMDecider` (DeepSeek, OpenAI-compatible) with a
  `RuleDecider` deterministic fallback. Every method falls back on *any* failure.
- **Privacy** (`src/model/privacy.py`): L2 clip → Gaussian DP noise → pairwise masking.
- **Attack demo** (`src/model/attack.py`): gradient inversion recovers a row from an
  unprotected update, fails on a masked+DP one.
- **Player** (`web/`): replays `trace.json`; never calls the model.

## Key files

```
privatelearn/
  src/agents/decider.py     LLM + rule decisions (describe/contract/plan/strategy)
  src/agents/hospital.py    SchemaAgent/Planner/Trainer/Challenger + HospitalAgent
  src/agents/coordinator.py contract negotiation, probe/decide/train/aggregate
  src/agents/runtime.py     Agent + MessageBus (privacy checks live here)
  src/model/{logistic,aggregation,privacy,attack}.py
  src/pipeline/{prep_data,run_loop}.py   data prep + the end-to-end run → trace.json
  src/shared/{schemas,io}.py             validators + JSON helpers
  data/hospitals/*.csv, data/holdout/test.csv, data/raw/diabetes.csv, manifest.json
  trace.json                the committed DeepSeek run (the demo replays this)
  web/{index.html,player.css,player.js}  the replay player
  tests/test_agents/pima_decider.py      TEST FIXTURE (not runtime) — Pima column mapping
  docs/{CONTRACTS,DESIGN,AGENT-DECISIONS,deck}.md
AGENTS.md / CONTRIBUTING.md / TASKS.md   repo rules, onboarding, roster
```

## How to run / test

```powershell
cd privatelearn

# run the full federation, serve the player, open a browser
python demo.py                # --no-run replays the committed trace without recomputing

# regenerate the trace with the LLM (needs the key)
$env:DEEPSEEK_API_KEY = "sk-..."
python src/pipeline/run_loop.py --llm

# inspect the recorded decisions
python -c "import json; [print(r['round'], r.get('decided_by'), '|', r.get('reasoning','')[:90]) for r in json.load(open('trace.json'))['rounds']]"

# tests — run each directory (the parent discovers nothing; see CONTRIBUTING.md)
python -m unittest discover -s tests/test_model
python -m unittest discover -s tests/test_agents
```

## Critical caveats (do not relearn these the hard way)

1. **The demo requires `--llm`.** The offline (no-key) fallback is *generic*: it normalises
   column names and infers nothing, so it **cannot** reconcile `bp_kpa` vs `blood_pressure`
   and produces a degenerate (empty) contract. That is by design — reconciliation is the
   agents' job. Only `prep_data.py` (the dataset loader), the holdout CSV format, and the
   `age_in_months` fault scenario still know Pima names.
2. **`trace.json` is the LLM run.** If you regenerate with `run_loop.py` (no `--llm`) you
   overwrite it with a degenerate/generic trace. `test_committed_trace_matches_a_fresh_run`
   **skips** when the trace's `meta.decision_maker.provider != "rule"` — that is intended.
3. **DP accounting is stated-ε**, not an audited accountant. The honesty note says so.
4. **One sandbox-only test fails**: `test_prep_data` uses `tempfile.TemporaryDirectory()`
   and the agent sandbox denies writes there (`PermissionError`). It passes on a normal
   machine. Don't "fix" it.
5. **The model can't read images on `deepseek-v4-pro`**; switch to an image-capable model
   and use `read_image` on screenshots (e.g. `player.png`) to check the UI.
6. **Numbers in docs/deck come from `trace.json`** — never retype them from memory.

## Honest next steps (in rough priority)

1. **Make `prep_data.py` generic** so a dataset swap (bank data, etc.) is one command:
   auto-detect missing-value sentinels, auto-split, write the manifest + holdout generically.
   Today you must hand-supply hospital CSVs + a manifest for a new dataset.
2. **Move the Pima mapping into the manifest as data** (currently a test fixture,
   `tests/test_agents/pima_decider.py`) so the offline path can reconcile without code.
3. **Tighten DP accounting** (Rényi / moments accountant) instead of basic composition.
4. **Add a membership-inference attack** as a second privacy demo (stronger than gradient
   inversion alone).
5. **Record the 5-minute demo video** (the "recorded demo" half of T12 — a human task);
   rehearse per `docs/EXECUTION-PLAN.md` §10, cold-start test on a clean clone.
6. **Surface the agent activity more** in the player (PR #34 added the teams; the
   per-round reasoning is already in the decision log).

## Workflow reminders

- Never push to `main`; branch `<username>/<task>`, open a PR, a *different* contributor
  merges. Commit messages `type: summary`, end the PR body with "Written with DeepSeek".
- The Issues board is the live claim list; assign yourself before starting.
- The agent cannot run `--llm` itself (its shell doesn't inherit your session-only key);
  you must run it in your terminal and commit the resulting `trace.json`.
