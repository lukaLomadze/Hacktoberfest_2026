# What the agents decide — and what is not an agent decision

Audit of every decision in PrivateLearn. The rule: **the runtime machinery holds no
data-specific knowledge.** Anything that depends on the data is either a contract
constant, the demo scenario, or an agent decision.

## Agent decisions (LLM, recorded in `trace.json`)

| Decision | Agent | If the model is unavailable |
|---|---|---|
| which local column maps to which canonical feature | Schema | offline alias seed |
| the unit conversion factor | Schema | offline alias seed |
| which column is the label | Schema | offline label seed |
| local epochs / lr / mu | Planner | contract defaults 4 epochs @ lr 0.1 |
| aggregation strategy + exclusions | Coordinator | `select_strategy` rules |
| the contract: agreed feature set + model type | Coordinator | intersection of every hospital's proposal |

## The negotiated contract

Which features everyone will use, and the model type, is produced at runtime by the
Coordinator agent from each hospital's proposal — a feature is usable only if every
hospital proposed it. `CoordinatorAgent.features`, `.model` and `.n_weights` come from
that decision, and the trace records it under `schema`.

Nothing in the runtime assumes six features or seven weights: the weight count is
`len(features) + 1`, the message validators accept any width, and the evaluation set is
sliced to the agreed features. Changing the model means adding a trainer and a name to
`SUPPORTED_MODELS`; the contract carries the choice.

**Not configured anywhere — observed:**

- whether a hospital is offline → no probe reply arrives this round
- whether a hospital is an outlier → its probe loss against the other responders
- whether a round is discarded → a participant went silent after masking

The fault scenario only *causes* those states; the coordinator never reads the
fault list.

## Contract constants (the federation's agreement, not data knowledge)

- the six canonical feature names and units — `CONTRACTS.md` §4; this is the
  **vocabulary the Schema agents propose in**, not a list the Coordinator enforces
- allowed message kinds and their schemas — `runtime.py`, `schemas.py`
- the three aggregation strategy names — `aggregation.py`
- DP policy: ε per round, δ, per-example clip — a stated policy, not a data fact

## Demo scenario (not infrastructure)

- `run_loop.FAULTS`: Hospital C offline in round 2, age-in-months in round 3
- the three hospital ids and labels
- the seed and the round count

These describe the story being told, not how the system works.

## Offline fallback (used only when no model is available)

`hospital.py` `ALIASES`, `LABEL_ALIASES`, `HEIGHT_ALIASES`, `WEIGHT_ALIASES` — a
deterministic seed so tests, CI and an offline demo still run. The LLM path never
reads them. Do not add data here that an agent should be deciding.

## Known data knowledge outside the agent runtime

`src/pipeline/prep_data.py` decides which zero values are missing (Pima-specific)
and orders rows by age for the non-IID split. That is one-off dataset preparation,
run before the agents start; it is not part of the runtime and it never inspects a
model's output.

## What the machinery must never do

- reject an agent's answer against a hard-coded table of "correct" values
- carry a list of answers the agent is expected to reproduce
- special-case a hospital, a column name or a dataset in the runtime path

If a change needs any of those, it belongs in the contract or the scenario, and
must say so here.
