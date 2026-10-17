# PrivateLearn Platform — Execution Plan (phase 2)

> PrivateLearn (phase 1, `EXECUTION-PLAN.md`) proved the loop: three simulated hospitals
> train one model, no row leaves, and the agents decide. Phase 2 turns it into a **platform**.
> Companies from any field register; an admin creates **groups**, each one model with its
> own task. Companies join, connect their data through their own **agent node** (a separate
> process), train together live, and then use the model. Late joiners fine-tune the trained
> model into a new version.

**How to use this file:** §5 is the task table and the source of truth. Create one GitHub
issue per task with its acceptance criteria. Every agent reads
`privatelearn/docs/PLATFORM-API.md` (the frontend ⇄ backend ⇄ node contract) before writing
code. The claim → branch → PR loop is unchanged (`CONTRIBUTING.md`).

---

## 1. The goal in one sentence

Any number of companies, each running its **own agent team in its own process with its own
data and its own LLM**, train one shared model of the admin's choosing, while admins and
companies watch every step live.

## 2. What the judges see (≈ 6 minutes, everything runs live)

1. **Admin creates a group.** They type "Does this patient have heart disease?"; the
   architect agent drafts the TaskSpec (inputs, label, model type, privacy); the admin
   edits and publishes it.
2. **Companies register and request to join**; the admin approves them from a queue.
3. **A company connects its data.** It runs `node.py` against its own SQLite database (or
   the admin clicks "start demo node"). The node's Schema agent profiles the columns, its
   LLM maps oddly named columns in odd units to the group's inputs, and the company
   reviews and confirms the mapping in the **Connect data** tab.
4. **Training, live.** The stepper moves through key exchange → schema → normalisation →
   rounds → published. Packets fly on the canvas, the **agent conversation feed** shows
   each agent's reasoning, and the AUC climbs. One node process is killed mid-run: it goes
   grey, the coordinator notices and carries on.
5. **Use the model.** A company fills the generated form in the **Use model** tab and gets a
   prediction with an explanation.
6. **A late joiner** connects, the admin starts a **fine-tune**, and v2 appears next to v1.
7. A second group (another dataset) shows the platform is not tied to hospitals.

## 3. Repository layout and lane ownership

All new code lives in `privatelearn/hub/`. It is not called `platform/`, because a package
named `platform` would shadow Python's standard-library module. Ownership is exclusive.

```
privatelearn/
├── docs/PLATFORM-API.md           # P0: contract owner (lukaLomadze)
├── src/                           # existing engine: lanes A/B; the X tasks below
├── hub/
│   ├── requirements.txt           # D  (fastapi, uvicorn: the only new dependencies)
│   ├── server/                    # D  FastAPI app, SQLite, auth, routes, billing, fake federation
│   │   ├── app.py  db.py  seed.py  auth.py  billing.py  fake_federation.py
│   │   └── routes/ meta.py companies.py groups.py memberships.py jobs.py models.py usage.py node.py demo.py
│   ├── federation/                # F  FederationService, NetworkBus, trace export
│   │   ├── service.py  netbus.py  keys.py  trace_export.py
│   ├── node/                      # F  the company's agent process
│   │   ├── node.py  connectors.py
│   ├── demo/                      # F  seeded demo datasets + make_demo_data.py
│   └── web/                       # E  vanilla JS single-page app
│       ├── index.html  app.css  app.js  api.js  router.js
│       ├── pages/   (login, register, company, admin, group, group-edit, membership, job, usage)
│       ├── live/    (stepper.js, canvas.js, feed.js, chart.js)
│       └── mock/    (JSON copied from PLATFORM-API.md examples, for ?mock=1)
└── tests/
    ├── test_hub_server/           # D
    └── test_hub_node/             # F (node + federation)
```

| Lane | Owns | Who |
|---|---|---|
| **X — engine fixes** | `src/model/`, `src/agents/`, `src/pipeline/`, `src/shared/` (existing code) | lane A/B owners; the friend taking the existing-code fixes |
| **D — backend** | `hub/server/`, `hub/requirements.txt`, `tests/test_hub_server/` | lukaLomadze (Claude session) |
| **E — frontend** | `hub/web/` | lukaLomadze (Claude session) |
| **F — federation + node** | `hub/federation/`, `hub/node/`, `hub/demo/`, `tests/test_hub_node/` | lukaLomadze (Claude session) |
| contract | `privatelearn/docs/PLATFORM-API.md`, this file | lukaLomadze |

**Dependency exception:** `hub/server/` may use `fastapi` and `uvicorn`. Everything else
stays stdlib + numpy (torch only inside the optional X6/X7 adapters). Nodes talk to the
server with `urllib`, and the UI gets live updates over SSE, so there is no websocket
dependency.

## 4. How parallel lanes avoid waiting on each other

- **E builds against mocks first.** `api.js` with `?mock=1` serves the examples from
  `PLATFORM-API.md` (`hub/web/mock/`), so every page works before the backend exists.
- **D builds against a fake federation.** `fake_federation.py` implements the §8
  `FederationService` interface by replaying the committed `trace.json` as §6 events, so the
  SSE stream and the live view work before any node exists.
- **F builds against the in-process engine.** The NetworkBus is tested with real
  `HospitalAgent`s over HTTP on localhost, independent of D's routes (the §8 interface is
  the seam).
- **X tasks are pure refactors** of existing code that keep `trace.json` reproducible. F
  depends on them; D and E do not.

## 5. Task list

`⟶` marks the critical path. Hours are estimates. A task may start only when its
dependencies are **merged**.

### P — contract

| ID | Title (lane) | Deps | Hrs | Deliverable | Acceptance criteria |
|---|---|---|---|---|---|
| **P0** ⟶ | Platform contract + plan (contract) | — | 2 | `docs/PLATFORM-API.md`, this file, `TASKS.md`/`CONTRIBUTING.md` updates | Every endpoint, object, enum, SSE event and node message has a shape and an example; D/E/F reviewed it; version 0.1 frozen. |

### X — engine fixes in existing code (friend; lanes A/B)

> **Updated after #44–#47 (merged 2026-10-18):** the runtime no longer hard-codes feature names
> or counts, and the agents reconcile a shared schema semantically for any table. X3 changed
> accordingly. X1, X2, X4 and X5 are still open: `ALLOWED_HOSPITALS = {h0,h1,h2}` and
> `int(id.lstrip("h"))` remain, σ is still calibrated from the constants, and the
> normalisation stats still come from `manifest.json`.

| ID | Title | Deps | Hrs | Deliverable | Acceptance criteria |
|---|---|---|---|---|---|
| **X1** | DP σ follows the plan actually run | — | 2 | `pipeline/run_loop.py`, `agents/hospital.py` | σ is computed from the epochs/lr the Planner chose (`record_sensitivity(n, lr, epochs, clip)`), not the constants; the trace records the per-update σ; a test with a 10-epoch plan shows a larger σ at the same ε. |
| **X2** ⟶ | Participant ids are any string | — | 2 | `shared/schemas.py`, `agents/hospital.py`, `agents/coordinator.py` | No `ALLOWED_HOSPITALS = {h0,h1,h2}` and no `int(id.lstrip("h"))`; the mask index comes from the participant's position in the sorted participant list; ids like `node_9f3a` pass every validator; the demo trace is unchanged. |
| **X3** ⟶ | Reconcile toward the group's inputs | X2 | 3 | `agents/coordinator.py`, `agents/decider.py`, `agents/hospital.py` | #44/#46 removed the fixed vocabulary; contract negotiation is now open-ended. Add an optional **target** (TaskSpec `input_schema`: names + units, PLATFORM-API §4.3): `decider.contract` maps every participant's described columns onto those target names, so the agreed features are the group's inputs and the Use-model form stays stable; the offline `RuleDecider` matches columns whose normalised name equals a target name (today it yields an empty contract); feature names and normalisation stats no longer come from `manifest.json` when a target is given; prompts say "participant", not "hospital"; demo numbers unchanged. |
| **X4** | Federated normalisation | X2 | 3 | `agents/coordinator.py`, `agents/hospital.py`, `model/privacy.py` | The coordinator learns the global mean/std from masked per-participant `(n, Σx, Σx²)` instead of `manifest.json` (still read in `run_loop.py`); matches the pooled statistics to 1e-9 without DP noise; protocol kinds `stats_request`/`stats`. |
| **X5** | Model adapter interface | X3 | 3 | `model/adapters.py` (+ tests) | `ModelAdapter` with `init_params / train_local / evaluate / predict` over a flat numpy vector; `LogisticAdapter` wraps `logistic.py`; Trainer, Challenger and Coordinator use the adapter; `SUPPORTED_MODELS` is the adapter registry; demo numbers unchanged. |
| **X6** | MLP adapter | X5 | 3 | `model/adapters_torch.py` | Torch MLP via `parameters_to_vector`; trains federated on sklearn `breast_cancer` split in 3; torch is imported lazily so the numpy-only path still works without it. |
| **X7** | Image adapter (stretch) | X5 | 4 | `model/adapters_torch.py` | Small CNN or tiny ViT on sklearn `digits` (8×8), federated over 3 nodes, < 60 s on CPU. |

### D — backend (`hub/server/`)

| ID | Title | Deps | Hrs | Deliverable | Acceptance criteria |
|---|---|---|---|---|---|
| **D1** ⟶ | Server scaffold | P0 | 3 | `app.py`, `db.py` (stdlib `sqlite3`), `auth.py`, `seed.py`, routes `meta`, `auth` | `uvicorn` serves `/api/v1` and `hub/web/` at `/`; error body per §1.1; demo login per §5.1; `GET /meta` returns every §2 enum; the seed creates 1 admin, 5 companies and 2 groups; CI installs `hub/requirements.txt` and runs `tests/test_hub_server`. |
| **D2** | Companies | D1 | 2 | `routes/companies.py` | §5.2 complete; a company cannot see another company; approve/reject follow §3; tests per endpoint. |
| **D3** | Groups + TaskSpec validation | D1 | 3 | `routes/groups.py` | §5.3 complete (draft-spec may return a template until D9); `task_spec` validated with `422` naming the field; `model.type` must be in `available_model_types`. |
| **D4** ⟶ | Memberships + state machine | D2, D3 | 3 | `routes/memberships.py` | §5.4 complete; every transition of §3 enforced (`409 invalid_transition` otherwise); node token issued, hashed in the DB, and revoked on reissue. |
| **D5** | Node endpoints | D4 | 3 | `routes/node.py` | §7 complete; `sanitize()` runs on every outbox payload (a payload with a `rows` key → `422`); polling updates `node.status`, with offline after 15 s; schema confirm is forwarded as `schema_confirmed`. |
| **D6** ⟶ | Jobs, events, SSE + fake federation | D4 | 4 | `routes/jobs.py`, `fake_federation.py` | §5.5 complete; events persisted with gapless `seq`; `/stream` honours `Last-Event-ID` and `?token=`; the fake service replays `trace.json` as §6 events over ~60 s, mapping h0/h1/h2 onto the job's participants; job and membership statuses update on finish. |
| **D7** | Model versions + predict | D6 | 2 | `routes/models.py` | §5.6 complete; a finished job creates a `ModelVersion` (v1, v2 …, `parent_version_id` for fine-tunes); predict validates inputs against `input_schema` and returns per-feature contributions for logistic. |
| **D8** | Usage ledger + payment seam | D4, D7 | 2 | `billing.py`, `routes/usage.py` | A `PaymentProvider` interface with a `NullProvider`; `UsageEvent` written at approval, per training round and per prediction (§4.7); `GET /usage` with totals. No money moves. |
| **D9** | Architect agent | D3 | 2 | `routes/groups.py` (draft-spec) | `POST /groups/draft-spec` asks the LLM (DeepSeek, as in `decider.py`) for a TaskSpec, validates it with D3's validator and falls back to a template; returns `made_by`. |
| **D10** | Demo helpers | D5, F2 | 2 | `routes/demo.py` | §5.8: start/stop a seeded node as a separate process (pid tracked, killed on server exit); reset re-seeds. |

### E — frontend (`hub/web/`)

Vanilla JS, no framework, colors only from `DESIGN.md` tokens. New platform tokens get added
to `DESIGN.md` in their own PR.

| ID | Title | Deps | Hrs | Deliverable | Acceptance criteria |
|---|---|---|---|---|---|
| **E1** ⟶ | App shell + API client + mocks | P0 | 3 | `index.html`, `app.js`, `router.js`, `api.js`, `app.css`, `mock/`, login & register pages | Hash router; `api.js` handles the bearer token and the §1.1 error shape; `?mock=1` serves the PLATFORM-API examples; role-based nav; a shared status-pill component that renders unknown enum values as text. |
| **E2** | Company home | E1 | 2 | `pages/company.js` | Browse groups (question, model type, members, current version); request to join; "my memberships" list with status pills. |
| **E3** | Admin approvals | E1 | 2 | `pages/admin.js` | Queues of pending companies and membership requests; approve/reject inline; the list updates without a reload. |
| **E4** | Admin: create / edit group | E1 | 3 | `pages/group-edit.js` | TaskSpec editor (question, task type, label, input-schema rows, model type from `/meta`, rounds, privacy); "Draft with agent" fills it from the question and shows the reasoning; save draft and publish. |
| **E5** | Membership page, tab 1: Connect data | E2 | 4 | `pages/membership.js` | Tabs **Connect data** / **Use model**; a progress tracker over the membership states (§3); copyable node command; live node status; profile table; the mapping proposal with the agent's reasoning and `made_by`, editable; confirm; validation checklist. |
| **E6** ⟶ | Live process view | E1 | 6 | `live/*.js`, `pages/job.js` | `#/jobs/{id}`: stepper over `job.step`; canvas with N participants + coordinator and animated packets (ported from `web/player.js`); **agent conversation feed**; decision cards; AUC/loss chart. Driven only by §6 events: SSE live, `/events` for catch-up; works on the mock event list. |
| **E7** | Membership page, tab 2: Use model | E5 | 3 | `pages/membership.js` | Form generated from `input_schema` (number/integer/category; image upload and text box shown disabled until those adapters exist); prediction with probability and contribution bars; version list with metrics. |
| **E8** | Admin group detail | E3, E4, E6 | 3 | `pages/group.js` | Members table with statuses and node status; Start training / Fine-tune (disabled with the reason when not allowed); start/stop demo node buttons; link to the live view; v1 vs v2 metrics side by side. |
| **E9** | Usage view | E1 | 1 | `pages/usage.js` | Ledger table and totals for the admin and the company; a note that payments are coming; the wallet field is shown read-only. |

### F — federation + node (`hub/federation/`, `hub/node/`)

| ID | Title | Deps | Hrs | Deliverable | Acceptance criteria |
|---|---|---|---|---|---|
| **F1** ⟶ | NetworkBus + FederationService | P0, X2 | 4 | `federation/service.py`, `netbus.py` | Implements the PLATFORM-API §8 interface for real: one `CoordinatorAgent` per job on a timed tick; node messages in and out through `deliver_from_node`/`inbox_for_node`; `sanitize()` on everything; emits every §6 event (participant states, `agent.message` from `activity`, packets, rounds, published). Tested with in-process nodes over localhost HTTP. |
| **F2** ⟶ | `node.py`: the company's agent process | P0, X2, X3 | 4 | `node/node.py`, `node/connectors.py` | CLI `--server --token --db`; connectors for CSV, SQLite (table or query) and any DB-API URL; hello → profile → schema proposal (its own LLM key, rule fallback) → waits for confirmation → validation → poll loop running a `HospitalAgent`; sends `activity`; never sends a row; reconnects if the server restarts. |
| **F3** | Diffie-Hellman mask keys | F1, F2 | 3 | `federation/keys.py`, node side | Nodes exchange public keys through the coordinator (`key_offer`/`key_set`, RFC 3526 group, stdlib `pow`); pair secrets exist only on nodes; a test shows the server cannot unmask a single update but the sum is exact. |
| **F4** | Federated normalisation over the network | X4, F1, F2 | 2 | service + node | The `normalization` step runs X4's protocol across processes; the stepper shows it; matches the pooled statistics. |
| **F5** | Fine-tune jobs | F1 | 2 | `federation/service.py` | `start_job(..., init_weights=parent)` starts from the parent model with FedProx anchored to it; the late joiner's AUC is reported before and after. |
| **F6** | Demo data + trace export | F2 | 3 | `demo/make_demo_data.py`, `demo/*.db`, `federation/trace_export.py` | 4 diabetes companies (the 3 existing splits + a late joiner from the unused Pima rows) as SQLite files with different column names and units; a second group's data (sklearn `breast_cancer`, 3 companies); `/jobs/{id}/trace` validates with `schemas.validate_trace`. |

### I — integration

| ID | Title | Deps | Hrs | Deliverable | Acceptance criteria |
|---|---|---|---|---|---|
| **I1** | End-to-end demo run + rehearsal script | D10, E8, F3, F5, F6 | 3 | `hub/README.md`, `docs/deck.md` update | One command starts the server; the §2 story runs live start to finish twice in a row; a fallback (`fake_federation` replay) is one flag away if the network or the LLM fails on stage. |

## 6. Dependency graph

```
P0              → D1, E1, F1, F2
D1              → D2, D3
D2, D3          → D4            D3 → D9
D4              → D5, D6        D6 → D7       D4, D7 → D8
D5, F2          → D10
E1              → E2, E3, E4, E6, E9
E2 → E5 → E7    E3, E4, E6 → E8
X2              → X3, X4, F1    X3 → X5 → X6, X7
X2, X3          → F2
F1, F2          → F3            X4, F1, F2 → F4        F1 → F5        F2 → F6
D10, E8, F3, F5, F6 → I1
X1              (independent)
```

**Critical path:** `P0 → D1 → D4 → D6 → D7 …` for the backend and
`X2 → X3 → F2 → D10 → I1` for live agents. E6 (the live view) is the biggest single
frontend task and has no backend dependency thanks to the mocks, so start it first.

**Suggested first wave (all parallel):** D1 · E1 then E6 · X1 · X2 · F6 (data only).

## 7. Payments later (Solana): what is already in place

- `companies.wallet_address`, `groups.pricing` and the `payment_pending` membership state
  are reserved in the contract.
- Every billable action already writes a `UsageEvent` (D8), and contributions (rows ×
  rounds) are stored per job, so revenue sharing is a query.
- Adding Solana means one new `PaymentProvider` class and filling `amount` / `payment_ref`.
  No endpoint shape changes.

## 8. Honesty for the judges

**Real:** separate node processes, each with its own data and its own LLM; the live agent
decisions; masking with keys the server never holds (after F3); DP noise; the training and
the metrics.
**Simulated:** the companies are demo datasets on one machine (a teammate's laptop can join
as an extra node); demo login is not real authentication; DP accounting is a stated-ε
demonstration, as in phase 1.
