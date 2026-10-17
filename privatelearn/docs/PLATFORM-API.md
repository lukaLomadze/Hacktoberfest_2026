# PrivateLearn Platform — API contract (frontend ⇄ backend ⇄ node)

> The agreement every platform lane codes against. The frontend (lane E), the backend
> (lane D) and the company node + federation (lane F) never need to read each other's code:
> they read this file. A change is its own small PR that bumps the version below and says in
> the PR which lanes must follow. The engine contracts (`CONTRACTS.md`) still hold for the
> math and the bus messages; this file adds everything around them.

Version: **0.1** (draft — frozen once P0 is merged)

---

## 0. The system in one picture

```
 browser (lane E)                 platform server (lane D + F)                 company node (lane F)
 hub/web/  ──── REST + SSE ────▶  FastAPI  /api/v1/*          ◀── HTTP poll ── hub/node/node.py
                                  SQLite   hub.db                              holds the DB connection
                                  FederationService (F)                        runs Schema·Planner·
                                   └ CoordinatorAgent per job                  Trainer·Challenger
                                     on a NetworkBus                           its own LLM key
```

- **One node per membership** (company × group): each group needs its own dataset, so the
  company starts one `node.py` per group it joins. The node is a separate OS process.
- **Rows never reach the server.** The node sends only what `CONTRACTS.md` allows (schema
  descriptions, column statistics, losses, masked updates). The connection string stays on
  the node and is never sent.
- The browser never talks to a node. Everything goes through the server.

---

## 1. Conventions

| | |
|---|---|
| Base path | `/api/v1` |
| Format | JSON, UTF-8. Field names `snake_case`. |
| Ids | strings with a type prefix: `usr_`, `co_`, `grp_`, `mem_`, `node_`, `job_`, `mv_`, `use_`. Never parse them. |
| Times | ISO-8601 UTC strings, e.g. `"2026-10-18T14:03:22Z"` |
| Lists | always wrapped: `{"items": [...]}` (no pagination in v0.1) |
| Auth (browser) | `Authorization: Bearer <session_token>` |
| Auth (node) | `X-Node-Token: <node_token>` |
| Unknown fields | clients must ignore fields they don't know; servers may add fields without a version bump |
| Unknown enum values | the frontend must render them as plain text, never crash (enums below have reserved values) |

### 1.1 Errors

Every non-2xx response has this body:

```json
{"error": {"code": "invalid_transition", "message": "membership mem_7 is 'requested'; approve it first", "field": null}}
```

| HTTP | `code` | when |
|---|---|---|
| 400 | `bad_request` | malformed JSON, missing field |
| 401 | `unauthorized` | no / unknown token |
| 403 | `forbidden` | wrong role, or not your company |
| 404 | `not_found` | |
| 409 | `invalid_transition` | the state machine (§3) does not allow this action now |
| 422 | `validation_error` | a field has a bad value; `field` names it (e.g. `"task_spec.training.rounds"`) |
| 503 | `node_offline` | the action needs the company's node and it is not polling |

---

## 2. Enums (the server also returns them from `GET /meta`)

```
role               admin | company
company.status     pending | approved | rejected | suspended
group.status       draft | open | training | trained | archived
membership.status  requested | approved | rejected | node_connected | schema_proposed
                   | schema_confirmed | ready | training | active | left
                   (reserved for payments: payment_pending)
node.status        offline | online | busy
job.kind           train | finetune
job.status         queued | running | succeeded | failed | cancelled
job.step           waiting_for_nodes | key_exchange | schema_negotiation | normalization
                   | training | evaluating | published            (in this order)
participant.state  waiting | online | probing | planning | training | masking | sent
                   | excluded | offline | done
task_type          binary_classification | multiclass_classification | regression
                   | image_classification | text_generation
model.type         logistic | mlp | cnn | vit | llm_lora
agent              schema | planner | trainer | challenger | validator | coordinator | architect
made_by            llm | rule
usage.kind         enrollment | training_round | prediction | contribution
```

`GET /meta` also returns `available_model_types` — the model types the engine can train
right now. The frontend offers only those (others shown disabled). v0.1 guarantees
`["logistic"]`.

---

## 3. State machines

**Company:** `pending → approved | rejected`, `approved ⇄ suspended`. Only approved companies
can request to join groups.

**Group:** `draft → open` (admin publishes) `→ training` (job running) `→ trained`
(a model version exists) `→ training` again for a fine-tune `→ trained`. `archived` from any.

**Membership** (one company in one group):

```
requested ─admin approve─▶ approved ─node says hello─▶ node_connected
   │                                                        │ node posts schema proposal
   └─admin reject─▶ rejected                                ▼
                                                     schema_proposed
                                                            │ company confirms (may edit)
                                                            ▼
                                                     schema_confirmed
                                                            │ node posts validation ok
                                                            ▼
                     active ◀─job succeeds─ training ◀─job starts─ ready
```

Any membership not in `training` may move to `left`. A node that says hello again (restart,
reconnect) never moves its membership backwards.

A membership joining a group that is already `trained` follows the same path; when it is
`ready`, the admin starts a `finetune` job. A failed or cancelled job puts its participants
back to `ready` (or `active` if they already had a model).

**Job:** `queued → running → succeeded | failed | cancelled`. While `running`, `step` moves
forward through the `job.step` order. `GET /jobs/{id}` always reports the current `step`.

---

## 4. Objects

### 4.1 User and session

```json
{"id": "usr_1", "email": "admin@privatelearn.dev", "role": "admin", "company_id": null}
```

### 4.2 Company

```json
{"id": "co_3", "name": "Tbilisi Heart Center", "field": "healthcare",
 "contact_email": "it@thc.ge", "status": "approved",
 "wallet_address": null, "created_at": "2026-10-18T10:00:00Z"}
```

`wallet_address` is reserved for Solana payments. The server stores it, nothing reads it yet.

### 4.3 Group and TaskSpec

```json
{"id": "grp_1", "name": "Diabetes risk", "status": "open",
 "description": "Three-hospital diabetes screening model",
 "task_spec": { ...TaskSpec... },
 "pricing": null,
 "member_count": 3, "current_version": null,
 "created_at": "2026-10-18T10:00:00Z"}
```

**TaskSpec** is what the admin configures: what the model learns and how.

```json
{
  "question": "Does this patient have diabetes?",
  "task_type": "binary_classification",
  "label": {"name": "has_diabetes", "classes": ["no", "yes"], "positive": "yes"},
  "input_schema": [
    {"name": "glucose", "type": "number", "unit": "mg/dL", "description": "plasma glucose", "required": true},
    {"name": "bmi", "type": "number", "unit": "kg/m^2", "description": "body-mass index", "required": true},
    {"name": "age", "type": "number", "unit": "years", "description": "", "required": true}
  ],
  "model": {"type": "logistic", "hyperparams": {"epochs": 4, "lr": 0.1}},
  "training": {"rounds": 4, "min_participants": 2, "aggregation": "adaptive"},
  "privacy": {"secure_aggregation": true,
              "dp": {"enabled": true, "epsilon_per_round": 1.5, "delta": 1e-5}},
  "metric": "auc"
}
```

- `input_schema[].type`: `number | integer | category | image | text`. An `image` field adds
  `"shape": [h, w], "channels": c`; a `category` field adds `"categories": [...]`.
- `input_schema` is the **vocabulary the Schema agents map into**: it replaces the fixed six
  diabetes features of the original demo.
- `label.classes` is `null` for regression and text generation.
- `pricing` (reserved): `null` now; later `{"currency": "SOL", "enrollment_fee": 0.5,
  "per_prediction": 0.001, "revenue_share": "by_rows"}`.

### 4.4 Membership

```json
{"id": "mem_4", "company_id": "co_3", "company_name": "Tbilisi Heart Center",
 "group_id": "grp_1", "group_name": "Diabetes risk", "status": "schema_proposed",
 "node": {"id": "node_9", "status": "online", "last_seen": "2026-10-18T14:03:20Z",
          "agent": {"provider": "deepseek", "model": "deepseek-chat"},
          "connector": {"kind": "sqlite", "display": "patients.db / table visits"}},
 "data": {"row_count": 120, "columns": ["glu_mmol", "bmi", "age_y", "outcome"],
          "profile": {"glu_mmol": {"min": 3.1, "p05": 4.0, "median": 6.2, "mean": 6.6,
                                   "p95": 10.9, "max": 13.4, "missing": 0}}},
 "schema": {"label": "outcome", "made_by": "llm",
            "reasoning": "glu_mmol has a median of 6.2, which is mmol/L; converted to mg/dL",
            "mapping": [{"feature": "glucose", "source": ["glu_mmol"], "unit": "mmol/L",
                         "factor": 18.0182, "transform": "x * 18.0182"}],
            "confirmed": false},
 "validation": null,
 "contribution": {"rows": 0, "rounds": 0},
 "requested_at": "2026-10-18T11:00:00Z"}
```

- `node` is `null` until a node says hello. `data` and `schema` are `null` until the node
  sends them. They hold names and statistics only, never rows.
- `validation` once posted: `{"ok": true, "checks": [{"name": "label_balance", "ok": true,
  "detail": "38% positive"}]}`.

### 4.5 Job

```json
{"id": "job_2", "group_id": "grp_1", "kind": "train", "status": "running",
 "step": "training", "round": 2, "rounds": 4,
 "parent_version_id": null,
 "participants": [{"membership_id": "mem_4", "node_id": "node_9", "company_name": "Tbilisi Heart Center",
                   "state": "training", "color": "#4d8bff"}],
 "metrics": {"auc_series": [0.5, 0.81], "loss_series": [0.69, 0.55], "eps_total": 1.5},
 "started_at": "2026-10-18T14:00:00Z", "finished_at": null, "error": null}
```

`participants[].color` is assigned by the server (stable per job) so every view draws a
company in the same color.

### 4.6 ModelVersion

```json
{"id": "mv_1", "group_id": "grp_1", "version": 1, "parent_version_id": null, "job_id": "job_2",
 "model_type": "logistic", "features": ["glucose", "bmi", "age"],
 "metrics": {"auc": 0.832, "auc_solo_by_company": {"co_3": 0.763}, "auc_centralized": null},
 "contributors": [{"company_id": "co_3", "rows": 120, "rounds": 4}],
 "sha256": "9f2c…", "created_at": "2026-10-18T14:05:00Z"}
```

### 4.7 UsageEvent (payments seam)

```json
{"id": "use_1", "company_id": "co_3", "group_id": "grp_1", "kind": "prediction", "units": 1,
 "amount": null, "currency": null, "payment_ref": null, "created_at": "2026-10-18T14:10:00Z"}
```

The server writes one on every billable action (approval → `enrollment`, every round a
company trains in → `training_round` + `contribution` with `units` = rows, every predict →
`prediction`). `amount` and `payment_ref` stay `null` until a payment provider exists.

---

## 5. REST endpoints (browser ⇄ server)

`A` = admin only, `C` = company (own data only), `*` = any signed-in user, `—` = no auth.

### 5.1 Meta and auth

| Method & path | Who | Body → Response |
|---|---|---|
| `GET /meta` | — | → `{"version": "0.1", "enums": {...§2...}, "available_model_types": ["logistic"]}` |
| `POST /auth/login` | — | `{"email"}` → `{"token", "user": User, "company": Company \| null}` (demo login: no password; seeded and registered emails only) |
| `POST /auth/logout` | * | → `204` |
| `GET /auth/me` | * | → `{"user": User, "company": Company \| null}` |

### 5.2 Companies

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /companies/register` | — | `{"name", "field", "contact_email", "wallet_address"?}` → `{"token", "user", "company"}` (status `pending`; the user can sign in and wait) |
| `GET /companies?status=pending` | A | → `{"items": [Company]}` |
| `GET /companies/{id}` | A, C(own) | → Company |
| `POST /companies/{id}/approve` | A | → Company |
| `POST /companies/{id}/reject` | A | `{"reason"?}` → Company |

### 5.3 Groups

| Method & path | Who | Body → Response |
|---|---|---|
| `GET /groups` | * | → `{"items": [Group]}`. Companies see `open` and `trained` groups plus groups they belong to. |
| `GET /groups/{id}` | * | → Group + `"members": [Membership]` (admin: all; company: own only) + `"versions": [ModelVersion]` |
| `POST /groups` | A | `{"name", "description", "task_spec"}` → Group (`draft`) |
| `PATCH /groups/{id}` | A | partial Group → Group (`task_spec` is editable only while `draft`) |
| `POST /groups/{id}/publish` | A | → Group (`draft → open`) |
| `POST /groups/draft-spec` | A | `{"question", "notes"?}` → `{"task_spec", "reasoning", "made_by"}`. The **architect agent** drafts a TaskSpec from a plain question; the admin edits it before saving. |

### 5.4 Memberships

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /groups/{id}/join` | C | → Membership (`requested`) |
| `GET /memberships?group_id=&status=` | A, C(own) | → `{"items": [Membership]}` |
| `GET /memberships/{id}` | A, C(own) | → Membership |
| `POST /memberships/{id}/approve` | A | → Membership |
| `POST /memberships/{id}/reject` | A | `{"reason"?}` → Membership |
| `POST /memberships/{id}/node-token` | C(own) | → `{"node_token", "command": "python hub/node/node.py --server http://… --token … --db <your connection>"}`. Calling again revokes the old token. Needs status ≥ `approved`. |
| `POST /memberships/{id}/schema/confirm` | C(own) | `{"mapping"?: [...], "label"?}`: optional edits → Membership (`schema_confirmed`). The server forwards it to the node, which validates and reports back. |
| `POST /memberships/{id}/leave` | C(own) | → Membership (`left`) |

### 5.5 Jobs and live events

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /groups/{id}/jobs` | A | `{"kind": "train" \| "finetune", "membership_ids"?: [...]}` → Job (`queued`). Default participants: every `ready` and `active` membership. `finetune` starts from the group's current version. `409` if a job is already running or fewer than `min_participants` are ready. |
| `GET /groups/{id}/jobs` | * (members) | → `{"items": [Job]}` |
| `GET /jobs/{id}` | * (members) | → Job |
| `POST /jobs/{id}/cancel` | A | → Job |
| `GET /jobs/{id}/events?since=<seq>` | * (members) | → `{"items": [Event]}`: the history, for replay or catch-up |
| `GET /jobs/{id}/stream` | * (members) | **Server-Sent Events**, §6. EventSource cannot set headers, so this endpoint also accepts `?token=<session_token>`. |
| `GET /jobs/{id}/trace` | * (members) | → a `trace.json` (CONTRACTS.md §5) for a finished job, so the original replay player can show it |

"Members" means the admin, or a company with a membership in that group.

### 5.6 Models and predictions

| Method & path | Who | Body → Response |
|---|---|---|
| `GET /groups/{id}/versions` | * (members) | → `{"items": [ModelVersion]}` |
| `POST /groups/{id}/predict` | C(`active`), A | `{"inputs": {"glucose": 148, "bmi": 33.6, "age": 50}, "version_id"?}` → `{"prediction": "yes", "probability": 0.73, "scores": {"no": 0.27, "yes": 0.73}, "version_id": "mv_1", "explanation": [{"feature": "glucose", "contribution": 0.91}]}` |

Inputs use the TaskSpec's canonical names and units. The server validates them against
`input_schema` (`422` names the field). `explanation` is optional: logistic returns
per-feature contributions, and other model types may return `[]`.

### 5.7 Usage (payments seam)

| Method & path | Who | Body → Response |
|---|---|---|
| `GET /usage?company_id=&group_id=` | A, C(own) | → `{"items": [UsageEvent], "totals": {"prediction": 12, "training_round": 8}}` |

### 5.8 Demo helpers (judges' demo, admin only)

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /demo/nodes/start` | A | `{"membership_id"}` → `{"pid"}`: starts that company's seeded demo `node.py` as a **separate process** on the server machine |
| `POST /demo/nodes/stop` | A | `{"membership_id"}` → `204`: kills it, to show a company going offline mid-training |
| `POST /demo/reset` | A | → `204`: wipes the database and re-seeds |

---

## 6. Live events (SSE): what the visualisation is driven by

`GET /jobs/{id}/stream` sends one SSE message per event:

```
id: 57
event: agent.message
data: {"seq":57,"ts":"2026-10-18T14:02:11Z","job_id":"job_2","round":2,"type":"agent.message", ...}
```

On reconnect the browser sends `Last-Event-ID` and the server resends everything after it.
The same objects come from `GET /jobs/{id}/events`. Every event has `seq` (per job,
increasing, no gaps), `ts`, `job_id`, `round` (`0` before training) and `type`, plus:

| `type` | extra fields | drives |
|---|---|---|
| `job.status` | `status`, `error` | header badge |
| `job.step` | `step` | the stepper |
| `participant.state` | `membership_id`, `node_id`, `state`, `detail` | node color / opacity on the canvas, the per-company status line |
| `agent.message` | `from: {"node_id" \| "coord", "agent", "company_name"}`, `to` (node id, `"coord"` or `"*"`), `text`, `made_by`, `data` | the **agent conversation feed** (one bubble per message) |
| `packet` | `from`, `to`, `kind` (`update` \| `broadcast` \| `probe` \| `key`), `masked`, `rejected` | the animated packets on the canvas |
| `round.decision` | `strategy`, `participants`, `excluded`, `reasoning`, `decided_by` | the decision-log card |
| `round.completed` | `auc_before`, `auc_after`, `loss`, `eps_total`, `applied` | the AUC/loss chart |
| `model.published` | `version_id`, `version`, `metrics` | the "Published vN" step, toast |

`text` is always human-readable and never contains a data row, for example
`"Planner: 4 epochs @ lr 0.1 — loss is falling steadily, keep the default"`.

---

## 7. Node protocol (node ⇄ server): lanes D and F

The node authenticates with `X-Node-Token`. It only ever talks to the server.

| Method & path | Body → Response |
|---|---|
| `POST /node/hello` | `{"node_version", "agent": {"provider", "model"}, "connector": {"kind", "display"}}` → `{"node_id", "membership_id", "group": {"id", "name", "task_spec"}, "poll_wait_s": 20}`. Membership `→ node_connected`. |
| `POST /node/profile` | `{"row_count", "columns", "profile": {col: {min,p05,median,mean,p95,max,missing}}}` → `204` |
| `POST /node/schema-proposal` | `{"label", "mapping": [...§4.4...], "reasoning", "made_by"}` → `204`. Membership `→ schema_proposed`. |
| `POST /node/validation` | `{"ok", "checks": [...]}` → `204`. If `ok` and the schema is confirmed: membership `→ ready`. |
| `GET /node/inbox?wait=20` | long-poll → `{"messages": [Message]}`, or an empty list after `wait` seconds. **Polling is the heartbeat:** no poll for 15 s means `node.status = offline`. |
| `POST /node/outbox` | `{"messages": [{"to", "kind", "payload"}]}` → `{"accepted": n}`. The server runs `agents.runtime.sanitize()` on every payload, so raw-data keys are rejected with `422` exactly as on the in-process bus. |

**Message** is `agents.runtime.Message.to_dict()`: `{"sender", "recipient", "kind",
"payload", "tick", "seq"}`. The `kind`s and payloads during a job are those in
`src/agents/hospital.py` and CONTRACTS.md §3 (`schema_request`, `schema`, `schema_agreed`,
`probe`, `probe_result`, `train`, `update`, `broadcast`), plus these platform kinds,
whose payloads lane F defines in the F1 PR:

| kind | direction | purpose |
|---|---|---|
| `schema_confirmed` | server → node | the company's confirmed mapping (sent outside a job) |
| `activity` | node → coord | `{"agent", "text", "made_by"}`: what a local agent just did; becomes an `agent.message` event (§6) |
| `key_offer` / `key_set` | node → coord / coord → nodes | Diffie-Hellman public keys for the pairwise masks (F3) |
| `stats_request` / `stats` | coord → nodes / node → coord | masked sums for federated normalisation (F4) |
| `job_end` | coord → nodes | the job finished; the node stores the final model locally |

On the bus a participant's id is its `node_id`; display names come from the membership.

---

## 8. Internal interface: backend (D) ⇄ federation (F)

So lanes D and F can work in parallel, the server routes call only this Python interface
(`hub/federation/service.py`, owned by F). D writes the routes and the DB; F writes
the bus, the coordinator hosting and the events.

```python
class FederationService:
    def __init__(self, emit: Callable[[dict], None], on_finished: Callable[[str, dict], None]): ...
        # emit(event)                  every §6 event; D persists it and fans it out over SSE
        # on_finished(job_id, result)  result = {"status", "error", "weights", "features",
        #                                "norm": {"mean", "std"}, "metrics", "contributors"}
    def start_job(self, job_id: str, task_spec: dict, participants: list[dict],
                  init_weights: list[float] | None) -> None: ...   # returns at once; runs on a thread
        # participants[i] = {"membership_id", "node_id", "company_name", "schema": {...confirmed...}}
    def cancel_job(self, job_id: str) -> None: ...
    def deliver_from_node(self, node_id: str, messages: list[dict]) -> None: ...  # from POST /node/outbox
    def inbox_for_node(self, node_id: str, wait_s: float) -> list[dict]: ...     # for GET /node/inbox
    def predict(self, model: dict, inputs: dict) -> dict: ...  # model = a stored on_finished result
```

Until F1 lands, D uses a **fake** `FederationService` that replays the events of the
committed `trace.json` on a timer. That lets lanes D and E build the whole live view first.

---

## 9. Rules both sides keep

- The server never stores or forwards a data row, and the node never sends one: `sanitize()`
  runs on everything a node posts.
- Every LLM decision is an `agent.message` with `made_by`, so the UI can show which
  decisions were the model's and which fell back to rules.
- The frontend reads enums and model types from `GET /meta`; it never hardcodes them.
- The payments seam is the `usage` table plus `pricing` and `wallet_address`. No endpoint
  moves money in v0.1.
