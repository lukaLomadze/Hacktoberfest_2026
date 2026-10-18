# PrivateLearn Platform — API contract (frontend ⇄ backend ⇄ node)

> The agreement every platform lane codes against. The frontend (lane E), the backend
> (lane D) and the company node + federation (lane F) never need to read each other's code:
> they read this file. A change is its own small PR that bumps the version below and says in
> the PR which lanes must follow. The engine contracts (`CONTRACTS.md`) still hold for the
> math and the bus messages; this file adds everything around them.

Version: **0.2** (draft). Implemented by the backend in `hub/server/` (tasks D1–D9).

**Changes since 0.1** (found while building the backend; frontend and node lanes, read these):
- Errors: new codes `conflict` (409, duplicate name/email/membership), `method_not_allowed`
  (405), `internal` (500). Request bodies must be sent with `Content-Type: application/json`,
  and unknown request fields are rejected with `422` (a typo is never silently ignored).
- New endpoints: `POST /companies/{id}/suspend`, `POST /groups/{id}/archive`.
- `Group.current_version` is `{"id", "version", "metrics", "created_at"}` or `null`.
- Membership adds `status_reason`, `has_node_token`, `activated_at`, `updated_at`,
  `schema.edited`, `schema.missing_required`; Job adds `version_id`, `participants[].company_id`
  and `participants[].contribution`; ModelVersion adds `synthetic`.
- Events: `participants` / `excluded` in `round.decision` are **membership ids**; the stream
  ends with `event: end` once the job is over (call `EventSource.close()`).
- Node: `poll_wait_s` is **10**; `hello` also returns `membership_status` and the confirmed
  `schema`; profiles need at least 10 rows and only the listed statistics.
- Predict: who may call it, `imputed`, `synthetic` and `version` in the response (§5.6).
- §8: the server calls `bind(emit, on_finished)` instead of passing them to `__init__`; new
  `send_to_node`; the server (not the federation) announces `published`.

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
| Lists | always wrapped: `{"items": [...]}` (no pagination, except job events: §5.5) |
| Request bodies | a JSON object with `Content-Type: application/json`; unknown fields → `422` naming the field |
| Status codes | `201` for a POST that creates something (register, group, join, job); `204` for no body; else `200` |
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
| 405 | `method_not_allowed` | the path exists, not with this method |
| 409 | `invalid_transition` | the state machine (§3) does not allow this action now |
| 409 | `conflict` | a duplicate: company name, email, or an open membership in the same group |
| 422 | `validation_error` | a field has a bad value; `field` names it (e.g. `"task_spec.training.rounds"`) |
| 500 | `internal` | a server bug; the message never contains internals |
| 503 | `node_offline` | reserved: no endpoint returns it yet (messages to an offline node are queued) |

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
right now. The frontend offers only those (others shown disabled). v0.2 guarantees
`["logistic"]`.

---

## 3. State machines

**Company:** `pending → approved | rejected`, `approved ⇄ suspended` (`approve` also
reinstates a suspended company; `rejected` is final). Only an approved company can join
groups, issue node tokens, confirm schemas or predict, and its nodes are refused (`403`)
while it is suspended.

**Group:** `draft → open` (admin publishes) `→ training` (job running) `→ trained`
(a model version exists) `→ training` again for a fine-tune `→ trained`. `archived` from any
state except while a job is queued or running; archiving rejects pending join requests. A
failed or cancelled job returns the group to `trained` if it has a model, else `open`.

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

Any membership not in `training` may move to `left` (this also revokes its node token). A node
that says hello again (restart, reconnect) never moves its membership backwards. Two things do
move it back, on purpose:
- a profile whose **columns changed** → `node_connected` (schema and validation cleared);
- a **failed validation** of a `ready`/`active` membership → `schema_confirmed`, so it is left
  out of jobs until the node reports `ok` again.

While a membership is `training`, its data, schema, token and membership are frozen (`409`).
A membership that has ever received a model keeps `activated_at`, and with it the right to
predict, even while it re-maps its data.

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
 "member_count": 3,
 "current_version": {"id": "mv_1", "version": 1, "metrics": {"auc": 0.832}, "created_at": "2026-10-18T14:05:00Z"},
 "created_at": "2026-10-18T10:00:00Z", "updated_at": "2026-10-18T10:00:00Z"}
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
- `label.classes` is `null` for regression and text generation; a binary label has exactly two
  classes and names the `positive` one. The label may not also be an input.
- The server fills every default and returns the normalised spec. It also checks the parts fit:
  the model type must be in `available_model_types` and support the task type and every input
  type (logistic: binary, number/integer inputs only); with `secure_aggregation`,
  `min_participants` must be ≥ 2 (one "masked" update is not hidden). With `dp.enabled: false`,
  `epsilon_per_round` and `delta` are `null`. `metric` defaults per task type.
- `member_count` counts memberships from `approved` to `active` (not requests or closed ones).
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
            "confirmed": false, "edited": false, "missing_required": ["bmi", "age"]},
 "validation": null,
 "contribution": {"rows": 0, "rounds": 0},
 "status_reason": null, "has_node_token": true, "activated_at": null,
 "requested_at": "2026-10-18T11:00:00Z", "updated_at": "2026-10-18T11:05:00Z"}
```

- `node` is `null` until a node says hello. `data` and `schema` are `null` until the node
  sends them. They hold names and statistics only, never rows.
- `schema.missing_required` lists required inputs the mapping does not cover; the company
  cannot confirm until it is empty. `schema.edited` is true once the company changed the
  node's proposal. `contribution.rounds` adds up over jobs; `contribution.rows` is the rows
  used in the latest job.
- `node.status`: `online` while a long-poll is open or within 15 s of the node's last request,
  `busy` when online and training, else `offline`. The node also reports `node_version`.
- `validation` once posted: `{"ok": true, "checks": [{"name": "label_balance", "ok": true,
  "detail": "38% positive"}]}`.

### 4.5 Job

```json
{"id": "job_2", "group_id": "grp_1", "kind": "train", "status": "running",
 "step": "training", "round": 2, "rounds": 4,
 "parent_version_id": null, "version_id": null,
 "participants": [{"membership_id": "mem_4", "node_id": "node_9", "company_id": "co_3",
                   "company_name": "Tbilisi Heart Center", "state": "training", "color": "#4d8bff",
                   "contribution": {"rows": 0, "rounds": 0}}],
 "metrics": {"auc_series": [0.5, 0.81], "loss_series": [0.69, 0.55], "eps_total": 1.5},
 "created_at": "2026-10-18T13:59:58Z", "started_at": "2026-10-18T14:00:00Z", "finished_at": null, "error": null}
```

`participants[].color` is assigned by the server (stable per job) so every view draws a
company in the same color. Participants are in join order. `node_id` is `null` for a member
that never connected a node (seeded dev data). `version_id` is set once the job published.

### 4.6 ModelVersion

```json
{"id": "mv_1", "group_id": "grp_1", "version": 1, "parent_version_id": null, "job_id": "job_2",
 "model_type": "logistic", "features": ["glucose", "bmi", "age"],
 "metrics": {"auc": 0.832, "auc_solo_by_company": {"co_3": 0.763}, "auc_centralized": null},
 "contributors": [{"company_id": "co_3", "rows": 120, "rounds": 4}],
 "sha256": "9f2c…", "synthetic": false, "created_at": "2026-10-18T14:05:00Z"}
```

`synthetic: true` marks a placeholder model from the fake federation (a group whose inputs are
not the recorded run's): it predicts, but was not trained. The UI must say so.

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
| `GET /meta` | — | → `{"version": "0.2", "enums": {...§2...}, "available_model_types": ["logistic"]}` |
| `POST /auth/login` | — | `{"email"}` → `{"token", "user": User, "company": Company \| null}` (demo login: no password; seeded and registered emails only) |
| `POST /auth/logout` | * | → `204` |
| `GET /auth/me` | * | → `{"user": User, "company": Company \| null}` |

### 5.2 Companies

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /companies/register` | — | `{"name", "field", "contact_email", "wallet_address"?}` → `{"token", "user", "company"}` (status `pending`; the user can sign in and wait) |
| `GET /companies?status=pending` | A | → `{"items": [Company]}` |
| `GET /companies/{id}` | A, C(own) | → Company |
| `POST /companies/{id}/approve` | A | `{"reason"?}` → Company (`pending` or `suspended` → `approved`) |
| `POST /companies/{id}/reject` | A | `{"reason"?}` → Company (`pending → rejected`) |
| `POST /companies/{id}/suspend` | A | `{"reason"?}` → Company (`approved → suspended`) |

### 5.3 Groups

| Method & path | Who | Body → Response |
|---|---|---|
| `GET /groups` | * | → `{"items": [Group]}`. Companies see `open`, `training` and `trained` groups plus groups they belong to. A group a company cannot see is `404`, never `403`. |
| `GET /groups/{id}` | * | → Group + `"members": [Membership]` (admin: all; company: own only) + `"versions"`: summaries like `current_version` (empty for non-members) |
| `POST /groups` | A | `{"name", "description"?, "task_spec", "pricing"?}` → Group (`draft`) |
| `PATCH /groups/{id}` | A | partial Group → Group (`task_spec` is editable only while `draft`) |
| `POST /groups/{id}/publish` | A | → Group (`draft → open`; the spec is re-checked against `available_model_types`) |
| `POST /groups/{id}/archive` | A | → Group (`archived`; `409` while a job runs) |
| `POST /groups/draft-spec` | A | `{"question", "notes"?}` → `{"task_spec", "reasoning", "made_by", "warnings": [...]}`. The **architect agent** drafts a TaskSpec from a plain question; the admin edits it before saving. Without a reachable LLM, or when its draft is invalid, it returns a template (`made_by: "rule"`) and `warnings` say what to fill in. Nothing is saved. |

### 5.4 Memberships

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /groups/{id}/join` | C | → Membership (`requested`). Group must be `open`, `training` or `trained`; `409 conflict` if the company already has an open membership there (a new request is allowed after `rejected`/`left`). |
| `GET /memberships?group_id=&status=` | A, C(own) | → `{"items": [Membership]}` |
| `GET /memberships/{id}` | A, C(own) | → Membership |
| `POST /memberships/{id}/approve` | A | → Membership |
| `POST /memberships/{id}/reject` | A | `{"reason"?}` → Membership |
| `POST /memberships/{id}/node-token` | C(own) | → `{"node_token", "command": "python hub/node/node.py --server http://… --token … --db <your connection>"}`. Calling again revokes the old token (the node keeps its `node_id`). Needs status ≥ `approved`, not `training`. The token is shown only here; the server stores its hash. |
| `POST /memberships/{id}/schema/confirm` | C(own) | `{"mapping"?: [...], "label"?}`: optional edits → Membership (`schema_confirmed`, validation cleared). Allowed from `schema_proposed` to `active`. Every required input must be mapped (`422` field `mapping`). The server queues it to the node as `schema_confirmed`; the node validates and reports back. |
| `POST /memberships/{id}/leave` | C(own) | → Membership (`left`) |

### 5.5 Jobs and live events

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /groups/{id}/jobs` | A | `{"kind": "train" \| "finetune", "membership_ids"?: [...]}` → Job (`queued`, `201`). Group must be `open` or `trained`; participants' companies must be approved. Default participants: every `ready` and `active` membership. `finetune` starts from the group's current version. `409` if a job is already running or fewer than `min_participants` are ready. |
| `GET /groups/{id}/jobs` | * (members) | → `{"items": [Job]}` |
| `GET /jobs/{id}` | * (members) | → Job |
| `POST /jobs/{id}/cancel` | A | → Job |
| `GET /jobs/{id}/events?since=<seq>` | * (members) | → `{"items": [Event], "has_more": bool}`: up to 1000 events after `since`; ask again from the last `seq` while `has_more` |
| `GET /jobs/{id}/stream` | * (members) | **Server-Sent Events**, §6. EventSource cannot set headers, so this endpoint also accepts `?token=<session_token>`. |
| `GET /jobs/{id}/trace` | * (members) | → a `trace.json` (CONTRACTS.md §5) for a finished job, so the original replay player can show it. **Not implemented yet** (task F6): `404` until then. |

"Members" means the admin, or a company with a membership in that group.

### 5.6 Models and predictions

| Method & path | Who | Body → Response |
|---|---|---|
| `GET /groups/{id}/versions` | * (members) | → `{"items": [ModelVersion]}` |
| `POST /groups/{id}/predict` | C (has received a model), A | `{"inputs": {"glucose": 148, "bmi": 33.6, "age": 50}, "version_id"?}` → `{"prediction": "yes", "probability": 0.73, "scores": {"no": 0.27, "yes": 0.73}, "version_id": "mv_1", "version": 1, "explanation": [{"feature": "glucose", "contribution": 0.91}], "imputed": [], "synthetic": false}` |

Inputs use the TaskSpec's canonical names and units. The server validates them against
`input_schema` (`422` names the field): unknown names are rejected, required inputs must be
present, numbers finite, integers whole (`2.0` is fine). An optional input left out is imputed
with the training mean and listed in `imputed`. `explanation` is sorted by |contribution|;
other model types may return `[]`. A company may predict once its membership has received a
model (`activated_at` set) and it has not left; each call is a `prediction` usage event (the
admin's calls are not billed). `409` if the group has no model yet; `404` for an unknown
`version_id`. Only binary tasks predict until more model adapters exist.

### 5.7 Usage (payments seam)

| Method & path | Who | Body → Response |
|---|---|---|
| `GET /usage?company_id=&group_id=` | A, C(own) | → `{"items": [UsageEvent], "totals": {"prediction": 12, "training_round": 8}}`. A company asking for another company's usage gets `403`. |

### 5.8 Demo helpers (judges' demo, admin only)

| Method & path | Who | Body → Response |
|---|---|---|
| `POST /demo/nodes/start` | A | `{"membership_id"}` → `{"pid"}`: starts that company's seeded demo `node.py` as a **separate process** on the server machine |
| `POST /demo/nodes/stop` | A | `{"membership_id"}` → `204`: kills it, to show a company going offline mid-training |
| `POST /demo/reset` | A | → `204`: wipes the database and re-seeds |

Not implemented yet (task D10 waits for the node, F2).

---

## 6. Live events (SSE): what the visualisation is driven by

`GET /jobs/{id}/stream` sends one SSE message per event:

```
id: 57
event: agent.message
data: {"seq":57,"ts":"2026-10-18T14:02:11Z","job_id":"job_2","round":2,"type":"agent.message", ...}
```

On reconnect the browser sends `Last-Event-ID` (or the page passes `?since=<seq>`) and the
server resends everything after it. When the job is over and every event has been sent, the
server sends `event: end` (`data: {}`) and closes: call `EventSource.close()` on it, or the
browser reconnects every 2 s. Comment lines (`: keep-alive`) may arrive; ignore them.
The same objects come from `GET /jobs/{id}/events`. Every event has `seq` (per job,
increasing, no gaps), `ts`, `job_id`, `round` (`0` before training) and `type`, plus:

| `type` | extra fields | drives |
|---|---|---|
| `job.status` | `status`, `error` | header badge |
| `job.step` | `step` | the stepper |
| `participant.state` | `membership_id`, `node_id`, `state`, `detail` | node color / opacity on the canvas, the per-company status line |
| `agent.message` | `from: {"node_id" \| "coord", "agent", "company_name"}`, `to` (node id, `"coord"` or `"*"`), `text`, `made_by`, `data` | the **agent conversation feed** (one bubble per message) |
| `packet` | `from`, `to` (a node id, the membership id for a member without a node, or `"coord"`), `kind` (`update` \| `broadcast` \| `probe` \| `key`), `masked`, `rejected` | the animated packets on the canvas |
| `round.decision` | `strategy`, `participants`, `excluded` (both **membership ids**), `reasoning`, `decided_by` | the decision-log card |
| `round.completed` | `auc_before`, `auc_after`, `loss`, `eps_total`, `applied` | the AUC/loss chart |
| `model.published` | `version_id`, `version`, `metrics` | the "Published vN" step, toast |

The server always ends a successful job with `job.step published`, `model.published`,
`job.status succeeded` (in that order, written together with the final job state), and a
failed or cancelled one with `job.status` carrying `error`. Events that arrive for a job
that is already over are dropped.

`text` is always human-readable and never contains a data row, for example
`"Planner: 4 epochs @ lr 0.1 — loss is falling steadily, keep the default"`.

---

## 7. Node protocol (node ⇄ server): lanes D and F

The node authenticates with `X-Node-Token`. It only ever talks to the server.

| Method & path | Body → Response |
|---|---|
| `POST /node/hello` | `{"node_version", "agent": {"provider", "model"}, "connector": {"kind", "display"}}` → `{"node_id", "membership_id", "membership_status", "group": {"id", "name", "task_spec"}, "schema": confirmed schema or null, "poll_wait_s": 10}`. Membership `approved → node_connected`; later states are kept. A restarted node learns its confirmed mapping from `schema`. |
| `POST /node/profile` | `{"row_count", "columns", "profile": {col: {min,p05,median,mean,p95,max,missing}}}` → `204`. One entry per column, only these keys, numbers or `null`. `row_count` ≥ 10: below that, min/max would be individual rows. |
| `POST /node/schema-proposal` | `{"label", "mapping": [...§4.4...], "reasoning", "made_by"}` → `204`. Needs a profile first. Sources must be reported columns, never the label; each input mapped at most once. Membership `→ schema_proposed` (re-proposing the confirmed mapping changes nothing). |
| `POST /node/validation` | `{"ok", "checks": [{"name", "ok", "detail"}]}` → `204`. `ok` may not be true while a check failed. If `ok` and the schema is confirmed: `schema_confirmed → ready` (or `active`). A failure moves `ready`/`active` back to `schema_confirmed`. |
| `GET /node/inbox?wait=10` | long-poll → `{"messages": [Message]}` as soon as one is queued, or an empty list after `wait` seconds (0 ≤ `wait` ≤ `poll_wait_s`). **Polling is the heartbeat** (§4.4 `node.status`). |
| `POST /node/outbox` | `{"messages": [{"to", "kind", "payload"}]}` → `{"accepted": n}`. The server runs `agents.runtime.sanitize()` on every payload, so raw-data keys are rejected with `422` exactly as on the in-process bus. At most 100 messages per call. Until engine task X2, `sanitize()` checks `update` messages against the fixed ids h0/h1/h2, so real node updates fail it. |

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
Messages from the server itself have `"sender": "server"`. The node endpoints other than
inbox and outbox answer `409` while the membership is `training` or the group is archived,
and `401` once the token is revoked or the membership has left.

---

## 8. Internal interface: backend (D) ⇄ federation (F)

So lanes D and F can work in parallel, the server routes call only this Python interface
(`hub/federation/service.py`, owned by F). D writes the routes and the DB; F writes
the bus, the coordinator hosting and the events.

```python
class FederationService:
    available_model_types: tuple                      # what /meta will advertise once X5 lands
    def bind(self, emit: Callable[[dict], None], on_finished: Callable[[str, dict], None]) -> None: ...
        # called once by the server before anything else.
        # emit(event)                  every §6 event except the closing ones (below); each event
        #                                carries "job_id"; D persists it and serves it over SSE
        # on_finished(job_id, result)  result = {"status": "succeeded" | "failed", "error",
        #                                "model_type", "features", "weights", "norm": {"mean", "std"},
        #                                "metrics", "contributors": [{"membership_id", "rows", "rounds"}],
        #                                "synthetic"?}
    def start_job(self, job_id: str, task_spec: dict, participants: list[dict],
                  init_weights: list[float] | None) -> None: ...   # returns at once; runs on a thread
        # participants[i] = {"membership_id", "node_id", "company_name", "schema": {...confirmed...}}
    def cancel_job(self, job_id: str) -> None: ...                # stop; do NOT call on_finished afterwards
    def deliver_from_node(self, node_id: str, messages: list[dict]) -> None: ...  # from POST /node/outbox
    def inbox_for_node(self, node_id: str, wait_s: float) -> list[dict]: ...     # for GET /node/inbox; blocks
    def send_to_node(self, node_id: str, kind: str, payload: dict) -> None: ...  # queue a server message
    def predict(self, model: dict, inputs: dict) -> dict: ...
        # model = {"model_type", "features", "norm", "weights"}; inputs already validated
        # → {"probability", "contributions": [{"feature", "contribution"}], "imputed": [...]}
    def shutdown(self) -> None: ...                               # wake long-polls, stop job threads
```

What the server does with the callbacks, so the federation does not have to:
- a result whose weights, features and normalisation do not line up, or contain NaN, fails
  the job instead of publishing a broken model;
- it creates the ModelVersion, moves members to `active`, writes the usage ledger and emits
  `job.step published`, `model.published` and `job.status` itself, in one transaction;
- `job.status` from `emit` may only say `running`; finishing goes through `on_finished`;
- NaN/Infinity in an event becomes `null`; a malformed event is logged and dropped;
- on startup, jobs a previous server process left running are failed and their members released.

Until F1 lands, D uses a **fake** `FederationService` (`hub/server/fake_federation.py`). It
replays the committed `trace.json` on a timer for the job's real participants (participant
*i* plays recorded hospital *i* mod 3, and recorded hospital names are replaced by company
names), then publishes the recording's real final model. For a group whose inputs are not the
recording's, the published model is `synthetic`. `HUB_FAKE_DELAY` scales the speed
(1 ≈ one minute per job, 0 = instant). That lets lanes D and E build the whole live view first.

---

## 9. Rules both sides keep

- The server never stores or forwards a data row, and the node never sends one: `sanitize()`
  runs on everything a node posts.
- Every LLM decision is an `agent.message` with `made_by`, so the UI can show which
  decisions were the model's and which fell back to rules.
- The frontend reads enums and model types from `GET /meta`; it never hardcodes them.
- The payments seam is the `usage` table plus `pricing` and `wallet_address`. No endpoint
  moves money in v0.2.
