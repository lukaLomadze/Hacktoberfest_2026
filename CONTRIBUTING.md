# Contributing — how to start work here

This repo is a hackathon collaboration: **PrivateLearn** (agentic federated learning across
three hospitals) is the main project. This file is the entry point. Read it once and you can
pick up work in the right order without asking anyone.

## Read these, in this order

1. **`AGENTS.md`** — the shared rules for every contributor and agent (branches, PRs,
   ownership, commit style). It is authoritative for *how* we work.
2. **`docs/EXECUTION-PLAN.md`** — the PrivateLearn plan. §6 is the task table, §4 is file
   ownership, §7 is the schedule, §8 is the judge Q&A.
3. **`privatelearn/docs/CONTRACTS.md`** — the shared contracts: dataset, model math, the
   message/trace JSON shapes, and the feature schema. Non-negotiable — code against these.
4. **The [Issues](https://github.com/lukaLomadze/Hacktoberfest_2026/issues) tab** — the live
   claim board. It is the single source of truth for **who is doing what right now**.
5. **`privatelearn/README.md`** — how to run and test.

## Correctly ordering tasks

Tasks are T0–T12. You may only start a task whose **dependencies are merged**, not merely
opened. The issue body lists each task's dependencies. Current shape:

```
DONE            T0 (contracts) · T2 (model) · T10 (player)
READY NOW       T1 (B) · T6 (B) · T11 (C)
IN PROGRESS     T3 (A)
AFTER T3        T4 (A) · T8 (B)
AFTER T4        T5 (A) · T9 (B)
AFTER T6        T7 (B)
AFTER T7,T8,T4  T9 (B)
AFTER T9,T11    T12 (C)
```

**Critical path:** `T0 → T2 → T3 → T4 → T8 → T9 → T12`.

If the Issues board says a task is assigned or its dependencies are still open, pick a
different one — do not start it anyway.

## The loop (this is what stops two agents colliding)

1. **Claim.** Open Issues, pick an *unassigned* task whose dependencies are all merged,
   and **assign it to yourself**. Assigned = taken.
2. **Branch from fresh `main`:**
   ```powershell
   git checkout main && git pull
   git checkout -b <your-username>/<short-task-name>
   ```
3. **Edit only your lane's files** (plan §4). Need a file in another lane? Open an issue,
   don't edit it.
4. **Test before pushing:**
   ```powershell
   cd privatelearn
   python -m unittest discover -s tests -v
   python -c "import sys; sys.path.insert(0,'src'); from shared import schemas, io; schemas.validate_trace(io.load_json('data/sample_trace.json')); print('trace VALID')"
   ```
5. **Commit** `type: summary`, push, open a PR, request a **different** contributor to review.
6. **Merge fast** once reviewed — an unmerged dependency blocks everyone downstream.

## Lanes (exclusive file ownership)

| Lane | Owns | Currently |
|---|---|---|
| **A — model/math** | `src/model/`, aggregation, privacy, attack | Gradnos (DeepSeek) |
| **B — agents/schema** | `src/agents/`, `src/pipeline/`, `data/` | open |
| **C — frontend/demo** | `web/`, the deck + recording | open |

Shared/cross-cutting files (`docs/EXECUTION-PLAN.md`, `privatelearn/docs/CONTRACTS.md`,
`privatelearn/src/shared/`) change only through a small dedicated PR.

## Rules that specifically prevent duplicated work

- **One issue, one owner.** Assignment is the lock; there is no second "I'm doing this".
- **Exclusive file ownership.** Two agents never open the same file in the same week.
- **Branch only off `main`.** Never build on top of someone's open PR branch.
- **Contracts are frozen.** A `CONTRACTS.md` change is its own PR with a version bump.

## Quick test / run reference

```powershell
# model tests
cd privatelearn && python -m unittest discover -s tests/test_model -v

# validate the sample trace
cd privatelearn && python -c "import sys; sys.path.insert(0,'src'); from shared import schemas, io; schemas.validate_trace(io.load_json('data/sample_trace.json')); print('trace VALID')"

# demo player (needs a browser)
cd privatelearn && python -m http.server 8000   # open http://localhost:8000/web/index.html

# end-to-end run (once T9 exists)
cd privatelearn && python src/pipeline/run_loop.py
```
