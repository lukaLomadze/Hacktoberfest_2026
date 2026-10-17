# Tasks

The live task list is the **[GitHub Issues board](https://github.com/lukaLomadze/Hacktoberfest_2026/issues)**:
one issue per task, assignment is the claim, and a task starts only when its dependencies are
merged. The dependency graph and the claim → branch → PR loop are in `CONTRIBUTING.md`.

## Contributors
| GitHub username | Agent / model | Lane |
|---|---|---|
| lukaLomadze | Claude | B — agents/data/pipeline · C — web player + deck · shared files · platform D/E/F (one agent session per lane) |
| Gradnos | DeepSeek | A — model, aggregation, privacy, attack · contracts · design spec |
| _friend (username TBD)_ | — | X — engine fixes in existing code for the platform |

Add a row when someone joins.

## Ownership
| Area / files | Owner |
|---|---|
| `privatelearn/src/model/`, `privatelearn/docs/CONTRACTS.md`, `privatelearn/docs/DESIGN.md` | Gradnos |
| `privatelearn/src/agents/`, `privatelearn/src/pipeline/`, `privatelearn/data/`, `privatelearn/web/`, `privatelearn/docs/deck.md` | lukaLomadze |
| Shared: `AGENTS.md`, `CLAUDE.md`, `TASKS.md`, `CONTRIBUTING.md`, `.github/`, root `README.md` | lukaLomadze |
| Platform contract: `privatelearn/docs/PLATFORM-API.md`, `docs/PLATFORM-PLAN.md` | lukaLomadze |
| Platform lane D (backend): `privatelearn/hub/server/`, `hub/requirements.txt`, `tests/test_hub_server/` | lukaLomadze (D session) |
| Platform lane E (frontend): `privatelearn/hub/web/` | lukaLomadze (E session) |
| Platform lane F (federation + node): `privatelearn/hub/federation/`, `hub/node/`, `hub/demo/`, `tests/test_hub_node/` | lukaLomadze (F session) |
| Platform lane X (engine fixes): the X tasks edit `privatelearn/src/` files; each X issue names its files, and the lane A/B owner reviews it | friend |

Two agent sessions of the same person still never share a lane: one session = one lane =
one branch at a time.

## Status (October 2026)
Phase 1 — T0–T12 are all merged: data → model → aggregation → privacy → attack → agents →
coordinator → end-to-end `trace.json` → player → deck. Remaining: the screen recording (needs a human).

Phase 2 — **the platform** (`docs/PLATFORM-PLAN.md`): companies, groups (one model each),
live multi-process agent nodes, fine-tuning. P0 (contract) is in review; tasks P0, X1–X7,
D1–D10, E1–E9, F1–F6 and I1 are listed in that plan's §5. First wave, all in parallel:
D1 · E1 → E6 · X1 · X2 · F6.
