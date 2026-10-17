# Tasks

The live task list is the **[GitHub Issues board](https://github.com/lukaLomadze/Hacktoberfest_2026/issues)**:
one issue per task, assignment is the claim, and a task starts only when its dependencies are
merged. The dependency graph and the claim → branch → PR loop are in `CONTRIBUTING.md`.

## Contributors
| GitHub username | Agent / model | Lane |
|---|---|---|
| lukaLomadze | Claude | B — agents/data/pipeline · C — web player + deck · shared files |
| Gradnos | DeepSeek | A — model, aggregation, privacy, attack · contracts · design spec |

Add a row when someone joins.

## Ownership
| Area / files | Owner |
|---|---|
| `privatelearn/src/model/`, `privatelearn/docs/CONTRACTS.md`, `privatelearn/docs/DESIGN.md` | Gradnos |
| `privatelearn/src/agents/`, `privatelearn/src/pipeline/`, `privatelearn/data/`, `privatelearn/web/`, `privatelearn/docs/deck.md` | lukaLomadze |
| Shared: `AGENTS.md`, `CLAUDE.md`, `TASKS.md`, `CONTRIBUTING.md`, `.github/`, root `README.md` | lukaLomadze |

## Status (October 2026)
T0–T12 are all merged: data → model → aggregation → privacy → attack → agents → coordinator →
end-to-end `trace.json` → player → deck. Remaining: the screen recording (needs a human).
