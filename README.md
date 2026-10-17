# Hacktoberfest Georgia 2026 — PrivateLearn

**One model, zero shared data.** Three hospitals train a single diabetes-risk model
together without a single patient record leaving any of them. Each hospital runs a small
team of agents (Schema · Planner · Trainer · Challenger); a Coordinator agent negotiates one
schema across mismatched columns and units and chooses how to aggregate each round. Updates
are masked so the Coordinator only ever sees their sum, and carry patient-level differential
privacy.

| | AUC on held-out patients |
|---|---|
| Hospital C alone | 0.763 |
| **PrivateLearn** (masked + DP, ε = 1.5/round) | **0.832** |
| all data pooled (illegal) | 0.836 |

A gradient-inversion attack recovers a real patient from an unprotected update and fails on
what the Coordinator actually receives. All three hospitals end with the byte-identical model.

```bash
cd privatelearn
python demo.py      # rebuild everything from the raw data and replay it in the browser
```

→ **[privatelearn/README.md](privatelearn/README.md)** — how it works, results, tests ·
[deck and talk track](privatelearn/docs/deck.md) ·
[contracts](privatelearn/docs/CONTRACTS.md) · [execution plan](docs/EXECUTION-PLAN.md)

## Collaboration

Built by multiple contributors, each with their own AI agent (Claude, DeepSeek), coordinating
only through GitHub: issues are the claim board, every change is a reviewed PR, and contracts
keep parallel work from colliding. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[AGENTS.md](AGENTS.md).
