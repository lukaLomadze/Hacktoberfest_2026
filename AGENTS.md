# AGENTS.md — Shared rules for all contributors and AI agents

This repo is built by several people, each possibly using a different AI agent (Claude, DeepSeek, Copilot, Gemini, GPT, Cursor, or none). Agents cannot talk to each other. **GitHub is the shared memory.** Every agent, whatever the model, follows this file.

The roster of contributors, their agents and their areas is in `TASKS.md`.

## Project
**PrivateLearn** (`privatelearn/`): agentic federated learning — three simulated hospitals
train one logistic-regression model with no patient row leaving a hospital (pairwise
masking + differential privacy). Python 3.9+ and numpy only; the replay player is vanilla JS.
Run everything with `cd privatelearn && python demo.py`. The plan is `docs/EXECUTION-PLAN.md`;
the contracts every component must follow are `privatelearn/docs/CONTRACTS.md`.

## Start of every session
1. `git checkout main && git pull`
2. Read `CONTRIBUTING.md`, `privatelearn/docs/CONTRACTS.md` and the open GitHub issues
   (the Issues board is the live task list; assigning yourself is the claim).
3. Pick only a task assigned to your contributor. If none is, ask your human to assign one.
4. `git checkout -b <github-username>/<short-task-name>` (example: `lukas/event-page`)

## Rules
- **Never push directly to `main`.** Open a PR; a *different* contributor reviews it.
- Stay inside the files your task owns (see `TASKS.md` → Ownership). If you need a change in someone else's area, open an issue instead of editing it.
- Shared files (`AGENTS.md`, `CONTRIBUTING.md`, `.github/`, `privatelearn/docs/CONTRACTS.md`) have a single owner listed in `TASKS.md`.
- Follow `privatelearn/docs/CONTRACTS.md` for message, trace and data shapes. To change a contract, update the doc in its own small PR and mention it in the relevant issue.
- Small PRs: one task, a few hours of work, no unrelated refactors or reformatting.
- Run format, lint and tests before pushing. Never commit secrets, `.env` files or build output.
- Commit messages: `type: short summary` (`feat`, `fix`, `docs`, `chore`, `refactor`, `test`).
- Mention the agent in the PR description (for example "written with DeepSeek") so reviewers know what to double-check.
- Hacktoberfest quality: no spammy, empty or trivial PRs.

## Handoff
When you finish, the PR description must say: what changed, how to test it, and anything other contributors must know. Put `Closes #<issue>` in the PR so the board updates when it merges.

## Style
- Python: stdlib + numpy only, `from __future__ import annotations`, snake_case, docstrings
  that say *why*. Tests are stdlib `unittest`, one directory per lane, run per directory.
- Every message and file shape follows `privatelearn/docs/CONTRACTS.md`; change it in its own PR.
- Web: vanilla JS, no framework; colours only via the CSS tokens in `privatelearn/docs/DESIGN.md`.
- Numbers in docs and the deck come from `trace.json`, never typed by hand from memory.
- CI (`.github/workflows/ci.yml`) runs every test, validates both traces and fails if the
  committed data or `trace.json` no longer match a fresh run.
