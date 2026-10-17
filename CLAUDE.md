# AGENTS.md — Shared rules for all contributors and AI agents

This repo is built by several people, each possibly using a different AI agent (Claude, DeepSeek, Copilot, Gemini, GPT, Cursor, or none). Agents cannot talk to each other. **GitHub is the shared memory.** Every agent, whatever the model, follows this file.

The roster of contributors, their agents and their areas is in `TASKS.md`.

## Project
Hacktoberfest Georgia. <!-- TODO: one-paragraph description, stack, how to run -->

## Start of every session
1. `git checkout main && git pull`
2. Read `TASKS.md`, `docs/CONTRACTS.md` and the open GitHub issues.
3. Pick only a task assigned to your contributor. If none is, ask your human to assign one.
4. `git checkout -b <github-username>/<short-task-name>` (example: `lukas/event-page`)

## Rules
- **Never push directly to `main`.** Open a PR; a *different* contributor reviews it.
- Stay inside the files your task owns (see `TASKS.md` → Ownership). If you need a change in someone else's area, open an issue instead of editing it.
- Shared files (`package.json`, configs, router, global CSS, `docs/CONTRACTS.md`) have a single owner listed in `TASKS.md`.
- Follow `docs/CONTRACTS.md` for APIs, data shapes and props. To change a contract, update the doc in its own small PR and mention it in the relevant issue.
- Small PRs: one task, a few hours of work, no unrelated refactors or reformatting.
- Run format, lint and tests before pushing. Never commit secrets, `.env` files or build output.
- Commit messages: `type: short summary` (`feat`, `fix`, `docs`, `chore`, `refactor`, `test`).
- Mention the agent in the PR description (for example "written with DeepSeek") so reviewers know what to double-check.
- Hacktoberfest quality: no spammy, empty or trivial PRs.

## Handoff
When you finish, the PR description must say: what changed, how to test it, and anything other contributors must know. Update your task's status in `TASKS.md`.

## Style
<!-- TODO: language/framework, formatter, naming conventions -->
