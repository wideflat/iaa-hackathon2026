---
name: record-progress
description: Use at a milestone, before a long wait, before stopping, or when context is large - sweeps the session for results, decisions, submissions and caught errors that are not yet written down, files each in its place under docs/ and experiments/, and refreshes the START HERE block of docs/workflow/HANDOFF.md so a compaction or a new session loses nothing.
---

# Record what this session produced

Hooks save facts (git state, jobs, last experiments) before a compaction; only you can save judgement. Ask each question in order, do not skip rows.

| # | Question | Where it goes |
|---|---|---|
| 1 | Did a run finish (CV folds, paired comparison, blend)? | `experiments/log.jsonl` (automatic for run_cv) + row in `docs/experiments/RESULTS.md` with fold A / B, cap250k, paired difference and CI |
| 2 | Did we decide, adopt or reject something? | `docs/workflow/DECISIONS.md` (id, decision, evidence, status) - negative results too |
| 3 | Did we submit? | RESULTS.md submissions table: file, what it is, local CV, public score (`python scripts/check_submissions.py`) |
| 4 | Did a check catch an AI/code error, a leak or a data problem? | `docs/workflow/AI_USAGE.md` ledger and `DEVELOPMENT_LOG.md` |
| 5 | Is a doc now stale (README table, ROADMAP next steps, VALIDATION/MODELLING text)? | edit it, and say what changed; do not silently overwrite old numbers |
| 6 | Is anything running or half done? | START HERE: command, expected finish, where the output goes |

Then rewrite the **START HERE** block at the top of `docs/workflow/HANDOFF.md` (<= 60 lines): current best model and scores, what is submitted, what is running, open decisions for the user, the next three steps, traps hit. Commit the docs with explicit paths and push `dev`.

Rules: numbers always carry their conditions (fold, model file, seeds); the public score is a claim until it is shown by the submissions list; never invent reviewer names or leaderboard facts.
