# IAA AI-Assisted Hackathon 2026 - Part 1 (Kaggle, finished: private rank 3) and Part 2 (emerging market, closes 2026-10-30 23:59 EDT)

Part 2: emerging-market test rows only on Kaggle (`iaa-ai-assisted-hackathon-2026-part-2`); plan in `docs/part2/PART2_STRATEGY.md`, data differences in `docs/part2/DATA_REVIEW.md`. Switch the market with `IAA_MARKET=emerging IAA_DATA_DIR=data/part2/emerging`; Part 2 scripts are `scripts/p2_*.py`; fold E (train 2023 -> validate 2024) is the emerging validation.

Predict the expected ultimate claim cost per 2025 policy row (motor / buildings / contents); score = pooled Tweedie D2 (p=1.5); judged 40% performance, 25% AI use, 20% trust/validation, 15% communication. Index of everything: `docs/README.md`.

## Start of a session / after a compaction
Read `docs/workflow/HANDOFF.md` ("START HERE": current state, open decisions, next steps). After a compaction a hook prints a pointer, and `artifacts/session_snapshot.md` holds facts saved just before it (git state, running jobs, last experiments). The hooks are in `.claude/settings.json` -> `scripts/hooks/`.

## Standing rules (from the project lead)
- Report times in EDT (America/New_York), not UTC. Docs written before 2026-10-18 12:00 EDT use UTC and say so.
- Work on branch `dev`; `main` is stable. Commit with explicit paths; push to `origin dev`. No pull request unless asked.
- Do not post to the competition forum, and do not probe the leaderboard (D21). Submissions: 10 per day; final choice by local validation plus public score (D20).
- Keep the validation setup fixed (folds A: train 2021-22 -> 2023; B: train 2021-23 -> 2024). Judge changes by paired bootstrap on the same rows, on both folds, never by the public score alone.
- Never use as features: `customer_service_contacts`, target-like columns, noise/timing columns (see `src/iaa/features.py`). A D2 above 0.35 is a leak alarm.
- One heavy job at a time (15 GB RAM, 4 cores). Rebuild the frame cache once before parallel runs.
- Reviewer / approver names in `docs/workflow/AI_USAGE.md` are placeholders for the team; do not invent names.

## Record as you go (so nothing depends on memory or on the context window)
At every milestone (a result, a decision, a submission, a bug caught) do the matching item, then update the START HERE block of `docs/workflow/HANDOFF.md`:
- result of a run -> `experiments/log.jsonl` (done by `scripts/run_cv.py`) and a row in `docs/experiments/RESULTS.md`
- decision or rejected idea -> `docs/workflow/DECISIONS.md` (one row, with the evidence)
- narrative / bug caught by a check -> `docs/workflow/DEVELOPMENT_LOG.md`; AI errors caught -> `docs/workflow/AI_USAGE.md`
- submission made -> row in RESULTS.md with the public score once `scripts/check_submissions.py` shows it
The skill `record-progress` runs this sweep (hooks cannot call skills or the model, so run it yourself before long waits and before stopping).

## Long runs and sandbox restarts
The container has restarted at the start of several user turns (about every 15-40 minutes), killing background jobs. Start long work with `nohup bash scripts/queue_runs.sh > artifacts/queue.log 2>&1 &`: it is resumable (`run_cv.py --resume` keeps a checkpoint per fold and line; finished runs are skipped). At the start of a turn check `uptime` and `pgrep -af run_cv`, and relaunch the queue if it is empty.
