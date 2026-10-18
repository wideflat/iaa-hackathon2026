# Roadmap to the Part 1 deadline (2026-10-23 16:00 UTC) and beyond

## Status at 2026-10-18 05:30 UTC
Best submission: `fs_gbm_v2_trend_comp` public 0.12455 (rank 5; top 0.12815). Model family at its ceiling on local validation (see `experiments/OVERNIGHT_2026-10-18.md`): hyperparameters, features, GLM, seeds and calibration all within noise. Remaining upside is mostly level/trend choices that only the leaderboard can check, plus noise. Done: items 1 (level, via trend + completeness), 2 (GLM), 4 (tuning) below. Next: per-peril trend (shrunk to line), optional single-line level probes, and the judging deliverables.

## Modelling (in priority order)
1. **Level calibration.** Probes suggest the hidden answer is ~16% above our predictions overall. Implement the model-based causes first: frequency completeness adjustment for 2024 exposure by peril (motor bodily injury mainly) and an explicit cost-trend step; then per-line level probes (1-2 submissions per line) only to confirm.
2. **Fix the GLM benchmark**: winsorise heavy-tailed inputs, stronger regularisation, cap predictions; must be a sane actuarial baseline (organisers' own basic GLM scores 0.1056 publicly).
3. **Blend** the frequency x severity GBM, the direct Tweedie GBM and the GLM; re-test on folds; keep per-line rebalancing.
4. **Tune** boosting rounds, learning rate, leaves and cap quantile on folds A and B (not on the leaderboard); ablate suspected-noise columns and `written_premium`/`discount_applied`.
5. **Diagnostics**: calibration by peril and segment, lift/Gini by line, bodily-injury handling, stability across seeds.

## Submissions
Up to 5 per day (rules). Select exactly one Final Submission before the deadline; record every submission in `experiments/log.jsonl` with its CV numbers.

## Judging deliverables (due at the end of Part 2, covering both parts)
- Video demo (<= 5 minutes graded) and written report (<= 5 pages graded); outline: data quality findings, validation design, model, AI-assisted workflow with verification, guardrails, results.
- Repository: clean structure, tests passing, reproducible commands in `README.md`, decision record, AI usage log (extend `workflow/AI_USAGE.md` with the human review checkpoints).

## Part 2 readiness (new emerging-market dataset)
- Re-run the drift screens (`src/iaa/drift.py`) and the target/feature checks on the new data; list what changed (missing columns, new levels, distribution shift, different claim maturity).
- Keep the pipeline parameterised (paths, line list, folds) so it can be re-fit quickly; document remaining uncertainty explicitly.
