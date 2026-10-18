# Overnight session, 2026-10-18 (about 03:20-05:30 UTC)

Written for the morning hand-over. Instruction received: explore as much as possible with the remaining submissions, refine the single model (no blending), iterate fast, parallelise where possible.

## Outcome in one paragraph
Public score went from 0.12153 (v1) to **0.12455 (v2, rank 5)** by adding a severity trend and a frequency completeness offset. Everything else tried (GLM, new features, row timing, hyperparameters, banding, smaller trees, seed averaging, calibration curve) moved fold-A/B scores by less than the noise, so the model family is at its ceiling on this data. Five submissions were used on 2026-10-18 UTC (the written limit); the sixth was deliberately **not** used because the only remaining candidate (v3, leaves 4) is a near copy of v2 (rank correlation 0.992, mean +0.8%) and would add no information.

## What was run
| Item | Result |
|---|---|
| v2: trend + completeness | Public 0.12455 (+0.0030 vs v1; the x1.16 probe predicted ~0.1247). Fold A 0.1320, fold B 0.1320 (observed space) |
| Stabilised GLM | Fold A 0.1411, B 0.1278, public 0.12133. Paired + capped analysis: fold A edge was one 11.9M claim; GBM stays |
| New features (customer cross-line history, address history, lagged weather) | GBM 0.1326 (paired +0.0006, ns), GLM 0.1399 vs 0.1411: no gain |
| Row timing features | 0.1328 / 0.1399: no gain; excluded |
| GBM hyperparameters (helper 1, 9 runs) | plateau; smaller trees marginally better |
| Feature ablations (helper 2, 7 runs) | only excluding suspected-noise columns matters (keeping them costs -0.022) |
| GLM regularisation (helper 3, 7 runs) | plateau (0.1322-0.1421, base 0.1399) |
| GLM quantile bands | 0.1390: no gain |
| GBM leaves 4 (v3 candidate) | A 0.1339 (cap 0.1323), B 0.1322 (cap 0.1331); paired vs v2 +0.0011 (P=0.95) on A, +0.0001 on B |
| Seed averaging (3 seeds, leaves 4) | A 0.1338: no gain |
| Calibration by predicted decile / power transform | no systematic compression; best power ~1.0: no gain |

## Method improvements made along the way
- Paired cluster bootstrap on the same rows (`scripts/compare_oof.py`): ~10x tighter than comparing unpaired CIs.
- Outlier-robust companion score (row cost capped at 250k) in every run.
- Fold B scored in observed space for models with a completeness offset.
- Frame cache (`iaa.cache`) so experiments start in seconds; flags for parallel-friendly runs.
- Leak alarm (pooled D2 > 0.35 aborts a run).

## Parallelism: what worked and what did not
- Helper sessions in separate containers (3 sessions, 23 runs) worked and reproduced our own baselines exactly (helper base 0.1320 and GLM base 0.1399 equal the local runs), which is the check that their numbers can be trusted. They made no commits and no submissions.
- One container: 4 cores and 15 GB. Two GBM runs or two GLM runs at once run out of memory (three OOM kills); heavy runs now go one at a time.

## Ready but not submitted
`submissions/fs_gbm_v3_leaves4.csv` (validated, dry-run OK). Recommendation: keep v2 (0.12455) as the Final candidate; submit v3 only if you want its leaderboard reading.

## Open items for you
1. Host answer on the submission limit (written 5/day vs platform 10/day). Until then we stay at 5 per UTC day.
2. Choice of the Final Submission (we have not selected one).
3. Whether to remove the internal draft questions in `docs/methodology/TARGET_DEFINITION.md` section 7.
4. The default branch on GitHub is still `claude/brave-mayer-b3poow`; set it to `main` in repository settings.
