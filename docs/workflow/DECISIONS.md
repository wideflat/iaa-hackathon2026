# Decision record

Each decision: what, why (evidence), status. "Reviewed by" is the human who confirmed the direction.

| ID | Decision | Evidence / reason | Status | Reviewed by |
|---|---|---|---|---|
| D1 | Target = accident-period net cost: sum of `incurred - recoveries` over non-repudiated claims whose `loss_date` falls in the row's period | Task text and glossary; paid is already net of excess; recoveries only on closed motor claims | Adopted | Project lead |
| D2 | Validate with fixed forward-in-time folds (A: 2021-22 -> 2023; B: 2021-23 -> 2024), no grouping by policy | Test is a future year; 85% of test policies also appear in training | Adopted | Project lead |
| D3 | Exclude `customer_service_contacts` | Outcome-generated in training (corr 0.55), baseline-only in 2025; CV 0.55 -> 0.10 without it | Adopted; guard test + D2 alarm | Assistant, confirmed by lead |
| D4 | Recompute exposure from period dates | Supplied column corrupted on ~2-3% of training rows; test clean | Adopted | Assistant |
| D5 | Do not use `renewal_decision`; filter no rows by it | Not in test; rows later declined carry 24-52% of cost (98% have claims) | Adopted | Project lead |
| D6 | Rebalance each line's predictions to the training total | Tweedie 1.5 log-link GBM under-predicts totals in-sample by 20-30%; +0.017 D2 on fold A | Adopted | Assistant |
| D7 | Keep all claims (open and closed) as labels | Closed-only: fold A 0.109-0.112 vs 0.1225; bodily-injury claims stay open for years (15-80% of cost) so closed-only biases recent years. The lead preferred closed-only in principle; validation did not support it | Adopted on validation; closed-only kept as an option (`--closed-only`) | Project lead (informed) |
| D8 | Do not build the target from link ratios; handle immaturity only as a completeness adjustment on 2024 frequency by peril | Task is pricing; severity independent of report lag; maturity-feature and closed-only tests gave no clear gain; development is secondary | Adopted; adjustment not yet implemented | Project lead |
| D9 | Primary model family = frequency x severity per peril (GBM), with GLM benchmark and direct Tweedie as comparison | Fold A 0.1329 vs 0.1225; standard actuarial structure | Adopted | Project lead |
| D10 | Cap severity at the 99.9th percentile per peril and load the excess back | One 11.9M claim vs typical 10-20k; stable development factors under capping | Adopted | Assistant |
| D11 | Submit only via the guarded script (`--confirm`), one validated file at a time; use level probes sparingly | Submissions are limited (5/day) and public | Adopted | Project lead |
| D12 | Work on `dev`; `main` holds stable states; default branch to be set to `main` by the owner | Repo workflow request | Adopted | Project lead |
| D13 | Treat the leaderboard as a level check only; choose models on local validation | Public score is a 33% noisy subset; CV and public ordering agree so far | Adopted | Project lead (principle set on day 1) |

| D14 | Adopt severity trend (one pooled log-linear trend per line, projected to the predicted year) and a frequency completeness offset for immature accident years | Probes showed hidden level ~x1.16 above v1; severity trend is +5-8%/yr in most perils, frequency has none; v2 reached 0.12455 (+0.0030) as predicted; completeness fixes fold B motor calibration 1.33 -> 1.08 | Adopted | Project lead (plan approved) |
| D15 | GBM, not GLM, is the main model | Paired + robust comparison and the leaderboard (section 6 of RESULTS) | Adopted | Assistant, shared with lead |
| D16 | Evaluate with paired bootstrap and a robust (capped) companion score | One outlier claim misled raw fold A | Adopted | Assistant |
| D17 | Exclude period-length / start-month / end-month features | No gain (+0.0002 / 0.000); avoids end-of-period information | Adopted | Assistant |
| D18 | Use helper sessions in separate containers for independent screens; they never commit, push or submit | CPU/RAM limits of one container (4 cores, 15 GB); lead asked for parallel work | Adopted | Project lead |
| D19 | Submission limit: 10 per UTC day | Rules text said 5, platform 10; the host confirmed 10 and updated the rules (2026-10-18). Until the answer we stayed at 5 | Resolved | Project lead (asked the host) |

| D20 | **Final Submission rule:** choose using local validation and the public leaderboard together. Candidates must be fully validated (tests passing, no leak alarm). Among candidates not significantly worse than the best on local validation (paired bootstrap on the same rows, raw and capped-at-250k scores, folds A and B with B at half weight), pick the one with the highest public score; if the public scores are within about 0.001, pick the simplest model. Written before looking at the final candidates. | Public score is a noisy 33% subset; local validation alone cannot resolve small differences; both views together hedge | Adopted | Project lead |
| D21 | **No more leaderboard probing** (no scaled or otherwise artificial submissions). Submissions are only for real, validated candidates. | Project lead's instruction on 2026-10-18; the level question is mostly settled (three probes, then model-based trend and completeness) | Adopted | Project lead |
| D22 | Run a tail and large-claim audit (cap choice, excess load, the open 11.9M claim) | Research report: tail handling is one of the few levers likely to exceed noise; one claim distorted fold A and enters the bodily-injury excess load | Done: single claims above 5x the peril cap are winsorised in the excess load and level targets (`outlier_mult=5`); neutral to slightly positive in CV (fold B motor +0.0004, paired CI excludes 0), lowers the 2025 total by 1.2%. Per-peril rebalancing tested and rejected (neutral). Candidate v4 = leaves 4 + `outlier_mult=5` | Project lead |
| D23 | Add a CANN-style neural-net frequency x severity model and blend it 50/50 with the GBM (candidate, first submitted blend) | Folds: NN v1 A 0.1365 / B 0.1296; blend A 0.1388 / B 0.1338 vs GBM 0.1339 / 0.1323; paired CI excludes 0 on both folds (+0.0049, +0.0014). Public: blend 0.12587 vs v2 0.12455 | Adopted as current best candidate; weights, seeds and further algorithms (XGBoost, CatBoost) under test | Project lead |
| D24 | Do not add XGBoost, CatBoost, the direct Tweedie NN or the GLM to the GBM + NN blend | Paired bootstrap vs the reference blend: none positive on both folds (RESULTS section 10) | Rejected; next: NN variants (wide, 3 seeds) | Project lead |
| D25 | Reject the risk-dependent tail (excess) load | No change at lambda 0.5 or 1.0 on both folds (RESULTS section 12): the excess loads are 0.1-0.9% of cost, so scaling them by risk cannot matter | Rejected; code kept behind `tail_lambda` (default off) | Project lead |
| D26 | Reject customer/address entity embeddings in the NN | Single-model scores below the no-entity net on both folds and metrics (RESULTS section 13) | Rejected; code kept behind `--entity` | Project lead |
| D27 | Reject recency weighting and domain-shift weighting | RESULTS section 14: no run positive on both folds and both metrics; adversarial weights near uniform | Rejected; code kept behind `--recency` / `--adv-weights` (default off) | Project lead |
| D28 | Adopt GBM + TabM-style NN (equal weights) as the next candidate | RESULTS section 15: above the submitted GBM + NN v1 blend on both folds and both metrics; B interval excludes 0 | Final TabM fit on 2021-24 running; submit after checks | Project lead |

## Open decisions
- Names of the team's independent reviewer and approver for the governance log (placeholders in `workflow/AI_USAGE.md`): to be filled in by the project lead.
- Whether open claims' case estimates are trusted as final (IBNER not measurable).
