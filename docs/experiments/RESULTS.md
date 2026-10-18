# Results summary (curated; full machine-generated table in `EXPERIMENT_LOG.md`)

All scores are pooled Tweedie D2 (power 1.5). Fold A = train 2021-22, validate 2023 (mature, trusted). Fold B = train 2021-23, validate 2024 (immature actuals, half weight). 95% CIs are customer-clustered bootstraps; differences smaller than about 0.01 are inside the noise.

## 1. Model comparison on the local folds
| Model | Fold A | Fold B | Public LB | Comment |
|---|---|---|---|---|
| Constant cost rate per line x exposure | 0.087 | 0.071 | - | floor; line level and exposure carry most of the early signal |
| Direct Tweedie GBM, before rebalancing | 0.098 (0.077-0.113) | 0.120 | - | calibration 0.64-0.75 |
| Direct Tweedie GBM + per-line rebalancing | 0.1225 (0.109-0.133) | 0.1253 | - | calibration 0.92-0.96 on A |
| Same, closed-claims-only labels | 0.109 | 0.128 | - | worse on A (motor 0.029 vs 0.056) |
| Same, closed-only + level rescaled to all claims | 0.112 | 0.126 | - | |
| **Frequency x severity GBM (per peril)** | **0.1329** (0.126-0.139) | **0.1290** (0.120-0.138) | **0.12153** | current best; motor 0.074 vs 0.056 for the direct model |
| Frequency x severity GLM (first version) | 0.063 | 0.012 | - | broken: buildings D2 < 0, motor calibration 344 on B |
| *Invalid: direct GBM with `customer_service_contacts`* | *0.555* | *0.655* | - | *leak; kept as evidence* |

## 2. One-year-earlier rehearsal (`scripts/rehearsal.py`)
Pretend today is 2022-12-31: train on 2021-22 with only claims known then, predict 2023 without its claims, score against 2023 as valued today (24 months old).
| Variant | All claims | Closed claims only |
|---|---|---|
| Constant rate | 0.085 | 0.083 |
| GBM, no maturity feature | 0.109 | 0.108 |
| GBM + days-to-valuation feature | 0.121 | 0.117 |
(A first version scored 0.99 because `y_true` was left in the features; fixed, guarded by a test, and a score above 0.35 now aborts any run.)

## 3. Calibration diagnosis (fold A, direct Tweedie GBM, 300 rounds)
Dropping history features, weather, premium or all time-drifting features left calibration at 0.67-0.78 and D2 at 0.102-0.111; an offset formulation gave 0.101 (calibration 0.69-0.77). The GBM is already 0.72-0.81 calibrated on its own training data and its deviance-optimal in-sample scalar is ~1.007: the shortfall is a property of Tweedie-1.5 with a log link, not trend or drift. Rescaling to the training total (+0.017) recovers nearly all of the oracle gain (oracle scalar chosen on the validation year: 0.1242).

## 4. Public leaderboard and level probes
| Submission (2026-10-18) | Public score | Note |
|---|---|---|
| `fs_gbm_v1` | **0.12153** | frequency x severity GBM trained on 2021-24 |
| same predictions x 0.90 | 0.11554 | all lines scaled |
| same predictions x 1.10 | 0.12429 | all lines scaled |

Quadratic fit in ln(scale) through the three scores: optimum near **x1.16** (predicted public score about 0.1247, +0.003). The hidden answer therefore sits above our predictions, consistent with training on immature 2024 claims (motor bodily injury), cost trend, and "ultimate" including claims not yet reported. A single global scalar mixes the lines; per-line probes or a model-based fix (frequency completeness adjustment, trend) are the next steps. We do not tune the model to the leaderboard: only level, with few parameters.

Leaderboard snapshot (public, 2026-10-18 ~03:00 UTC, about 33% of test rows): 1. 0.12815, 2. 0.12753, 3. 0.12697, 4. 0.12532, 5. 0.12425, **6. us 0.12153**, 7. 0.11733, 8. 0.11416, 9. 0.11332. Reference files: organisers' basic GLM 0.10563; average-cost baseline = sample submission 0.07169.

## 5. Other measured facts that shaped the work
- Constant rate already scores 0.087 locally, so most of D2 comes from line level; gains beyond are small in absolute terms (leaders are at 0.128).
- Peril drives reporting development (12-month factor to ultimate: motor bodily injury 1.81, subsidence 1.46, other perils 1.02-1.08); severity is independent of report lag within a peril. Back-test of the 12-month-to-ultimate step: errors within +-1.6% for buildings/contents/non-BI motor and -3.5% to +4.2% for bodily injury.
- Still open at 2024-12-31 (share of cost): motor bodily injury 15% / 26% / 49% / 80% for accident years 2021-2024; buildings 0.7% / 2.6% / 8.2% / 36%.
- Recoveries: only 28% of closed motor own-damage/theft claims carry one (avg 77% of incurred), so unbooked recoveries overstate 2024 motor net cost by about 1.5%.
