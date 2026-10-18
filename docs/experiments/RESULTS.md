# Results summary (curated; full machine-generated table in `EXPERIMENT_LOG.md`)

All scores are pooled Tweedie D2 (power 1.5). Fold A = train 2021-22, validate 2023 (mature, trusted). Fold B = train 2021-23, validate 2024 (immature actuals, half weight). 95% CIs are customer-clustered bootstraps; differences smaller than about 0.01 are inside the noise.

## 1. Model comparison on the local folds
Fold B for models with a completeness offset is scored in *observed space* (prediction scaled by the expected reported share), so it is comparable with the immature actuals; older rows scored raw are marked (raw). "cap" = outlier-robust score with row cost capped at 250k.
| Model | Fold A | Fold B | Public LB | Comment |
|---|---|---|---|---|
| Constant cost rate per line x exposure | 0.087 | 0.071 (raw) | - | floor; line level and exposure carry most of the early signal |
| Direct Tweedie GBM, before rebalancing | 0.098 | 0.120 (raw) | - | calibration 0.64-0.75 |
| Direct Tweedie GBM + per-line rebalancing | 0.1225 | 0.1253 (raw) | - | |
| Same, closed-claims-only labels (+ level to all claims) | 0.109 (0.112) | 0.128 (0.126) | - | worse on A |
| Frequency x severity GBM v1 (per peril) | 0.1329 | 0.1290 (raw) | **0.12153** | first submission |
| + severity trend only | 0.1318 | 0.1265 (raw) | - | motor calibration on B 1.33 |
| **+ severity trend + completeness offset (v2)** | 0.1320 (cap 0.1315) | 0.1320 (cap 0.1330) | **0.12455** | current best submission; paired vs v1 on A: -0.0009 (-0.0015..-0.0003), tiny |
| v2 + cross-line customer / address history / lagged weather | 0.1326 | - | - | paired +0.0006 (ns): no gain |
| v2 + row timing features (period_days, start/end month) | 0.1328 | - | - | no gain; excluded by default |
| v2 with num_leaves 4 (350/300 rounds) | **0.1339** (cap 0.1323) | 0.1322 (cap 0.1331) | - | paired vs v2: +0.0011 (P=0.95) on A, +0.0001 on B; simplest, weakly best |
| Frequency x severity GLM (stabilised, alpha 1e-3/1e-2) | 0.1411 (cap 0.1316) | 0.1278 (cap 0.1292) | 0.12133 | GLM edge on A was one claim (see section 6) |
| GLM + quantile bands | 0.1390 | - | - | no gain |
| First GLM version (unstabilised) | 0.063 | 0.012 | - | broken: buildings D2 < 0 |
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

## 4. Public leaderboard, level probes and submissions (2026-10-18)
| # | Submission | Public score | Note |
|---|---|---|---|
| 1 | `fs_gbm_v1` | 0.12153 | frequency x severity GBM trained on 2021-24 |
| 2 | v1 x 0.90 (all lines) | 0.11554 | level probe |
| 3 | v1 x 1.10 (all lines) | 0.12429 | level probe |
| 4 | `fs_gbm_v2_trend_comp` | **0.12455** | v1 + severity trend + completeness: +0.0030 vs v1, as the x1.16 probe predicted (about 0.1247); rank 5 |
| 5 | `fs_glm_v1` | 0.12133 | GLM; its level (mean 376 vs 359) accounts for at most ~0.001 of the 0.0032 gap |

Quadratic fit in ln(scale) through probes 1-3: optimum near x1.16. v2 reached the same level through model-based changes (motor +23%, buildings +19%, contents +11% vs v1) rather than hand scaling. Five submissions were used on 2026-10-18 (UTC). The rules text then said 5 per day and the platform setting was 10; we stayed at 5 until the host confirmed the limit is 10 per day and updated the rules.
Leaderboard snapshot at ~03:00 UTC (about 33% of test rows): 0.12815, 0.12753, 0.12697, 0.12532, 0.12425, then v1 0.12153 (later v2 0.12455: 5th). Reference files: organisers' basic GLM 0.10563; average-cost baseline 0.07169.

## 5. Other measured facts that shaped the work
- Constant rate already scores 0.087 locally, so most of D2 comes from line level; gains beyond are small in absolute terms (leaders are at 0.128).
- Peril drives reporting development (12-month factor to ultimate: motor bodily injury 1.81, subsidence 1.46, other perils 1.02-1.08); severity is independent of report lag within a peril. Back-test of the 12-month-to-ultimate step: errors within +-1.6% for buildings/contents/non-BI motor and -3.5% to +4.2% for bodily injury.
- Still open at 2024-12-31 (share of cost): motor bodily injury 15% / 26% / 49% / 80% for accident years 2021-2024; buildings 0.7% / 2.6% / 8.2% / 36%.
- Recoveries: only 28% of closed motor own-damage/theft claims carry one (avg 77% of incurred), so unbooked recoveries overstate 2024 motor net cost by about 1.5%.

## 6. Model choice: GLM vs GBM, and why raw fold A misled us
- Raw fold A favoured the GLM (0.1411 vs 0.1320), driven by motor (+0.018). Paired bootstrap on the same rows: +0.0071 overall (CI -0.0015..+0.0207).
- Fold A's validation year contains one open 11.9M motor bodily-injury claim. Capping row cost at 250k: GLM 0.1316 vs GBM 0.1315, difference +0.0001; the motor edge shrinks from +0.0184 to +0.0037.
- Fold B (2024): GBM better, -0.0042 raw and -0.0038 capped (both significant); buildings and contents favour the GBM on both folds.
- The leaderboard (2025) agrees: GBM 0.12455 vs GLM 0.12133. Decision: GBM is the main model.

## 7. Overnight screens (helper sessions and local runs, 2026-10-18)
| Study | Result (fold A, base 0.1320) | Read |
|---|---|---|
| GBM hyperparameters (helper 1) | fastlr 0.1328, slowlr 0.1320, leaves7 0.1336, leaves31 0.1309, minleaf100 0.1324, minleaf800 0.1325, cap99 0.1328, cap9999 0.1310 | plateau (all within CI ~0.014); smaller trees marginally better; leaves 4 confirmed (above) |
| Feature ablations (helper 2) | keep suspected-noise columns 0.1096 (-0.022); drop premium/discount 0.1311; drop claim history 0.1319; drop weather 0.1312; drop customer 0.1328; drop service/channel 0.1318 | only the noise-column exclusion matters |
| GLM regularisation (helper 3, base 0.1399) | alpha-freq 3e-4 0.1421, 3e-3 0.1373, 1e-2 0.1322; alpha-sev 3e-3 0.1392, 3e-2 0.1408, 1e-1 0.1417 | plateau |
| New features / timing | GBM 0.1326 / 0.1328; GLM 0.1399 / 0.1399 | no gain |
Helpers 1-3 ran in their own containers (no commits, no submissions); each GLM run needs ~5.5 GB RAM and two GBM runs together exceed the 15 GB of this container (three OOM kills were handled by re-running).

## 8. Tail audit follow-through (2026-10-18)
| Run | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| fs_gbm_l4 (reference) | 0.1339 [0.1323] | 0.1322 [0.1331] | |
| fs_gbm_l4_peril (per-peril rebalancing) | 0.1338 [0.1321] | 0.1321 [0.1330] | neutral; rejected, keep per-line |
| fs_gbm_l4_out5 (winsorise claims above 5x cap in excess load and level) | 0.1339 [0.1323] | 0.1323 [0.1333] | paired vs reference: A 0.0000, B +0.0002 (CI excludes 0, motor +0.0004); adopted for robustness |

The open 11.9M bodily-injury claim inflated the BI excess load (0.083 vs 0.008) and the motor level by about 2.7%; winsorising lowers the fitted 2025 total by 1.2% relative to the plain leaves-4 file (rank correlation 0.9999). Candidate file: `submissions/fs_gbm_v4_l4_out5.csv` (not yet submitted). Full audit: `docs/data/TAIL_AUDIT.md` (generated by `scripts/tail_audit.py`).

## 9. Neural net and blend (2026-10-18)
| Model | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Public LB |
|---|---|---|---|
| GBM leaves 4 + out5 (`fs_gbm_l4_out5`) | 0.1339 [0.1323] | 0.1323 [0.1333] | not submitted |
| NN v1, CANN-style, 1 seed (`fs_nn_v1`, motor 0.0848 / 0.0830, buildings 0.1061 / 0.1098, contents 0.0837 / 0.0832) | 0.1365 [0.1319] | 0.1296 [0.1309] | not submitted alone |
| 50/50 blend (OOF) | 0.1388 [0.1356] | 0.1338 [0.1349] | |
| Submitted `blend_gbm_nn_v1.csv` (= 0.5 `fs_gbm_v4_l4_out5` + 0.5 `fs_nn_v1`, both fitted on 2021-24) | | | **0.12587** (v2 0.12455, v1 0.12153, GLM 0.12133) |

Paired bootstrap, blend minus GBM: fold A +0.0049 (95% CI +0.0024..+0.0082), fold B +0.0014 (CI +0.0003..+0.0025). Blend weights 0.25 / 0.75 were also positive on both folds but 0.5 was the best pooled; no per-line weights used. The public gain over v2 (+0.0013) has the same sign as the local gain over the GBM; the public score is one 2025-type sample and is not used to tune weights (D20/D21). Reading from the Kaggle submissions page (screenshot supplied by the project lead; `scripts/check_submissions.py` returned an authentication error at the time).

## 10. Other algorithms and blend additions (2026-10-18)
Reference blend R = equal-weight GBM (`fs_gbm_l4_out5`) + NN (`fs_nn_v1`): fold A 0.1388 [cap250k 0.1356], fold B 0.1338 [0.1349]. Candidates are added with equal weights (`scripts/blend_multi.py`, paired bootstrap on the same rows, customer-clustered).

| Model (single) | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| XGBoost freq x sev, depth 3 (`fs_xgb_d3`, helper session) | 0.1322 [0.1317] | 0.1321 [0.1332] | about LightGBM; 17 min for both folds |
| CatBoost freq x sev, depth 4 (`fs_cat_d4`, helper session) | 0.1165 [0.1235] | 0.1201 [0.1208] | worse; motor calibration 1.24 on fold B; 27 min per fold |
| Direct Tweedie NN (`nn_tweedie_v1`) | 0.1336 [0.1297] | 0.1215 [0.1228] | no trend / completeness handling; fold B motor calibration 1.23 |

| Blend | A: cand - R (95% CI) | B: cand - R (95% CI) | Verdict |
|---|---|---|---|
| R + direct Tweedie NN | +0.0007 (-0.0000..+0.0015) | -0.0016 (-0.0024..-0.0008) | reject (worse on B) |
| R + XGBoost | -0.0009 (-0.0027..+0.0004) | +0.0003 (-0.0001..+0.0006) | no gain |
| R + CatBoost | -0.0027 (-0.0074..+0.0004) | -0.0011 (-0.0020..-0.0002) | reject |
| R + all three | -0.0009 (-0.0046..+0.0015) | -0.0006 (-0.0012..-0.0000) | reject |

Rule (set before looking, as in DECISIONS D23): a model joins the blend only if the paired difference is positive on both folds. None does. The direct net is structurally disadvantaged on fold B because its immature-year training rows are not completeness-adjusted; the GLM (tuned, fold A 0.1411) helped on fold A but not fold B (3-way 0.1331 vs 0.1338), so it is not adopted either.

## 11. Wider, higher-dropout NN (2026-10-18, 13:15 EDT)
`fs_nn_wide`: hidden 384,192, dropout 0.2, 1 seed. Single: fold A 0.1387 [cap250k 0.1310], fold B 0.1305 [0.1320] (NN v1: 0.1365 [0.1319] / 0.1296 [0.1309]; GBM: 0.1339 [0.1323] / 0.1323 [0.1333]). Motor stronger (A 0.0900 / B 0.0840), buildings A weaker (0.1043).

| Blend (equal weights) | A: minus R (95% CI) | B: minus R (95% CI) | Read |
|---|---|---|---|
| GBM + NN wide (replace NN v1) | +0.0013 (-0.0012..+0.0051) | +0.0004 (-0.0003..+0.0009) | positive on both folds, not significant; cap250k scores about equal (A 0.1351 vs 0.1356, B 0.1354 vs 0.1349) |
| GBM + NN v1 + NN wide | +0.0015 (-0.0008..+0.0050) | -0.0000 (-0.0007..+0.0004) | no gain |

R = GBM + NN v1 (A 0.1388 [0.1356], B 0.1338 [0.1349]). The wide net is within noise of NN v1 in the blend: not worth a new submission by itself. The 3-seed NN (variance reduction, running) is the next test; a decision on a final NN configuration waits for it.

## 12. 3-seed NN and risk-dependent tail load (2026-10-18, 15:15 EDT)
| Run | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| `fs_nn_s3` (NN v1, 3 seeds averaged per stage) | 0.1406 [0.1330] | 0.1313 [0.1326] | best single model so far; motor 0.0927 / 0.0848 |
| GBM + `fs_nn_s3` (equal weights), minus R (GBM + NN v1) | +0.0018 (-0.0005..+0.0055) | +0.0006 (-0.0000..+0.0012) | positive on both folds, small; cap250k +0.0000 (A: 0.1356) / +0.0006 (B: 0.1355) |
| `fs_gbm_tail05` (risk-dependent excess load, lambda 0.5; helper 2) | 0.1338 [0.1323] | 0.1323 [0.1333] | no change vs GBM 0.1339 / 0.1323 |
| `fs_gbm_tail10` (lambda 1.0; helper 2) | 0.1337 [0.1322] | 0.1323 [0.1332] | no change |

Tail model: fitted for 4 of 5 motor perils on fold A training (theft skipped, too few large claims); tail ratio median about 0.6-0.75, max 2.9-4.2, but the base excess loads are tiny (0.001-0.009), so the load barely moves predictions. Rejected (D25 below): not worth its code.

## 13. Entity embeddings (customer_id, address_id as learned embeddings; helper 3, 15:14 EDT)
| Run | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| `fs_nn_entity` (1 seed) | 0.1338 [0.1299] | 0.1281 [0.1290] | worse than the same NN without entities (0.1365 [0.1319] / 0.1296 [0.1309]) |
| `fs_nn_entity_d3` (dropout 0.3) | 0.1367 [0.1304] | 0.1292 [0.1304] | no gain either |
No paired bootstrap run: both single-model scores are below the no-entity net on both folds and both metrics, so the idea is rejected (D26). Reading: the history features already carry the customer signal; unseen-customer and rare-customer rows get no benefit, and the extra parameters overfit.

## 14. Recency and domain-shift weights on the GBM (helper 1, 17:45 EDT)
LightGBM leaves 4 + outlier 5; reference A 0.1339 [cap250k 0.1323], B 0.1323 [0.1333].
| Run | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| `fs_gbm_rec03` (recency rho 0.3) | 0.1344 [0.1318] | 0.1322 [0.1332] | within noise |
| `fs_gbm_rec07` (rho 0.7) | 0.1354 [0.1310] | 0.1312 [0.1323] | A up, B down, capped scores down on both |
| `fs_gbm_adv05` (domain-shift weights, strength 0.5) | 0.1339 [0.1323] | 0.1322 [0.1332] | weights near uniform; no change |
| `fs_gbm_adv10` (strength 1.0) | 0.1345 [0.1318] | 0.1320 [0.1331] | no change |
Rejected (D27): none is positive on both folds on both metrics. The train and 2025-type rows are hard to tell apart (classifier weights near uniform), so there is little covariate shift for the weights to correct; recency does not help because the severity trend and completeness handling already model the drift. Predictions were not saved (helper stopped after the logs).
