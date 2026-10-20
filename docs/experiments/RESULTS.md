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

## 15. TabM-style NN (k=4 BatchEnsemble members, 1 net per stage; 17:55 EDT)
| Run | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| `fs_nn_tabm4` | 0.1413 [0.1331] | 0.1328 [0.1338] | best single model: above the 3-seed net (0.1406 [0.1330] / 0.1313 [0.1326]) on both folds and metrics, at similar cost |
| GBM + TabM (equal weights) minus R (GBM + NN v1) | +0.0027 (-0.0003..+0.0074) | +0.0013 (+0.0007..+0.0020) | cap250k: 0.1360 vs 0.1356 (A), 0.1361 vs 0.1349 (B) |
| GBM + TabM minus GBM + NN 3 seeds | +0.0008 (-0.0002..+0.0020) | +0.0007 (+0.0001..+0.0015) | TabM better in the blend on both folds |
| GBM + TabM + NN 3 seeds minus R | +0.0033 (-0.0004..+0.0094) | +0.0009 (+0.0003..+0.0017) | no better than GBM + TabM on fold B |
Blend GBM + TabM: A 0.1414 [0.1360], B 0.1351 [0.1361]. Candidate for the next submission (D28).

Submitted 2026-10-18 about 18:05 EDT (user approved): `blend_gbm_tabm4_v2.csv` = 0.5 `fs_gbm_v4_l4_out5` + 0.5 `fs_nn_tabm4`. Public score: to be filled in from the Kaggle submissions page (the submissions API returned authentication errors this afternoon).

Component checks submitted 2026-10-18 about 18:15 EDT at the user's request (public scores to be filled in from the Kaggle page): `fs_nn_tabm4.csv` (TabM alone; CV A 0.1413 / B 0.1328) and `fs_gbm_v4_l4_out5.csv` (GBM alone; CV A 0.1339 / B 0.1323). Purpose: see whether the public ordering GBM < blend, TabM vs blend matches the local one. They are real models, not level probes, and are not used to tune weights (D20/D21).

## 16. Public scores of the 2026-10-18 evening submissions (read 18:20 EDT from the Kaggle REST list; times in the list are UTC)
| Submission | Public | Local CV (A / B, D2) |
|---|---|---|
| Blend GBM v4 + TabM (equal weights) | **0.12679** | 0.1414 / 0.1351 |
| Blend GBM v4 + NN v1 (equal weights) | 0.12587 | 0.1388 / 0.1338 |
| GBM v4 alone (leaves 4, outlier 5) | 0.12548 | 0.1339 / 0.1323 |
| TabM alone | 0.12346 | 0.1413 / 0.1328 |
| (earlier) v2 GBM 0.12455, v1 GBM 0.12153, GLM 0.12133 | | |
Reading: the public order of the two blends matches the local order (+0.0009 public, +0.0027 / +0.0013 local); the blend beats both of its components (GBM +0.0013, TabM +0.0033), which is the diversity effect we want. TabM alone is worse than the GBM publicly although it was better on fold A and similar on fold B: fold A single-model scores overstate the NN, and fold B (the one that looks like 2025: immature last year) is the better guide, where the NN is about equal to the GBM. Do not choose a NN-heavy blend from fold A alone. No weights were tuned on these public scores (D20/D21); they are the first public evidence that supports the equal-weight GBM + TabM blend as the current best (0.12679). The top public score seen earlier was 0.12815.

## 17. Per-line blend weights for GBM + TabM (18:25 EDT)
Weights on TabM per line chosen from the per-line single-model ordering (TabM better on motor on both folds; GBM better on contents on both; buildings close), not by maximising D2: motor 0.7, buildings 0.4, contents 0.3 (equal weights = 0.5 each). Checked neighbours (0.6/0.5/0.4 and 0.7/0.5/0.3) to see that the result does not hinge on the exact values.
| TabM weights (motor/buildings/contents) | Fold A vs equal (95% CI) | Fold B vs equal (95% CI) |
|---|---|---|
| 0.7 / 0.4 / 0.3 | +0.0021 (+0.0006..+0.0044); D2 0.1435, cap250k 0.1369 (equal 0.1414 / 0.1360) | +0.0004 (+0.0002..+0.0007); D2 0.1355, cap250k 0.1365 (equal 0.1351 / 0.1361) |
| 0.6 / 0.5 / 0.4 | +0.0011 (+0.0003..+0.0023) | +0.0002 (+0.0001..+0.0004) |
| 0.7 / 0.5 / 0.3 | +0.0019 (+0.0005..+0.0042) | +0.0004 (+0.0002..+0.0006) |
Small but positive on both folds in every variant, with intervals excluding zero. File: `submissions/blend_gbm_tabm4_perline_v3.csv` (made with `scripts/blend_submissions.py`, which reproduces the submitted equal-weight blend exactly). Not submitted yet; one submission left today.

## 18. TabM variants (18:30 EDT)
| Run | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| `fs_nn_tabm4` (k=4, hidden 192,96, dropout 0.1) | 0.1413 [0.1331] | 0.1328 [0.1338] | reference |
| `fs_nn_tabm4_wide` (k=4, hidden 384,192, dropout 0.2) | 0.1381 [0.1326] | 0.1318 [0.1330] | worse on both folds and metrics: wider is not better |
Helper runs `fs_nn_tabm8` (k=8) and `fs_nn_tabm4_s3` (3 seeds) are pending.

Update 18:40 EDT. `fs_nn_tabm4_small` (hidden 128,64): A 0.1424 [0.1340], B 0.1308 [0.1322]: mixed vs the default (A up, B down); adding it to GBM + TabM: A +0.0013 (-0.0001..+0.0037), B -0.0006 (-0.0012..-0.0002): rejected. The default TabM configuration (k=4, 192,96, dropout 0.1) stays.
Level check of the GBM + TabM blend (actual / predicted, fold A, fold B): motor 0.944 / 0.949, buildings 0.999 / 1.017, contents 1.072 / 0.991, overall 0.971 / 0.978. Motor is over-predicted by about 5% on both folds; correcting it with a scalar fitted on the other fold gains only +0.0001..+0.0002 D2, so no level correction is made (D2 is insensitive to a 5% level error in one line, and the 2025 level is not known from these folds).

## 19. TabM with 3 seeds (helper 3, finished 18:42 EDT; 30 min for both folds)
`fs_nn_tabm4_s3`: A 0.1429 [0.1347], B 0.1328 [0.1340] (1 seed: 0.1413 [0.1331] / 0.1328 [0.1338]). In the blend with the GBM: equal weights A +0.0005 (+0.0001..+0.0010), B -0.0001 (-0.0007..+0.0004) vs the 1-seed blend; per-line weights A +0.0003, B -0.0002 (both CIs include 0). Seed averaging does not change the blend; the single-seed TabM already fitted on 2021-24 is as good, so no 3-seed refit is made.

## 20. Per-line-weight blend submitted (19:56 EDT)
`blend_gbm_tabm4_perline_v3.csv` (TabM weight 0.7 motor / 0.4 buildings / 0.3 contents): public **0.12718**, vs 0.12679 for the equal-weight blend (+0.0004; the local gain was +0.0021 on fold A and +0.0004 on fold B). New best public score; the top public score seen earlier was 0.12815. This is the current final-submission candidate under D20 (local validation positive on both folds and the public score supports it).

## 21. Stacking per-line blend weights across folds (`scripts/stack_oof.py`, 00:00 EDT 2026-10-19)
Per-line non-negative weights (sum 1) fitted by maximum Tweedie likelihood on one fold's out-of-fold predictions and evaluated on the other.
| Components | fitted A -> eval B | fitted B -> eval A | vs hand-set per-line weights (0.7/0.4/0.3 on TabM; B 0.1355, A 0.1435) |
|---|---|---|---|
| GBM + TabM | 0.1353 [cap 0.1362] | 0.1439 [0.1371] | equal to within 0.0004: the hand weights are about the cross-fold optimum (fitted weights: motor TabM 0.75-1.0, buildings 0.26-0.33, contents 0.17-0.27) |
| + GLM (fs_glm_v3) | 0.1349 | 0.1440 | no gain (worse on B, +0.0005 on A) |
| + GLM + XGBoost + wide NN + small TabM | 0.1347 | 0.1440 | no gain; weights unstable between folds |
Conclusion (D29): no stacking layer or extra components beyond GBM + TabM with per-line weights. The remaining headroom is in better components (features, representations), not in combining.

## 22. Engineered features, GBM (20:18 EDT)
`--engineered` (src/iaa/engineered.py; label-free): premium per sum insured / vehicle value group, licence age, mileage per vehicle age, history frequency and severity, weather anomalies (vs region-year mean) and year-on-year changes, and the premium residual (log premium minus a cross-fitted regression on the other rating features = the underwriter's load beyond the rating factors).
`fs_gbm_eng` (LightGBM leaves 4, outlier 5): A 0.1340 [0.1325], B 0.1323 [0.1334] vs the same GBM without: 0.1339 [0.1323] / 0.1323 [0.1333]. No change: trees already find these interactions. Weak linear correlations of the new columns with cost (all below 0.06). The NN (TabM) with the same features and with piecewise-linear encodings of the numeric inputs is being tested next, since neural nets are weaker at ratios and thresholds.

`fs_nn_tabm4_eng` (TabM k=4 + the engineered features): A 0.1420 [0.1330], B 0.1313 [0.1326] vs plain TabM 0.1413 [0.1331] / 0.1328 [0.1338]: motor better (0.0985 vs 0.0954 on A), buildings worse. In the blend with the GBM vs GBM + plain TabM: A +0.0005 (-0.0006..+0.0018), B -0.0006 (-0.0014..+0.0001); with both TabMs: A +0.0015, B -0.0003. Not positive on both folds: rejected (D30). Engineered ratios, weather anomalies and the premium residual do not add signal beyond the raw columns for either model family.

## 23. TabM with piecewise-linear encoding of numeric inputs (helper 3, 20:25 EDT; 19 min for both folds)
`fs_nn_tabm4_ple` (8 quantile bins per numeric input, added to the standardised value): A 0.1430 [cap250k 0.1364], B 0.1340 [0.1351] vs plain TabM 0.1413 [0.1331] / 0.1328 [0.1338]: better on both folds and both metrics (cap +0.0033 / +0.0013). In the blend (GBM + NN, equal weights) vs GBM + plain TabM: A 0.0000 [cap +0.0010], B +0.0004 (CI -0.0002..+0.0010) [cap +0.0004]; with the per-line weights: A -0.0002 [cap +0.0011], B +0.0004 [cap +0.0005]. Mixing both TabMs (GBM + (TabM + TabM-PLE)/2, per-line weights): A 0.1437 [0.1377], B 0.1359 [0.1369] vs GBM + TabM 0.1435 [0.1369] / 0.1355 [0.1365]: small but positive on both folds and both metrics (+0.0002 / +0.0004 uncapped, +0.0008 / +0.0004 capped). Candidate (D31): final fit of the PLE TabM and blend GBM + (TabM + TabM-PLE)/2 with per-line weights.

## 24. Submitted 2026-10-18 21:12 EDT: GBM + mean(TabM, TabM-PLE), per-line weights
`blend_gbm_tabm_ple_perline_v4.csv`: public **0.12779** (previous best 0.12718; equal-weight GBM + TabM 0.12679). Local: A 0.1437 [cap 0.1377], B 0.1359 [0.1369]. Public gain +0.0006 over the previous best, same sign as the local gain (+0.0002 / +0.0004). Top public score seen earlier: 0.12815 (gap 0.0004). Current final-submission candidate (D20).

## 25. Alternative categorical encodings for the GBM (21:27 EDT)
Native = LightGBM categorical splits (reference A 0.1339 [cap250k 0.1323], B 0.1323 [0.1333]). `--cat-encoding both` = frequency-ranked integer label code plus log level count, as numeric columns.
| Run | Fold A D2 [cap250k] | Fold B D2 [cap250k] | Read |
|---|---|---|---|
| `fs_gbm_both` (helper 3) | 0.1368 [0.1353] | 0.1334 [0.1346] | better than native on both folds and both metrics (A +0.0029, B +0.0011); motor 0.0780 vs 0.0717 on A |
Blend (equal weights) with TabM: replacing the native GBM by the `both` GBM beats GBM + TabM by +0.0010 (CI +0.0004..+0.0015) on A and +0.0002 (-0.0003..+0.0008) on B (cap250k: 0.1370 vs 0.1360, 0.1364 vs 0.1361). Using both GBMs together adds nothing (A -0.0004, B +0.0002). Label, count and target variants still running (`fs_gbm_label` locally, `fs_gbm_count` helper 1); not yet compared.

Update 21:40 EDT: `fs_gbm_label` (frequency-ranked integer label codes only): A 0.1375 [cap250k 0.1354], B 0.1336 [0.1347]: better than `both` (0.1368 / 0.1334) and than native (0.1339 / 0.1323) on both folds and both metrics. Numeric-input encodings for the trees (`--num-encoding lowcard|count|both`, `--max-bin`) and for the NN (`--nn-numtf log|quantile`, PLE already tested) are being tested on top of the label encoding.

## 26. Categorical encodings: summary and public check (21:57 EDT)
Single GBM (LightGBM leaves 4, outlier 5), local D2 [cap250k] fold A / fold B:
| Encoding of categoricals | A | B |
|---|---|---|
| native categorical splits (reference) | 0.1339 [0.1323] | 0.1323 [0.1333] |
| label (frequency-ranked integers) | 0.1375 [0.1354] | 0.1336 [0.1347] |
| count (log level counts; helper 1, log line only) | 0.1377 [0.1357] | 0.1331 [0.1343] |
| both (label + count) | 0.1368 [0.1353] | 0.1334 [0.1346] |
| **target (smoothed out-of-fold claim rate)** | **0.1376 [0.1365]** | **0.1339 [0.1350]** |
Target encoding is the best single model on both folds and both metrics. Submitted alone (`fs_gbm_target.csv`, fitted on 2021-24): public **0.12720** vs 0.12548 for the native GBM (+0.0017; local gain +0.0037 A / +0.0016 B) and equal to the previous best blend (0.12718): the encoding change is a real gain that the public score confirms.
Blends (per-line weights 0.7/0.4/0.3 on the mean of TabM and TabM-PLE; OOF, A / B): submitted blend (native GBM) 0.1437 [0.1377] / 0.1359 [0.1369]; with the target GBM instead 0.1442 [0.1384] / 0.1361 [0.1371] (+0.0006 (CI +0.0001..+0.0011) A, +0.0001 (-0.0004..+0.0006) B); adding label and/or both and native GBMs on top changes nothing (+/-0.0001): the encoding variants are highly correlated, so averaging them adds little. Candidate: `blend_gbmtarget_tabm_ple_perline_v5.csv`.

## 27. Numeric-input encodings for the GBM (helper 3, 22:06 EDT; on top of label encoding, reference A 0.1375 [0.1354], B 0.1336 [0.1347])
| Run | Fold A [cap250k] | Fold B [cap250k] | Read |
|---|---|---|---|
| `fs_gbm_label_num` (`--num-encoding both`: low-cardinality numerics also as categoricals + log counts of numeric values) | 0.1347 [0.1345] | 0.1308 [0.1324] | worse (-0.0028 on both folds): rejected |
| `fs_gbm_label_bin1023` (LightGBM max_bin 1023 instead of 255) | 0.1375 [0.1354] | 0.1335 [0.1347] | no change |
Trees are invariant to monotone transforms of numeric inputs, so only information-adding encodings can matter; the extra copies add noise. Numeric transforms for the NN (rank-gauss, log1p) are running (`fs_nn_tabm4_q`, `fs_nn_tabm4_log`; PLE already helped).

Numeric transforms for the TabM net (22:23 EDT; plain TabM: A 0.1413 [0.1331], B 0.1328 [0.1338]; PLE: A 0.1430 [0.1364], B 0.1340 [0.1351]):
| Run | Fold A [cap250k] | Fold B [cap250k] | Read |
|---|---|---|---|
| `fs_nn_tabm4_q` (rank-gauss / quantile transform) | 0.1397 [0.1336] | 0.1308 [0.1321] | worse than plain on both folds |
| `fs_nn_tabm4_log` (log1p of right-skewed non-negative inputs) | 0.1420 [0.1338] | 0.1316 [0.1327] | A up, B down: no gain |
Conclusion (D32): of all numeric encodings tried, only the piecewise-linear encoding (8 quantile bins) helps the NN; extra numeric copies hurt the trees. Categorical encoding is where the gain was (target encoding for the trees: D31 -> see section 26).

## 28. Submitted 2026-10-18 22:28 EDT: v5 = target-encoded GBM + mean(TabM, TabM-PLE), per-line NN weights 0.7/0.4/0.3
`blend_gbmtarget_tabm_ple_perline_v5.csv`: public **0.12790** (previous best 0.12779; target GBM alone 0.12720; native-GBM blend 0.12779). Local A 0.1442 [cap 0.1384], B 0.1361 [0.1371] (+0.0006 / +0.0001 vs the previous best blend). Gap to the top public score seen (0.12815): 0.0003. Current final-submission candidate (D20). The public gain over the previous blend (+0.0001) is smaller than the single-model gain from target encoding (+0.0017) because the neural nets already carry most of what the encoding adds.

`fs_nn_tabm4_te` (TabM k=4 + target-encoded categorical inputs, 22:41 EDT): A 0.1395 [cap250k 0.1340], B 0.1314 [0.1325] vs plain TabM 0.1413 [0.1331] / 0.1328 [0.1338]: no gain (A -0.0018 uncapped, +0.0009 capped; B -0.0014 / -0.0013). The embeddings already capture what the target encoding adds for the NN; the TE + PLE combination is running on helper 3.

`fs_nn_tabm4_ple_te` (TabM + PLE + target-encoded categorical inputs; helper 3, 22:52 EDT): A 0.1427 [cap250k 0.1352], B 0.1341 [0.1351] vs TabM + PLE 0.1430 [0.1364] / 0.1340 [0.1351]: no gain. Target-encoded inputs do not help the NN (D33): the embeddings already carry it.

`fs_gbm_target_peril` (per-peril claim-rate encodings of the categoricals instead of one any-claim encoding; 23:00 EDT): A 0.1380 [cap250k 0.1362], B 0.1340 [0.1350] vs `fs_gbm_target` 0.1376 [0.1365] / 0.1339 [0.1350]: equal within noise. Not adopted (D34): more code, no gain.

`fs_gbm_target_hl1` (target-encoded GBM, severity trend weighted to recent years, half-life 1 year; 23:10 EDT): A 0.1384 [cap250k 0.1364], B 0.1340 [0.1351] vs `fs_gbm_target` 0.1376 [0.1365] / 0.1339 [0.1350]: equal within noise; calibration (pred / actual) unchanged (A: motor 1.05, buildings 1.02, contents 0.94; B: 1.03, 0.99, 1.00). Trend multipliers x1.5 / x0.5 are running on helper 1.

## 29. Severity trend ('inflation factor') optimisation (23:40 EDT; helper 1 for the multipliers)
Estimated pooled trend per line (log-linear, peril effects removed, claims capped at the 99.9th percentile; `scripts/trend_check.py`): fitted on 2021-22 / 2021-23 / 2021-24: motor 6.4% / 4.4% / 5.7%; buildings 2.7% / 4.9% / 6.7%; contents 1.8% / 3.5% / 4.3% per year. Realised year-on-year change in severity: motor +6.3%, +2.5%, +9.5% (2022, 2023, 2024); buildings +2.8%, +6.9%, +10.1%; contents +1.8%, +5.0%, +5.7%: inflation is accelerating, so a pooled linear estimate lags the latest year.
Backtest of the trend multiplier on the target-encoded GBM (calibration = predicted / actual, motor / buildings / contents; fold A 2023, fold B 2024; D2 [cap250k]):
| Trend | Fold A calibration | Fold B calibration | A D2 [cap] | B D2 [cap] |
|---|---|---|---|---|
| x0.5 | 1.014 / 1.007 / 0.929 | 1.006 / 0.950 / 0.972 | 0.1375 [0.1368] | 0.1339 [0.1351] |
| x1.0 (current) | 1.045 / 1.020 / 0.937 | 1.035 / 0.985 / 0.996 | 0.1376 [0.1365] | 0.1339 [0.1350] |
| x1.5 | 1.080 / 1.034 / 0.943 | 1.065 / 1.017 / 1.019 | 0.1376 [0.1356] | 0.1336 [0.1346] |
| recent-weighted (half-life 1 y) | 1.05 / 1.02 / 0.94 | 1.03 / 0.99 / 1.00 | 0.1384 [0.1364] | 0.1340 [0.1351] |
Reading: D2 does not react to the trend (all within 0.0004); calibration says buildings and contents are about right at x1.0 on both folds (the buildings optimum lies between x1.0 and x1.5 on fold B and below x1.0 on fold A), while motor is over-predicted by 3-5% at x1.0 on both folds and is closest to 1.0 at x0.5. The acceleration seen in the year-by-year changes is not confirmed as a bias by the backtests. Decision (D35): keep the pooled trend (multiplier 1.0); do not raise it on the strength of the 2024 jump alone. A motor-specific multiplier of about 0.5-0.7 would be supported by both folds but is worth under +0.0003 D2 and conflicts with the recent motor change (+9.5%); left unchanged and noted as an open uncertainty in methodology/MODELLING.md.

## 30. Forward-looking inflation for 2025 (00:15 EDT)
User question: apply a higher severity factor for 2025 than the historical pooled average? Evidence: (a) year-on-year severity change is accelerating in buildings (2.8, 6.9, 10.1%) and contents (1.8, 5.0, 5.7%) but not in motor (6.3, 2.5, 9.5%); (b) backtest of a 'latest year-on-year change' rule (half-life 0.3 y; identical to the pooled estimate on fold A, which has two training years): `fs_gbm_target_hl03` A 0.1382 [cap250k 0.1363], B 0.1338 [0.1350]; fold B calibration (pred / actual) motor 1.02, buildings 1.01, contents 1.01 vs 1.035, 0.985, 0.996 with the pooled trend: closer to 1 for motor and similar elsewhere, D2 unchanged. (c) Level effect on 2025 of switching the final fit to the latest-year trend: motor +5.3% (pooled 5.7% -> 7.7%), buildings +7.6% (6.7% -> 9.6%), contents +3.1% (4.3% -> 5.5%).
Judgement: motor's latest change (+9.5%) follows +2.5%, so it is noisy, and the backtests show motor over-predicted at the pooled trend: no uplift for motor. Buildings and contents accelerate steadily: uplift supported. Candidate `blend_v5_trend2025_v6.csv` = v5 x (motor 1.0, buildings 1.0757, contents 1.0312) (`scripts/scale_submission.py`). This is a modelling assumption about 2025, backed by the backtest direction; the public score is not used to choose the factor (D21). Not yet submitted.
Re-weighted blend (NN weights 0.9 / 0.36 / 0.26, average of the per-fold likelihood optima, vs v5 0.7 / 0.4 / 0.3): A +0.0010 (CI -0.0002..+0.0029) [cap 0.0000], B +0.0000 (-0.0002..+0.0002): does not beat v5 on both folds, so not submitted (D36).

Submitted 2026-10-19 00:20 EDT (user approved): `blend_v5_trend2025_v6.csv` (v5 x buildings 1.0757, contents 1.0312): public **0.12761** vs v5 0.12790 (-0.0003). The forward-looking uplift did not help: the pooled historical trend is at least as good for 2025 as the latest-year trend on buildings and contents. Not adopted (D37); v5 remains the final candidate. The uplift was chosen from the year-by-year severity changes and the fold-B backtest, not from the public score, and the public score is not used to search further factors (D21); a downward motor adjustment suggested by the backtests (motor over-predicted 3-5%, worth under +0.0003) is left as an open uncertainty rather than tested on the board.

## 31. Residual scan and richer claim history (15:15 EDT 2026-10-19)
Residual scan of the v5 blend (`scripts/residual_scan.py`, `docs/data/RESIDUAL_SCAN.md`): calibration (actual / predicted) by prediction decile and by feature bin on the pooled out-of-fold predictions. Findings: (1) motor is over-predicted by about 5-10% in nearly every bin (uniform, not feature-specific); (2) contents: the lowest-prediction decile is under-predicted (actual / predicted 1.6); (3) nothing else systematic. Leave-one-fold-out checks: a per-line level scalar gains only +0.0001..+0.0002 D2 (motor 0.945), and shrinking predictions toward a flat rate per exposure gains nothing (optimal shrinkage 0): the blend is about optimally scaled and shaped.
`fs_gbm_target_hist` (`--history-extra`: per-peril counts of earlier claims on the policy, days since the last claim, largest earlier cost, open earlier claims): A 0.1380 [cap250k 0.1360], B 0.1339 [0.1351] vs `fs_gbm_target` 0.1376 [0.1365] / 0.1339 [0.1350]: no gain (D38). The policy's own claim history is already captured by the existing history counts.

`fs_gbm_target_l7` (target-encoded GBM, leaves 7; helper 3, 15:36 EDT): A 0.1386 [cap250k 0.1364], B 0.1346 [0.1357] vs leaves 4 (`fs_gbm_target`) 0.1376 [0.1365] / 0.1339 [0.1350]: uncapped +0.0010 / +0.0007, capped -0.0001 / +0.0007: slightly better; target encoding changes the hyperparameter optimum (with native categoricals leaves 31 was worse and leaves 7 about +0.0016 on fold A). Leaves 12, 600 rounds and min-leaf 100 are being screened.

## 32. Multi-seed final NNs and GBM hyper-parameters with target encoding (16:40 EDT 2026-10-19)
Submitted `blend_gbmtarget_tabm_ple_s3_perline_v7.csv` (v5 recipe, TabM and TabM-PLE finals each averaged over 3 seeds): public **0.12785** vs v5 0.12790: no difference (the 3-seed averaging changes predictions by about 0.1% overall, rank correlation 0.999, so the single-seed finals were not noise-limited). Not adopted over v5 (D39).
Target GBM re-screen (helper 3; reference leaves 4, 350/300 rounds: A 0.1376 [cap250k 0.1365], B 0.1339 [0.1350]): leaves 7 A 0.1386 [0.1364], B 0.1346 [0.1357]; 600/500 rounds A 0.1374 [0.1366], B 0.1346 [0.1357]; min leaf 100: A 0.1377 [0.1365], B 0.1339 [0.1350]. Leaves 7 is the only candidate better on three of four numbers (+0.0010 / +0.0007 uncapped); testing it in the blend and as a final fit.

Blend test of the leaves-7 target GBM (OOF, equal NN mix, per-line weights): leaves 7 instead of leaves 4: A 0.1442 [0.1382], B 0.1360 [0.1371] vs v5 0.1442 [0.1384] / 0.1361 [0.1371] (diff 0.0000 on both folds); averaging leaves 4 and 7: +0.0001 / +0.0001 (CIs include 0). Saturated: no change to the blend (D40); the leaves-7 final fit is not used.

## 33. Structural checks (2026-10-19 17:30 EDT)
Leaderboard seen at 17:15 EDT: 1st 0.12996, 2nd 0.12940, 3rd 0.12923, 4th 0.12851, 5th 0.12815; we are 6th with 0.12790 (gap to 1st 0.0021).
- 2025 weather vs training years: all seven weather variables have 2025 means inside the range of the yearly training means (wind exposure 4.88 vs 4.35-4.60 is the highest) and only 0-4% of 2025 rows lie outside the 1-99% training range: no extrapolation problem.
- Ultimate-equivalent claim frequency by peril and accident year (completeness-adjusted): flat within noise except mild trends (motor bodily injury -3%/yr, buildings subsidence +11%/yr, motor glass +3%/yr); no case for a frequency trend.
- Second-stage correction of the blend (`scripts/stage2.py`; LightGBM Tweedie 1.5, init_score = log blend, 60 rounds, 4 leaves, min leaf 3000, trained on the out-of-fold rows of one fold and evaluated on the other): with raw features A -0.0058 (CI -0.0162..+0.0006), B -0.0036 (CI -0.0052..-0.0018); with only the three prediction columns A -0.0002, B -0.0006. A learned correction does not transfer across years: what is left in the residuals is year-specific noise, not structure (D41: no stacking layer, no residual boosting on the blend).

## 34. GBM-offset (CANN-style) TabM-PLE net (17:40 EDT 2026-10-19)
`fs_nn_tabm4_ple_off` (`--nn-gbm-offset`): the TabM-PLE net gets the cross-fitted (4 customer-hash folds) log rate and log severity of the target-encoded GBM (leaves 4, 350/300 rounds, outlier 5) as offsets and learns only a multiplicative correction (zero-initialised head, so training starts exactly at the GBM). Prediction = full-train GBM x correction.
- Alone: A 0.1387 [cap250k 0.1366], B 0.1339 [0.1348] (target GBM 0.1376 [0.1365] / 0.1339 [0.1350]; TabM-PLE 0.1430 [0.1364] / 0.1340 [0.1351]). It behaves like the GBM, not like the NN: the correction stays small and does not recover the NN's diversity.
- In the v5 blend (per-line weights, paired bootstrap vs v5): as a third member of the NN mean A -0.0010, B +0.0001; as a second tree-like member A +0.0002 (CI -0.0002..+0.0006), B -0.0003; 50/50 with v5 A -0.0017, B -0.0004; 20% weight A -0.0004, B +0.0000. None positive on both folds.
Conclusion: a boosted-residual NN is a GBM-like model and adds nothing the blend does not already have (D42).
- Helper 3 NN-config screen, 3 hidden layers (192,96,48) dropout 0.15, TabM-PLE (`fs_nn_tabm4_ple_h3`): alone A 0.1443, B 0.1339 (reference TabM-PLE 0.1430 / 0.1340). In the v5 blend, replacing TabM-PLE by it: A +0.0007 (CI -0.0003..+0.0023), B +0.0000; adding it as a third NN member: A +0.0005 (CI -0.0000..+0.0012), B +0.0001. Below the bar (not positive with a CI above 0 on both folds); no final fit. Remaining helper screens (wd 1e-3, dropout 0.25, lr 1e-3, linear-leaf GBM) pending.
- Helper 3, lr 1e-3 / 24 epochs TabM-PLE (`fs_nn_tabm4_ple_lr1`): alone A 0.1413 [0.1356], B 0.1348 [0.1359] (reference 0.1430 [0.1364] / 0.1340 [0.1351]). Added as a third NN member of the v5 blend: A -0.0005 (CI -0.0014..+0.0001), B +0.0001 (CI 0.0000..+0.0003): mixed, no gain. NN training-config screens (width, depth, lr, dropout, wd) are exhausted: all within +/-0.001 of the reference.
- Direct Tweedie LightGBM (`lgbm_v2_rebalanced`, A 0.1225 / B 0.1253) mixed into the v5 blend at 10/20/30%: A -0.0003 / -0.0010 / -0.0019, B 0.0000 / -0.0003 / -0.0008 (paired vs v5; B CI excludes 0 at 30%). Worse at every weight on both folds; a target-encoded direct-Tweedie refit would need to gain 0.01+ alone to change this, so it is not run (D43).

## 35. Review of the target construction and data quality (19:00 EDT 2026-10-19)
Systematic scan of every table (duplicates, sentinels, unit changes, level-by-year recodes, date consistency, claim-to-row matching) and of the D1/D7/D8 assumptions. Defects found (DATA_QUALITY items 11-14, repairs in `src/iaa/cleaning.py`, applied in the loaders, tested in `tests/test_cleaning.py`):
- 604 exact duplicate claims (11-character ids, 600 in loss year 2022): dropped.
- 252 claims with the placeholder loss date 2020-12-01 (reported through 2021): loss date imputed from report date and peril lag, so they enter the 2021 target (they were dropped as unmatched).
- `sum_insured` 1000x too large on 0.2% of buildings/contents rows, train and test (1,395 / 1,402 train, 370 / 360 test rows): divided by 1000.
- `construction_type`: T (2021-22) is W (2023+) recoded, plus case variants (11 levels -> 4): the 2025 rows contain only W/w, so a model trained before 2023 never saw the test level.
- D1 check: recoveries are booked only on closed claims, but whether a claim recovers is fixed at the outset (own_damage: 30% of closed claims in every accident year including 2024, always 80% of incurred; theft 13% at 30%). Open own_damage/theft claims therefore carry an expected recovery of 23.6% / 4.0% of incurred, not yet booked: net cost of open claims reduced accordingly (`target.open_recovery_rates`; -1.37M, 1.2% of motor 2024).
- Checked and clean: duplicate policy rows or periods, overlapping periods, orphan claims, claim line vs policy line, report before loss, ages vs dates of birth, numeric medians by year, level-by-year of every other categorical, near-duplicate claims. `customer_service_contacts` lagged one year carries no information (claim frequency flat in last year's contacts given no claim), confirming D3.

Local effect, target-encoded GBM (A / B pooled D2; paired bootstrap vs the unrepaired `fs_gbm_target` 0.1376 / 0.1339):
| Repairs | A | B |
|---|---|---|
| all four (`fs_gbm_target_clean`) | 0.1369 (-0.0007, CI -0.0022..+0.0003) | 0.1338 (-0.0001, CI -0.0005..+0.0003) |
| claims only (duplicates, placeholder dates) | 0.1368 (-0.0008, CI -0.0022..+0.0002; motor 0.0795 -> 0.0781) | 0.1338 (-0.0002) |
| policy rows only (sum_insured, construction) | 0.1376 (0.0000, CI -0.0002..+0.0002) | 0.1338 (-0.0001) |
| all + expected recoveries on open claims (`fs_gbm_target_clean_rec`, scored on its own y) | 0.1369 (+0.0000 vs clean) | 0.1333 (0.0000 vs clean) |
No repair changes the validation score measurably (all intervals include 0; the fold-A motor dip comes from the claims repairs and fold A trains on two years only, so its severity trend is the least stable estimate). They are kept because each is a correction whose truth is known from the data, not because they score; a refit on 2021-24 sees all of them and the 2025 rows no longer contain the 730 corrupted sum_insured values (D44).

## 36. v8: the v5 recipe refit on the repaired data (19:55 EDT 2026-10-19)
`blend_v8_clean_perline.csv` = target-encoded GBM + mean(TabM, TabM-PLE) (single-seed finals, `scripts/queue_final_clean.sh`), per-line NN weights 0.7 / 0.4 / 0.3, all four data repairs and the expected recoveries on open claims. Submitted 23:52 UTC (19:52 EDT) with the lead's go-ahead: public **0.12867**, a new best (v5 0.12790, +0.00077; 5 of 10 submissions used on the 2025-10-19 UTC day before this one).
Versus v5 the 2025 predictions are 1.7% lower overall (motor -2.5%, buildings -0.6%, contents -0.8%), rank correlation 0.996-0.998. The change is mostly a motor level correction: the pooled motor severity trend falls from 5.7% to 5.1% a year (open own_damage claims in 2024 no longer carry unbooked recoveries, 4% of that peril's 2024 cost, and the 2022 duplicates no longer lift the 2022 level), and the 2024 level it projects from is lower. Locally the same repairs moved the GBM by -0.0007 / -0.0001 (RESULTS 35): the validation folds (calibration motor 1.04-1.05) could not resolve a level shift of this size, the public score did. Not attributed to a single repair (no ablation submissions, D21).

## 37. Regime breaks, level and trend screens on the repaired data (20:50 EDT 2026-10-20)
New structural facts (scripts inline, numbers from the repaired claims):
- **Bodily-injury severity steps down on 2023-01-01**: median 13.4k / 13.2k (2021 / 2022) -> 9.8k / 9.9k (2023 / 2024); the whole distribution moves (5th-90th percentiles -25% to -30%), quarterly medians are flat before and after, closed and open claims alike (so it is not case-reserve development). A pooled log-linear trend cannot represent it.
- **Claim frequency steps on 2023 too** (ultimate-equivalent, per exposure): own_damage 0.0641 / 0.0645 -> 0.0595 / 0.0598, third_party_prop 0.0437 / 0.0441 -> 0.0419 / 0.0409, bodily_injury -9% over four years, glass +9%, buildings subsidence +45% and fire +23% (steady), contents theft +8%. The model assumes none of it (rate = four-year average, no frequency trend), which is why predicted counts are 3-10% above actual in both backtests (motor fold B 1.03-1.05 for every peril; fold A own_damage 1.105).
- Severity calibration of the production GBM (`scripts/sev_calibration.py`): bodily injury predicted/actual 1.22 (fold A), 1.05 (fold B); other motor perils 0.85-0.99 (under).
- Backtest of next-year mean severity per peril (train <= 2022 -> 2023, train <= 2023 -> 2024; claim-weighted mean absolute log error): no trend 0.092 / 0.107, pooled trend 0.050 / 0.051, per-peril trend (clipped to [0, 0.12]) 0.038 / 0.035, shrunk per-peril (negative allowed) 0.046 / 0.032. Per-peril trends cut the error by about a third.
- Paired noise of the public leaderboard: the standard deviation of the D2 difference between two near-identical models on random one-third customer subsets is 0.0003 on fold-B-like data and 0.0010 on fold-A-like data (one 11.9M claim). A 2.5% motor level shift moves local D2 by about 0.0001, so v8's +0.0008 cannot be attributed to the level change.

Screens, target-encoded GBM on the repaired data (reference `fs_gbm_target_clean_rec` A 0.1369 / B 0.1333; paired bootstrap on the same rows):
| Run | A | B | Verdict |
|---|---|---|---|
| rebalance per peril (all years) | -0.0001 | -0.0000 | neutral |
| per-peril rebalance to recent-weighted cost, half-life 2 y | -0.0001 | -0.0000 | neutral |
| same, half-life 1 y | -0.0001 | -0.0000 | neutral |
| line-level rebalance, half-life 1 y | 0.0000 | +0.0000 | neutral |
| written_premium and sum_insured divided by their median in the row's own year | +0.0000 (CI -0.0003..+0.0003) | -0.0002 (-0.0006..+0.0002) | rejected |
| policy year as a numeric input (`--year-feature`) | +0.0003 (-0.0007..+0.0017; motor +0.0010) | +0.0002 (-0.0002..+0.0006; buildings +0.0006) | positive on both folds, not significant |
Level and mix changes are invisible to the folds, as the sensitivity analysis predicts; they can only be judged by the backtest above and, noisily, by the public score.
- Per-peril severity trends (`--trend-mode peril`, `fs_gbm_c_tpp`): A +0.0003 (CI +0.0000..+0.0005, motor 0.0782 -> 0.0790), B +0.0000. Per-peril trends plus per-peril recent-weighted rebalance (half-life 1 y): A +0.0002, B -0.0000.
- Per-peril trends plus the policy-year input (`fs_gbm_c_tpp_year`): **A +0.0011 (CI -0.0001..+0.0027; motor 0.0782 -> 0.0804), B +0.0002 (CI -0.0002..+0.0006; buildings +0.0006)**: the best GBM variant, positive on both folds, not significant. Queued: this plus per-peril recent-weighted rebalancing (`fs_gbm_c_all`, so the rebalance target and the year input agree on the post-break level) and the TabM / TabM-PLE nets with per-peril trends and recent-weighted rebalancing.
