# Part 1 techniques: what Part 2 already uses, what was tested, what is open

2026-10-25 (EDT). Part 1 final recipe (v10): tree half = mean(frequency x severity GBM, mean of two direct Tweedie GBMs), net half = TabM + TabM-PLE (3 seeds each), net weight per line 0.7 / 0.4 / 0.3, repaired data, severity trend, completeness offset, outlier winsorisation, per-line rebalancing, claim-history features.

| Part 1 technique | Part 1 evidence | Part 2 status |
|---|---|---|
| Frequency x severity per peril, Poisson / Gamma, exposure offset | core | **used** (developed model) |
| Severity trend, completeness offset, excess load, outlier multiple 5 | core | **used** (borrowed from the developed book; emerging severity trend checked, D59) |
| Data repairs (duplicates, placeholder dates, sum_insured x1000, construction codes, open-claim recoveries) | RESULTS 35 | **used**, with emerging bands (0 duplicates, 0 placeholders expected) |
| Target encoding of categoricals (GBM) | +0.002 | **used** |
| LightGBM with 4 leaves | +0.001 | **used**; 8 leaves, slower learning, larger leaf size tested: no gain (D64) |
| TabM and TabM-PLE nets | +0.004 | **used**, one seed each, 50% weight; per-line Part 1 weights 0.7 / 0.4 / 0.3 tested early (+0.0018 vs +0.0023 for 50/50) |
| 3-seed nets (v7) | within noise | **open**: would reduce level noise of the net half; hours of compute |
| Direct Tweedie GBM in the tree half (v10) | +0.0003 local, +0.0002 public | **in test** (helper `p2-tw`, `p2_calib2.py --tw-names`) |
| XGBoost depth 3 (v12) | no gain | not tried; no reason to expect more |
| CatBoost, GLM, direct Tweedie net, GBM-offset net, linear trees, peril-level stacking | rejected | not tried (rejected in Part 1) |
| Claim-history features (`hist_*`, `cust_n_*`, `addr_n_*`) | in the model | **in test** as one-year windows `hist1`: the emerging data shows last-year claimants 27% above the model among motor renewals (A/E 1.195 vs 0.944) |
| Lag-1 weather | in the model | dropped (empty for the first emerging year) |
| Year-on-year change features (`yoy`), engineered ratios, deflation, seasonality offset, event thinning, per-peril trends, year feature | tested, no gain | ratios / market scaling / lines-held tested: no gain (D64); the others not retried |
| Domain-shift weights | no gain | tested for the market shift: no gain (D67) |
| Recency weights | no gain | not applicable (two emerging years) |
| Paired cluster bootstrap, two folds | validation | **used**, plus the reverse direction (new) |
| Public leaderboard as level probe | level x0.9 / x1.1 | used once (p2_t1) as a sanity check |
| Selection by local validation, not public score | D20 | **used** |

New in Part 2 (not in Part 1): the market layer (`market.py`), the transfer calibration (robust level, gender factor), both-direction validation, the blind second opinion, the scenario table.
