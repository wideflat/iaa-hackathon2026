# Part 1 (developed market): final results

> Competition closed 2026-10-23 16:00 UTC. Everything below was read from the Kaggle API on 2026-10-24 (submissions list with private scores, final leaderboard). Scores are pooled Tweedie D2 (power 1.5).

## Outcome

- **Final private leaderboard: 3rd of 54 teams, 0.13864** (1st SnorlaxCode 0.13923, 2nd Simone 0.13875; the 4th team has the identical 0.13864).
- Public leaderboard: 14th (best public 0.12887). The public ranking was a poor guide: public and private scores of our submissions correlate at Spearman 0.91, but the private set has a higher score level (about +0.010) and reshuffled the top: we moved from 14th to 3rd.
- **Final submission for judging: `blend_v10_perline.csv`** (public 0.12887, private 0.13864). The final leaderboard shows 0.13864, which is v10's private score; this is what Kaggle's default (best public score) selects, and D20 (local validation plus public score) pointed to the same file. 21 submissions in 6 days.
- Week 1 prediction file for the final submission form: `submissions/blend_v10_perline.csv`, sha256 `2223edb5502401f3f5b1833ae4281df5e3d43bdd60452f993d2ae5ec01caa8bd` (gitignored, 626,639 rows).

### Final private leaderboard, top 10
| rank | team | private score |
|---|---|---|
| 1 | SnorlaxCode | 0.13923 |
| 2 | Simone | 0.13875 |
| 3 | Kohei Kudo | 0.13864 |
| 4 | Manuel Rodríguez | 0.13864 |
| 5 | kaiY | 0.13777 |
| 6 | Krishna Shrestha | 0.13768 |
| 7 | CST | 0.13766 |
| 8 | HS | 0.13755 |
| 9 | Pramod Misra | 0.13741 |
| 10 | BenDavid313 | 0.13740 |

## All 21 submissions, in order

| when (EDT) | submission | public | private |
|---|---|---|---|
| 10-17 22:58 | GBM freq x sev v1 | 0.12153 | 0.13363 |
| 10-17 23:03 | level probe x1.10 | 0.12429 | 0.13482 |
| 10-17 23:03 | level probe x0.90 | 0.11554 | 0.12941 |
| 10-17 23:36 | GBM + severity trend + completeness | 0.12455 | 0.13333 |
| 10-18 00:12 | GLM freq x sev | 0.12133 | 0.13193 |
| 10-18 10:08 | GBM + CANN NN 50/50 | 0.12587 | 0.13740 |
| 10-18 18:01 | GBM + TabM 50/50 | 0.12679 | 0.13782 |
| 10-18 18:03 | GBM leaves 4 (alone) | 0.12548 | 0.13457 |
| 10-18 18:03 | TabM (alone) | 0.12346 | 0.13627 |
| 10-18 19:55 | GBM + TabM per-line weights (v3) | 0.12718 | 0.13830 |
| 10-18 21:12 | GBM + TabM/PLE per-line (v4) | 0.12779 | 0.13864 |
| 10-18 21:56 | GBM target-encoded (alone) | 0.12720 | 0.13610 |
| 10-18 22:28 | v5: target GBM + TabM/PLE | 0.12790 | 0.13859 |
| 10-19 00:20 | v6: v5 + 2025 inflation uplift | 0.12761 | 0.13768 |
| 10-19 16:33 | v7: v5 with 3-seed nets | 0.12785 | 0.13840 |
| 10-19 19:52 | v8: v5 on repaired data | 0.12867 | 0.13879 |
| 10-20 07:51 | v9: v8 + per-peril trend/year GBM | 0.12854 | 0.13906 |
| 10-20 14:26 | v10: v8 + Tweedie GBM in tree half (SELECTED) | 0.12887 | 0.13864 |
| 10-20 14:51 | v11: v10 + raw/repaired net mix | 0.12876 | 0.13868 |
| 10-20 17:01 | XGBoost depth 3 (alone) | 0.12732 | 0.13796 |
| 10-20 18:49 | v12: v10 with XGBoost d3 | 0.12863 | 0.13899 |

Machine-readable copy: `docs/experiments/part1_submissions.csv`.

## Reading the scores

1. **Most of the gain came early and from structure, not tuning.** Private: single GBM 0.1336 -> GBM + neural net blend 0.1374 (+0.0038, the largest single step) -> target-encoded categoricals and per-line weights 0.1386 (v5). Everything after v5 (data repairs, Tweedie GBM, XGBoost, 3 seeds, trend variants) moved the private score by at most +0.0005 (v8 0.13879, v9 0.13906, v12 0.13899), the same size as the noise between near-identical models.
2. **Selection noise is real.** Our private scores for the 14 blends that mattered range from 0.13840 to 0.13906 (spread 0.0007). Had v9 (private 0.13906, public 0.12854) been the selected submission we would have ranked 2nd instead of 3rd; v10 was picked because its public score was the best. Local validation (folds A / B) could not separate these candidates either (paired intervals include 0). D20 (select by local validation and public score) was followed; the outcome was a lucky-or-unlucky coin flip of about one rank.
3. **Single models, private:** GLM 0.13193, GBM 0.13457, target-encoded GBM 0.13610, TabM 0.13627, **XGBoost depth 3 0.13796**. Public said XGBoost (0.12732) was close to the LightGBM single model (0.12720); on the private set it was clearly better, in line with the local folds (+0.0020 / +0.0013). Neural nets alone were weaker than the GBM family but the blend added +0.004.
4. **Level probes (x0.90 / x1.10 of the first GBM):** private 0.12941 / 0.13482 vs 0.13363: a +/-10% level error costs 0.004 (-10%) / gains 0.001 (+10%); the private optimum lay above our level, as the pooled severity trend suggested.
5. **Data repairs (duplicates, placeholder dates, unit error, recoded levels) improved the public score by +0.0008 and the private by +0.0002 (v5 -> v8)**; their value for the judging is the documented treatment of data quality (`docs/data/DATA_QUALITY.md`, items 11-17), not the score.

## What worked, what did not (pointers)

- Adopted: frequency x severity per peril with severity trend and completeness offset (D14), per-line rebalancing (D6), winsorisation (D22), CANN/TabM nets and the GBM + net blend with per-line weights 0.7 / 0.4 / 0.3 (D23, D28, D29), target encoding for the GBM (D31), data repairs and expected recoveries on open claims (D44), direct Tweedie GBM on the adjusted target (D51).
- Rejected with evidence: see `docs/workflow/DECISIONS.md` D24-D55 and `docs/experiments/RESULTS.md` sections 10-44 (XGBoost/CatBoost in the early blend, recency/domain-shift weights, entity embeddings, engineered features, tail load, forward-looking inflation uplift, stage-2 residual models, GBM-offset nets, peril-level blend weights, seasonality, year-on-year features, CatBoost).

## Lessons for Part 2

- A held-out score difference below about 0.001 cannot be ranked (paired bootstrap on 1/3 subsets: SD 0.0003-0.001). Choose by robustness and by design, not by the last decimal.
- The biggest real wins were found by auditing the data (duplicates, unit errors, regime breaks) and by changing the target/loss family (Tweedie on the adjusted target), not by hyper-parameter tuning.
- Keep the validation setup frozen and the decision rule written before each run (kept in Part 2).
