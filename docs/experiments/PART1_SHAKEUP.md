# Part 1: public vs private leaderboard, and what the shake-up was

2026-10-25 (EDT). Data: the public leaderboard (Kaggle download, 54 rows) and the private leaderboard (`/leaderboard/view`), joined on team id; our 21 submissions with both scores (`part1_submissions.csv`). The three organiser baseline files (`sample_submission`, two sandbox baselines) are removed: 51 real teams. Figure: `part1_public_vs_private.png` (script `scripts/plot_part1_shakeup.py`); data: `part1_public_vs_private_teams.csv`.

## Short answer
The shake-up is mostly real sampling noise inside a very tight pack, not a general overfitting story. Public and private agree on the big picture (Pearson 0.997 over the 51 teams, Spearman 0.87) but not on the order *inside* the top group (Spearman 0.20 among the public top 25). Teams that submitted a lot are somewhat over-represented among the biggest drops, but the biggest rises also have many submissions, and the statistical signal for "heavy submitters overfit the public board" is weak (see below).

## What the numbers say
**1. Everybody moved up by the same amount.** The private score is about 0.0082 higher than the public score for all teams (private = 0.0147 + 0.938 x public). That is a level difference between the two halves of the test set (the private half is "easier" for everyone), not a shake-up.

**2. The competition is decided in a band narrower than the noise.** The public top 25 span 0.1264-0.1305 (0.004). Around the trend line, the private score of a team scatters by SD 0.0015 (top 25 only), which is as large as the spread of the private scores in that group (SD about 0.0014). So the public score tells almost nothing about the order within the group: the top 10 teams by public ended at private ranks 1, 3, 6, 8, 13, 14, 17, 22, 23, 27 (mean 13.4; with no noise it would be 5.5). 43% of teams moved 5 or more places, 18% moved 10 or more; the median move is 4.

**3. This is the winner's curse.** Picking the best of many near-equal teams on a noisy measure picks lucky ones; they regress on the other half. The same mechanism lifted teams that were mid-pack on public (kaiY 18 to 5, BenDavid313 23 to 10, CST 20 to 7, Kohei Kudo = us 14 to 3).

**4. Overfitting to the public board: possible but small.**
* Across all 51 teams, the number of submissions is unrelated to the residual (Spearman -0.08).
* The three biggest drops (overfitter... 3 to 27, Lion is a Cat 4 to 23, Tevfik Dogukan Isik 6 to 22) made 40, 25 and 75 submissions; but Simone (48), kaiY (38) and BenDavid313 (32) made as many and rose 9-13 places.
* Within the public top 25, teams with 36 or more submissions (6 teams) have a lower mean residual than the rest (+0.0004 against +0.0015, Spearman of submissions against residual -0.30, p = 0.14): the sign is what overfitting predicts, the size is about 0.001 and it is not statistically significant with 25 teams.
* Across all teams |rank shift| correlates with submissions (0.54), but mostly because heavy submitters sit in the dense top cluster where a small score change moves many ranks.

**5. Our own 21 submissions show the same thing in miniature.** Over all 21 the two scores correlate 0.94 (Spearman 0.92), driven by the big differences (level probes x0.9 / x1.1, the GLM, the first GBM). In the last 12 submissions, whose public scores differ by only 0.0017 (0.12718-0.12887), the correlation drops to 0.69 (Spearman 0.77): inside that range public order is mostly noise. We selected v10 because it had the best public score (0.12887); its private score was 0.13864 (rank 3), whereas v9 (public 0.12854) scored 0.13906 and would have been 2nd. Selecting by public among near-equal candidates cost about 0.0004.

## Was there real volatility in the scores?
Yes, in the sense that differences below about 0.0015 between teams are within the public-to-private noise, so ranks 3 to 25 were largely a lottery relative to the public board. The top four private scores (0.13923, 0.13875, 0.13864, 0.13864) lie within 0.0006 of each other, which is also inside that noise: the final order near the top should be read as "a group of equally good solutions", not as a clean ranking.

## Caveats
* Only the final selected submission of each team is visible, so the analysis cannot see how many candidates each team really chose between.
* The residual SD is estimated from the same data as the trend; it is a descriptive noise level, not a formal test.
* The number of submissions is a crude proxy for overfitting (it ignores how teams selected their final file).

## What to carry into Part 2
Do not choose between candidates whose public scores differ by less than about 0.001-0.0015; rely on the two-direction local validation, use the public score only to catch level or format mistakes, and expect the private order among near-identical candidates to differ. Our robust-level candidates t2, t3, t4 are within 0.0005 of each other publicly, i.e. indistinguishable.
