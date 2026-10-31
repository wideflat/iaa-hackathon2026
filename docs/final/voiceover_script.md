# Voice-over script with timecodes (relaxed pace, about 141 words per minute)


## 01 · The approach  (0.6s - 45.3s)

* [0:00.6] I'm Kohei, and this is one pricing engine applied to two markets.
* [0:05.8] Week 1: a developed market with 2.4 million policy rows. Rank 3 of 54.
* [0:11.7] Week 2: an emerging market with only 2 years of data, and a test book that is 76% new business. Our final file scored 0.1646 on the public leaderboard.
* [0:23.5] The approach: keep what transfers from the developed market, re-measure what does not, and test every step in both directions.
* [0:31.8] And AI works alongside me throughout: I used Claude Code, steered from my phone by voice, to explore the data, write and test code, run experiments, and verify its own work.

## 02 · What changed  (45.3s - 71.2s)

* [0:45.3] So what changed in the new market?
* [0:48.5] Motor claims are 34% more frequent, but 42% cheaper.
* [0:52.5] Glass, third-party and injury claims are far more frequent. Escape of water is far less.
* [0:58.8] And 3 out of 4 test policies are brand-new customers with no history.
* [1:04.4] We measured that these differences are stable across both years, instead of assuming it.

## 03 · The method  (71.2s - 105.0s)

* [1:11.2] The method keeps what transfers, and re-measures what does not.
* [1:15.6] A developed-market model, trained on the variables both markets share, ranks who is risky. I add last year's claim history.
* [1:23.9] An ensemble of 5 tree models, 7 neural nets and 4 Tweedie models averages out the noise.
* [1:31.0] Then the emerging book sets the level: claim counts and winsorised severities for each peril, credibility-weighted, plus one gender factor.
* [1:39.3] The emerging data decides how much. The developed data decides who.

## 04 · The final ensemble  (105.0s - 143.8s)

* [1:45.0] Here is the structure of the final submission.
* [1:48.5] 5 tree models: LightGBM with 4 and 8 leaves, single and 3-seed fits, a column-sampled variant, and an XGBoost model.
* [1:56.8] 7 neural nets: TabM, TabM with piecewise-linear encoding, and TabM with quantile inputs, each as a single and a 3-seed fit, plus one wider TabM.
* [2:07.1] 4 direct Tweedie models complete it, at 4 and 8 leaves, single and 3-seed.
* [2:13.0] Each group is averaged with equal weights: trees and nets 37.5% each, Tweedie 25%. Then the emerging level and the gender factor are applied.

## 05 · How we trust it  (143.8s - 181.4s)

* [2:23.8] With 2 years of data, a single split can mistake noise for signal.
* [2:29.3] So every change is tested in both directions, calibrating on 2023 and scoring 2024 and the reverse, and kept only if neither gets worse.
* [2:39.2] That took the score from 0.1110, for an emerging-only model, to 0.1405.
* [2:44.4] More than 30 ideas were rejected this way, including pooled training and a GLM blend.
* [2:50.7] A simulation with a known truth confirmed the level method, and a ceiling analysis showed that a perfect level is worth at most 0.005.

## 06 · AI in the workflow  (181.4s - 215.6s)

* [3:01.4] Here is how that looked in practice.
* [3:04.6] Claude Code explored both markets, wrote the code with 125 tests, and ran the experiments, while more than 10 helper sessions refit models in parallel.
* [3:14.9] A blind second agent compared the markets from the raw tables alone.
* [3:20.1] And verification caught real errors: unsupported statements in a first draft, a simulator bug that looked like a model bias, and a net that Part 1 had already rejected.
* [3:31.9] Every number traces to a command.

## 07 · A new way to work  (215.6s - 248.5s)

* [3:35.6] One example of how this changes the way I work.
* [3:39.9] Part of Week 2 was done from my phone, on a bus, with no laptop. Claude Code ran in the cloud, and I controlled it remotely from the Claude app.
* [3:52.2] I gave my instructions by voice. The app transcribes my speech, Claude Code picks it up, runs the work, and reports back while I am on the move.
* [4:03.6] That is AI-assisted actuarial work, away from the desk.

## 08 · Guardrails and uncertainty  (248.5s - 272.1s)

* [4:08.5] Guardrails: a leak alarm, reproducibility checks, a model card, and 84 recorded decisions.
* [4:14.0] The biggest uncertainty is the buildings level. Plus or minus 20% is about 0.01 of D².
* [4:20.7] And gender rating is flagged for regulatory review.
* [4:24.3] Built with an AI teammate, from a bus seat. Thank you.
