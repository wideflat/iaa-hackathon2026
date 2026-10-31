# Voice-over script with timecodes (relaxed pace, about 141 words per minute)


## 01 · The approach  (0.6s - 48.1s)

* [0:00.6] I'm Kohei, and this is one pricing engine applied to two markets.
* [0:05.8] Week one: a developed market with two point four million policy rows. Rank three of fifty-four.
* [0:12.5] Week two: an emerging market with only two years of data, and a test book that is seventy-six percent new business. Our final file scored point one six four six on the public leaderboard.
* [0:26.3] The approach: keep what transfers from the developed market, re-measure what does not, and test every step in both directions.
* [0:34.6] And AI works alongside me throughout: I used Claude Code, steered from my phone by voice, to explore the data, write and test code, run experiments, and verify its own work.

## 02 · What changed  (48.1s - 74.7s)

* [0:48.1] So what changed in the new market?
* [0:51.3] Motor claims are thirty-four percent more frequent, but forty-two percent cheaper.
* [0:56.0] Glass, third-party and injury claims are far more frequent. Escape of water is far less.
* [1:02.4] And three out of four test policies are brand-new customers with no history.
* [1:07.9] We measured that these differences are stable across both years, instead of assuming it.

## 03 · The method  (74.7s - 108.5s)

* [1:14.7] The method keeps what transfers, and re-measures what does not.
* [1:19.1] A developed-market model, trained on the variables both markets share, ranks who is risky. I add last year's claim history.
* [1:27.4] An ensemble of five tree models, seven neural nets and four Tweedie models averages out the noise.
* [1:34.5] Then the emerging book sets the level: claim counts and winsorised severities for each peril, credibility-weighted, plus one gender factor.
* [1:42.8] The emerging data decides how much. The developed data decides who.

## 04 · The final ensemble  (108.5s - 148.9s)

* [1:48.5] Here is the structure of the final submission.
* [1:52.1] Five tree models: LightGBM with four and eight leaves, single and three-seed fits, a column-sampled variant, and an XGBoost model.
* [2:00.4] Seven neural nets: TabM, TabM with piecewise-linear encoding, and TabM with quantile inputs, each as a single and a three-seed fit, plus one wider TabM.
* [2:10.6] Four direct Tweedie models complete it, at four and eight leaves, single and three-seed.
* [2:16.6] Each group is averaged with equal weights: trees and nets thirty-seven and a half percent each, Tweedie twenty-five. Then the emerging level and the gender factor are applied.

## 05 · How we trust it  (148.9s - 190.9s)

* [2:28.9] With two years of data, a single split can mistake noise for signal.
* [2:34.4] So every change is tested in both directions, calibrating on 2023 and scoring 2024 and the reverse, and kept only if neither gets worse.
* [2:44.3] That took the score from point one one one oh, for an emerging-only model, to point one four oh five.
* [2:52.6] More than thirty ideas were rejected this way, including pooled training and a GLM blend.
* [2:58.9] A simulation with a known truth confirmed the level method, and a ceiling analysis showed that a perfect level is worth at most half of one hundredth.

## 06 · AI in the workflow  (190.9s - 226.2s)

* [3:10.9] Here is how that looked in practice.
* [3:14.1] Claude Code explored both markets, wrote the code with one hundred and twenty-five tests, and ran the experiments, while more than ten helper sessions refit models in parallel.
* [3:25.5] A blind second agent compared the markets from the raw tables alone.
* [3:30.6] And verification caught real errors: unsupported statements in a first draft, a simulator bug that looked like a model bias, and a net that Part one had already rejected.
* [3:42.5] Every number traces to a command.

## 07 · A new way to work  (226.2s - 259.1s)

* [3:46.2] One example of how this changes the way I work.
* [3:50.5] Part of week two was done from my phone, on a bus, with no laptop. Claude Code ran in the cloud, and I controlled it remotely from the Claude app.
* [4:02.8] I gave my instructions by voice. The app transcribes my speech, Claude Code picks it up, runs the work, and reports back while I am on the move.
* [4:14.2] That is AI-assisted actuarial work, away from the desk.

## 08 · Guardrails and uncertainty  (259.1s - 283.9s)

* [4:19.1] Guardrails: a leak alarm, reproducibility checks, a model card, and eighty-four recorded decisions.
* [4:24.6] The biggest uncertainty is the buildings level. Plus or minus twenty percent is about one hundredth of a point.
* [4:32.5] And gender rating is flagged for regulatory review.
* [4:36.1] Built with an AI teammate, from a bus seat. Thank you.
