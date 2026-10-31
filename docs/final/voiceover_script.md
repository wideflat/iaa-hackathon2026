# Voice-over script with timecodes (relaxed pace, about 141 words per minute)


## 01 · The approach  (0.6s - 44.6s)

* [0:00.6] I'm Kohei, and this is one pricing engine applied to two markets.
* [0:06.2] Week one: a developed market with two point four million policy rows. Rank three of fifty-four.
* [0:13.4] Week two: an emerging market with only two years of data, and a test book that is seventy-six percent new business. We finished second.
* [0:24.1] The approach: keep what transfers from the developed market, re-measure what does not, and test every step in both directions.
* [0:33.0] And AI works alongside me through all of it: exploring the data, writing and testing the code, running experiments, and checking its own work.

## 02 · What changed  (44.6s - 73.3s)

* [0:44.6] So what changed in the new market?
* [0:48.0] Motor claims are thirty-four percent more frequent, but forty-two percent cheaper.
* [0:53.2] Glass, third-party and injury claims are far more frequent. Escape of water is far less.
* [0:60.0] And three out of four test policies are brand-new customers with no history.
* [1:06.0] We measured that these differences are stable across both years, instead of assuming it.

## 03 · The method  (73.3s - 109.6s)

* [1:13.3] The method keeps what transfers, and re-measures what does not.
* [1:18.0] A developed-market model, trained on the variables both markets share, ranks who is risky. I add last year's claim history.
* [1:26.9] An ensemble of five tree models, seven neural nets and four Tweedie models averages out the noise.
* [1:34.6] Then the emerging book sets the level: claim counts and winsorised severities for each peril, credibility-weighted, plus one gender factor.
* [1:43.6] The emerging data decides how much. The developed data decides who.

## 04 · How we trust it  (109.6s - 157.5s)

* [1:49.6] With two years of data, a single split can mistake noise for signal.
* [1:55.6] So every change is tested in both directions: calibrate on 2023 and score 2024, then the reverse. We keep it only if neither direction gets worse.
* [2:07.1] That took the score from point one one one oh, for an emerging-only model, to point one four oh five.
* [2:16.1] More than thirty ideas were rejected this way, including pooled training, a GLM blend and clipping of high predictions.
* [2:24.6] A simulation with a known truth confirmed the level method, and a ceiling analysis showed that a perfect level is worth at most half of one hundredth.

## 05 · AI in the workflow  (157.5s - 195.5s)

* [2:37.5] Here is how that looked in practice.
* [2:40.9] Claude Code explored both markets, wrote the code with one hundred and twenty-five tests, and ran the experiments, while more than ten helper sessions refit models in parallel.
* [2:53.2] A blind second agent compared the markets from the raw tables alone.
* [2:58.8] And verification caught real errors: unsupported statements in a first draft, a simulator bug that looked like a model bias, and a net that Part one had already rejected.
* [3:11.6] Every number traces to a command.

## 06 · A new way to work  (195.5s - 222.5s)

* [3:15.5] One example of how this changes the way I work.
* [3:20.2] Part of week two was modelled from my phone, on a bus, with no laptop.
* [3:27.0] I spoke my instructions, Claude Code transcribed them, ran the experiments in the cloud, and sent back results I could read between stops.
* [3:37.3] That is AI-assisted actuarial work, away from the desk.

## 07 · Guardrails and uncertainty  (222.5s - 249.4s)

* [3:42.5] Guardrails: a leak alarm, reproducibility checks, a model card, and eighty-four recorded decisions.
* [3:48.4] The biggest uncertainty is the buildings level. Plus or minus twenty percent is about one hundredth of a point.
* [3:57.0] And gender rating is flagged for regulatory review.
* [4:00.8] Built with an AI teammate, from a bus seat. Thank you.
