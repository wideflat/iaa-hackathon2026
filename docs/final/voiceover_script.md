# Voice-over script with timecodes (relaxed pace, about 141 words per minute)


## 01 · A different way to work  (0.6s - 25.0s)

* [0:00.6] Most of this model was built on a bus.
* [0:04.9] No laptop. Just my phone, and my voice.
* [0:08.7] I spoke, Claude Code transcribed it, ran the experiments in the cloud, and sent back results I could read between stops.
* [0:18.1] This is what AI-assisted actuarial work looks like when it leaves the desk.

## 02 · The result  (25.0s - 46.4s)

* [0:25.0] Two weeks, two markets.
* [0:27.2] Week one: a developed market with two point four million policy rows. Rank three of fifty-four.
* [0:34.4] Week two: a new emerging market, only two years of data, and a test book that is seventy-six percent new business.
* [0:43.8] We finished second.

## 03 · What changed  (46.4s - 75.1s)

* [0:46.4] So what changed in the new market?
* [0:49.9] Motor claims are thirty-four percent more frequent, but forty-two percent cheaper.
* [0:55.0] Glass, third-party and injury claims are far more frequent. Escape of water is far less.
* [1:01.8] And three out of four test policies are brand-new customers with no history.
* [1:07.8] We measured that these differences are stable across both years, instead of assuming it.

## 04 · The method  (75.1s - 111.4s)

* [1:15.1] The method keeps what transfers, and re-measures what does not.
* [1:19.8] A developed-market model, trained on the variables both markets share, ranks who is risky. I add last year's claim history.
* [1:28.8] An ensemble of five tree models, seven neural nets and four Tweedie models averages out the noise.
* [1:36.5] Then the emerging book sets the level: claim counts and winsorised severities for each peril, credibility-weighted, plus one gender factor.
* [1:45.4] The emerging data decides how much. The developed data decides who.

## 05 · How we trust it  (111.4s - 159.3s)

* [1:51.4] With two years of data, a single split can mistake noise for signal.
* [1:57.4] So every change is tested in both directions: calibrate on 2023 and score 2024, then the reverse. We keep it only if neither direction gets worse.
* [2:08.9] That took the score from point one one one oh, for an emerging-only model, to point one four oh five.
* [2:17.9] More than thirty ideas were rejected this way, including pooled training, a GLM blend and clipping of high predictions.
* [2:26.4] A simulation with a known truth confirmed the level method, and a ceiling analysis showed that a perfect level is worth at most half of one hundredth.

## 06 · AI in the workflow  (159.3s - 196.9s)

* [2:39.3] AI was part of every step.
* [2:42.3] Claude Code explored both markets, wrote the code with one hundred and twenty-five tests, and ran the experiments, while more than ten helper sessions refit models in parallel.
* [2:54.6] A blind second agent compared the markets from the raw tables alone.
* [3:00.2] And verification caught real errors: unsupported statements in a first draft, a simulator bug that looked like a model bias, and a net that Part one had already rejected.
* [3:13.0] Every number traces to a command.

## 07 · Guardrails and uncertainty  (196.9s - 223.8s)

* [3:16.9] Guardrails: a leak alarm, reproducibility checks, a model card, and eighty-four recorded decisions.
* [3:22.9] The biggest uncertainty is the buildings level. Plus or minus twenty percent is about one hundredth of a point.
* [3:31.4] And gender rating is flagged for regulatory review.
* [3:35.3] Built with an AI teammate, from a bus seat. Thank you.
