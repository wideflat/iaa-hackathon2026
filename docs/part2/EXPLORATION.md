# Part 2 exploration: findings and recommendations (2026-10-24, EDT)

Everything here was measured on the data in this repository; each number has a script or a RESULTS section behind it (RESULTS 45-49). "Forward" = calibrate on emerging 2023, test on 2024; "reverse" = calibrate on 2024, test on 2023. Paired differences are on the same rows with a customer-cluster bootstrap.

## 1. What the emerging market looks like

**The market difference has structure, it is not noise.** Comparing accident year 2023 (mature) of both books, per peril (`scripts` output in the session, same logic as `p2_explore_ae.py`):

| | Frequency ratio emerging / developed | Median claim cost ratio |
|---|---|---|
| Motor | own damage 1.2, third-party property 1.4, glass/hail 1.5-1.8, bodily injury 1.4-1.5, theft 1.3 | 0.52-0.71, mostly ~0.6 |
| Buildings | escape of water 0.66-0.69, accidental damage 0.75-0.84, storm 1.3-1.7, flood 0.9-1.0, subsidence 0.85-1.1, fire 0.84-2.3 (49 vs 21 claims) | 0.56-0.79, mostly ~0.7; **fire ~1.0** |
| Contents | accidental damage 0.75, escape of water 0.65, theft 1.0-1.1, flood 1.0-1.2 | 0.66-0.88, mostly ~0.75 |

The frequency multipliers of the high-count perils repeat in both years (motor third-party property 1.42 / 1.40, contents escape of water 0.67 / 0.63, buildings escape of water 0.69 / 0.66), so they are real. Median severity scales are stable per line (motor ~0.6, buildings ~0.7, contents ~0.75). Mean-based cost ratios are not stable (fire 3.0 then 0.9, subsidence 0.77 then 0.34, bodily injury 0.62 then 0.95): a few large claims move them.
Plain-words model of the market: *developed relativities x a frequency multiplier per peril x a severity scale per line (fire and bodily injury are exceptions) + a gender effect in motor.*

**Where the developed model's relativities do not hold (A/E by feature, `ae_by_feature.csv`):**
* Motor: gender (male +14%, female -13% in frequency, z = 8), occupation (builder +34%, mechanic +20%, student -16%, care worker -16%), age band (17-24 -11%, 25-34 +12%), voluntary excess (1000: -19%), written premium (top bucket -12% / -18% in cost), NCB 6-8 years +20% (one band), named young driver +19%.
* Buildings: non-standard (MIX) construction +26% (z 3: the BRK / STN / WOD / MIX mapping is right in direction, too low in size), year built 1900-1932 +25%; no strong feature.
* Contents: accidental damage cover: rows without cover +16%, with cover -10% (z 3.1 / -2.4).
* Region (E01-E06), channel, cover type, vehicle use and new business show nothing beyond chance.

**The 2025 test book.** Adversarial AUC (train rows vs test rows, shared features, without time-stamped fields): new-business rows 0.55 / 0.58 / 0.61 (motor / buildings / contents), renewals 0.69 / 0.63 / 0.62 (ageing: years licensed, vehicle age, NCB advance by a year). So the 76% new business looks like the new business already seen; the shift is its share. Exposure: 53-56% of test rows have exposure 0.75-0.99 (policies incepting through January) against 6-8% of training rows; 30-32% are full-year rows against 67-72%.

**Data quality, unchanged from DATA_REVIEW:** no duplicates or placeholder dates; sum_insured x1000, free-text occupation, local construction codes, 36 pre-inception claims, the customer_service_contacts leak.

## 2. What mattered for the score

* **Level matters more than relativities.** On the 2024 validation rows, a model with the level fitted *on the same rows* scores 0.1405 pooled D2; with the level fitted on 2023 it scores 0.1265 (buildings 0.111 -> 0.065). The buildings level moved by a factor of about 1.5 between the two years (fire: 49 claims / 3.2M in 2023, 21 / 1.1M in 2024). Relativity improvements are worth +0.002 to +0.004; getting the level right is worth up to 0.014.
* **Closing the total to the calibration rows copies large-loss noise into the level.** Count-based frequency factors and median-based severity factors per peril, with *no* final closure: forward +0.0054 (+0.0019..+0.0087), reverse +0.0009 (-0.0023..+0.0044) over closure + gender (RESULTS 49; `scripts/p2_calib2.py`). With closure the same factors are worse (-0.0031 forward).
* **Gender is the only emerging-only motor correction that survives both directions:** +0.0039 forward, +0.0028 reverse. Age band, occupation and voluntary excess add +0.002 forward and nothing (-0.002..+0.001) in reverse: overfitting to one year.
* **NN base:** the TabM base is better than the GBM base on all three lines, 50/50 blend +0.0023 (+0.0005..+0.0042) (RESULTS 48b).
* **Not the cause of the public-score gap:** new-business vs renewal D2 are 0.126 / 0.119, prediction dispersion is the same in validation and test (sd of log rate 0.93), in-sample 2023 and 2024 D2 are 0.130 / 0.141. The gap (0.16077 vs 0.1265) stays unexplained; the public leaderboard may be a subset of the test rows (noise of about +-0.01 for 60k rows). Do not chase it.

## 3. Recommendations, in order

1. **Change the level method (highest value).** Replace closure + per-peril cost factors by robust frequency x severity factors fitted on both emerging years pooled (counts for frequency, medians against the developed medians for severity, credibility k = 20 / 30), no final closure. Test the pooled-year version with a leave-one-year-out check (already done: forward, reverse), then a scenario table for buildings (level x0.8 / 1.0 / 1.2, fire severity at 0.7 vs 1.0 of the developed). *Cost: half a day; risk: low.*
2. **Keep gender as the single emerging-only correction** (effect stable in both directions) and do not add age / occupation / excess corrections. *Decision on gender in the report stays with the project lead.*
3. **Adopt the TabM + GBM base (50/50) after the TabM-PLE check and a reverse-direction check.** It was the part-1 recipe too. *Cost: waiting for the background job (hours, restarts kill it).*
4. **Handle fire and bodily injury explicitly.** Their severity ratios differ from the line scale (fire ~1.0 against ~0.7; bodily injury mean 0.45 against median 0.6); with the peril-level severity credibility (k = 30) fire gets ~0.9. Report it as an assumption with the sensitivity.
5. **Cheap checks worth doing:** the MIX construction mapping (A/E +26%), the contents accidental-damage-cover pattern, a developed model refit without `written_premium` / `discount_applied` (market-specific pricing variables; the top premium bucket is over-predicted). Each is one refit (25-30 min) and judged by fold E in both directions.
6. **Validation upgrade:** use both directions (forward and reverse) for every future decision, plus the paired bootstrap; report the pair. This answers "one fold, many variants".
7. **Do not spend time on:** more motor feature corrections, region, new-business multiplier, stacking with emerging features, further tuning of the GBM, chasing the public-score gap.
8. **Submissions:** p2_t1 (0.16077) stays the reference. Next candidate = recommendation 1 + 2 (+3 when ready). Judge it by both-direction fold E; the public score only checks the level.
9. **For the report (30 points of judging sit on the story):** the structure finding (relativities x peril frequency multipliers x line severity scale + gender), the level-first insight (0.014 vs 0.003), the rejected list with evidence, and the uncertainty table. These are the "adaptation to the emerging market" evidence.

## 4. Tools and workflow ideas that came out of this session

* A **both-directions validation** and an **A/E by feature screen** (`p2_explore_ae.py`) are reusable, cheap and good evidence for the validation criterion.
* Ask a second model family to propose the "what differs between markets" list from the raw tables and compare with section 1 (verification evidence).
* Set up the background NN job as a resumable queue (done) and run it on a GPU notebook if restarts continue to kill it (it has lost hours twice).
* Keep a one-page "market difference table" (section 1) as the first exhibit of the report; the video can show it in 30 seconds.
