# What exactly is the target? (accident basis, ultimate, valuation)

Status: working interpretation + open questions. Reviewed against standard ratemaking definitions
(calendar / accident / policy / report year; paid, case-incurred, IBNR; development; trend). The
CAS Werner & Modlin chapter could not be read from the sandbox (casact.org blocked): check this file against it.

## 1. Which basis? Accident period, not calendar year
Task text: "the final cost of all claims with a **loss date in the row's period**, net of excess and
recoveries, excluding declined claims". Glossary: "Ultimate: the final cost of claims once all are reported and settled".

| Basis | Definition | Fits the text? |
|---|---|---|
| Calendar year | Payments + change in reserves during 2025, on any accident date | No. It would include development on 2021-24 accidents, which cannot be attributed to a 2025 policy row. |
| Policy year | All losses on policies incepting in 2025 | No. The text selects by loss date, not inception. |
| Accident year | Losses with accident date in 2025 | Yes (and in fact the exposure period of each row: accident date matched to earned exposure). |
| Report year | Claims reported in 2025 | No. |

So the target is **accident-period ultimate cost per exposure row**, i.e. a pure-premium forecast. Needing no 2025 claims
data is not a contradiction: it is a prospective forecast (as in pricing), scored later against a hidden answer.
A calendar-year reading *would* be contradictory (it needs 2025 payments), which is further evidence against it.

## 2. What "ultimate" requires, and what we can and cannot measure
Ultimate = paid + case reserves (= incurred) + IBNR, where IBNR = pure IBNR (not yet reported) + IBNER (development on known claims).

| Component | In the data? | Our treatment |
|---|---|---|
| Paid, case incurred | Yes (single snapshot at 2024-12-31, no triangle) | Net cost = incurred - recoveries |
| Pure IBNR | Estimable from report lags (report_date - loss_date) of mature years 2021-22; lag distributions are stable across years | Completeness factor by line/peril/loss date; est. count uplift 2023: +0.2-0.6%, 2024: +3-6% (cost +3% contents, +8% buildings, +17.5% motor) |
| IBNER (case reserve development) | **No**: one snapshot cannot show how case estimates move | Assumed zero; flagged as sensitivity. Open 2022 bodily-injury claims average 20.5k vs 22.8k closed: no sign of under-reserving |
| Recoveries on open claims | Recoveries exist only on closed own_damage/theft (80% of incurred) | Will arrive later on open claims: unmodelled upward-to-downward adjustment, flag |
| Trend | Severity/frequency drift across years | Not yet explicit. Likely cause of GBM under-calibration (0.64-0.75) on fold A. Testable on mature fold A |

## 3. Open questions (could be asked on the competition forum)
1. At what date and basis is the 2025 answer valued? Includes claims not yet reported at 2025-12-31? Open claims at final settled cost?
2. Are recoveries on open claims included at their eventual amounts?
3. Is the answer the simulator's full claim set (a true "ultimate") or an incurred snapshot at a later date?

## 4. Ways to settle it empirically
- Fold A (2023, ~mature) tests the accident-basis + trend treatment without any ultimate assumption.
- A few public-leaderboard submissions of one model at different global scale factors reveal the level the answer sits at
  (D2 is maximised near the right level). This is 3 numbers, not a fit to the leaderboard, but it is noisy on heavy-tailed lines.

## 5. Reported-claim triangles (built with `src/iaa/development.py`)
Accident quarter x report quarter, volume-weighted age-to-age factors, cumulated to ultimate (as on p.119 of the chapter).
Link ratios are stable across accident quarters (no diagonal / calendar effect visible), so all-year volume-weighted averages are used.

| Line / peril | 3m->6m | 6m->9m | cum. factor to ultimate at 3m / 6m / 12m | implied count uplift AY2023 / AY2024 |
|---|---|---|---|---|
| Motor, all perils | 1.193 | 1.016 | 1.246 / 1.044 / 1.019 | +0.6% / +7.9% |
| **Motor bodily injury** | **2.50** | 1.371 | **4.85 / 1.94 / 1.21** | **+6.1% / +82%** |
| Buildings | 1.178 | 1.011 | 1.207 / 1.025 / 1.009 | +0.3% / +5.8% |
| Contents | 1.127 | 1.000 | 1.134 / 1.006 / 1.006 | +0.2% / +3.5% |

Reading: bodily injury is reported very slowly (only ~21% of claims known at 3 months) and is the heavy-cost peril, so it
drives the motor 2024 shortfall. The 2024 BI factors rest on ~100 claims per quarter and are the least certain number in the project.
This agrees with the day-lag estimate (motor 2024 +6.2% count, +17.5% cost) to within the quarter-alignment difference.

Not estimable from one snapshot: paid or case-incurred development (IBNER), settlement-driven recovery emergence.

## 6. Resolved: the GBM under-calibration was not trend
Earlier (and in the first draft of the research report) the 0.64-0.75 predicted/actual on fold A was attributed to cost trend. Diagnosis:
- The same GBM is already 0.72 (buildings) / 0.81 (motor) calibrated **on its own training data**; out of time it is only 3-4 points worse.
- Dropping hist, weather, premium or all drifting features, or switching to an offset formulation, did not change it.
- The deviance-optimal scalar on the training data is ~1.007: the model is deviance-optimal in-sample. A Tweedie power-1.5 log-link GBM simply does not preserve totals on heavy-tailed data.
- Rebalancing predictions per line to the training total (`LGBMTweedie(rebalance=True)`) lifts fold A from D2 0.0975 to 0.1225 (oracle scalar chosen on the validation year: 0.1242), fold B 0.1200 -> 0.1253.
So trend remains untested rather than established; the first trend estimate should be made on the rebalanced model.

## 7. Draft question for the organizers (competition forum)
"The target is described as the ultimate cost of claims with a loss date in the row's period. (1) At what valuation date and maturity are the 2025 labels measured: all claims eventually reported and settled, or an incurred snapshot at a later date such as 31 Dec 2025? (2) Are open claims at their final cost, and do recoveries on open claims count at their eventual amount? (3) Is the 2025 label built by the same process that produced the 2021-2024 claims (including claims not yet reported at the 2024-12-31 snapshot)?"

## 8. Annual accident-year triangles and LDFs (`scripts/ldf.py` -> `docs/LDF_TABLES.md`)
Textbook layout: accident year x age 12/24/36/48 months, cumulative reported claims at their current net cost (as-if triangle: reporting
emergence only, no IBNER), volume-weighted link ratios over AY2021-24 (AY2021 is fully observed to 48m so it is kept), exponential-decay tail.
CDF to ultimate at 12 / 24 / 36 months and implied uplift of the as-valued amount:

| Group | CDF 12m | CDF 24m | CDF 36m | AY2024 | AY2023 |
|---|---|---|---|---|---|
| Motor, all perils (net cost) | 1.258 | 1.030 | 1.004 | +25.8% | +3.0% |
| Motor bodily injury, net cost, 11.9M claim in | 1.710 | 1.070 | 1.009 | +71.0% | +7.0% |
| Motor bodily injury, cost capped at 250k | 1.812 | 1.070 | 1.009 | see LDF_TABLES.md | |
| Motor excl. bodily injury | 1.052 | 1.001 | 1.000 | +5.2% | +0.1% |
| Buildings | 1.091 | 1.002 | 1.000 | +9.1% | +0.2% |
| Contents | 1.041 | 1.002 | 1.000 | +4.1% | +0.2% |

Findings: (1) the single open 11.9M bodily-injury claim (paid 0.74M, case estimate 11.2M; next largest 1.1M) lowers the AY2023 12-24 link to 1.414
(others 1.77, 1.67) and the BI 12m CDF from 1.81 to 1.71; capping at 100k / 250k / 500k gives a stable 1.82 / 1.81 / 1.80. Large losses are capped before
triangles are built. (2) Sanity check: BI ultimate claim counts implied for AY2024 (1,178 x 1.79 = about 2,110) match AY2021-23 (about 2,150-2,200).
(3) Only 3 / 2 / 1 accident years inform the 12-24 / 24-36 / 36-48 links, so the 24-36 and later factors are thin; the tail beyond 48m is an extrapolation.
