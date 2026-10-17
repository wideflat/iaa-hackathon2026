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
