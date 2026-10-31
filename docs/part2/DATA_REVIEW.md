# Part 2 data review: developed folder vs emerging folder

Written 2026-10-24 (EDT). Evidence: `scripts/download_part2.py` (download), `scripts/p2_build_frames.py` (frames), the profiling queries in
the session log (`docs/workflow/DEVELOPMENT_LOG.md`), `tests/test_transfer.py` (harmonisation rules).

## 1. What Part 2 gives us

| Folder | Content | Training years | Test (2025) |
|---|---|---|---|
| `developed/` | **byte-identical to the Part 1 data** (sha256 of `claims`, `policies_motor`, `test_motor` equal to `data/raw/public`) | 2021-24 | 254,986 + 184,155 + 187,498 rows (not scored in Part 2, Week 1 file already final) |
| `emerging/` | new portfolio, own customers / addresses / weather | 2023-24 | motor 90,521, buildings 52,295, contents 66,247 rows (209,063 = sample submission) |
| `emerging/expenses`, `reference_policies` | cost assumptions of both markets (8 items); 20 test rows for the office-premium build-up in the report (judged by the panel, not scored on Kaggle) | | |

The Week 2 leaderboard scores the emerging test rows only. The target is unchanged: ultimate cost of claims with loss date in the row,
net of excess and recoveries, excluding declined claims, as the **risk premium** (expenses, tax, reinsurance and profit are *not* added to the prediction).

## 2. Size and level: the emerging book is 5-10 times smaller and behaves differently

Claims counted by loss year 2023-24, not repudiated, per exposure year (frequency) and mean incurred per claim; exposure is days in force / 365 capped at 1.

| Line | Rows (emerging / developed) | Claims (em / dev) | Frequency em / dev | Mean incurred em / dev | Written premium per row em / dev |
|---|---|---|---|---|---|
| motor | 47,283 / 509,275 | 7,050 / 53,884 | **0.181 / 0.135 (+34%)** | **2,377 / 4,114 (-42%)** | 612 / 577 |
| buildings | 27,447 / 368,133 | 1,056 / 13,395 | 0.045 / 0.045 (same) | 9,052 / 9,878 (-8%) | 358 / 339 |
| contents | 34,566 / 375,137 | 998 / 12,153 | 0.034 / 0.041 (-15%) | 2,589 / 2,886 (-10%) | 128 / 143 |

Reading: motor is the line that moved (more claims, much cheaper claims); buildings frequency and severity are close; contents is a little lighter.
The number of claims decides what can be learned: ~1,000 buildings and ~1,000 contents claims over two years (about 100-400 per peril, fire 70, subsidence 52)
is far too thin to estimate a peril-level severity model on its own, which is why levels will be borrowed from the developed book (strategy in `PART2_STRATEGY.md`).

Perils (emerging, claims 2023-24): motor own_damage 2,712, third_party_prop 2,263, glass_hail 1,350, bodily_injury 450, theft 408;
buildings storm 466, escape_of_water 268, accidental_damage 122, flood 81, fire 70, subsidence 52; contents theft 418, accidental_damage 340, escape_of_water 172, flood 86.

## 3. The 2025 test book is a different population (the key fact)

* **New business share: 76% of the 2025 test rows (all three lines) vs 14-15% of the 2023-24 training rows** (developed test: 15%). Only 24% of the test policies are seen in training.
* Channel mix is unchanged (broker 70%, direct 25%, comparison site 5%, train = test), so the shift is *new vs renewing customers* (renewal_count = 0), not a channel shift.
* New business claims more often in **both** markets: developed motor frequency 0.18-0.22 (new) vs 0.13-0.15 (renewal), +45%; emerging motor 0.25-0.28 vs 0.17-0.20, +35-45% (n = 2,600 new-business rows in 2023, noisy); buildings and contents +5-20%.
  The 2025 level is therefore dominated by the new-business relativity. The emerging data estimates it from the 2023 new-business rows only (motor 2,620, buildings 1,507, contents 1,847), the developed data from 2023 rows of 38,907 / 28,257 / 28,767.
* Mean `exposure_calc` per row is 0.82-0.86 in the emerging training rows and 0.87-0.91 in the test rows (part-year starts); the raw `exposure` column is corrupted on a few rows (see 5.3), which is why the supplied column is not used.

## 4. Schema differences (what we harmonise and what we drop)

| Item | Developed | Emerging | Treatment (`src/iaa/market.py`) |
|---|---|---|---|
| Age | exact `driver_age` / `policyholder_age`, `date_of_birth` | bands 17-24, 25-34, 35-49, 50-64, 65+ (`driver_age_band`, `policyholder_age_band`) | shared `age_band` (developed cut at the band edges); exact ages only in developed-only models |
| Gender | absent | M/F, allowed for pricing in the emerging market (competition text) | emerging-only models; governance decision recorded (D57) |
| Credit band, mileage, telematics (opt-in, score) | present | absent (telematics not offered) | dropped from the shared feature space |
| Occupation | 20 clean values | free text: case variants, trailing blanks, `Self-employed <occupation>` prefix (≈ 60 distinct strings, 20 after normalisation, 8.8% self-employed) | `normalise_occupation` -> 20 base occupations + `self_employed` flag |
| Construction type | brick, stone, timber (W/T) and non_standard (+ case variants fixed in Part 1) | local codes BRK 70%, WOD 12%, MIX 10%, STN 8% | map BRK->brick, STN->stone, WOD->timber, MIX->non_standard (MIX is an assumption: "mixed or non-standard"; sensitivity tested) |
| Region | R01-R12 | E01-E06 | market specific, not shared |
| Weather | 2021-24 | 2023-24 (2025 weather present for test rows) | lag-1 weather unavailable for 2023 rows -> dropped |
| Claim history | 2021-24 | starts in 2023: every 2023 row has empty history by construction | claim-history inputs dropped for Part 2 modelling (`HISTORY_COLS`) |
| Expenses | fixed 45 CU, variable 12%, commission broker 15% / comparison site 10% / direct 0, tax 12%, cat reinsurance 6%, margin 5% | fixed 85, variable 15%, commission broker 20% / direct 0 / **comparison site blank**, tax 8%, cat reinsurance 12%, margin 8% | office-premium build-up (strategy doc); the blank comparison-site commission must be an explicit assumption |

## 5. Data-quality audit of the emerging folder

Same checks as Part 1 (`docs/data/DATA_QUALITY.md`), run on the emerging tables.

| # | Check | Result | Action |
|---|---|---|---|
| 5.1 | Exact duplicate claims (all fields except `claim_id`) | **0** (Part 1: 604) | none; `cleaning` bands for the emerging market are (0, 0), so a duplicate would fail fast |
| 5.2 | Placeholder loss date 2020-12-01 | **0** (Part 1: 252) | none |
| 5.3 | `exposure` vs period dates | supplied exposure disagrees on 0.5% of training rows (motor 241, buildings 148, contents 190); Part 1: 2-3% | `exposure_calc` (days / 365, capped at 1) as in Part 1 |
| 5.4 | `sum_insured` unit error (x1000) | **0.19% of buildings and 0.17% of contents rows**, test rows included (104 + 135); maximum 470,000,000 vs median 229,000 | same repair as Part 1 (divide by 1000 when > 20 x median), band < 1% enforced |
| 5.5 | Claims dated before the policy exists | **36 claims** (loss dates 2021-22, report dates 2023-24: 28 motor, 3 buildings, 5 contents) | they match no policy row and are dropped by the target construction (0.4% of claims); not imputed, flagged |
| 5.6 | Report before loss, duplicate row_id, duplicate claim_id | none | |
| 5.7 | Missing values | 12% of each of vehicle_age, engine_size_cc, overnight_parking (motor), days_unoccupied, smoke_alarms (buildings), security_level, high_value_items (contents), **independently per column** (joint patterns match independent 12% draws) in train *and* test; developed has 2.5%. Motor: claim frequency identical for rows with and without a missing value (0.1809 vs 0.1808): missing completely at random. Buildings: frequency 0.040 with a missing value vs 0.047 without (-14%) | trees use their native missing handling; no imputation by the transfer base model; the buildings difference is tested, not assumed |
| 5.8 | Report-lag pattern and open share (completeness) | 2023: open 3.3% (dev 3.2%), median lag 9 days (dev 9); 2024: open 19.7% (dev 20.2%), median lag 8 (dev 8); lag percentiles per line match | the developed completeness table is applied to the emerging accident years |
| 5.9 | Customer service contacts | **same outcome-generated leak as Part 1**: correlation with the claim count 0.77 / 0.59 / 0.51 (motor / buildings / contents) in the training rows, mean 0.81 in training vs 0.40 in the 2025 rows (motor) | never a feature (leak guard, `OUTCOME_LEAK_COLS`) |
| 5.11 | `vehicle_age` range (blind review, checked) | emerging minimum is 3.0 (developed 0.0, 6.8% of developed rows below 3); mean 10.25 vs 7.8 | covariate shift; trees are flat beyond the training range; no action |
| 5.10 | Renewal decision | same mix as developed (renewed 78%, lapsed 9%, mid-term change 7%, cancelled 5%) from 2024 (2023 has none of lapsed / cancelled because the book starts in 2023) | training label only, never a feature |

## 6. What this means for the model

1. Levels (frequency and severity by peril) must be re-estimated for the emerging market, relativities can be borrowed.
2. The new-business relativity is the single biggest driver of the 2025 level and has to come mostly from the developed book.
3. 2024 accident-year claims are immature in both markets in the same way: the developed completeness factors apply.
4. History features are unusable in the emerging market (book starts 2023, 76% of the test rows are new business); `prior_claims_5y` (declared) and `ncb_years` are the substitutes.
5. Pooled D2 is dominated by motor (mean written premium per row 612 vs 358 buildings and 128 contents), as in Part 1.
