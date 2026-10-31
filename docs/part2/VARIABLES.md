# Variables: Part 1 vs Part 2 (developed and emerging)

Part 1 data = Part 2 `developed/` folder (byte-identical). So there are two column sets: **developed** and **emerging**. Source: parquet schemas (`data/part2/*`), `src/iaa/features.py`, `src/iaa/market.py`.

Legend: **yes** = present · **no** = not in the file · **band** = only as an age band.
"Part 1 model" = Part 1 final blend. "Part 2 model" = the transferred developed model on shared features (`market.to_common`) + the emerging-only gender factor.
Use codes: **F** feature · **X-leak** never (outcome-generated) · **X-noise** excluded as noise (`NOISE_COLS`) · **X-label** training label only · **ID** identifier/date · **X-mkt** market-specific, left out of the shared model · **X-hist** claim-history input, unusable in the emerging market · **adj** feature after a harmonisation step · **G** emerging-only calibration factor.

**General treatment (applies to every feature row below):** frequency x severity per peril. Frequency is Poisson with a log(`exposure_calc`) offset and a completeness offset for immature years; severity is Gamma on claim cost capped at the 99.9% quantile of the peril and trended to the predicted year; the cost above the cap is added back as an excess load. LightGBM (4 leaves): categorical variables are replaced by the out-of-fold smoothed claim rate of their level; numeric and boolean variables enter as they are and missing values are left to the trees. Neural nets (TabM, TabM-PLE): numerics are winsorised at the 0.5%/99.5% quantiles, missing values get the median plus a missing flag, then z-scored (PLE adds quantile-bin encodings); categorical levels with fewer than 200 rows share one "other" embedding. Predictions are then rescaled to the emerging level (per-line closure, per-peril credibility factors) and multiplied by the gender factor.

## 1. Policy rows, all three lines (`policies_<line>`, `test_<line>`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model | How it is treated |
|---|---|---|---|---|---|---|
| row_id, policy_id, customer_id, address_id | yes | yes | ids; ids are not comparable across markets | ID | ID | Never a feature. row_id keys predictions; customer_id groups the bootstrap; policy_id and address_id join claims, customers, addresses, weather. |
| period_start, period_end | yes | yes | | ID (exposure, year) | ID | Not a feature. Give `exposure_calc`, the policy `year` (trend, completeness) and the row's days in force. |
| exposure | yes | yes | supplied column is corrupted on some rows (2-3% developed, 0.5% emerging); `exposure_calc` = days/365 capped at 1 is used | offset via `exposure_calc` | same | Not used as supplied. `exposure_calc` = days/365 (cap 1) is the log offset of the frequency model. |
| voluntary_excess, compulsory_excess | yes | yes | compulsory_excess is constant per line in both markets (250 motor, 350 buildings and contents): no information | F | F | Voluntary excess: numeric feature. Compulsory excess is constant per line: carries nothing. |
| channel | yes | yes | direct / comparison_site / broker; same mix in emerging train and test | F | F | Categorical: target-encoded (GBM), embedded (NN). |
| payment_method | yes | yes | | F | F | Categorical, as channel. |
| missed_payments_12m | yes | yes | | F | F | Numeric, as is. |
| new_business | yes | yes | **76% of emerging 2025 rows** vs 15% developed | F | F | Boolean feature. Its relativity comes from the developed book; no emerging-specific multiplier (D58). |
| renewal_count | yes | yes | | F | F | Numeric, as is. |
| days_quote_to_inception | yes | yes | | F | F | Numeric, as is. |
| discount_applied | yes | yes | | F | F | Numeric, as is. |
| written_premium | yes | yes | insurer's current price | F | F | Numeric, as is (not deflated). |
| customer_service_contacts | yes | yes | outcome-generated (claims cause contacts): corr with claim count 0.5-0.8, level halves in 2025 rows | **X-leak** | **X-leak** | Dropped before modelling (`OUTCOME_LEAK_COLS`); a test fails if it appears as a feature. |
| renewal_decision | yes (train only) | yes (train only) | | X-label | X-label | Dropped (training files only). |

## 2. Motor (`policies_motor`, `test_motor`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model | How it is treated |
|---|---|---|---|---|---|---|
| driver_age | yes | **no** | exact age | F | replaced by `age_band` (cut at the emerging band edges) | Not in the Part 2 model: cut into `age_band` for the developed book. |
| driver_age_band | no | yes | 17-24, 25-34, 35-49, 50-64, 65+ | n/a | F as `age_band` | Used as the categorical `age_band` (the five bands); in the developed book `age_band` is cut from the exact age at the same edges. |
| years_licensed, ncb_years | yes | yes | | F | F | Numeric, as is. |
| vehicle_value_group, vehicle_age, make_model, engine_size_cc, body_type, transmission, seats, fuel_type, driver_assist, cover_type, overnight_parking, vehicle_use, named_young_driver | yes | yes | vehicle_age, engine_size_cc, overnight_parking ~12% missing in emerging (2.5% developed), missing at random | F | F | Numeric / categorical as typed; `make_model` (high cardinality) target-encoded; trees take missing values natively (12% in emerging), nets impute the median plus a missing flag. |
| vehicle_colour | yes | yes | noise | X-noise | X-noise | Dropped (noise list). |
| annual_mileage | yes | **no** | not collected in the emerging market | F | X-mkt | Not in the shared model (absent in emerging); the developed model is trained without it. |
| telematics_opt_in, telematics_score | yes | **no** | not offered in the emerging market (score 88% blank in developed) | F | X-mkt | Not in the shared model; trained without it. |

## 3. Buildings (`policies_buildings`, `test_buildings`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model | How it is treated |
|---|---|---|---|---|---|---|
| policyholder_age | yes | **no** | exact age | F | replaced by `age_band` | Not in the Part 2 model: cut into `age_band`. |
| policyholder_age_band | no | yes | same five bands | n/a | F as `age_band` | Used as the categorical `age_band` (the five bands); in the developed book `age_band` is cut from the exact age at the same edges. |
| sum_insured | yes | yes | x1000 unit error on 0.2% of rows in both markets (repaired) | F | F | x1000 error repaired (value > 20 x median is divided by 1000, bands enforced); numeric feature. |
| construction_type | yes | yes (**different codes**) | developed: brick/stone/timber(W,T)/non_standard with case variants; emerging: BRK, STN, WOD, MIX | F (cleaned) | F (emerging mapped BRK->brick, STN->stone, WOD->w, MIX->non_standard; MIX is an assumption) | Cleaned: lower-cased, T merged into W; emerging codes mapped BRK->brick, STN->stone, WOD->w, MIX->non_standard; categorical. Mapping sensitivity not yet tested. |
| occupancy, accidental_damage_cover, mortgage | yes | yes | | F | F | Categorical / boolean, as is. |
| days_unoccupied, smoke_alarms | yes | yes | ~12% missing in emerging (2.5% developed) | F | F | Numeric / boolean; missing values handled natively by trees (12% missing in emerging). |

## 4. Contents (`policies_contents`, `test_contents`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model | How it is treated |
|---|---|---|---|---|---|---|
| policyholder_age / policyholder_age_band | yes / no | no / yes | as buildings | F | `age_band` | As buildings: developed exact age cut into `age_band`; emerging band used directly. |
| sum_insured | yes | yes | x1000 unit error on 0.2% of rows (repaired) | F | F | x1000 error repaired (value > 20 x median is divided by 1000, bands enforced); numeric feature. |
| tenure, accidental_damage_cover, works_from_home, lodgers, bicycle_cover | yes | yes | | F | F | Categorical / boolean, as is. |
| security_level, high_value_items | yes | yes | ~12% missing in emerging | F | F | Categorical / numeric; missing handled natively by trees. |

## 5. Customers (`customers`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model | How it is treated |
|---|---|---|---|---|---|---|
| customer_id | yes | yes | | ID | ID | Join key only. |
| date_of_birth | yes | **no** | gives `customer_age` | F (customer_age) | X-mkt | Not in the Part 2 model (age enters as `age_band`). |
| gender | **no** | yes (M/F, balanced) | allowed for pricing in the emerging market | n/a | **G** (credibility-weighted factor, +0.0032 D2; see D57) | Not in the transferred model. Applied afterwards as a multiplicative factor per line fitted on emerging 2023-24 (credibility k = 300 claims; final factors M/F: motor 1.125/0.875, buildings 1.013/0.988, contents 0.913/1.085. `--no-gender` removes it. |
| credit_band | yes | **no** | A (best) to E | F | X-mkt | Not in the shared model; the developed model is trained without it. |
| occupation | yes (20 clean values) | yes (free text, ~60 spellings) | emerging: case, blanks, "Self-employed <occupation>" | F | F as normalised occupation + `self_employed` flag (adj) | Trimmed, lower-cased, 'self-employed ' prefix stripped (20 base values) plus a `self_employed` flag; categorical. |
| marital_status, household_size, prior_claims_5y, licence_points, years_at_address, customer_since_year | yes | yes | | F | F | Numeric / categorical, as is. |
| surname, email_domain, marketing_consent, document_preference, previous_insurer, campaign_code, branch_id, marketing_segment | yes | yes | noise / marketing artefacts | X-noise | X-noise | Dropped (noise list). |

## 6. Addresses (`addresses`) and weather (`weather`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model | How it is treated |
|---|---|---|---|---|---|---|
| region | R01-R12 | **E01-E06** | different code sets, not comparable | F | X-mkt (tested in the emerging residual: no gain) | Not in the shared model (different code sets). Tested as an emerging residual: no gain. |
| urban_density, flood_zone, distance_to_river_km, elevation_m, soil_type, year_built, property_type, roof_type, basement, trees_within_5m, listed_building, bedrooms, energy_rating, solar_panels | yes | yes | same schema, near-identical distributions | F | F | Numeric / categorical / boolean, as is. |
| weather: hail_days, wind_exposure_index, freezing_days, dry_summer_index, wildfire_index, rainfall_mm, mean_temperature_c | years 2021-2025 | years 2023-2025 (2025 is the test year) | | F (year of the row) | F (year of the row) | Joined on address and the row's policy year; numeric. |
| lag-1 weather (previous year's values) | derived | derived, but empty for 2023 rows | | F | X-hist | Dropped (empty for 2023 rows in the emerging book). |

## 7. Claims (`claims`) and derived inputs

| Variable | Developed | Emerging | Note |
|---|---|---|---|
| claim_id, policy_id, line, peril, loss_date, report_date, status, paid_to_date, incurred_to_date, recoveries, repairer_id, third_party_id | yes | yes | same schema; target = incurred - recoveries (open claims: expected recoveries), excluding repudiated; valued 2024-12-31 in both markets |
| duplicates / placeholder loss date 2020-12-01 | 604 / 252 | **0 / 0** | emerging has 36 claims dated before the policy starts instead |
| Derived: hist_n_claims, hist_cost, hist_n_last_year, cust_n_*, addr_n_* (claim history) | Part 1 model: F | empty for every 2023 row (book starts 2023) | Part 2 model: X-hist |
| Derived: customer_age, period_days, start_month, end_month | customer_age F; timing X | no customer_age; timing X | |
| Derived: age_band, occupation (normalised), self_employed | n/a | n/a | Part 2 model F; built by `market.to_common` in both markets |

## 8. Counts

| | Developed | Emerging |
|---|---|---|
| Training rows motor / buildings / contents | 979,968 / 707,726 / 721,617 (2021-24) | 47,283 / 27,447 / 34,566 (2023-24) |
| Test rows motor / buildings / contents | 254,986 / 184,155 / 187,498 | 90,521 / 52,295 / 66,247 |
| Claims | 163,150 | 9,258 |
| Columns, policy files (motor / buildings / contents) | 39 / 27 / 28 | 36 / 27 / 28 |
| Customer / address / weather columns | 18 / 16 / 9 | 17 / 16 / 9 |
