# Variables: Part 1 vs Part 2 (developed and emerging)

Part 1 data = Part 2 `developed/` folder (byte-identical). So there are two column sets: **developed** and **emerging**. Source: parquet schemas (`data/part2/*`), `src/iaa/features.py`, `src/iaa/market.py`.

Legend: **yes** = present · **no** = not in the file · **band** = only as an age band.
"Part 1 model" = Part 1 final blend. "Part 2 model" = the transferred developed model on shared features (`market.to_common`) + the emerging-only gender factor.
Use codes: **F** feature · **X-leak** never (outcome-generated) · **X-noise** excluded as noise (`NOISE_COLS`) · **X-label** training label only · **ID** identifier/date · **X-mkt** market-specific, left out of the shared model · **X-hist** claim-history input, unusable in the emerging market · **adj** feature after a harmonisation step · **G** emerging-only calibration factor.

## 1. Policy rows, all three lines (`policies_<line>`, `test_<line>`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model |
|---|---|---|---|---|---|
| row_id, policy_id, customer_id, address_id | yes | yes | ids; ids are not comparable across markets | ID | ID |
| period_start, period_end | yes | yes | | ID (exposure, year) | ID |
| exposure | yes | yes | supplied column is corrupted on some rows (2-3% developed, 0.5% emerging); `exposure_calc` = days/365 capped at 1 is used | offset via `exposure_calc` | same |
| voluntary_excess, compulsory_excess | yes | yes | compulsory_excess is constant per line in both markets (250 motor, 350 buildings and contents): no information | F | F |
| channel | yes | yes | direct / comparison_site / broker; same mix in emerging train and test | F | F |
| payment_method | yes | yes | | F | F |
| missed_payments_12m | yes | yes | | F | F |
| new_business | yes | yes | **76% of emerging 2025 rows** vs 15% developed | F | F |
| renewal_count | yes | yes | | F | F |
| days_quote_to_inception | yes | yes | | F | F |
| discount_applied | yes | yes | | F | F |
| written_premium | yes | yes | insurer's current price | F | F |
| customer_service_contacts | yes | yes | outcome-generated (claims cause contacts): corr with claim count 0.5-0.8, level halves in 2025 rows | **X-leak** | **X-leak** |
| renewal_decision | yes (train only) | yes (train only) | | X-label | X-label |

## 2. Motor (`policies_motor`, `test_motor`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model |
|---|---|---|---|---|---|
| driver_age | yes | **no** | exact age | F | replaced by `age_band` (cut at the emerging band edges) |
| driver_age_band | no | yes | 17-24, 25-34, 35-49, 50-64, 65+ | n/a | F as `age_band` |
| years_licensed, ncb_years | yes | yes | | F | F |
| vehicle_value_group, vehicle_age, make_model, engine_size_cc, body_type, transmission, seats, fuel_type, driver_assist, cover_type, overnight_parking, vehicle_use, named_young_driver | yes | yes | vehicle_age, engine_size_cc, overnight_parking ~12% missing in emerging (2.5% developed), missing at random | F | F |
| vehicle_colour | yes | yes | noise | X-noise | X-noise |
| annual_mileage | yes | **no** | not collected in the emerging market | F | X-mkt |
| telematics_opt_in, telematics_score | yes | **no** | not offered in the emerging market (score 88% blank in developed) | F | X-mkt |

## 3. Buildings (`policies_buildings`, `test_buildings`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model |
|---|---|---|---|---|---|
| policyholder_age | yes | **no** | exact age | F | replaced by `age_band` |
| policyholder_age_band | no | yes | same five bands | n/a | F as `age_band` |
| sum_insured | yes | yes | x1000 unit error on 0.2% of rows in both markets (repaired) | F | F |
| construction_type | yes | yes (**different codes**) | developed: brick/stone/timber(W,T)/non_standard with case variants; emerging: BRK, STN, WOD, MIX | F (cleaned) | F (emerging mapped BRK->brick, STN->stone, WOD->w, MIX->non_standard; MIX is an assumption) |
| occupancy, accidental_damage_cover, mortgage | yes | yes | | F | F |
| days_unoccupied, smoke_alarms | yes | yes | ~12% missing in emerging (2.5% developed) | F | F |

## 4. Contents (`policies_contents`, `test_contents`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model |
|---|---|---|---|---|---|
| policyholder_age / policyholder_age_band | yes / no | no / yes | as buildings | F | `age_band` |
| sum_insured | yes | yes | x1000 unit error on 0.2% of rows (repaired) | F | F |
| tenure, accidental_damage_cover, works_from_home, lodgers, bicycle_cover | yes | yes | | F | F |
| security_level, high_value_items | yes | yes | ~12% missing in emerging | F | F |

## 5. Customers (`customers`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model |
|---|---|---|---|---|---|
| customer_id | yes | yes | | ID | ID |
| date_of_birth | yes | **no** | gives `customer_age` | F (customer_age) | X-mkt |
| gender | **no** | yes (M/F, balanced) | allowed for pricing in the emerging market | n/a | **G** (credibility-weighted factor, +0.0032 D2; see D57) |
| credit_band | yes | **no** | A (best) to E | F | X-mkt |
| occupation | yes (20 clean values) | yes (free text, ~60 spellings) | emerging: case, blanks, "Self-employed <occupation>" | F | F as normalised occupation + `self_employed` flag (adj) |
| marital_status, household_size, prior_claims_5y, licence_points, years_at_address, customer_since_year | yes | yes | | F | F |
| surname, email_domain, marketing_consent, document_preference, previous_insurer, campaign_code, branch_id, marketing_segment | yes | yes | noise / marketing artefacts | X-noise | X-noise |

## 6. Addresses (`addresses`) and weather (`weather`)

| Variable | Developed | Emerging | Note | Part 1 model | Part 2 model |
|---|---|---|---|---|---|
| region | R01-R12 | **E01-E06** | different code sets, not comparable | F | X-mkt (tested in the emerging residual: no gain) |
| urban_density, flood_zone, distance_to_river_km, elevation_m, soil_type, year_built, property_type, roof_type, basement, trees_within_5m, listed_building, bedrooms, energy_rating, solar_panels | yes | yes | same schema, near-identical distributions | F | F |
| weather: hail_days, wind_exposure_index, freezing_days, dry_summer_index, wildfire_index, rainfall_mm, mean_temperature_c | years 2021-2025 | years 2023-2025 (2025 is the test year) | | F (year of the row) | F (year of the row) |
| lag-1 weather (previous year's values) | derived | derived, but empty for 2023 rows | | F | X-hist |

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
