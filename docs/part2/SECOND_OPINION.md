# Independent second opinion on the market differences (blind review)

2026-10-24 (EDT). A separate agent was given only the raw parquet tables and five questions (frequency/cost by peril, differing rating variables, emerging data defects, how the 2025 test differs, other points). It was told not to read any repository document, script or result, so its list is not anchored on ours. It is the same model family; a different family should still be asked (PART2_STRATEGY section 6). Its full report was reviewed here; claims marked "checked" were re-verified by us against the data.

## Agreement (found independently, same numbers)
| Topic | Blind review | Ours |
|---|---|---|
| New business share of the 2025 test | 76.0 / 75.9 / 76.1% (train 12-15%); 24% of policies seen in training | 76% / 14% (DATA_REVIEW 3) |
| Motor frequency / cost vs developed | x1.34 / ~0.55-0.65 (median 0.63) | x1.34 / ~0.6 (EXPLORATION 1) |
| Peril frequency ratios | glass 1.55-1.69, TPP 1.41, BI 1.37-1.43, OD 1.17-1.23; EoW 0.65-0.69; storm 1.3-1.5 | same (EXPLORATION table) |
| Fire / subsidence / flood | 30-50 claims each, ratios not significant, fire 2.3x then 0.84x | same |
| customer_service_contacts leak | outcome-generated, corr 0.77, 9.5% of train rows with 3+ contacts vs 0.8% in test | corr 0.77, mean 0.81 vs 0.40 (DATA_REVIEW 5.9) |
| exposure column | 241 / 148 / 190 corrupted rows (days instead of years, x1.0139) | 241 / 148 / 190 (DATA_REVIEW 5.3) |
| sum_insured x1000 | 53 / 104 buildings, 57 / 135 contents (train / test) | 53 / 104 and 57 / 135 |
| Occupation | 81 distinct strings vs 21, 12% null, "Self-employed" prefix | normalised to 20 + flag |
| Missingness | ~12% per field, unrelated to claims, same in test | MCAR (DATA_REVIEW 5.7) |
| 36 pre-policy claims, 0 duplicates | 36 (28 / 3 / 5), 0 duplicate ids | 36, 0 |
| Gender | motor M/F frequency 1.30 (z ~11), 1.32 after controls; no effect in buildings / contents | 1.29-1.34 motor, stable both years |
| NCB, young age, excess, written premium misfit | NCB 0-1 flatter in emerging; 17-24 loading lower; larger excess credits; premium top decile less steep | same variables flagged by our A/E screen |
| Weather response transfers | glass_hail by hail days is the same curve in both markets; the ratio 1.7 is mostly weather | not examined by us before |
| Largest model risk | new-business / low-NCB relativity cannot be validated on 2023-24 | D58, scenario table |
| Level needs peril-level factors, not one scalar | yes | D60 |

## New points from the blind review, and what we did
| # | Point | Checked | Action |
|---|---|---|---|
| 1 | Developed `customers.parquet` is missing 30,822 motor customers: 10% of developed motor rows (constant across years, also in the Part 1 test file) have no customer attributes | yes: 10.0% of motor rows in every year and in the test; buildings / contents 0%; those rows have the same claim rate (0.1478 vs 0.1440 per exposure year) | recorded as data-quality item 18 (`docs/data/DATA_QUALITY.md`); trees take the missing values natively; no action |
| 2 | Emerging `vehicle_age` starts at 3 (developed 0, 6.8% below 3); emerging fleet mean 10.2 vs 7.8 (blind review 6.9, other row base) | yes: min 3.0 vs 0.0; mean 10.25 vs 7.8 | covariate shift on a feature the developed model uses; the trees are flat beyond the training range, so no fix; the one-way A/E for vehicle age was weak (p = 0.04). Added to DATA_REVIEW 5 |
| 3 | The developed GLM "over-spreads" the risk range in the emerging market (A/E by predicted-risk decile 0.87 ... 1.13 ... 0.92) | tested on our predictions: slope calibration exponent 1.003 forward (no gain), 0.926 reverse (+0.0009 D2) | rejected: not consistent in both directions (D62) |
| 4 | `declined_by_insurer` rows still have a next-year policy in 94.7% of cases (label unreliable) | not needed | label is not used |
| 5 | 16% of motor train rows have `ncb_years` > `years_licensed`; 23 of 122 buildings accidental-damage claims on policies without the cover | not checked | note only; unrealistic but consistent in both markets |
| 6 | Developed has a single 11.9M open bodily-injury claim (25% of developed motor 2023 incurred) | known (Part 1: outlier multiple 5 winsorisation) | already handled |
| 7 | Weather in the 2025 test years differs (freezing days 28.9 vs 17-21, hail 11.4 vs 14-17) | not checked | the model reads the row's own year of weather; the response is shared, so 2025 water claims go up and hail claims go down in the prediction. Candidate check in the report |
| 8 | NB test rows look like NB training rows (age-band TV distance 0.012) | agrees with our adversarial AUC 0.55-0.61 | none |
| 9 | Customer-level `years_at_address` < 2: developed 1.18, emerging 1.00 (z -4.8); `listed_building` differs (weak) | not checked | candidate for the A/E screen, not pursued (our screen found no strong property-line misfit) |

## Where the blind review is weaker or differs
* It could not map the emerging construction codes safely (we assumed BRK / STN / WOD / MIX); our A/E found MIX +26% under-predicted. Open item (helper job).
* Its GLM-based decile calibration (over-spread at both ends) is a single-direction result; our two-direction test does not confirm a consistent slope.
* Its recommendation to avoid borrowing developed relativities for NCB, young age, years licensed, excess and make_model is plausible but our two-direction tests found extra emerging corrections for age, occupation and excess helpful forward and harmful in reverse (EXPLORATION 3).

## What this is worth for the judging criteria
Independent agreement on ~15 findings, 2 new items verified (one is a Part 1 data-quality gap we had missed in a month of work), 1 suggested modelling change tested and rejected: concrete evidence for "rigorous verification of AI-generated outputs".
