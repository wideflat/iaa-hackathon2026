# Model card: Part 2 emerging-market risk premium

Status: candidate t3 (GBM + TabM, robust level, gender factor). Written 2026-10-24 (EDT); numbers are from RESULTS 45-50.

**Purpose.** Predict, for each 2025 emerging-market policy row (motor, buildings, contents), the expected ultimate claim cost (net of excess and recoveries, excluding declined claims): the *risk premium*, before expenses, tax, reinsurance and profit. The office premium for the 20 reference policies is built from it in the report (`src/iaa/office_premium.py`).

**Method in one paragraph.** A frequency x severity model per peril (LightGBM, and TabM nets, Part 1 recipe) is trained on the developed book (2021-24) using only variables both markets record. It scores every emerging row. The emerging book then sets (a) a frequency factor per peril from claim counts, (b) a severity factor per peril from winsorised claim means against the developed claims, both credibility-weighted towards the line value, and (c) a gender factor in each line; the developed trend and the claim-reporting completeness are carried over. No emerging-only machine-learning model is used for the final level or relativities.

**Data.** Developed: 980k / 708k / 722k policy rows, 163k claims. Emerging: 47k / 27k / 35k rows (2023-24), 9.3k claims, claims valued at 2024-12-31. `DATA_REVIEW.md`, `VARIABLES.md`.

**Inputs.** Shared variables only (`VARIABLES.md`). Never used: `customer_service_contacts` (outcome-generated), `renewal_decision`, noise fields, claim history, region (different code sets), exact age (replaced by the five age bands), mileage, telematics, credit band. Gender is not a model input; it enters as a factor fitted on the emerging book (decision D57).

**Validation.** Emerging fold E in both directions (calibrate 2023 / score 2024 and the reverse), paired customer-cluster bootstrap. Pooled D2 (Tweedie 1.5) of the final ensemble (3 tree bases, 6 nets, 4 direct Tweedie fits, claim-history inputs, robust level, gender): forward 0.1408, reverse 0.1272; the same figure for an emerging-only GBM is 0.1110 forward (RESULTS 45). Public leaderboard: first file 0.16077, final ensemble 0.16462, best single file 0.16474 (differences among the last seven candidates are noise). A simulation with known truth and a ceiling analysis are in RESULTS 76-78 and 84.

**Known limits and uncertainty.**
* The 2025 test is 76% new business, the validation years 15%: new-business relativity comes from the developed book and is only weakly testable (ratio of observed to predicted new-business/renewal frequency 1.02-1.15 in motor, noisy elsewhere).
* Buildings level: the two emerging years differ by a factor of about 1.5 (fire 49 vs 21 claims). A +-20% error in the buildings level moves the validation D2 by up to -0.011 / +0.007 depending on the direction; motor and contents levels are well pinned (RESULTS 51).
* The nets' level relative to the GBM moves by +-5% between training cut-offs (single seed); the net half is averaged over more than one net before final use.
* Gender rating: allowed for pricing in the emerging market by the competition's data description; its regulatory status outside the competition must be confirmed. A gender-free file is one switch away (`--no-gender`).
* Assumptions not yet tested: emerging construction code MIX read as non-standard (A/E +26%); developed claim-reporting pattern for 2024 (matches in lag percentiles and open share).

**Monitoring triggers (if this were deployed).** Actual / expected claim cost by line and by new-business flag after 6 months and after 12 months; escalate to the pricing committee if outside 0.85-1.15 for a line or if the buildings experience includes a claim above 10x the median; re-estimate the peril factors once a third year exists.

**Governance.** Every decision has a row in `docs/workflow/DECISIONS.md` with its evidence; AI contributions and the errors caught are in `docs/workflow/AI_USAGE.md`; human reviewer / approver names are placeholders to be filled by the team. Reproduction: `scripts/reproduce_part2.sh`. CI runs the tests on every push.
