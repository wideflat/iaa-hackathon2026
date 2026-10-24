# Part 2 strategy: emerging market, workflow and judging criteria

Written 2026-10-24 (EDT). Part 2 closes 2026-10-31 03:59 UTC = **2026-10-30 23:59 EDT**; 10 Kaggle submissions per day. Facts below come from
`DATA_REVIEW.md` and `docs/experiments/RESULTS.md` sections 45-46; anything not measured is marked as an assumption.

## 1. What is scored, and what we hold

| Criterion | Weight | Where we stand on 2026-10-24 |
|---|---|---|
| Pricing performance & adaptation | 40% | Part 1 file final (private rank 3 of 54, 0.13864). Part 2: transfer model validated on emerging fold E (pooled D2 0.1265 vs 0.1110 for an emerging-only model); first prediction file built, not yet submitted. Part 2 score = Kaggle D2 on the 209,063 emerging test rows; the panel also reads the adaptation story and the uncertainty statement. |
| Effective use of AI | 25% | Strong log from Part 1 (`workflow/AI_USAGE.md`: leak found, 4 planted defects found, helper sessions, verification of every claim). Part 2 additions needed: AI used for the market-difference analysis with independent verification (section 6). |
| Trust, validation, guardrails | 20% | 102 automated tests, CI on every push (`.github/workflows/ci.yml`), fail-fast data bands, leak alarm, reproducibility assertion for the developed model re-run. Needs: a model card, a human-review log with real names (placeholders now), a one-command reproduction (section 7). |
| Communication & documentation | 15% | Docs are extensive but long; the graded part is the **first 5 pages** and the **first 5 minutes**. Needs the report and the video script (section 8). |

## 2. What changed between the markets (the adaptation story)

1. **Levels:** emerging motor has +34% claim frequency and -42% mean claim cost; buildings similar frequency and -8% cost; contents -15% frequency, -10% cost (`DATA_REVIEW` section 2). The developed model overprices emerging motor by ~30% (calibration closure 1.0 after the per-line step, raw ratio 0.70).
2. **Information:** exact ages, mileage, telematics, credit band missing; gender, free-text occupation, local construction codes added; history starts in 2023; 12% of selected fields missing at random (`DATA_REVIEW` sections 4-5).
3. **Population:** the 2025 test book is 76% new business against 14% in training; same channel mix. The new-business relativity (motor +45% in both markets) therefore drives the 2025 level, and the emerging data cannot estimate it (315-382 new-business motor claims per year).
4. **Data quality:** the Part 1 planted defects are absent (0 duplicates, 0 placeholder dates) but the same families reappear in new forms: sum_insured x1000 on 0.2% of rows, free-text occupation, local construction codes, 36 pre-inception claims, supplied exposure column wrong on 0.5% of rows, customer_service_contacts leak (corr 0.77 with claim count in training, half the level in the test rows).
5. **What did not change:** report-lag pattern and open-claim share (completeness), severity trend within the noise, the relativities that matter (the developed model beats the emerging-only model on all three lines).

## 3. Modelling strategy

**Base (adopted, D56-D59):** the Part 1 recipe trained on the developed book on the features both markets share (`src/iaa/market.py`), scored on every emerging row; then per-line level closure (+ mild per-peril credibility factors, k = 300 claims) and a credibility-weighted gender factor fitted on the emerging 2023-24 rows (`src/iaa/transfer.py`, `scripts/p2_make_submission.py`).

| Idea | Evidence | Status |
|---|---|---|
| Emerging-only GBM / GLM | fold E pooled D2 0.1110 / far worse | rejected as the base; GBM kept as a challenger |
| Developed model, no calibration | 0.1254 (level wrong, relativities right) | superseded |
| Peril-level cost factors | nil over line level; k = 30 worse | mild k = 300 kept for transparency |
| Frequency / severity split calibration | worse than cost factors | rejected |
| New-business multiplier from emerging data | worse; estimated 1.57 in 2023, 0.9 in 2024 | rejected (D58) |
| Region and other emerging-only features | residual GBM -0.0013..-0.0054 | rejected |
| Gender factor | +0.0032 (95% +0.0013..+0.0055), stable across years | adopted (D57) |
| NN half (TabM, TabM-PLE nets of Part 1) on the developed book | Part 1 gain +0.004 D2; running in the background (`scripts/queue_p2_nn.sh`) | pending: adopt only if fold E paired difference is positive |
| Stacking with emerging features / blends with the emerging-only GBM | +0.004 stack, blends lower than `dev_cal` alone | rejected |
| Claim-history inputs (`hist_*`), lag-1 weather | unusable (book starts 2023; 76% of test rows are new business) | excluded; `prior_claims_5y` and `ncb_years` carry the information |

**Next experiments, in order (stop rule: a change is adopted only with a positive paired difference on fold E and no line worse than -0.002):**
1. NN half of the base (running), then the Part 1 per-line NN weights 0.7 / 0.4 / 0.3 as the prior.
2. `hist_n_last_year` (claims in the previous year) as a renewal-only input: needs the developed model refit with the feature (45 min of compute); expected small because 76% of the rows are new business.
3. Customer-grouped cross-validation on 2023-24 together (a second validation design, section 4) to confirm the ranking `dev_cal` > emerging-only and the gender factor.
4. Level scenarios for buildings (two emerging years disagree: fire 49 vs 21 claims).

## 4. Validation design and its limits

* **Fold E (primary):** train 2023, validate 2024 on 59,202 emerging rows; paired cluster bootstrap on customers; the developed completeness table makes the immature 2024 year comparable. Only the **relativity and level transfer** is measured here.
* **What fold E cannot measure:** the 76% new-business share of 2025 (2024 validation has 15%), the 2025 trend step and buildings catastrophe years. These are treated as scenarios, not as validated results.
* **Second design (to run):** customer-grouped 5-fold CV over 2023-24 for the emerging-only challengers, as a check on fold E.
* **Leak guards:** `customer_service_contacts` excluded (same leak, DATA_REVIEW 5.9); a pooled D2 above 0.35 fails the run; the developed model re-run is asserted to reproduce the first run (1e-6).
* **Public score:** Week 2 leaderboard, 10 submissions per day, noise SD of paired differences 0.0003-0.001 in Part 1; used only as a sanity check on the level, never to choose between candidates closer than that (D20, D21).

## 5. Uncertainty statement (for the report)

| Source | Size | How it enters |
|---|---|---|
| Buildings level (two years, fire / subsidence luck) | 2023 vs 2024 total differs by ~40% | scenario table: level x0.8 / x1.0 / x1.2 for buildings; report D2 sensitivity on fold E |
| New-business relativity | developed motor +45%; emerging ratio 1.0 (0.9..1.15) | scenario: relativity x0.9 / x1.0 / x1.1 on new-business rows |
| Severity trend | emerging intervals 0-10% | scenario: trend 0% / model / +3 points |
| Completeness of 2024 | borrowed from developed, patterns match | sensitivity: +-2 points on 2024 factors |
| Gender factor | 0.905 / 1.096 (motor, k = 300) | file without gender kept (`--no-gender`) |

Each scenario is a re-run of `scripts/p2_make_submission.py` with a documented switch and is reported as a change in the total predicted cost and in the fold-E D2.

## 6. Effective use of AI (25%): what to show

* **Throughout the workflow:** reading the competition pages (headless browser), EDA and profiling, writing the harmonisation and transfer code with tests first, running experiments, drafting documents.
* **Demonstrated improvement:** (a) the transfer decision (D56) came from a pre-registered comparison, not from preference; (b) the audit of the emerging data reused the Part 1 checklist and covered all ten checks of `DATA_REVIEW` section 5 in one working session, where the Part 1 defects were found only after three days of modelling.
* **Rigorous verification of AI output (ledger in `workflow/AI_USAGE.md`):** every number in the documents is traced to a command or a log line. Three AI drafting errors in the first version of `DATA_REVIEW.md` were caught by re-reading it against fresh queries before it was committed: an unmeasured explanation of the mean exposure (the raw column mean of 1.15 was inflated by a few corrupted rows; `exposure_calc` means are 0.82-0.86), invented new-business row counts (replaced by the measured 2023 counts 2,620 / 1,507 / 1,847), and the statement that the customer-service-contacts leak "has the same pattern" (measured: correlation 0.77 / 0.59 / 0.51 with the claim count, mean 0.81 in training vs 0.40 in 2025 rows).
* **Independent second opinion (to do):** give a second model family (outside Claude) the raw tables and the question "what differs between the markets?" without our conclusions, and compare the list with ours. Report agreements and misses. This is the cheapest strong evidence of verification.
* **Human review checkpoints:** the project lead approves every submission and every decision row; names of the reviewer / approver stay placeholders until the team provides them (do not invent).

## 7. Trust, validation, guardrails (20%): to do

1. `docs/part2/MODEL_CARD.md`: purpose, data, features and exclusions, calibration, known limits, monitoring triggers (for example, actual / expected by line and by new-business flag after 6 months; escalation if outside 0.85-1.15).
2. `scripts/reproduce_part2.sh`: from the downloaded data to the submission file in one command (frames, developed predictions, submission), with the sha256 of inputs and outputs printed.
3. Data contracts: checks for the emerging tables (column set, dtypes, value domains, the repair bands) as tests on small fixtures.
4. Submission validator (exists: `iaa.submission.validate_submission`) plus level checks printed by `p2_make_submission.py` (predicted vs actual per exposure, vs written premium).
5. Governance: gender rating decision record (D57) including the regulatory caveat; the human-in-the-loop sign-off list.

## 8. Communication (15%): report and video

* **Report (first 5 pages are graded):** p1 summary and result table (Part 1 rank, Part 2 fold E, public score); p2 what changed in the new market (table from section 2); p3 method (transfer + calibration + gender, with the rejected list); p4 validation, guardrails, AI use with verification; p5 uncertainty, office-premium build-up for the 20 reference policies (`src/iaa/office_premium.py`), limits. Everything else in an appendix.
* **Video (first 5 minutes are graded):** 0:00 problem and result; 0:45 what changed (three numbers: frequency, severity, 76% new business); 1:45 method in one picture; 2:45 how AI was used and checked (one concrete catch); 3:45 guardrails and uncertainty; 4:30 what we would do next. Record the screen with the repo and the result table open.
* **Repository:** README with a 10-line map; `docs/README.md` index extended with `part2/`; `EVIDENCE_MAP.md` maps each rubric bullet to files and commits so a reviewer or an AI screener finds it in one hop.

## 9. Tools beyond Claude Code that would raise quality or speed

| Need | Tool | Why / how |
|---|---|---|
| Independent second opinion on methodology and on the data differences | another assistant family (Gemini / ChatGPT / a local model) with the raw tables | verification evidence; ask it to attack the transfer assumptions |
| Parallel compute | helper Claude sessions already used; a GPU notebook (Kaggle / Colab) for the NN base | the nets are the slowest step (hours on 4 CPU cores) |
| Experiment tracking | keep `experiments/log.jsonl` (append-only, rendered); optional MLflow if the team wants a UI | no new dependency is needed to meet the criterion |
| Data validation | `pandera` or `great_expectations` schemas for the emerging tables | executable data contracts as evidence for the guardrails criterion; the same checks already exist as assertions |
| Explainability | SHAP for the GBM and a relativity table (GLM-style) comparing developed and emerging | explains "what carried over" to actuaries |
| Report | Markdown -> PDF with pandoc or Quarto, figures from matplotlib; page limit check | the 5-page limit is easy to break |
| Video | OBS / Loom screen recording, script drafted by AI and cut to time; captions | the first 5 minutes are graded, so rehearse to 4:30 |
| Rubric self-check | an LLM given the rubric table, the report and the video script; ask for missing evidence per bullet | AI screeners are the first reviewers; make their job easy |
| CI | GitHub Actions (added): tests on every push | automated-tests evidence with a public badge |
| Pre-commit | ruff + a secret / large-file check | governance evidence; low effort |

## 10. Timeline to the deadline

| When (EDT) | Do |
|---|---|
| Oct 24-25 | submit candidate T1 (with the project lead's go-ahead); NN base finishes; fold E for the NN half; customer-grouped CV |
| Oct 26 | pick the candidate; scenario runs; model card; reproduction script; data-contract tests |
| Oct 27 | report draft (5 pages) + office-premium exhibit; second-opinion run; AI-use ledger |
| Oct 28 | video script, recording, edit; evidence map; README |
| Oct 29 | final candidate selection on fold E + public score sanity; freeze; submit; final form |
| Oct 30 | buffer; form submission before 23:59 EDT (final submission form link not yet provided) |
