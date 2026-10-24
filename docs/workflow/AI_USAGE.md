# AI-assisted workflow log

Judged criterion: meaningful use of AI across the workflow, with rigorous verification of its output.
Record each significant AI contribution and how a human or a test verified it.

| Stage | What AI did | How it was verified |
|-------|-------------|---------------------|
| Competition reading | Read overview, data, evaluation, rules via the Kaggle API and summarised them. | Spot-checked against `competitions/.../data` file list and metric metadata. |
| EDA / target construction | Wrote the claim-to-row matching and net-cost logic; found the exposure corruption and recoveries pattern. | Unit tests on hand-built examples (`tests/test_target.py`); row/claim counts reconciled (all but ~400 claims match, zero double matches). |
| Leakage detection | First model scored D2 0.55 vs public leaderboard ~0.12; AI traced it to `customer_service_contacts` through feature importance, outcome correlation and train-vs-test distribution comparison. | Re-ran CV without it: D2 0.10, consistent with the leaderboard. Added `drift.py` and a guard test so it cannot recur silently. |
| Metric / bootstrap | Implemented D2 and an exact cluster bootstrap. | Compared to `sklearn.metrics.d2_tweedie_score` to 1e-9 (`tests/test_metrics.py`). |

| Parallel experiments | Delegated three screening grids (23 runs) to helper sessions in separate containers; ran paired analyses locally | Helper baselines reproduced our own runs exactly (0.1320 GBM, 0.1399 GLM); helpers forbidden from committing or submitting; every number re-logged in `experiments/log.jsonl` |
| Evaluation method | Built a paired bootstrap and an outlier-robust score after noticing that fold-A differences were not resolvable | Unit tests (identical predictions give zero difference; real difference detected; paired SE smaller than unpaired); the GLM-vs-GBM conclusion cross-checked on fold B and the leaderboard |
| Claim checking | Research subagents' conclusions (e.g. recovery bias) were re-tested on the data | Corrected the recovery estimate from "several-fold" to ~1.5% after measuring recoveries per claim |
| Incident handling | Logged and fixed: `y_true`/`y_all` leakage into features, Gamma failure on zero-cost claims, cache race, OOM kills | Guard tests, a D2 > 0.35 alarm, and process rules (run heavy jobs one at a time) |

| Data-quality audit (missed, then caught) | For three days the modelling work assumed the tables were clean apart from the leak and the corrupted exposure column; a systematic scan requested by the lead (duplicates, sentinels, unit changes, level-by-year recodes, date consistency) found four more planted defects: 604 duplicate claims, 252 placeholder loss dates, `sum_insured` 1000x on 0.2% of rows (train and test), a recoded `construction_type` | Each defect reproduced with an independent query (e.g. the 11-character claim ids match a 10-character original on all 11 other fields), repaired in `src/iaa/cleaning.py` with fail-fast count bands and unit tests, effect measured by ablation (RESULTS 35: no measurable D2 change). Lesson recorded: audit first, tune second |

| Part 2 data review (errors caught before commit) | First draft of `docs/part2/DATA_REVIEW.md` contained three unsupported statements: a cause for the mean exposure of 1.15 (corrupted raw column; not measured), invented new-business row counts, and "same leak pattern" for customer_service_contacts | Each re-checked with a fresh query before committing: exposure_calc means 0.82-0.86, 2023 new-business rows 2,620 / 1,507 / 1,847, leak correlation 0.77 / 0.59 / 0.51 and means 0.81 vs 0.40; the text was replaced by the measured values |
| Part 2 transfer experiments | Wrote the harmonisation layer, calibration helpers (13 unit tests), fold-E scripts; ran the paired comparisons | Developed-model re-run asserted to reproduce the first run to 1e-6; every adopted change needs a positive paired bootstrap on fold E (D56-D59); an estimated new-business multiplier was rejected because it did not hold out of sample |

Human review checkpoints: (add here: who reviewed which decision and when).
