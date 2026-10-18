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

Human review checkpoints: (add here: who reviewed which decision and when).
