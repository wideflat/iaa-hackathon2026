# AI-assisted workflow log

Judged criterion: meaningful use of AI across the workflow, with rigorous verification of its output.
Record each significant AI contribution and how a human or a test verified it.

| Stage | What AI did | How it was verified |
|-------|-------------|---------------------|
| Competition reading | Read overview, data, evaluation, rules via the Kaggle API and summarised them. | Spot-checked against `competitions/.../data` file list and metric metadata. |
| EDA / target construction | Wrote the claim-to-row matching and net-cost logic; found the exposure corruption and recoveries pattern. | Unit tests on hand-built examples (`tests/test_target.py`); row/claim counts reconciled (all but ~400 claims match, zero double matches). |
| Leakage detection | First model scored D2 0.55 vs public leaderboard ~0.12; AI traced it to `customer_service_contacts` through feature importance, outcome correlation and train-vs-test distribution comparison. | Re-ran CV without it: D2 0.10, consistent with the leaderboard. Added `drift.py` and a guard test so it cannot recur silently. |
| Metric / bootstrap | Implemented D2 and an exact cluster bootstrap. | Compared to `sklearn.metrics.d2_tweedie_score` to 1e-9 (`tests/test_metrics.py`). |

Human review checkpoints: (add here: who reviewed which decision and when).
