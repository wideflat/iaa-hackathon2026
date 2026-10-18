# Validation principles (stick to these unless evidence says otherwise)

1. **Mimic the test.** The test is a future year (2025), ~85% of its policies also in training, scored by
   pooled Tweedie D2 (power 1.5) on a heavy-tailed target. So: time-based folds, no grouping by policy.
2. **Folds are fixed** (`src/iaa/validation.py`): A = train 2021-22 / validate 2023 (mature, trusted);
   B = train 2021-23 / validate 2024 (immature claims: sanity check, half weight until the 2024 target is developed).
3. **Same metric as the leaderboard** (`src/iaa/metrics.py`), pooled across lines, plus per-line D2 and
   calibration (sum predicted / sum actual) for diagnosis. Unit-tested against sklearn.
4. **Know the noise.** Every score has a customer-clustered bootstrap CI. Only changes that beat the noise are accepted.
5. **No look-ahead.** Claim-history features use only claims lost and reported by 31 Dec of the prior year.
   All fitting happens inside the training folds. Tests: `tests/test_features.py`, `tests/test_validation.py`.
6. **CV can be fooled by a feature that is wrong in every training year.** Time-based CV cannot see it, so
   before trusting a feature compare train with test (`src/iaa/drift.py`): (a) too-good-to-be-true correlation with
   the outcome, (b) shift between the last training year and the test year. This caught `customer_service_contacts`.
7. **Trust CV over the public leaderboard while they agree** (the public score is a 33% slice of a noisy target).
   If they diverge, investigate how the test differs from the validation before changing anything.
8. **Log every experiment** (`experiments/log.jsonl`): config, CV scores, public score, what changed.
