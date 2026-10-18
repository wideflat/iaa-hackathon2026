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

9. **Compare models with a paired bootstrap, not two unpaired CIs.** Unpaired 95% CIs are about +-0.014 wide on fold A and cannot resolve the differences we care about (0.001-0.01); the paired difference on the same rows (`scripts/compare_oof.py`, `metrics.bootstrap_d2_diff`) is about 10x tighter because the same rare large claims hit both models.
10. **Also report the outlier-robust score** (row cost capped at 250k, `d2_pooled_cap250k`). One 11.9M claim in fold A swung the GLM-vs-GBM comparison by +0.007 (raw) vs +0.0001 (capped). Decisions use the paired difference on both versions, and weigh the future-year evidence (fold B, leaderboard) at least as much as fold A.
11. **Scoring immature folds in observed space.** Models with a completeness offset expose `predict_observed` (prediction x expected reported share); fold B is scored that way, which cut motor calibration from 1.33 to 1.08.
12. **Leak alarm**: any run with pooled D2 above 0.35 aborts (`metrics.assert_not_too_good`).
