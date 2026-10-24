# Evidence map: judging criteria -> where to look

One hop from each rubric bullet to the file, test or log line that supports it. Status: **done** = in the repository now, **todo** = planned (`PART2_STRATEGY.md`).

| Criterion bullet | Evidence | Status |
|---|---|---|
| Quantitative accuracy, Week 1 | `docs/experiments/PART1_FINAL.md` (private rank 3, 0.13864; all 21 submissions); Week 1 file `submissions/blend_v10_perline.csv` | done |
| Quantitative accuracy, Week 2 | `docs/experiments/RESULTS.md` sections 45-46 (fold E); Kaggle public score once submitted; `submissions/p2_*.csv` | in progress |
| Appropriate modelling choices and adaptation to the emerging market | `docs/part2/DATA_REVIEW.md` (what differs), `docs/part2/PART2_STRATEGY.md` section 3 (adopted / rejected with evidence), `docs/workflow/DECISIONS.md` D56-D59, `src/iaa/market.py`, `src/iaa/transfer.py` | done |
| Clear treatment of data quality, limitations, remaining uncertainty | `DATA_REVIEW.md` section 5 (ten checks), `docs/data/DATA_QUALITY.md` (Part 1 items 1-17), `PART2_STRATEGY.md` section 5 (uncertainty table); scenario runs | scenarios todo |
| Meaningful integration of AI across the workflow | `docs/workflow/AI_USAGE.md`, `docs/workflow/DEVELOPMENT_LOG.md`, helper-session use in Part 1 | done |
| Demonstrated improvement in quality, efficiency, challenge, reliability | AI_USAGE rows "Data-quality audit (missed, then caught)" and "Part 2 data review"; `PART2_STRATEGY.md` section 6 | done |
| Rigorous verification of AI-generated outputs | AI_USAGE.md (error ledger); unit tests written before code (`tests/test_transfer.py`, `tests/test_office_premium.py`); reproducibility assertion in `scripts/p2_dev_predictions.py`; independent second-opinion run | second opinion todo |
| Comprehensive validation checks, automated tests, diagnostics | `tests/` (102+ tests), `.github/workflows/ci.yml`, `docs/methodology/VALIDATION.md`, paired bootstrap in `src/iaa/metrics.py`, leak alarm (`assert_not_too_good`), repair bands in `src/iaa/cleaning.py` | done |
| Governance, reproducibility, human review / escalation | `DECISIONS.md` (who decided what, with evidence), model card, `scripts/reproduce_part2.sh`, sign-off list | todo |
| Video and written report | `docs/part2/REPORT.md` (5 pages), `docs/part2/VIDEO_SCRIPT.md` (5 minutes) | todo |
| Well-structured code repository with version control | `README.md`, `docs/README.md`, `src/iaa/` (modules with header docstrings), `scripts/`, branch `dev` history | done |
