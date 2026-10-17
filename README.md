# IAA AI-Assisted Actuarial Hackathon 2026

Pricing model and tooling for [Part 1](https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1):
predict the expected ultimate claim cost (net of excess and recoveries, excluding declined claims) of every
2025 motor, buildings and contents policy row. Metric: pooled Tweedie D2 (power 1.5). Closes 2026-10-23 16:00 UTC.
Final prizes are judged after Part 2 on performance, AI use, validation/guardrails and documentation.

## Setup
    pip install -e ".[dev]"
    python scripts/download_data.py        # needs KAGGLE_API_TOKEN (or the sandbox proxy); data/ is gitignored
    pytest                                 # unit tests, no data needed

Never commit competition data or credentials (competition rules forbid redistribution).

## Workflow
    python scripts/run_cv.py --model mean_rate          # no-model baseline through the fixed folds
    python scripts/run_cv.py --model lgbm               # LightGBM Tweedie per line
    python scripts/make_submission.py --model lgbm --out submissions/lgbm.csv   # fits 2021-24, writes validated CSV

    python scripts/submit.py submissions/lgbm.csv -m "lgbm v1"            # validate + dry run
    python scripts/submit.py submissions/lgbm.csv -m "lgbm v1" --confirm   # submit (uses a daily slot)

Results are appended to `experiments/log.jsonl`. Branches: `main` is stable, work happens on `dev`.

## Layout
- `src/iaa/target.py` target definition, exposure fix, claim matching
- `src/iaa/validation.py`, `metrics.py`, `cv.py` fixed time-based folds, official metric, harness
- `src/iaa/features.py` leak-safe features and column policy; `drift.py` train-vs-test screens
- `src/iaa/models/` models; `submission.py` validated submission writer
- `docs/` DATA_QUALITY, VALIDATION, AI_USAGE; `tests/` guardrail tests

## Status
See `experiments/log.jsonl`. Open items: develop 2024 claims to ultimate, large-loss treatment, calibration
(GBM predicts ~65-75% of actual on fold A), frequency x severity models, GLM benchmark.
