#!/bin/bash
# Seasonal frequency offset (calendar months in force) on the target-encoded GBM, repaired data, folds A,B. Resumable.
cd "$(dirname "$0")/.."
name=fs_gbm_c_season
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 2 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --seasonality --note "repaired data, per-peril seasonal frequency offset from the months a row is in force" > "artifacts/run_${name}.txt" 2>&1
