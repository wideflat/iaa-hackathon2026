#!/bin/bash
# Year-on-year change features on the target-encoded GBM, repaired data, folds A,B. Resumable.
cd "$(dirname "$0")/.."
name=fs_gbm_c_yoy
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --yoy --note "repaired data, year-on-year change features (ncb, premium, excess, ...)" > "artifacts/run_${name}.txt" 2>&1
