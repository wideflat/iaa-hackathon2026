#!/bin/bash
# Linear-leaf LightGBM (linear_tree) on the target-encoded GBM, repaired data, folds A,B. Resumable.
cd "$(dirname "$0")/.."
name=fs_gbm_c_lin
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --lgb-params linear_tree=true,linear_lambda=10.0 --note "repaired data, linear-leaf trees (linear_lambda 10)" > "artifacts/run_${name}.txt" 2>&1
