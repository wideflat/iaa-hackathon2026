#!/bin/bash
# XGBoost Tweedie on the adjusted pure-premium target (kind twx), depth 3, 2 threads (runs next to another job). Resumable.
cd "$(dirname "$0")/.."
name=fs_twx_d3
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_twx --folds A,B --threads 2 --outlier-mult 5 --freq-rounds 350 --cat-encoding target --depth 3 --note "XGBoost Tweedie on the adjusted pure-premium target, depth 3" > "artifacts/run_${name}.txt" 2>&1
