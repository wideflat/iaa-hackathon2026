#!/bin/bash
# Two XGBoost screens in parallel (2 threads each): depth 5, and depth 3 with gamma 1. Waits for queue_chain.sh (the Tweedie net). Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "^bash scripts/queue_chain.sh" > /dev/null; do sleep 15; done
common="--model fs_xgb --folds A,B --threads 2 --outlier-mult 5 --freq-rounds 350 --sev-rounds 300 --cat-encoding target"
[ -f artifacts/oof_fs_xgb_c_d5.parquet ] || python scripts/run_cv.py --resume --name fs_xgb_c_d5 $common --depth 5 --note "XGBoost depth 5" > artifacts/run_fs_xgb_c_d5.txt 2>&1 &
[ -f artifacts/oof_fs_xgb_c_d3_g1.parquet ] || python scripts/run_cv.py --resume --name fs_xgb_c_d3_g1 $common --depth 3 --xgb-params gamma=1.0 --note "XGBoost depth 3 gamma 1" > artifacts/run_fs_xgb_c_d3_g1.txt 2>&1 &
wait
