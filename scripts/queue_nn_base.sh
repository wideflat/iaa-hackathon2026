#!/bin/bash
# TabM on the repaired data with the default (pooled) trend and line rebalancing: separates the effect of the repairs from the effect of per-peril trends for the net. Waits for queue_all.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_all.sh" > /dev/null || pgrep -f "bash scripts/queue_nn_tpp.sh" > /dev/null; do sleep 20; done
name=fs_nn_tabm4_c
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --note "repaired data, TabM, default pooled trend and line rebalance (separates repairs from per-peril trends)" > "artifacts/run_${name}.txt" 2>&1
