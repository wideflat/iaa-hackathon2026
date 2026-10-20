#!/bin/bash
# Direct Tweedie TabM-PLE net on the adjusted pure-premium row target (kind twnn), repaired data, folds A,B. Waits for queue_xgb.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_xgb.sh" > /dev/null; do sleep 15; done
name=fs_twnn_tabm4_ple
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_twnn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --nn-ple 8 --note "direct Tweedie TabM-PLE net on the adjusted pure-premium row target" > "artifacts/run_${name}.txt" 2>&1
