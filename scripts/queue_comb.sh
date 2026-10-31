#!/bin/bash
# Per-peril trends + policy year feature, repaired data, folds A,B. Waits for queue_tpp.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_tpp.sh" > /dev/null; do sleep 20; done
name=fs_gbm_c_tpp_year
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --trend-mode peril --year-feature --note "repaired data, per-peril severity trends + policy year input" > "artifacts/run_${name}.txt" 2>&1
