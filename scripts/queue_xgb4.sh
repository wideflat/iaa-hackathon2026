#!/bin/bash
# XGBoost depth 4 and 5 (depth 3 beat LightGBM leaves 4 on both folds), target-encoded categoricals, repaired data. Waits for nothing (runs first). Resumable.
cd "$(dirname "$0")/.."
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_xgb --folds A,B --threads 4 --outlier-mult 5 --freq-rounds 350 --sev-rounds 300 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_xgb_c_d4 --depth 4 --note "XGBoost depth 4, target-encoded categoricals, repaired data"
run fs_xgb_c_d5 --depth 5 --note "XGBoost depth 5, target-encoded categoricals, repaired data"
