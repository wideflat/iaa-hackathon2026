#!/bin/bash
# XGBoost (Poisson frequency + Gamma severity, exposure offset) with target-encoded categoricals on the repaired data, folds A,B, as a blend candidate. Waits for queue_final_v10.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_final_v10.sh" > /dev/null; do sleep 15; done
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_xgb --folds A,B --threads 4 --outlier-mult 5 --freq-rounds 350 --sev-rounds 300 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_xgb_c_d2 --depth 2 --note "XGBoost depth 2, target-encoded categoricals, repaired data"
run fs_xgb_c_d3 --depth 3 --note "XGBoost depth 3, target-encoded categoricals, repaired data"
