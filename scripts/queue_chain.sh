#!/bin/bash
# After the running XGBoost depth-4 CV: depth 5, then the direct Tweedie TabM-PLE net. Resumable (finished runs are skipped).
cd "$(dirname "$0")/.."
while pgrep -f "name fs_xgb_c_d4" > /dev/null; do sleep 15; done
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --threads 4 --outlier-mult 5 "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_xgb_c_d4 --model fs_xgb --folds A,B --freq-rounds 350 --sev-rounds 300 --cat-encoding target --depth 4 --note "XGBoost depth 4, target-encoded categoricals, repaired data"
run fs_xgb_c_d5 --model fs_xgb --folds A,B --freq-rounds 350 --sev-rounds 300 --cat-encoding target --depth 5 --note "XGBoost depth 5, target-encoded categoricals, repaired data"
run fs_twnn_tabm4_ple --model fs_twnn --folds A,B --nn-tabm-k 4 --nn-ple 8 --note "direct Tweedie TabM-PLE net on the adjusted pure-premium row target"
