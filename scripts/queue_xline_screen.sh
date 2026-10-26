#!/bin/bash
# Screen of the cross-line covariates on the developed folds A / B (resumable).
cd "$(dirname "$0")/.."
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" "$@" > "artifacts/run_${name}.txt" 2>&1
}
G="--model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target"
run xs_gbm_base $G --note "screen: target-encoded GBM, repaired data, reference for the cross-line test"
run xs_gbm_xline $G --crossline --note "screen: + cross-line covariates of the same customer and year"
