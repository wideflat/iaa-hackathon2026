#!/bin/bash
# Event-thinned claims (regional surges) on the target-encoded GBM, repaired data, folds A,B. Resumable.
cd "$(dirname "$0")/.."
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_c_ev --thin-events --note "repaired data, regional claim surges thinned to baseline for fitting"
