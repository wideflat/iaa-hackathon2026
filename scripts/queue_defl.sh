#!/bin/bash
# Index-adjusted nominal features (premium, sum insured) on the target-encoded GBM, repaired data, folds A,B. Waits for queue_level.sh to finish first. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_level.sh" > /dev/null; do sleep 20; done
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_c_defl --deflate written_premium,sum_insured --note "repaired data, written_premium and sum_insured divided by their median in the row's own year"
