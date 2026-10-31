#!/bin/bash
# Policy year as a GBM input (regime shifts by segment), repaired data, folds A,B. Waits for queue_defl.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_defl.sh" > /dev/null || pgrep -f "bash scripts/queue_level.sh" > /dev/null; do sleep 20; done
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_c_year --year-feature --note "repaired data, policy year as a numeric GBM input"
