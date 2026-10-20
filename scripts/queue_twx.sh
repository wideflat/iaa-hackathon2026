#!/bin/bash
# XGBoost Tweedie(1.5) on the adjusted pure-premium row target (kind twx), depth 3 and 4, repaired data, folds A,B. Waits for queue_chain.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_chain.sh" > /dev/null; do sleep 20; done
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_twx --folds A,B --threads 4 --outlier-mult 5 --freq-rounds 350 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_twx_d3 --depth 3 --note "XGBoost Tweedie on the adjusted pure-premium target, depth 3"
run fs_twx_d4 --depth 4 --note "XGBoost Tweedie on the adjusted pure-premium target, depth 4"
