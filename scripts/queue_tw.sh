#!/bin/bash
# Direct pure-premium Tweedie(1.5) GBM on the trend- and completeness-adjusted row target (kind tw), repaired data, folds A,B. Resumable.
cd "$(dirname "$0")/.."
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_tw --folds A,B --threads 4 --outlier-mult 5 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_tw_l4 --leaves 4 --freq-rounds 350 --note "adjusted pure-premium Tweedie GBM, leaves 4, 350 rounds"
run fs_tw_l8 --leaves 8 --freq-rounds 300 --note "adjusted pure-premium Tweedie GBM, leaves 8, 300 rounds"
