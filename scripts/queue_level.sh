#!/bin/bash
# Peril-level / recency-weighted rebalancing screens on the target-encoded GBM (repaired data), folds A,B. Resumable.
cd "$(dirname "$0")/.."
run() {  # name, extra args...
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_c_peril --rebalance-level peril --note "repaired data, rebalance per peril to all-year trended cost"
run fs_gbm_c_peril_hl2 --rebalance-level peril --level-halflife 2 --note "repaired data, per-peril rebalance to recent-weighted cost, half-life 2 y"
run fs_gbm_c_peril_hl1 --rebalance-level peril --level-halflife 1 --note "repaired data, per-peril rebalance to recent-weighted cost, half-life 1 y"
run fs_gbm_c_line_hl1 --level-halflife 1 --note "repaired data, line-level rebalance to recent-weighted cost, half-life 1 y"
