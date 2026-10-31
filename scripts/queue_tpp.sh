#!/bin/bash
# Per-peril severity trends (with / without recent-weighted per-peril rebalancing), repaired data, folds A,B. Waits for queue_year.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_year.sh" > /dev/null; do sleep 20; done
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_c_tpp --trend-mode peril --note "repaired data, per-peril severity trends"
run fs_gbm_c_tpp_peril_hl1 --trend-mode peril --rebalance-level peril --level-halflife 1 --note "repaired data, per-peril severity trends, per-peril rebalance to recent-weighted cost (half-life 1 y)"
