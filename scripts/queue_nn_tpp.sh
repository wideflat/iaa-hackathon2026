#!/bin/bash
# TabM and TabM-PLE with per-peril severity trends and per-peril recent-weighted rebalancing (half-life 1 y), repaired data, folds A,B. Waits for queue_comb.sh. Resumable.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/queue_comb.sh" > /dev/null; do sleep 20; done
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --trend-mode peril --rebalance-level peril --level-halflife 1 "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_nn_tabm4_c_tpp --note "repaired data, TabM, per-peril trends + per-peril recent-weighted rebalance"
run fs_nn_tabm4_ple_c_tpp --nn-ple 8 --note "repaired data, TabM-PLE, per-peril trends + per-peril recent-weighted rebalance"
