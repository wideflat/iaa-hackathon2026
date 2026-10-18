#!/bin/bash
# Resumable run queue. Safe to start again after a sandbox restart: finished (fold, line) pieces are reused (--resume) and finished runs are skipped.
# Usage: nohup bash scripts/queue_runs.sh > artifacts/queue.log 2>&1 &
cd "$(dirname "$0")/.."
run() {  # name, args...
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_nn_wide --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-hidden 384,192 --nn-dropout 0.2 --note "NN freq x sev, hidden 384,192 dropout 0.2, 1 seed"
run fs_nn_s3 --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-seeds 3 --note "NN v1 with 3 seeds averaged per stage"
run fs_nn_tabm4 --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --note "TabM-style BatchEnsemble NN, k=4 members, 1 net"
run fs_nn_tabm4_wide --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --nn-hidden 384,192 --nn-dropout 0.2 --note "TabM k=4, hidden 384,192, dropout 0.2"
