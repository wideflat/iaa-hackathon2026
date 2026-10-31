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
run fs_nn_tabm4_small --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --nn-hidden 128,64 --note "TabM k=4, hidden 128,64"
run fs_gbm_label --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding label --note "GBM with frequency-ranked integer label encoding of categoricals"
run fs_gbm_target --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --note "GBM with smoothed out-of-fold claim-rate (target) encoding of categoricals"
run fs_nn_tabm4_q --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --nn-numtf quantile --note "TabM k=4 + rank-gauss (quantile) numeric transform"
run fs_nn_tabm4_log --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --nn-numtf log --note "TabM k=4 + log1p of right-skewed numeric inputs"
run fs_nn_tabm4_te --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-tabm-k 4 --nn-cat-te --note "TabM k=4 + target-encoded categorical inputs"
run fs_gbm_target_peril --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target_peril --note "GBM with per-peril smoothed out-of-fold claim-rate encodings of categoricals"
run fs_gbm_target_hl1 --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --trend-halflife 1 --note "target-encoded GBM, severity trend weighted to recent years (half-life 1 year)"
run fs_gbm_target_hl03 --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --trend-halflife 0.3 --note "target GBM, severity trend = essentially the latest year-on-year change (half-life 0.3 y)"
run fs_gbm_target_hist --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --history-extra --note "target GBM + per-peril claim history, gap since last claim, max prior cost, open claims"
