#!/bin/bash
# Repaired-data GBM, TabM and TabM-PLE with per-peril OOF outputs (pp_* columns), folds A,B, for peril-level blend weights. Resumable.
cd "$(dirname "$0")/.."
run() {
  name=$1; model=$2; shift 2
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model "$model" --folds A,B --threads 4 --outlier-mult 5 "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_c_pp fs_gbm --leaves 4 --freq-rounds 350 --sev-rounds 300 --cat-encoding target --note "repaired data target GBM with per-peril outputs"
run fs_nn_tabm4_c_pp fs_nn --nn-tabm-k 4 --note "repaired data TabM with per-peril outputs"
run fs_nn_tabm4_ple_c_pp fs_nn --nn-tabm-k 4 --nn-ple 8 --note "repaired data TabM-PLE with per-peril outputs"
