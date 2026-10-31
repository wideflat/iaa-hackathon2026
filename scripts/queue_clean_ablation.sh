#!/bin/bash
# Ablation of the data-quality repairs on the target-encoded GBM (folds A,B). Resumable. Usage: nohup bash scripts/queue_clean_ablation.sh > artifacts/queue_clean.log 2>&1 &
cd "$(dirname "$0")/.."
run() {  # name, fixes, note
  name=$1; fixes=$2; note=$3
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  IAA_CLEAN_FIXES=$fixes python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --note "$note" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_target_clean_claims duplicates,placeholders "target GBM, repairs to claims only (duplicates, placeholder loss dates)"
run fs_gbm_target_clean_pol sum_insured,construction "target GBM, repairs to policy rows only (sum_insured x1000, construction_type)"
