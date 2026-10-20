#!/bin/bash
# Row length / timing as GBM inputs (buildings and contents short rows are mis-calibrated by up to 2x in both folds), repaired data, folds A,B. 2 threads so it can run next to another job. Resumable.
cd "$(dirname "$0")/.."
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 2 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_gbm_c_expo --extra-features exposure_calc --note "repaired data, exposure_calc as an input"
run fs_gbm_c_timing --extra-features exposure_calc,start_month,end_month --note "repaired data, exposure_calc + start_month + end_month as inputs"
