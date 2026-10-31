#!/bin/bash
# Stack of the small positive GBM changes: per-peril trends + policy year + exposure/start/end month inputs, repaired data, folds A,B. Resumable.
cd "$(dirname "$0")/.."
name=fs_gbm_c_combo
[ -f "artifacts/oof_${name}.parquet" ] || python scripts/run_cv.py --resume --name "$name" --model fs_gbm --folds A,B --threads 4 --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --trend-mode peril --year-feature --extra-features exposure_calc,start_month,end_month --note "repaired data, per-peril trends + year + exposure/start/end month inputs" > "artifacts/run_${name}.txt" 2>&1
