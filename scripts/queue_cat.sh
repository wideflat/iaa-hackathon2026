#!/bin/bash
# CatBoost (Poisson frequency x Tweedie-1.99 severity, exposure baseline) with its own ordered target statistics for the categoricals (cat_encoding native), repaired data, folds A,B. Resumable.
cd "$(dirname "$0")/.."
run() {
  name=$1; shift
  [ -f "artifacts/oof_${name}.parquet" ] && { echo "skip $name (done)"; return; }
  python scripts/run_cv.py --resume --name "$name" --model fs_cat --folds A,B --threads 4 --outlier-mult 5 --freq-rounds 350 --sev-rounds 300 "$@" > "artifacts/run_${name}.txt" 2>&1
}
run fs_cat_c_d6 --depth 6 --note "CatBoost depth 6 with native ordered target statistics, repaired data"
run fs_cat_c_d4 --depth 4 --note "CatBoost depth 4 with native ordered target statistics, repaired data"
