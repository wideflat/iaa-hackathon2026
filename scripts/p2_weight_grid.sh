#!/bin/bash
# Blend-weight grid of the final recipe (t7 settings), both directions: net weight and Tweedie share, one setting at a time. Output: artifacts/p2_weight_grid.txt
cd "$(dirname "$0")/.."
OUT=artifacts/p2_weight_grid.txt
: > $OUT
BASE="--gbm-name hist1 --nn-models nn_tabm_hist1,nn_tabm_ple_hist1 --tw-names twl4,twl8 --k-freq motor=20,buildings=400,contents=20"
for nw in 0.3 0.5 0.7 0.85; do
  for ts in 0.0 0.15 0.25 0.35; do
    echo "== net weight $nw tweedie share $ts" >> $OUT
    python scripts/p2_calib2.py $BASE --nn-weight $nw --tw-share $ts 2>&1 | grep -E "^===|robustw99_tw_noclose\+gender  |robustw99_noclose\+gender  " >> $OUT
  done
done
