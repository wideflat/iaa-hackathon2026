#!/bin/bash
# Final fit of the single XGBoost depth-3 model (Poisson frequency x Gamma severity, target-encoded categoricals) on the repaired 2021-24 data. Resumable.
cd "$(dirname "$0")/.."
[ -f submissions/fs_xgb_d3_final.csv ] || python scripts/make_submission.py --model fs_xgb --depth 3 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --out submissions/fs_xgb_d3_final.csv > artifacts/make_xgb_d3.txt 2>&1
