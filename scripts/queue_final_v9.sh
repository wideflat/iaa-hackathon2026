#!/bin/bash
# v9: GBM refit with per-peril severity trends + policy year + per-peril recent-weighted rebalance (half-life 1 y), blended with the v8 repaired-data nets. Resumable.
cd "$(dirname "$0")/.."
[ -f submissions/fs_gbm_c_all.csv ] || python scripts/make_submission.py --model fs_gbm --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --trend-mode peril --year-feature --rebalance-level peril --level-halflife 1 --out submissions/fs_gbm_c_all.csv > artifacts/make_gbm_c_all.txt 2>&1
[ -f submissions/blend_v9_perline.csv ] || python scripts/blend_submissions.py --a submissions/fs_gbm_c_all.csv --b submissions/nn_tabm4_plus_ple_clean.csv --wb motor=0.7,buildings=0.4,contents=0.3 --out submissions/blend_v9_perline.csv > artifacts/blend_v9.txt 2>&1
echo done
