#!/bin/bash
# v10: tree half = mean(frequency x severity GBM, mean of two adjusted pure-premium Tweedie GBMs), nets as v8 (repaired data), per-line NN weights 0.7 / 0.4 / 0.3. Resumable.
cd "$(dirname "$0")/.."
[ -f submissions/fs_tw_l4_final.csv ] || python scripts/make_submission.py --model fs_tw --leaves 4 --freq-rounds 350 --outlier-mult 5 --cat-encoding target --out submissions/fs_tw_l4_final.csv > artifacts/make_tw_l4.txt 2>&1
[ -f submissions/fs_tw_l8_final.csv ] || python scripts/make_submission.py --model fs_tw --leaves 8 --freq-rounds 300 --outlier-mult 5 --cat-encoding target --out submissions/fs_tw_l8_final.csv > artifacts/make_tw_l8.txt 2>&1
[ -f submissions/tw_mean_final.csv ] || python scripts/blend_submissions.py --a submissions/fs_tw_l4_final.csv --b submissions/fs_tw_l8_final.csv --wb 0.5 --out submissions/tw_mean_final.csv > artifacts/blend_tw.txt 2>&1
[ -f submissions/tree_half_v10.csv ] || python scripts/blend_submissions.py --a submissions/fs_gbm_target_clean.csv --b submissions/tw_mean_final.csv --wb 0.5 --out submissions/tree_half_v10.csv > artifacts/blend_tree.txt 2>&1
[ -f submissions/blend_v10_perline.csv ] || python scripts/blend_submissions.py --a submissions/tree_half_v10.csv --b submissions/nn_tabm4_plus_ple_clean.csv --wb motor=0.7,buildings=0.4,contents=0.3 --out submissions/blend_v10_perline.csv > artifacts/blend_v10.txt 2>&1
echo done
