#!/bin/bash
# Final fits of the v5 recipe on the repaired data (src/iaa/cleaning.py, expected recoveries): target GBM, TabM, TabM-PLE, then the per-line blend v8.
# Resumable: finished files are skipped. Usage: nohup bash scripts/queue_final_clean.sh > artifacts/queue_final_clean.log 2>&1 &
cd "$(dirname "$0")/.."
[ -f submissions/fs_gbm_target_clean.csv ] || python scripts/make_submission.py --model fs_gbm --leaves 4 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target --out submissions/fs_gbm_target_clean.csv > artifacts/make_gbm_target_clean.txt 2>&1
[ -f submissions/fs_nn_tabm4_clean.csv ] || python scripts/make_submission.py --model fs_nn --outlier-mult 5 --nn-tabm-k 4 --out submissions/fs_nn_tabm4_clean.csv > artifacts/make_tabm4_clean.txt 2>&1
[ -f submissions/fs_nn_tabm4_ple_clean.csv ] || python scripts/make_submission.py --model fs_nn --outlier-mult 5 --nn-tabm-k 4 --nn-ple 8 --out submissions/fs_nn_tabm4_ple_clean.csv > artifacts/make_tabm4_ple_clean.txt 2>&1
[ -f submissions/nn_tabm4_plus_ple_clean.csv ] || python scripts/blend_submissions.py --a submissions/fs_nn_tabm4_clean.csv --b submissions/fs_nn_tabm4_ple_clean.csv --wb 0.5 --out submissions/nn_tabm4_plus_ple_clean.csv > artifacts/blend_nn_clean.txt 2>&1
[ -f submissions/blend_v8_clean_perline.csv ] || python scripts/blend_submissions.py --a submissions/fs_gbm_target_clean.csv --b submissions/nn_tabm4_plus_ple_clean.csv --wb motor=0.7,buildings=0.4,contents=0.3 --out submissions/blend_v8_clean_perline.csv > artifacts/blend_v8_clean.txt 2>&1
echo done
