#!/bin/bash
cd /home/user/iaa-hackathon2026
[ -f submissions/fs_nn_tabm4_s3.csv ] || python scripts/make_submission.py --model fs_nn --outlier-mult 5 --nn-tabm-k 4 --seeds 3 --out submissions/fs_nn_tabm4_s3.csv > artifacts/make_tabm4_s3.txt 2>&1
[ -f submissions/fs_nn_tabm4_ple_s3.csv ] || python scripts/make_submission.py --model fs_nn --outlier-mult 5 --nn-tabm-k 4 --nn-ple 8 --seeds 3 --out submissions/fs_nn_tabm4_ple_s3.csv > artifacts/make_tabm4_ple_s3.txt 2>&1
