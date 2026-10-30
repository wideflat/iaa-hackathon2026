#!/bin/bash
# Ensemble E3 members (resumable): XGBoost depth 3, GBM with feature_fraction 0.7, wide TabM; hist1 inputs, tags B,C,A.
cd "$(dirname "$0")/.."
python scripts/p2_dev_predictions.py --model xgb --depth 3 --freq-rounds 350 --sev-rounds 300 --tags B,C,A --features hist1 --variant hist1 >> artifacts/p2_dev_ens3.log 2>&1
python scripts/p2_dev_predictions.py --model gbm --feature-fraction 0.7 --tags B,C,A --features hist1 --variant hist1ff >> artifacts/p2_dev_ens3.log 2>&1
python scripts/p2_dev_predictions.py --model nn_tabm_w --tags B,C,A --features hist1 --variant hist1 >> artifacts/p2_dev_ens3.log 2>&1
