#!/bin/bash
# Extra ensemble members for the final blend (resumable; tags B,C then A): TabM-PLE and TabM-quantile with 3 averaged nets per stage, and a GBM with 8 leaves (hist1 inputs).
cd "$(dirname "$0")/.."
python scripts/p2_dev_predictions.py --model nn_tabm_ple --seeds 3 --tags B,C,A --features hist1 --variant hist1s3 >> artifacts/p2_dev_ens.log 2>&1
python scripts/p2_dev_predictions.py --model gbm --leaves 8 --freq-rounds 300 --sev-rounds 250 --tags B,C,A --features hist1 --variant hist1l8 >> artifacts/p2_dev_ens.log 2>&1
python scripts/p2_dev_predictions.py --model nn_tabm_q --seeds 3 --tags B,C,A --features hist1 --variant hist1s3 >> artifacts/p2_dev_ens.log 2>&1
