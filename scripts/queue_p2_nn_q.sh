#!/bin/bash
# Third net (TabM with quantile inputs, hist1): validation tags B and A first, then the final tag C (resumable: finished files are skipped).
cd "$(dirname "$0")/.."
python scripts/p2_dev_predictions.py --model nn_tabm_q --tags B,A --features hist1 --variant hist1 >> artifacts/p2_dev_nn_tabm_q.log 2>&1
