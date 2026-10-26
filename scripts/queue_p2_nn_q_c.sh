#!/bin/bash
# Final fit (tag C = developed 2021-24) of the TabM-quantile net with hist1 inputs (resumable).
cd "$(dirname "$0")/.."
python scripts/p2_dev_predictions.py --model nn_tabm_q --tags C --features hist1 --variant hist1 >> artifacts/p2_dev_nn_tabm_q.log 2>&1
