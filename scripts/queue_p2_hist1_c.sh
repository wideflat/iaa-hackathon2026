#!/bin/bash
# After the A,B history refit of the GBM has finished, run cut-off C (resumable).
cd "$(dirname "$0")/.."
until [ -f artifacts/p2/dev_pred2_hist1_B_contents.parquet ]; do sleep 20; done
python scripts/p2_dev_predictions.py --tags C --features hist1 --variant hist1 >> artifacts/p2_dev_hist1.log 2>&1
