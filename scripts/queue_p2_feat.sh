#!/bin/bash
# Resumable screening of engineered feature sets on the developed GBM (tags A and B are enough for fold E in both directions).
cd "$(dirname "$0")/.."
python scripts/p2_dev_predictions.py --tags A,B --features xline --variant xline >> artifacts/p2_dev_xline.log 2>&1
python scripts/p2_dev_predictions.py --tags A,B --features ratios --variant ratios >> artifacts/p2_dev_ratios.log 2>&1
