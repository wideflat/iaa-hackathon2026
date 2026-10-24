#!/bin/bash
# Resumable: the TabM and TabM-PLE nets of the developed market, tags B (for validation) and C (for the final), then A (calibration rows).
cd "$(dirname "$0")/.."
for m in nn_tabm nn_tabm_ple; do
  python scripts/p2_dev_predictions.py --model $m --tags B,C,A >> artifacts/p2_dev_$m.log 2>&1
done
