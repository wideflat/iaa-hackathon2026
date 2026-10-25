#!/bin/bash
# Pooled training (developed + labelled emerging rows, market flag), forward (PF) and reverse (PR) validation tags for emerging row weights W (resumable: finished files are skipped).
cd "$(dirname "$0")/.."
for W in 1 5 20; do
  for T in PF PR; do
    python scripts/p2_dev_predictions.py --pool $T --pool-weight $W --features hist1 --variant poolw${W}_hist1 >> artifacts/p2_dev_pool.log 2>&1
  done
done
