#!/bin/bash
# One-command reproduction of the Part 2 prediction file from the downloaded competition data.
#   bash scripts/reproduce_part2.sh [candidate]      candidate: t2 (GBM base) or t3 (GBM + TabM, default)
# Steps (each resumable; finished outputs are skipped): download data -> feature frames -> developed models scored on the emerging rows -> calibrated prediction file -> validation.
# Prints the sha256 of every input table and of the output. Expect several hours on 4 cores (the nets are the slow part); `nohup bash scripts/reproduce_part2.sh > artifacts/reproduce.log 2>&1 &`.
set -euo pipefail
cd "$(dirname "$0")/.."
cand="${1:-t3}"
python scripts/download_part2.py
echo "== input fingerprints"; sha256sum data/part2/emerging/*.parquet data/part2/developed/*.parquet | awk '{print substr($1,1,16), $2}'
python scripts/p2_build_frames.py
python scripts/p2_dev_predictions.py --tags A,B,C
if [ "$cand" = "t3" ]; then
  python scripts/p2_dev_predictions.py --model nn_tabm --tags B,C,A
  python scripts/p2_make_submission.py --level robust --nn-share motor=0.5,buildings=0.5,contents=0.5 --nn-models nn_tabm --out submissions/p2_repro_t3.csv
  out=submissions/p2_repro_t3.csv
else
  python scripts/p2_make_submission.py --level robust --out submissions/p2_repro_t2.csv
  out=submissions/p2_repro_t2.csv
fi
python scripts/submit.py "$out" -m "reproduction check (dry run)" --part2   # dry run: validates the format against the sample submission, submits nothing
echo "== output fingerprint"; sha256sum "$out" | awk '{print substr($1,1,16), $2}'
python -m pytest -q
