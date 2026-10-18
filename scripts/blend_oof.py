"""Evaluate blends of saved out-of-fold predictions on the same rows (arithmetic mean on the cost scale, weight w on the first file).

  python scripts/blend_oof.py fs_gbm_l4_out5 fs_nn_v1 A,B --weights 0,0.25,0.5,0.75,1
Weights are an exploration; a blend weight chosen on one fold is optimistic, so judge it on the other fold and use simple fixed weights.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import ARTIFACTS_DIR  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, d2_tweedie  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("folds", default="A,B")
    ap.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--reference-weight", type=float, default=1.0, help="bootstrap the blend against this weight on the first file")
    x = ap.parse_args()
    a = pd.read_parquet(ARTIFACTS_DIR / f"oof_{x.a}.parquet")
    b = pd.read_parquet(ARTIFACTS_DIR / f"oof_{x.b}.parquet")
    ws = [float(w) for w in x.weights.split(",")]
    for fold in x.folds.split(","):
        m = a[a["fold"].str.startswith(fold)].merge(b[b["fold"].str.startswith(fold)][["row_id", "pred"]], on="row_id", suffixes=("_a", "_b"))
        print(f"fold {fold} ({len(m):,} rows): weight on {x.a} (rest on {x.b})")
        for w in ws:
            p = w * m["pred_a"] + (1 - w) * m["pred_b"]
            cap = m["y"].clip(upper=250_000)
            print(f"  w={w:4.2f} pooled {d2_tweedie(m['y'], p):.4f} [cap250k {d2_tweedie(cap, p):.4f}] " +
                  " ".join(f"{l}={d2_tweedie(d['y'], w * d['pred_a'] + (1 - w) * d['pred_b']):.4f}" for l, d in m.groupby("line")))
        ref = x.reference_weight * m["pred_a"] + (1 - x.reference_weight) * m["pred_b"]
        for w in ws:
            if w == x.reference_weight:
                continue
            r = bootstrap_d2_diff(m["y"], w * m["pred_a"] + (1 - w) * m["pred_b"], ref, m["customer_id"], n_boot=200)
            print(f"  blend w={w:4.2f} minus w={x.reference_weight:4.2f}: {r['diff']:+.4f} (95% CI {r['lo']:+.4f}..{r['hi']:+.4f}) P(blend better)={r['p_a_better']:.2f}")


if __name__ == "__main__":
    main()
