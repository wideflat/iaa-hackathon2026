"""Equal-weight blends of several saved out-of-fold prediction files, compared with a reference blend by paired bootstrap on the same rows.

  python scripts/blend_multi.py --ref fs_gbm_l4_out5,fs_nn_v1 --cand fs_gbm_l4_out5,fs_nn_v1,nn_tweedie_v1 --folds A,B
A name may be given per fold as name:A=file_for_A:B=file_for_B when a model was saved in separate files (e.g. fs_nn_v1=A:fs_nn_v1,B:fs_nn_v1_B).
Equal weights only: weights tuned on two folds are optimistic (see docs/workflow/DECISIONS.md D23).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import ARTIFACTS_DIR  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, d2_tweedie  # noqa: E402

SPLIT_FILES = {"fs_nn_v1": {"A": "fs_nn_v1", "B": "fs_nn_v1_B"}}


def load(name: str, fold: str) -> pd.DataFrame:
    f = SPLIT_FILES.get(name, {}).get(fold, name)
    d = pd.read_parquet(ARTIFACTS_DIR / f"oof_{f}.parquet")
    return d[d["fold"].str.startswith(fold)]


def frame(names: list[str], fold: str) -> pd.DataFrame:
    base = load(names[0], fold)[["row_id", "customer_id", "line", "y", "pred"]].rename(columns={"pred": names[0]})
    for n in names[1:]:
        base = base.merge(load(n, fold)[["row_id", "pred"]].rename(columns={"pred": n}), on="row_id")
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--folds", default="A,B")
    a = ap.parse_args()
    ref, cand = a.ref.split(","), a.cand.split(",")
    for fold in a.folds.split(","):
        m = frame(sorted(set(ref) | set(cand), key=(ref + cand).index), fold)
        pr, pc = m[ref].mean(axis=1), m[cand].mean(axis=1)
        cap = m["y"].clip(upper=250_000)
        print(f"fold {fold} ({len(m):,} rows)")
        for n in sorted(set(ref) | set(cand), key=(ref + cand).index):
            print(f"  single {n:22s} {d2_tweedie(m['y'], m[n]):.4f} [cap {d2_tweedie(cap, m[n]):.4f}]")
        print(f"  ref  {'+'.join(ref)}: {d2_tweedie(m['y'], pr):.4f} [cap {d2_tweedie(cap, pr):.4f}]")
        print(f"  cand {'+'.join(cand)}: {d2_tweedie(m['y'], pc):.4f} [cap {d2_tweedie(cap, pc):.4f}] " +
              " ".join(f"{l}={d2_tweedie(d['y'], pc[d.index]):.4f}" for l, d in m.groupby("line")))
        r = bootstrap_d2_diff(m["y"], pc, pr, m["customer_id"], n_boot=200)
        print(f"  cand minus ref: {r['diff']:+.4f} (95% CI {r['lo']:+.4f}..{r['hi']:+.4f}) P(cand better)={r['p_a_better']:.2f}")


if __name__ == "__main__":
    main()
