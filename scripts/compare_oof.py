"""Paired comparison of two saved out-of-fold prediction files (artifacts/oof_<name>.parquet) on the same rows.

  python scripts/compare_oof.py fs_glm_v3_stabilised fs_trend_comp A
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import ARTIFACTS_DIR  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, d2_tweedie  # noqa: E402


def main() -> None:
    a_name, b_name, fold = sys.argv[1], sys.argv[2], sys.argv[3]
    a = pd.read_parquet(ARTIFACTS_DIR / f"oof_{a_name}.parquet")
    b = pd.read_parquet(ARTIFACTS_DIR / f"oof_{b_name}.parquet")
    a, b = a[a["fold"].str.startswith(fold)], b[b["fold"].str.startswith(fold)]
    m = a.merge(b[["row_id", "pred"]], on="row_id", suffixes=("_a", "_b"))
    r = bootstrap_d2_diff(m["y"], m["pred_a"], m["pred_b"], m["customer_id"])
    print(f"fold {fold}: {a_name} {d2_tweedie(m['y'], m['pred_a']):.4f} vs {b_name} {d2_tweedie(m['y'], m['pred_b']):.4f} | "
          f"diff {r['diff']:+.4f} (95% CI {r['lo']:+.4f}..{r['hi']:+.4f}), P(first better) = {r['p_a_better']:.2f}, rows {len(m):,}")
    for line, d in m.groupby("line"):
        rl = bootstrap_d2_diff(d["y"], d["pred_a"], d["pred_b"], d["customer_id"], n_boot=200)
        print(f"   {line:10s} diff {rl['diff']:+.4f} (CI {rl['lo']:+.4f}..{rl['hi']:+.4f})")


if __name__ == "__main__":
    main()
