"""Peril-level blend weights for GBM vs the NN mean, fitted by Tweedie(1.5) likelihood on one fold and evaluated on the other.

  python scripts/stack_peril.py gbm=fs_gbm_c_pp tabm=fs_nn_tabm4_c_pp ple=fs_nn_tabm4_ple_c_pp
Each OOF file must carry the pp_<peril> columns (cv.run_cv). The blend of one row is sum_p [(1 - w_p) gbm_p + w_p (tabm_p + ple_p) / 2]; w_p is fitted
per peril on the other fold (bounds 0..1, shrunk towards the line's fixed weight by --shrink) and compared with the fixed line weights 0.7 / 0.4 / 0.3.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import ARTIFACTS_DIR, LINES  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, d2_tweedie  # noqa: E402

P = 1.5
FIXED = {"motor": 0.7, "buildings": 0.4, "contents": 0.3}


def dev(y, mu):
    mu = np.maximum(mu, 1e-6)
    return float(np.sum(-y * mu ** (1 - P) / (1 - P) + mu ** (2 - P) / (2 - P)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs=3)
    ap.add_argument("--shrink", type=float, default=0.5, help="weight on the fitted peril weight; the rest goes to the line's fixed weight")
    a = ap.parse_args()
    names = dict(s.split("=", 1) for s in a.specs)
    d = {k: pd.read_parquet(ARTIFACTS_DIR / f"oof_{v}.parquet") for k, v in names.items()}
    pp = {k: [c for c in v.columns if c.startswith("pp_")] for k, v in d.items()}
    perils = sorted(set(pp["gbm"]) & set(pp["tabm"]) & set(pp["ple"]))
    key = ["row_id", "fold"]
    base = d["gbm"][key + ["customer_id", "line", "y"]].copy()
    for k in ("gbm", "tabm", "ple"):
        base = base.merge(d[k][key + pp[k]].rename(columns={c: f"{k}_{c}" for c in pp[k]}), on=key)
    base = base.fillna(0.0)
    results = {}
    for fit, test in (("A", "B"), ("B", "A")):
        tr, te = base[base.fold.str.startswith(fit)], base[base.fold.str.startswith(test)]
        w = {}
        for line in LINES:
            lt = tr[tr.line == line]
            ps = [p for p in perils if (lt[f"gbm_{p}"] > 0).any()]
            g = np.column_stack([lt[f"gbm_{p}"] for p in ps]); n = np.column_stack([(lt[f"tabm_{p}"] + lt[f"ple_{p}"]) / 2 for p in ps])
            w0 = np.full(len(ps), FIXED[line])
            res = minimize(lambda x: dev(lt.y.to_numpy(), ((1 - x) * g + x * n).sum(axis=1)) / len(lt), w0, method="L-BFGS-B", bounds=[(0, 1)] * len(ps))
            for p, x in zip(ps, res.x):
                w[(line, p)] = a.shrink * x + (1 - a.shrink) * FIXED[line]
        pred_p = np.zeros(len(te)); pred_f = np.zeros(len(te))
        for line in LINES:
            m = (te.line == line).to_numpy()
            for (l, p), x in w.items():
                if l != line:
                    continue
                g, n = te.loc[m, f"gbm_{p}"].to_numpy(), ((te.loc[m, f"tabm_{p}"] + te.loc[m, f"ple_{p}"]) / 2).to_numpy()
                pred_p[m] += (1 - x) * g + x * n; pred_f[m] += (1 - FIXED[line]) * g + FIXED[line] * n
        r = bootstrap_d2_diff(te.y.to_numpy(), pred_p, pred_f, te.customer_id.to_numpy())
        print(f"fitted on {fit}, tested on {test}: peril weights {d2_tweedie(te.y, pred_p):.4f} fixed line weights {d2_tweedie(te.y, pred_f):.4f} diff {r['diff']:+.4f} (CI {r['lo']:+.4f}..{r['hi']:+.4f})")
        results[fit] = w
    print("peril weights on the NN mean (fitted on A | fitted on B):")
    for k in sorted(results["A"]):
        print(f"  {k[0]:10s} {k[1]:24s} {results['A'][k]:.2f} | {results['B'].get(k, float('nan')):.2f}")


if __name__ == "__main__":
    main()
