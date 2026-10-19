"""Per-line non-negative blend weights (sum to 1, cost scale) that maximise the Tweedie(1.5) likelihood on one fold, evaluated on the other fold.

  python scripts/stack_oof.py gbm=fs_gbm_l4_out5 tabm=fs_nn_tabm4 glm=fs_glm_v3_stabilised:A,fs_glm_v3_foldB:B ...
Each model is name=file or name=fileA:A,fileB:B when its two folds are in separate files. Reference blends are printed for comparison.
Weights fitted on both folds' union are printed too (for use in a submission), but judge a stacker by the cross-fold numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import ARTIFACTS_DIR, LINES  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, d2_tweedie  # noqa: E402

P = 1.5


def dev(y, mu):
    mu = np.maximum(mu, 1e-6)
    return float(np.sum(-y * mu ** (1 - P) / (1 - P) + mu ** (2 - P) / (2 - P)))


def load(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        f, _, fold = part.partition(":")
        d = pd.read_parquet(ARTIFACTS_DIR / f"oof_{f}.parquet")
        for fl in ([fold] if fold else ["A", "B"]):
            out[fl] = d[d["fold"].str.startswith(fl)]
    return out


def fit_weights(P_, y):
    k = P_.shape[1]
    res = minimize(lambda w: dev(y, P_ @ w) / len(y), np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 200})
    return res.x


def main() -> None:
    specs = dict(a.split("=", 1) for a in sys.argv[1:])
    names = list(specs)
    data = {n: load(s) for n, s in specs.items()}
    frames = {}
    for f in "AB":
        m = data[names[0]][f][["row_id", "customer_id", "line", "y"]].copy()
        m[names[0]] = data[names[0]][f]["pred"].to_numpy() if False else None
        base = data[names[0]][f][["row_id", "customer_id", "line", "y", "pred"]].rename(columns={"pred": names[0]})
        for n in names[1:]:
            base = base.merge(data[n][f][["row_id", "line", "pred"]].rename(columns={"pred": n}), on=["row_id", "line"])
        frames[f] = base
    W = {}
    for fit, test in (("A", "B"), ("B", "A")):
        parts = []
        for line in LINES:
            tr, te = frames[fit][frames[fit].line == line], frames[test][frames[test].line == line]
            w = fit_weights(tr[names].to_numpy(), tr["y"].to_numpy())
            W[(fit, line)] = w
            parts.append(te.assign(stack=te[names].to_numpy() @ w))
        te_all = pd.concat(parts)
        eq = te_all[names].mean(axis=1)
        cap = te_all["y"].clip(upper=250_000)
        r = bootstrap_d2_diff(te_all["y"], te_all["stack"], eq, te_all["customer_id"], n_boot=200)
        print(f"weights fitted on {fit}, evaluated on {test}: stack {d2_tweedie(te_all['y'], te_all['stack']):.4f} [cap {d2_tweedie(cap, te_all['stack']):.4f}] "
              f"equal-weight {d2_tweedie(te_all['y'], eq):.4f} [cap {d2_tweedie(cap, eq):.4f}] diff {r['diff']:+.4f} (CI {r['lo']:+.4f}..{r['hi']:+.4f})")
    print("weights per line (fitted on A | fitted on B):  " + " ".join(names))
    for line in LINES:
        print(f"  {line:10s} A: {np.round(W[('A', line)], 2)}  B: {np.round(W[('B', line)], 2)}")


if __name__ == "__main__":
    main()
