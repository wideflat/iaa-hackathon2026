"""Second-stage correction of the blend: a small LightGBM with Tweedie(1.5) loss and init_score = log(blend prediction), trained on the out-of-fold rows of one fold
and evaluated on the other fold (both directions). The boosting rounds are few and the leaves large, so it can only learn broad corrections.

  python scripts/stage2.py [--rounds 60] [--lr 0.03] [--leaves 4] [--min-leaf 3000]
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cache  # noqa: E402
from iaa.config import ARTIFACTS_DIR, LINES, SEED  # noqa: E402
from iaa.features import feature_columns  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, d2_tweedie  # noqa: E402

W = {"motor": 0.7, "buildings": 0.4, "contents": 0.3}


def load(n):
    return pd.read_parquet(ARTIFACTS_DIR / f"oof_{n}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=3000)
    ap.add_argument("--no-features", action="store_true", help="use only the blend prediction (a pure re-calibration)")
    a = ap.parse_args()
    g = load("fs_gbm_target")[["row_id", "customer_id", "line", "y", "fold", "pred"]].rename(columns={"pred": "g"})
    for n, k in (("fs_nn_tabm4", "t"), ("fs_nn_tabm4_ple", "p")):
        g = g.merge(load(n)[["row_id", "fold", "pred"]].rename(columns={"pred": k}), on=["row_id", "fold"])
    w = g["line"].map(W)
    g["blend"] = (1 - w) * g["g"] + w * (0.5 * g["t"] + 0.5 * g["p"])
    frames = cache.load_frames()
    parts = []
    for line in LINES:
        d = g[g["line"] == line]
        f = frames[line].set_index("row_id")
        cols = feature_columns(frames[line][frames[line]["split"] == "train"], True, ())
        X = f.loc[d["row_id"], cols].copy()
        for c in X.columns:
            if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
                X[c] = X[c].astype("category")
            elif pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype("int8")
        X = X.reset_index(drop=True)
        X["log_blend"] = np.log(np.maximum(d["blend"].to_numpy(), 1e-6))
        X["log_gbm"] = np.log(np.maximum(d["g"].to_numpy(), 1e-6))
        X["log_nn"] = np.log(np.maximum((0.5 * d["t"] + 0.5 * d["p"]).to_numpy(), 1e-6))
        X["expo"] = f.loc[d["row_id"], "exposure_calc"].to_numpy()
        if a.no_features:
            X = X[["log_blend", "log_gbm", "log_nn"]]
        d = d.reset_index(drop=True)
        out = d.copy()
        out["stage2"] = d["blend"].to_numpy()
        for fit, test in (("A", "B"), ("B", "A")):
            tr, te = d["fold"].str.startswith(fit).to_numpy(), d["fold"].str.startswith(test).to_numpy()
            params = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=1, lambda_l2=50.0, verbose=-1, seed=SEED, num_threads=0)
            b = lgb.train(params, lgb.Dataset(X[tr], label=d.loc[tr, "y"], init_score=np.log(np.maximum(d.loc[tr, "blend"], 1e-6))), num_boost_round=a.rounds)
            out.loc[te, "stage2"] = np.exp(b.predict(X[te], raw_score=True) + np.log(np.maximum(d.loc[te, "blend"], 1e-6)))
        parts.append(out)
    o = pd.concat(parts, ignore_index=True)
    for f in "AB":
        d = o[o["fold"].str.startswith(f)]
        cap = d["y"].clip(upper=250_000)
        r = bootstrap_d2_diff(d["y"], d["stage2"], d["blend"], d["customer_id"], n_boot=200)
        print(f"fold {f} (stage 2 trained on the other fold): blend {d2_tweedie(d['y'], d['blend']):.4f} [cap {d2_tweedie(cap, d['blend']):.4f}] -> stage2 {d2_tweedie(d['y'], d['stage2']):.4f} [cap {d2_tweedie(cap, d['stage2']):.4f}] "
              f"diff {r['diff']:+.4f} (CI {r['lo']:+.4f}..{r['hi']:+.4f}) " + " ".join(f"{l}={d2_tweedie(x['y'], x['stage2']):.4f}/{d2_tweedie(x['y'], x['blend']):.4f}" for l, x in d.groupby("line")))


if __name__ == "__main__":
    main()
