"""Adversarial validation on the emerging book: which features separate the 2025 test rows from the 2023-24 training rows, for new business and for renewals separately.

  python scripts/p2_explore_drift.py

A LightGBM classifier (train rows vs test rows, 5-fold, out-of-fold AUC) on the shared features; AUC 0.5 = same population. Reports the AUC per line and segment and the
top features by gain, so the report can say in what way the transferred 2025 book differs. Features only, no labels.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from iaa import cache, market  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.features import feature_columns  # noqa: E402
import p2_calib as pc  # noqa: E402


def main() -> None:
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    for line in LINES:
        e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"])
        # mechanically time-stamped fields would separate the years trivially (AUC 1.0): the weather of the row's year, the year the customer joined, the renewal counter
        stamped = {"region", "customer_since_year", "renewal_count", "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm", "mean_temperature_c"}
        cols = [c for c in feature_columns(e) if c not in stamped]
        for seg_name, seg in (("new business", e["new_business"]), ("renewals", ~e["new_business"]), ("all", pd.Series(True, index=e.index))):
            d = e[seg]
            X = d[cols].copy()
            for c in X.columns:
                if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
                    X[c] = X[c].astype("category") if not pd.api.types.is_bool_dtype(X[c]) else X[c].astype(int)
            y = (d["split"] == "test").astype(int).to_numpy()
            oof, imp = np.zeros(len(d)), pd.Series(0.0, index=X.columns)
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
                b = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=100, verbose=-1, num_threads=2, seed=1), lgb.Dataset(X.iloc[tr], y[tr]), 100)
                oof[va] = b.predict(X.iloc[va])
                imp += pd.Series(b.feature_importance("gain"), index=X.columns)
            print(f"{line:9s} {seg_name:13s} train {int((y == 0).sum()):6d} test {int(y.sum()):6d} AUC {roc_auc_score(y, oof):.3f}  top: " + ", ".join(f"{k}" for k in (imp / imp.sum()).sort_values(ascending=False).head(6).index), flush=True)


if __name__ == "__main__":
    main()
