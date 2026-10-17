"""Train-vs-test shift screen: the guard that catches features whose meaning changes at test time.

Time-based CV cannot catch a feature that is generated *by the outcome* in every training year
(e.g. customer_service_contacts: claims cause contacts in training rows, but 2025 rows show only
baseline contacts). Such a feature looks superb in CV and collapses on the leaderboard.
Compare the last training year with the test year, column by column, before trusting any feature.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _shift(a: pd.Series, b: pd.Series) -> float:
    """0 = identical distributions, 1 = disjoint. KS statistic (numeric) or total variation (categorical)."""
    a, b = a.dropna(), b.dropna()
    if a.empty or b.empty:
        return 1.0 if (a.empty != b.empty) else 0.0
    if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_bool_dtype(a):
        return float(ks_2samp(a.astype(float), b.astype(float)).statistic)
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())


def feature_shift_table(ref: pd.DataFrame, new: pd.DataFrame, cols, max_rows: int = 150_000, seed: int = 0) -> pd.DataFrame:
    """Per-column shift between a reference year and the test year, largest first."""
    ref = ref.sample(min(len(ref), max_rows), random_state=seed)
    new = new.sample(min(len(new), max_rows), random_state=seed)
    out = pd.DataFrame({"feature": list(cols), "shift": [_shift(ref[c], new[c]) for c in cols]})
    return out.sort_values("shift", ascending=False, ignore_index=True)


def target_signal(rows: pd.DataFrame, cols, y_col: str = "n_claims") -> pd.Series:
    """Spearman |rank correlation| of each numeric column with claim count: flags 'too good' features."""
    num = [c for c in cols if pd.api.types.is_numeric_dtype(rows[c]) or pd.api.types.is_bool_dtype(rows[c])]
    return rows[num].astype(float).corrwith(rows[y_col].astype(float), method="spearman").abs().sort_values(ascending=False)
