"""Competition metric: Tweedie D^2 with power 1.5, plus diagnostics.

D^2 = 1 - deviance(y, pred) / deviance(y, mean(y)), i.e. sklearn.metrics.d2_tweedie_score.
The host's scorer is not public: we assume predictions are floored at a tiny positive value
(sklearn rejects pred <= 0 for 1 < power < 2). `PRED_FLOOR` mirrors that assumption.
"""
import numpy as np
import pandas as pd

from .config import SEED, TWEEDIE_POWER

PRED_FLOOR = 1e-6


def _dev_terms(p: float):
    return 1.0 / ((1.0 - p) * (2.0 - p)), 1.0 / (1.0 - p), 1.0 / (2.0 - p)


def tweedie_deviance(y, mu, power: float = TWEEDIE_POWER) -> np.ndarray:
    """Per-row unit deviance for 1 < power < 2 (y >= 0, mu > 0)."""
    a, b, c = _dev_terms(power)
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), PRED_FLOOR)
    return 2.0 * (np.power(y, 2.0 - power) * a - y * np.power(mu, 1.0 - power) * b + np.power(mu, 2.0 - power) * c)


def d2_tweedie(y, pred, power: float = TWEEDIE_POWER) -> float:
    y = np.asarray(y, dtype=float)
    base = np.full_like(y, y.mean())
    return 1.0 - tweedie_deviance(y, pred, power).sum() / tweedie_deviance(y, base, power).sum()


def _baseline_dev_total(n, sum_y, sum_ypow, power):
    a, b, c = _dev_terms(power)
    mu = sum_y / n
    return 2.0 * (sum_ypow * a - sum_y * mu ** (1.0 - power) * b + n * mu ** (2.0 - power) * c)


def bootstrap_d2(y, pred, groups, n_boot: int = 300, power: float = TWEEDIE_POWER, seed: int = SEED) -> dict:
    """Cluster bootstrap of D^2 (resample groups, e.g. customer_id, not rows).

    Exact and fast: deviance is additive, so each group is reduced to four sums once.
    Rows of one customer are correlated (several policies, several years), so row-level
    resampling would understate the noise.
    """
    y = np.asarray(y, dtype=float)
    dev = tweedie_deviance(y, pred, power)
    g = pd.factorize(np.asarray(groups))[0]
    k = g.max() + 1
    n = np.bincount(g, minlength=k).astype(float)
    sy = np.bincount(g, weights=y, minlength=k)
    sp = np.bincount(g, weights=np.power(y, 2.0 - power), minlength=k)
    sd = np.bincount(g, weights=dev, minlength=k)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        base = _baseline_dev_total(w @ n, w @ sy, w @ sp, power)
        scores[i] = 1.0 - (w @ sd) / base
    return {"se": float(scores.std(ddof=1)), "lo": float(np.quantile(scores, 0.025)), "hi": float(np.quantile(scores, 0.975))}


def score_report(df: pd.DataFrame, y_col: str = "y", pred_col: str = "pred", line_col: str = "line",
                 group_col: str | None = "customer_id", n_boot: int = 200) -> dict:
    """Pooled D^2 (the leaderboard number), per-line D^2, per-line calibration and bootstrap SE."""
    out = {"d2_pooled": d2_tweedie(df[y_col], df[pred_col]), "n_rows": int(len(df))}
    if group_col is not None and n_boot:
        out["d2_pooled_boot"] = bootstrap_d2(df[y_col], df[pred_col], df[group_col], n_boot=n_boot)
    for line, d in df.groupby(line_col):
        out[f"d2_{line}"] = d2_tweedie(d[y_col], d[pred_col])
        out[f"calibration_{line}"] = float(d[pred_col].sum() / d[y_col].sum())  # 1.0 = unbiased
    return out
