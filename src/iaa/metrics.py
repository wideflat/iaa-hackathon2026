"""Competition metric: Tweedie D^2 with power 1.5, plus diagnostics.

D^2 = 1 - deviance(y, pred) / deviance(y, mean(y)), i.e. sklearn.metrics.d2_tweedie_score.
The host's scorer is not public: we assume predictions are floored at a tiny positive value
(sklearn rejects pred <= 0 for 1 < power < 2). `PRED_FLOOR` mirrors that assumption.
"""
import numpy as np
import pandas as pd

from .config import SEED, TWEEDIE_POWER

PRED_FLOOR = 1e-6
# The public leaderboard leaders sit near 0.13. A pooled D2 above this on honest validation means a leak, not skill.
LEAK_ALARM_D2 = 0.35
ROBUST_CAP = 250_000.0   # row cost cap used by the robust companion score


def assert_not_too_good(d2: float, context: str = "") -> None:
    """Fail loudly on an implausibly high score (e.g. a target or outcome-generated column leaked into the features)."""
    if d2 > LEAK_ALARM_D2:
        raise AssertionError(f"D2={d2:.3f} exceeds {LEAK_ALARM_D2} {context}: suspected target leakage, do not trust this result")


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
    # Outlier-robust companion: one 11.9M claim in 2023 swung fold-A model comparisons by ~0.02 (see docs/experiments/RESULTS.md)
    out["d2_pooled_cap250k"] = d2_tweedie(df[y_col].clip(upper=ROBUST_CAP), df[pred_col])
    if group_col is not None and n_boot:
        out["d2_pooled_boot"] = bootstrap_d2(df[y_col], df[pred_col], df[group_col], n_boot=n_boot)
    for line, d in df.groupby(line_col):
        out[f"d2_{line}"] = d2_tweedie(d[y_col], d[pred_col])
        out[f"calibration_{line}"] = float(d[pred_col].sum() / d[y_col].sum())  # 1.0 = unbiased
    return out


def bootstrap_d2_diff(y, pred_a, pred_b, groups, n_boot: int = 400, power: float = TWEEDIE_POWER, seed: int = SEED) -> dict:
    """Paired cluster bootstrap of D2(a) - D2(b) on the SAME rows (resample customers). Much tighter than comparing two unpaired CIs:
    most of the noise (the same rare large claims) is common to both models and cancels in the difference."""
    y = np.asarray(y, dtype=float)
    g = pd.factorize(np.asarray(groups))[0]
    k = g.max() + 1
    n = np.bincount(g, minlength=k).astype(float)
    sy = np.bincount(g, weights=y, minlength=k)
    sp = np.bincount(g, weights=np.power(y, 2.0 - power), minlength=k)
    sa = np.bincount(g, weights=tweedie_deviance(y, pred_a, power), minlength=k)
    sb = np.bincount(g, weights=tweedie_deviance(y, pred_b, power), minlength=k)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        base = _baseline_dev_total(w @ n, w @ sy, w @ sp, power)
        diffs[i] = (w @ sb - w @ sa) / base          # D2(a) - D2(b) = (dev_b - dev_a) / dev_base
    full = (sb.sum() - sa.sum()) / _baseline_dev_total(n.sum(), sy.sum(), sp.sum(), power)
    return {"diff": float(full), "se": float(diffs.std(ddof=1)), "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975)),
            "p_a_better": float((diffs > 0).mean())}
