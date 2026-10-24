"""Transfer of the developed-market model to the emerging market: calibration and combination helpers (pure functions).

The developed model is trained on four years of a mature book. The emerging book has two years, a different mix and different levels
(frequency, severity). Transfer keeps the developed *relativities* and re-estimates *levels* on the emerging book:

  * calibration_factors: actual / predicted cost per (line x peril) on emerging rows the developed model never saw, shrunk towards the
    line factor with weight `shrink` on the peril's own evidence (a peril with few claims should not move on noise);
  * apply_factors: scale per-peril predictions by those factors and sum;
  * geometric_blend: exp of a weighted mean of logs, for combining a calibrated transfer prediction with an emerging-only model.
"""
import numpy as np
import pandas as pd


def calibration_factors(actual: pd.DataFrame, predicted: pd.DataFrame, n_claims: pd.Series, k: float = 100.0) -> pd.Series:
    """Factor per peril column: actual total / predicted total, credibility-weighted towards the all-peril factor.

    actual, predicted: rows x perils (same index and columns); n_claims: claims per peril in `actual`.
    Credibility Z = n / (n + k): a peril with n claims keeps Z of its own ratio and (1 - Z) of the line ratio."""
    assert list(actual.columns) == list(predicted.columns), "actual and predicted must have the same perils"
    assert actual.index.equals(predicted.index), "actual and predicted must cover the same rows"
    line_ratio = float(actual.to_numpy().sum() / predicted.to_numpy().sum())
    own = actual.sum() / predicted.sum()
    z = n_claims.reindex(actual.columns).fillna(0.0) / (n_claims.reindex(actual.columns).fillna(0.0) + k)
    return (z * own + (1.0 - z) * line_ratio).rename("factor")


def apply_factors(predicted: pd.DataFrame, factors: pd.Series) -> np.ndarray:
    assert set(predicted.columns) <= set(factors.index), f"no factor for {set(predicted.columns) - set(factors.index)}"
    return (predicted * factors.reindex(predicted.columns)).sum(axis=1).to_numpy()


def geometric_blend(a: np.ndarray, b: np.ndarray, weight_a: float) -> np.ndarray:
    assert 0.0 <= weight_a <= 1.0
    return np.exp(weight_a * np.log(np.maximum(a, 1e-9)) + (1.0 - weight_a) * np.log(np.maximum(b, 1e-9)))


def credibility(n: pd.Series | float, k: float) -> pd.Series | float:
    """Z = n / (n + k): the weight a segment's own evidence gets (k = the number of claims worth as much as the prior)."""
    return n / (n + k)


def split_factors(actual_n: pd.Series, expected_n: pd.Series, actual_cost: pd.Series, expected_cost: pd.Series, k_freq: float, k_sev: float) -> pd.DataFrame:
    """Per-peril frequency and severity factors, each shrunk towards the all-peril value of its own kind.

    Inputs are totals per peril over the calibration rows (actual claims / cost vs the developed model's expected claims / cost).
    freq = actual_n / expected_n; the cost ratio is freq x sev, so sev = (actual_cost / expected_cost) / freq.
    Frequency is Poisson (credibility Z = n / (n + k_freq)); severity is noisier (k_sev larger). The line-level severity factor is the pooled
    ratio after the frequency factors are applied; shrinkage keeps the line total close to, not exactly at, the observed total (a final line rebalance can close the gap)."""
    assert actual_n.index.equals(expected_n.index) and actual_cost.index.equals(expected_cost.index) and actual_n.index.equals(actual_cost.index)
    f_line = float(actual_n.sum() / expected_n.sum())
    f_own = actual_n / expected_n
    f = credibility(actual_n, k_freq) * f_own + (1.0 - credibility(actual_n, k_freq)) * f_line
    s_line = float(actual_cost.sum() / (f * expected_cost).sum())
    s_own = (actual_cost / expected_cost) / f_own.replace(0.0, np.nan)
    z_s = credibility(actual_n, k_sev)
    s = (z_s * s_own.fillna(s_line) + (1.0 - z_s) * s_line)
    return pd.DataFrame({"freq": f, "sev": s, "n_claims": actual_n})
