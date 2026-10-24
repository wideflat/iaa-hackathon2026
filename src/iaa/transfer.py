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
