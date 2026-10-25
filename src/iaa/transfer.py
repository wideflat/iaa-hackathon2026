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


def group_factors(actual: pd.Series, predicted: pd.Series, groups: pd.Series, n_claims: pd.Series, k: float) -> pd.Series:
    """Multiplicative factor per level of a rating factor that exists in one market only (e.g. gender in the emerging market).

    ratio(level) = sum actual / sum predicted over the rows of the level, credibility-weighted towards 1 on the log scale
    (Z = claims / (claims + k)), then divided by the exposure-free weighted mean (weights = predicted) so the factors leave the
    predicted total unchanged."""
    assert actual.index.equals(predicted.index) and actual.index.equals(groups.index) and actual.index.equals(n_claims.index)
    ratio = actual.groupby(groups).sum() / predicted.groupby(groups).sum()
    z = credibility(n_claims.groupby(groups).sum().astype(float), k)
    raw = np.exp(z * np.log(ratio))
    w = predicted.groupby(groups).sum()
    return raw / float((raw * w).sum() / w.sum())


def calibrate_and_apply(pred_train: pd.DataFrame, act_by_peril: pd.DataFrame, n_by_peril: pd.Series, y_train: pd.Series, n_claims_train: pd.Series,
                        gender_train: pd.Series, pred_test: pd.DataFrame, gender_test: pd.Series, k_peril: float = 300.0, k_group: float = 300.0) -> tuple[np.ndarray, dict]:
    """Final transfer step for one line: peril factors -> line closure -> group (gender) factors, fitted on the emerging training rows and applied to the test rows.

    pred_train / pred_test: developed-model expected cost per peril (training rows at the as-valued basis, i.e. already multiplied by the completeness
    of their accident year; test rows on the ultimate basis). act_by_peril: actual net cost per row x peril of the training rows. Returns (prediction, factors)."""
    f_peril = calibration_factors(act_by_peril, pred_train, n_by_peril, k=k_peril)
    base_train = pd.Series(apply_factors(pred_train, f_peril), index=pred_train.index)
    closure = float(y_train.sum() / base_train.sum())
    f_group = group_factors(y_train, base_train * closure, gender_train, n_claims_train, k=k_group)
    pred = apply_factors(pred_test, f_peril) * closure * gender_test.map(f_group).to_numpy()
    return pred, {"peril": f_peril.to_dict(), "closure": closure, "group": f_group.to_dict()}


def robust_factors(act_n: pd.Series, exp_n: pd.Series, med_em: pd.Series, med_dev: pd.Series, k_freq: float = 20.0, k_sev: float = 30.0) -> pd.DataFrame:
    """Frequency factor and severity factor per peril that do not depend on a few large claims.

    freq_p: actual claim count / developed expected count, shrunk towards the line ratio (Poisson credibility, k_freq claims).
    sev_p: ratio of the emerging to the developed MEDIAN claim cost of the peril (robust to large losses), shrunk towards the claim-weighted line ratio of the
    medians with Z = n / (n + k_sev). The cost factor of a peril is freq_p x sev_p; expected developed cost already carries the developed severity trend and excess load."""
    assert act_n.index.equals(exp_n.index) and act_n.index.equals(med_em.index) and act_n.index.equals(med_dev.index)
    f_line = float(act_n.sum() / exp_n.sum())
    z_f = credibility(act_n, k_freq)
    freq = z_f * (act_n / exp_n) + (1.0 - z_f) * f_line
    ratio = med_em / med_dev
    s_line = float((ratio * act_n).sum() / act_n.sum())
    z_s = credibility(act_n, k_sev)
    sev = z_s * ratio + (1.0 - z_s) * s_line
    return pd.DataFrame({"freq": freq, "sev": sev, "cost_factor": freq * sev, "n_claims": act_n})


def calibrate_robust(cost_train: pd.DataFrame, count_train: pd.DataFrame, n_by_peril: pd.Series, y_train: pd.Series, n_claims_train: pd.Series, gender_train: pd.Series,
                     med_em: pd.Series, med_dev: pd.Series, cost_test: pd.DataFrame, gender_test: pd.Series, k_freq: float = 20.0, k_sev: float = 30.0,
                     k_group: float = 300.0, tw_train: pd.Series | None = None, tw_test: pd.Series | None = None, tw_share: float = 0.0) -> tuple[np.ndarray, dict]:
    """Final transfer step with the robust level: peril cost factor = frequency factor (counts) x severity factor (medians), a gender factor on top, NO final line closure.

    The closure used before copies the large-loss noise of the calibration years into the level (two-direction test, docs/part2/EXPLORATION.md)."""
    rf = robust_factors(n_by_peril, count_train.sum(), med_em, med_dev, k_freq, k_sev)
    fs_train = pd.Series(apply_factors(cost_train, rf["cost_factor"]), index=cost_train.index)
    fs_test = apply_factors(cost_test, rf["cost_factor"])
    scale = 1.0
    if tw_share > 0:   # direct Tweedie GBM: relativity only, its level follows the robust level on the calibration rows
        assert tw_train is not None and tw_test is not None
        scale = float(fs_train.sum() / tw_train.sum())
        base_train = (1.0 - tw_share) * fs_train + tw_share * scale * tw_train
        base_test = (1.0 - tw_share) * fs_test + tw_share * scale * tw_test.to_numpy()
    else:
        base_train, base_test = fs_train, fs_test
    f_group = group_factors(y_train, base_train, gender_train, n_claims_train, k=k_group)
    pred = base_test * gender_test.map(f_group).to_numpy()
    return pred, {"peril": rf.to_dict("index"), "group": f_group.to_dict(), "train_actual_over_predicted": float(y_train.sum() / base_train.sum()), "tw_scale": scale}
