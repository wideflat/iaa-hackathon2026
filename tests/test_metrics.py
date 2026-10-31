import numpy as np
import pytest
from sklearn.metrics import d2_tweedie_score

from iaa import metrics


def _data(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    pred = rng.gamma(2.0, 50.0, n)
    y = np.where(rng.random(n) < 0.1, rng.gamma(2.0, pred * 5), 0.0)
    return y, pred


def test_d2_matches_sklearn():
    y, pred = _data()
    assert metrics.d2_tweedie(y, pred) == pytest.approx(d2_tweedie_score(y, pred, power=1.5), rel=1e-9)


def test_constant_prediction_scores_zero_and_perfect_close_to_one():
    y, _ = _data()
    assert metrics.d2_tweedie(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-12)
    assert metrics.d2_tweedie(y, np.maximum(y, 1e-3)) > 0.99


def test_bootstrap_with_unique_groups_is_consistent_with_point_estimate():
    y, pred = _data()
    b = metrics.bootstrap_d2(y, pred, groups=np.arange(len(y)), n_boot=200)
    point = metrics.d2_tweedie(y, pred)
    assert b["lo"] <= point <= b["hi"] and b["se"] > 0


def test_cluster_bootstrap_is_wider_than_row_bootstrap_when_rows_are_correlated():
    y, pred = _data(n=6000)
    y = np.repeat(y[:1000], 6)  # each "customer" has 6 identical rows
    pred = np.repeat(pred[:1000], 6)
    groups = np.repeat(np.arange(1000), 6)
    assert metrics.bootstrap_d2(y, pred, groups, 200)["se"] > metrics.bootstrap_d2(y, pred, np.arange(6000), 200)["se"]


def test_leak_alarm_fires_on_implausibly_good_scores_only():
    metrics.assert_not_too_good(0.13)
    with pytest.raises(AssertionError):
        metrics.assert_not_too_good(0.99, "test")


def test_paired_bootstrap_detects_a_real_difference_and_is_tighter_than_unpaired_cis():
    y, pred = _data(n=20000)
    better = pred
    worse = pred * np.random.default_rng(1).lognormal(0, 0.25, len(pred))   # the same model plus noise
    groups = np.arange(len(y))
    d = metrics.bootstrap_d2_diff(y, better, worse, groups, n_boot=200)
    assert d["diff"] > 0 and d["lo"] > 0 and d["p_a_better"] > 0.97
    unpaired = metrics.bootstrap_d2(y, better, groups, n_boot=200)["se"]
    assert d["se"] < unpaired                                              # common noise cancels in the paired difference
    same = metrics.bootstrap_d2_diff(y, better, better, groups, n_boot=50)
    assert abs(same["diff"]) < 1e-12 and same["se"] < 1e-12


def test_robust_score_ignores_a_single_extreme_row_but_the_official_score_does_not():
    y, pred = _data(n=5000)
    y_out = y.copy()
    y_out[0] = 1e7
    import pandas as pd
    df = pd.DataFrame({"y": y_out, "pred": pred, "line": "motor", "customer_id": np.arange(len(y))})
    r = metrics.score_report(df, n_boot=0)
    clean = pd.DataFrame({"y": np.minimum(y_out, metrics.ROBUST_CAP), "pred": pred, "line": "motor", "customer_id": np.arange(len(y))})
    assert r["d2_pooled_cap250k"] == pytest.approx(metrics.d2_tweedie(clean["y"], clean["pred"]))
    assert r["d2_pooled_cap250k"] > r["d2_pooled"] + 0.02 or abs(r["d2_pooled_cap250k"] - r["d2_pooled"]) > 0.01
