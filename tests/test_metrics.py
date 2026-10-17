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
