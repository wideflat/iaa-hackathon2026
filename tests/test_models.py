import numpy as np
import pandas as pd

from iaa.models.baselines import LGBMTweedie, MeanRateModel


def _rows(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    expo = rng.uniform(0.2, 1.0, n)
    freq = np.exp(-2.0 + 0.8 * x)
    y = np.where(rng.random(n) < 1 - np.exp(-freq * expo), rng.gamma(2.0, 500.0, n), 0.0)
    return pd.DataFrame({"x": x, "exposure_calc": expo, "y": y, "n_claims": (y > 0).astype(int), "row_id": range(n)})


def test_mean_rate_predicts_total_exactly_on_training_data():
    df = _rows()
    assert np.isclose(MeanRateModel().fit(df).predict(df).sum(), df["y"].sum())


def test_rebalanced_gbm_matches_training_total_and_is_nonnegative():
    df = _rows()
    m = LGBMTweedie(n_rounds=60, params={"min_data_in_leaf": 50}).fit(df)
    p = m.predict(df)
    assert np.isclose(p.sum(), df["y"].sum(), rtol=1e-9) and (p >= 0).all()


def test_gbm_ranks_risk_in_the_right_direction():
    df = _rows()
    m = LGBMTweedie(n_rounds=60, params={"min_data_in_leaf": 50}).fit(df)
    lo = df.assign(x=-2.0, exposure_calc=1.0)
    hi = df.assign(x=2.0, exposure_calc=1.0)
    assert m.predict(hi).mean() > m.predict(lo).mean()
