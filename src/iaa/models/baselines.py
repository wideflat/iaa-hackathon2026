import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import SEED, TWEEDIE_POWER
from ..features import feature_columns


class MeanRateModel:
    """Constant pure premium per line: sum(y)/sum(exposure) x exposure. The floor any model must beat."""

    name = "mean_rate"

    def fit(self, train: pd.DataFrame):
        self.rate_ = train["y"].sum() / train["exposure_calc"].sum()
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.rate_ * rows["exposure_calc"].to_numpy()


class LGBMTweedie:
    """Direct Tweedie GBM on the claim *rate* (y / exposure), weighted by exposure."""

    name = "lgbm_tweedie"

    def __init__(self, drop_noise: bool = True, extra_drop=(), params: dict | None = None, n_rounds: int = 400):
        self.drop_noise, self.extra_drop, self.n_rounds = drop_noise, tuple(extra_drop), n_rounds
        self.params = {
            "objective": "tweedie", "tweedie_variance_power": TWEEDIE_POWER, "learning_rate": 0.03,
            "num_leaves": 31, "min_data_in_leaf": 400, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 10.0, "cat_smooth": 50, "verbose": -1, "seed": SEED, "num_threads": 0,
        }
        if params:
            self.params.update(params)

    def _x(self, rows: pd.DataFrame) -> pd.DataFrame:
        x = rows[self.cols_].copy()
        for c, cats in self.cats_.items():
            x[c] = pd.Categorical(x[c].astype(object), categories=cats)
        return x

    def fit(self, train: pd.DataFrame):
        self.cols_ = feature_columns(train, self.drop_noise, self.extra_drop)
        self.cats_ = {c: sorted(train[c].dropna().astype(str).unique()) for c in self.cols_
                      if not pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])}
        for c in self.cats_:
            train = train.assign(**{c: train[c].astype(str).where(train[c].notna())})
        x = self._x(train)
        for c in x.columns:
            if pd.api.types.is_bool_dtype(x[c]):
                x[c] = x[c].astype("int8")
        rate = train["y"] / train["exposure_calc"]
        ds = lgb.Dataset(x, label=rate, weight=train["exposure_calc"], free_raw_data=True)
        self.booster_ = lgb.train(self.params, ds, num_boost_round=self.n_rounds)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        for c in self.cats_:
            rows = rows.assign(**{c: rows[c].astype(str).where(rows[c].notna())})
        x = self._x(rows)
        for c in x.columns:
            if pd.api.types.is_bool_dtype(x[c]):
                x[c] = x[c].astype("int8")
        return self.booster_.predict(x) * rows["exposure_calc"].to_numpy()
