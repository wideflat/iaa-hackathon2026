"""XGBoost and CatBoost stages for FreqSev (kind='xgb' / 'cat'): same Poisson-frequency (exposure offset) and Gamma-severity set-up as the
LightGBM stages, to give a different tree algorithm for blending (Allstate-style diversity).

Each fitted stage exposes predict_raw(X) = log-scale model output without the offset/base (frequency: add log base outside; severity: includes log mean).
* XGBoost: count:poisson / reg:gamma with base_margin (exposure offset; log mean for the severity), depth-limited histogram trees, native categoricals.
* CatBoost: Poisson with Pool baseline (offset); severity as Tweedie(variance_power=1.99) (CatBoost has no Gamma loss), symmetric trees, native categoricals.
"""
import numpy as np
import pandas as pd

from ..config import SEED


def _cat_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CatBoost wants string categories without NaN."""
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    out = X.copy()
    for c in cats:
        out[c] = out[c].astype(object).where(out[c].notna(), "missing").astype(str)
    return out, cats


class XgbStage:
    def __init__(self, booster, base_margin_const: float | None):
        self.booster, self.const = booster, base_margin_const

    @classmethod
    def fit(cls, X, label, margin, objective, params, rounds, seed, threads, const=None):
        import xgboost as xgb
        p = dict(objective=objective, eta=0.05, max_depth=3, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
                 tree_method="hist", max_cat_to_onehot=8, nthread=threads, seed=seed, base_score=1.0, verbosity=0)
        p.update(params)
        dm = xgb.DMatrix(X, label=label, base_margin=margin, enable_categorical=True)
        return cls(xgb.train(p, dm, num_boost_round=rounds), const)

    def predict_raw(self, X):
        import xgboost as xgb
        m = None if self.const is None else np.full(len(X), self.const)
        return self.booster.predict(xgb.DMatrix(X, base_margin=m, enable_categorical=True), output_margin=True)


class CatStage:
    def __init__(self, model, cats):
        self.model, self.cats = model, cats

    @classmethod
    def fit(cls, X, label, baseline, loss, params, rounds, seed, threads):
        from catboost import CatBoostRegressor, Pool
        Xc, cats = _cat_frame(X)
        p = dict(loss_function=loss, iterations=rounds, learning_rate=0.05, depth=4, l2_leaf_reg=10.0, border_count=64, random_seed=seed,
                 thread_count=threads, verbose=0, allow_writing_files=False)
        p.update(params)
        m = CatBoostRegressor(**p)
        m.fit(Pool(Xc, label=label, cat_features=cats, baseline=baseline))
        return cls(m, cats)

    def predict_raw(self, X):
        from catboost import Pool
        Xc, cats = _cat_frame(X)
        return self.model.predict(Pool(Xc, cat_features=cats), prediction_type="RawFormulaVal")


def fit_freq(kind, X, n, expo, params, rounds, n_seeds, threads):
    base = n.sum() / expo.sum()
    off = np.log(expo * base)
    if kind == "xgb":
        return [XgbStage.fit(X, n, off, "count:poisson", params, rounds, SEED + i, threads) for i in range(n_seeds)], base
    return [CatStage.fit(X, n, off, "Poisson", params, rounds, SEED + i, threads) for i in range(n_seeds)], base


def fit_sev(kind, X, y, params, rounds, n_seeds, threads):
    mu = float(np.mean(y))
    if kind == "xgb":
        margin = np.full(len(y), np.log(mu))
        p = {"min_child_weight": 30.0, **params}
        return [XgbStage.fit(X, y, margin, "reg:gamma", p, rounds, SEED + i, threads, const=np.log(mu)) for i in range(n_seeds)]
    return [CatStage.fit(X, y, np.full(len(y), np.log(mu)), "Tweedie:variance_power=1.99", params, rounds, SEED + i, threads) for i in range(n_seeds)]
