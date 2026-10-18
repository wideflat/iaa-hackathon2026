"""Frequency x severity pricing models, per peril, combined into an expected cost per policy row.

expected cost(row) = sum over perils p of  E[claim count_p | row] x E[capped severity_p | row] x (1 + excess load_p)

* Frequency: Poisson with log link and log(exposure) offset (GLM: sample weight = exposure on the rate, which gives the same fit).
* Severity: Gamma with log link on claim-level net cost capped at a per-peril quantile (large losses distort a Gamma fit).
* Excess load: sum(cost above cap) / sum(capped cost) from training claims, loaded back multiplicatively (basic-limit / excess-loss-factor idea).
* kind='glm' (sklearn regularised GLMs on one-hot + standardised inputs) or kind='gbm' (LightGBM Poisson / Gamma).
Claims are matched to rows beforehand (target.matched_claims); repudiated claims are excluded.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import SEED
from ..features import feature_columns


def _split_cols(df: pd.DataFrame, cols):
    cat = [c for c in cols if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    num = [c for c in cols if c not in cat]
    return num, cat


class FreqSev:
    name = "freqsev"

    def __init__(self, claims_line: pd.DataFrame, kind: str = "gbm", cap_quantile: float = 0.999, rebalance: bool = True,
                 drop_noise: bool = True, extra_drop=(), freq_rounds: int = 250, sev_rounds: int = 200,
                 glm_alpha_freq: float = 1e-4, glm_alpha_sev: float = 1e-3, min_sev_claims: int = 150):
        assert kind in ("glm", "gbm")
        c = claims_line[claims_line["row_id"].notna() & (claims_line["status"] != "repudiated")]
        self.claims = c[["row_id", "peril", "net_cost"]]
        self.kind, self.cap_quantile, self.rebalance = kind, cap_quantile, rebalance
        self.drop_noise, self.extra_drop = drop_noise, tuple(extra_drop)
        self.freq_rounds, self.sev_rounds = freq_rounds, sev_rounds
        self.alpha_f, self.alpha_s, self.min_sev_claims = glm_alpha_freq, glm_alpha_sev, min_sev_claims

    # ---- feature matrices -------------------------------------------------------------------
    def _fit_features(self, train: pd.DataFrame):
        self.cols_ = feature_columns(train, self.drop_noise, self.extra_drop)
        self.num_, self.cat_ = _split_cols(train, self.cols_)
        if self.kind == "glm":
            num = train[self.num_].astype(float)
            self.imp_ = SimpleImputer(strategy="median", add_indicator=True).fit(num)
            self.scaler_ = StandardScaler().fit(self.imp_.transform(num))
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", min_frequency=1000, sparse_output=True).fit(self._cat(train))
        else:
            self.levels_ = {c: sorted(train[c].dropna().astype(str).unique()) for c in self.cat_}

    def _cat(self, df):
        return df[self.cat_].astype(object).where(df[self.cat_].notna(), "missing").astype(str)

    def _x(self, df: pd.DataFrame):
        if self.kind == "glm":
            num = self.scaler_.transform(self.imp_.transform(df[self.num_].astype(float)))
            return sparse.hstack([sparse.csr_matrix(num), self.ohe_.transform(self._cat(df))]).tocsr()
        x = df[self.cols_].copy()
        for c, lv in self.levels_.items():
            x[c] = pd.Categorical(x[c].astype(object).where(x[c].notna(), None).map(lambda v: None if v is None else str(v)), categories=lv)
        for c in self.num_:
            if pd.api.types.is_bool_dtype(x[c]):
                x[c] = x[c].astype("int8")
        return x

    # ---- fitting ------------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame):
        self._fit_features(train)
        X = self._x(train)
        expo = train["exposure_calc"].to_numpy()
        pos = pd.Index(train["row_id"])
        cl = self.claims[self.claims["row_id"].isin(pos)]
        counts = cl.groupby(["row_id", "peril"]).size().unstack(fill_value=0).reindex(pos, fill_value=0)
        self.freq_, self.sev_, self.excess_, self.cap_ = {}, {}, {}, {}
        for p in counts.columns:
            n = counts[p].to_numpy(dtype=float)
            if n.sum() < 20:
                continue
            self.freq_[p] = self._fit_freq(X, n, expo)
            cp = cl[cl["peril"] == p]
            cap = float(cp["net_cost"].quantile(self.cap_quantile)) if len(cp) >= self.min_sev_claims else float(cp["net_cost"].max())
            capped = cp["net_cost"].clip(upper=cap)
            self.cap_[p], self.excess_[p] = cap, float((cp["net_cost"] - capped).sum() / capped.sum())
            idx = pos.get_indexer(cp["row_id"])
            self.sev_[p] = self._fit_sev(X, idx, capped.to_numpy())
        self.k_ = 1.0
        if self.rebalance:
            self.k_ = float(train["y"].sum() / self._predict_raw(train).sum())
        return self

    def _sub(self, X, idx):
        return X[idx] if self.kind == "glm" else X.iloc[idx]

    def _fit_freq(self, X, n, expo):
        if self.kind == "glm":
            return PoissonRegressor(alpha=self.alpha_f, max_iter=200).fit(X, n / expo, sample_weight=expo)
        base = n.sum() / expo.sum()
        params = dict(objective="poisson", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
        b = lgb.train(params, lgb.Dataset(X, label=n, init_score=np.log(expo * base)), num_boost_round=self.freq_rounds)
        return b, base

    def _fit_sev(self, X, idx, y):
        Xs = self._sub(X, idx)
        if self.kind == "glm":
            return GammaRegressor(alpha=self.alpha_s, max_iter=200).fit(Xs, y)
        params = dict(objective="gamma", learning_rate=0.05, num_leaves=15, min_data_in_leaf=60, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
        return lgb.train(params, lgb.Dataset(Xs, label=y), num_boost_round=self.sev_rounds)

    # ---- prediction ---------------------------------------------------------------------------
    def _freq_rate(self, p, X):
        if self.kind == "glm":
            return self.freq_[p].predict(X)
        b, base = self.freq_[p]
        return np.exp(b.predict(X, raw_score=True)) * base

    def _sev(self, p, X):
        return self.sev_[p].predict(X)

    def expected_by_peril(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = self._x(rows)
        expo = rows["exposure_calc"].to_numpy()
        return pd.DataFrame({p: self._freq_rate(p, X) * expo * self._sev(p, X) * (1.0 + self.excess_[p]) for p in self.freq_}, index=rows.index)

    def _predict_raw(self, rows: pd.DataFrame) -> np.ndarray:
        return self.expected_by_peril(rows).sum(axis=1).to_numpy()

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.k_ * self._predict_raw(rows)
