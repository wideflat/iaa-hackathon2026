"""Frequency x severity pricing models, per peril, combined into an expected cost per policy row.

expected cost(row) = sum over perils p of  E[claim count_p | row] x E[capped severity_p | row] x (1 + excess load_p)

* Frequency: Poisson with log link and log(exposure) offset (GLM: sample weight = exposure on the rate, which gives the same fit).
* Severity: Gamma with log link on claim-level net cost capped at a per-peril quantile (large losses distort a Gamma fit).
* Excess load: sum(cost above cap) / sum(capped cost) from training claims, loaded back multiplicatively (basic-limit / excess-loss-factor idea).
* kind='glm' (sklearn regularised GLMs on one-hot + standardised inputs) or kind='gbm' (LightGBM Poisson / Gamma).
* Severity trend (trend=True): one pooled log-linear trend per line, estimated on capped claim cost with peril fixed effects, projects every
  training claim to the cost level of the year after the last training year (= the year being predicted).
* Completeness (completeness=True): claims of immature years are under-reported at the valuation date, so the frequency offset of those rows is
  multiplied by the expected reported share of the peril and year (from the annual count triangle); predictions are at completeness 1 (ultimate).
Claims are matched to rows beforehand (target.matched_claims); repudiated claims are excluded.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from .. import development as dev
from ..config import SEED, VALUATION_DATE
from ..features import feature_columns


def _split_cols(df: pd.DataFrame, cols):
    cat = [c for c in cols if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    num = [c for c in cols if c not in cat]
    return num, cat


class FreqSev:
    name = "freqsev"

    def __init__(self, claims_line: pd.DataFrame, kind: str = "gbm", cap_quantile: float = 0.999, rebalance: bool = True,
                 drop_noise: bool = True, extra_drop=(), freq_rounds: int = 250, sev_rounds: int = 200,
                 glm_alpha_freq: float = 1e-3, glm_alpha_sev: float = 1e-2, min_sev_claims: int = 150,
                 trend: bool = True, completeness: bool = True, params_freq: dict | None = None, params_sev: dict | None = None,
                 bands: bool = False, n_bands: int = 6):
        assert kind in ("glm", "gbm")
        c = claims_line[claims_line["row_id"].notna() & (claims_line["status"] != "repudiated")]
        self.claims = c[["row_id", "peril", "net_cost"]]
        self.kind, self.cap_quantile, self.rebalance = kind, cap_quantile, rebalance
        self.drop_noise, self.extra_drop = drop_noise, tuple(extra_drop)
        self.freq_rounds, self.sev_rounds = freq_rounds, sev_rounds
        self.alpha_f, self.alpha_s, self.min_sev_claims = glm_alpha_freq, glm_alpha_sev, min_sev_claims
        self.trend, self.completeness = trend, completeness
        self.bands, self.n_bands = bands, n_bands
        self.params_freq, self.params_sev = params_freq or {}, params_sev or {}
        self.sev_trend_ = 0.0
        has_dates = {"loss_date", "report_date"} <= set(claims_line.columns)
        self.c_by_year = self._completeness_table(claims_line) if (completeness and has_dates) else {}

    @staticmethod
    def _completeness_table(claims_line: pd.DataFrame) -> dict:
        """{peril: {year: expected reported share of that accident year at the valuation date}} from annual count triangles."""
        c = claims_line[claims_line["status"] != "repudiated"]
        last = VALUATION_DATE.year
        out = {}
        for p, d in c.groupby("peril"):
            tri = dev.annual_triangle(d, None, years=tuple(range(2021, last + 1)), n_ages=last - 2021 + 1)
            links = pd.Series(dev.link_ratios(tri.set_axis(range(tri.shape[1]), axis=1))).fillna(1.0).to_numpy()
            cdf = dev.factors_to_ultimate(links, dev.tail_exp_decay(links))
            out[p] = {y: float(1.0 / cdf[last - y]) for y in range(2021, last + 1)}
        return out

    def _c_vec(self, p, years) -> np.ndarray:
        """Completeness per row for peril p (1.0 for years without a factor, e.g. the year being predicted)."""
        table = self.c_by_year.get(p, {})
        return pd.Series(np.asarray(years)).map(table).fillna(1.0).to_numpy()

    @staticmethod
    def _estimate_trend(cl: pd.DataFrame) -> float:
        """Pooled annual log-trend of capped claim cost with peril fixed effects (clipped to [0, 0.12])."""
        if cl["year"].nunique() < 2:
            return 0.0
        y = np.log(cl["cost"].clip(lower=1.0))
        x = cl["year"].astype(float)
        dy = y - y.groupby(cl["peril"]).transform("mean")
        dx = x - x.groupby(cl["peril"]).transform("mean")
        den = float((dx ** 2).sum())
        return float(np.clip((dx * dy).sum() / den, 0.0, 0.12)) if den > 0 else 0.0

    # ---- feature matrices -------------------------------------------------------------------
    def _fit_features(self, train: pd.DataFrame):
        self.cols_ = feature_columns(train, self.drop_noise, self.extra_drop)
        self.num_, self.cat_ = _split_cols(train, self.cols_)
        if self.kind == "glm":
            num = train[self.num_].astype(float)
            self.lo_, self.hi_ = num.quantile(0.005), num.quantile(0.995)      # winsorise heavy-tailed inputs
            num = num.clip(self.lo_, self.hi_, axis=1)
            self.imp_ = SimpleImputer(strategy="median", add_indicator=True).fit(num)
            self.scaler_ = StandardScaler().fit(self.imp_.transform(num))
            self.band_idx_ = []
            if self.bands:   # actuarial banding: quantile bands (sparse one-hot) for continuous inputs, on top of the linear terms
                self.band_idx_ = [i for i, c in enumerate(self.num_) if num[c].nunique() > 10]
                z = np.clip(self.scaler_.transform(self.imp_.transform(num)), -6.0, 6.0)[:, self.band_idx_]
                self.band_ = KBinsDiscretizer(n_bins=self.n_bands, encode="onehot", strategy="quantile", subsample=200000,
                                              dtype=np.float32, random_state=SEED).fit(z)
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", min_frequency=1000, sparse_output=True).fit(self._cat(train))
        else:
            self.levels_ = {c: sorted(train[c].dropna().astype(str).unique()) for c in self.cat_}

    def _cat(self, df):
        return df[self.cat_].astype(object).where(df[self.cat_].notna(), "missing").astype(str)

    def _x(self, df: pd.DataFrame):
        if self.kind == "glm":
            num = df[self.num_].astype(float).clip(self.lo_, self.hi_, axis=1)
            num = np.clip(self.scaler_.transform(self.imp_.transform(num)), -6.0, 6.0)
            blocks = [sparse.csr_matrix(num), self.ohe_.transform(self._cat(df))]
            if self.band_idx_:
                blocks.append(self.band_.transform(num[:, self.band_idx_]).tocsr())
            return sparse.hstack(blocks).tocsr()
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
        years = pd.Series(train["year"].to_numpy(), index=pos)
        ref = int(train["year"].max()) + 1
        cl = self.claims[self.claims["row_id"].isin(pos)].copy()
        cl["year"] = cl["row_id"].map(years).to_numpy()
        counts = cl.groupby(["row_id", "peril"]).size().unstack(fill_value=0).reindex(pos, fill_value=0)
        # A) caps and excess loads per peril
        self.freq_, self.sev_, self.excess_, self.cap_ = {}, {}, {}, {}
        for p, cp in cl.groupby("peril"):
            cap = float(cp["net_cost"].quantile(self.cap_quantile)) if len(cp) >= self.min_sev_claims else float(cp["net_cost"].max())
            capped = cp["net_cost"].clip(upper=cap)
            self.cap_[p], self.excess_[p] = cap, float((cp["net_cost"] - capped).sum() / capped.sum())
        cl["cost"] = cl["net_cost"].clip(upper=cl["peril"].map(self.cap_))
        # B) severity trend: projects every claim to the cost level of year `ref`
        self.sev_trend_ = self._estimate_trend(cl) if self.trend else 0.0
        self.ref_year_ = ref
        cl["tf"] = np.exp(self.sev_trend_ * (ref - cl["year"]))
        # C) models per peril
        level = 0.0
        for p in counts.columns:
            n = counts[p].to_numpy(dtype=float)
            if n.sum() < 20:
                continue
            cvec = self._c_vec(p, train["year"].to_numpy()) if self.completeness else np.ones(len(train))
            self.freq_[p] = self._fit_freq(X, n, expo * cvec)
            cp = cl[cl["peril"] == p]
            idx = pos.get_indexer(cp["row_id"])
            self.sev_[p] = self._fit_sev(X, idx, (cp["cost"] * cp["tf"]).clip(lower=1.0).to_numpy())
            cc = self._c_vec(p, cp["year"].to_numpy()) if self.completeness else 1.0
            level += float((cp["net_cost"] * cp["tf"] / cc).sum())   # ultimate-equivalent, trended training cost of this peril
        self.k_ = 1.0
        if self.rebalance:
            self.k_ = float(level / self._predict_raw(train).sum())
        return self

    def _sub(self, X, idx):
        return X[idx] if self.kind == "glm" else X.iloc[idx]

    def _fit_freq(self, X, n, expo):
        if self.kind == "glm":
            m = PoissonRegressor(alpha=self.alpha_f, max_iter=200).fit(X, n / expo, sample_weight=expo)
            m.base_ = float(n.sum() / expo.sum())
            return m
        base = n.sum() / expo.sum()
        params = dict(objective="poisson", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
        params.update(self.params_freq)
        b = lgb.train(params, lgb.Dataset(X, label=n, init_score=np.log(expo * base)), num_boost_round=self.freq_rounds)
        return b, base

    def _fit_sev(self, X, idx, y):
        Xs = self._sub(X, idx)
        if self.kind == "glm":
            m = GammaRegressor(alpha=self.alpha_s, max_iter=200).fit(Xs, y)
            m.mean_ = float(np.mean(y))
            return m
        params = dict(objective="gamma", learning_rate=0.05, num_leaves=15, min_data_in_leaf=60, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
        params.update(self.params_sev)
        return lgb.train(params, lgb.Dataset(Xs, label=y), num_boost_round=self.sev_rounds)

    # ---- prediction ---------------------------------------------------------------------------
    def _freq_rate(self, p, X):
        if self.kind == "glm":
            m = self.freq_[p]
            return np.clip(m.predict(X), 0.0, 20.0 * m.base_)   # bounded: a GLM can explode on extrapolation
        b, base = self.freq_[p]
        return np.exp(b.predict(X, raw_score=True)) * base

    def _sev(self, p, X):
        m = self.sev_[p]
        if self.kind == "glm":
            return np.clip(m.predict(X), 0.1 * m.mean_, 10.0 * m.mean_)
        return m.predict(X)

    def expected_by_peril(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = self._x(rows)
        expo = rows["exposure_calc"].to_numpy()
        return pd.DataFrame({p: self._freq_rate(p, X) * expo * self._sev(p, X) * (1.0 + self.excess_[p]) for p in self.freq_}, index=rows.index)

    def _predict_raw(self, rows: pd.DataFrame) -> np.ndarray:
        return self.expected_by_peril(rows).sum(axis=1).to_numpy()

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.k_ * self._predict_raw(rows)

    def predict_observed(self, rows: pd.DataFrame) -> np.ndarray:
        """Prediction scaled by the expected reported share of each row's accident year: comparable with an immature (as-valued) actual."""
        by = self.expected_by_peril(rows)
        c = np.column_stack([self._c_vec(p, rows["year"].to_numpy()) for p in by.columns])
        return self.k_ * (by.to_numpy() * c).sum(axis=1)
