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
* Optional (all off by default): recency=rho weights training rows (and claims) by exp(rho x (year - last year)); adv_strength=s adds domain-shift weights
  (cross-fitted classifier train-vs-target rows, odds^s, clipped; needs fit(train, target=rows to be predicted), features only, no labels);
  tail_lambda=l makes the excess load risk-dependent: excess_p x (p_large(x) / mean p_large)^l with p_large from a shallow GBM on claims above the 99th percentile (gbm only).
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


def _nn():   # torch is only needed for kind='nn'
    from . import nn
    return nn


def _boosters():   # xgboost / catboost are only needed for kind='xgb' / 'cat'
    from . import boosters
    return boosters


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
                 bands: bool = False, n_bands: int = 6, n_seeds: int = 1, rebalance_level: str = "line", outlier_mult: float | None = None,
                 nn_params: dict | None = None, threads: int = 4,
                 recency: float = 0.0, adv_strength: float = 0.0, tail_lambda: float = 0.0, entity_cols=(),
                 cat_encoding: str = "native", num_encoding: str = "native", cat_te: bool = False):
        assert kind in ("glm", "gbm", "nn", "xgb", "cat")
        c = claims_line[claims_line["row_id"].notna() & (claims_line["status"] != "repudiated")]
        self.claims = c[["row_id", "peril", "net_cost"]]
        self.kind, self.cap_quantile, self.rebalance = kind, cap_quantile, rebalance
        self.drop_noise, self.extra_drop = drop_noise, tuple(extra_drop)
        self.freq_rounds, self.sev_rounds = freq_rounds, sev_rounds
        self.alpha_f, self.alpha_s, self.min_sev_claims = glm_alpha_freq, glm_alpha_sev, min_sev_claims
        self.trend, self.completeness = trend, completeness
        self.bands, self.n_bands = bands, n_bands
        self.outlier_mult = outlier_mult   # winsorise any single claim above outlier_mult x its peril's cap in the excess load and level targets (not in the severity fit)
        assert rebalance_level in ("line", "peril")
        self.rebalance_level = rebalance_level   # 'peril': rescale each peril to its own trended training total (fixes the Gamma/Poisson total bias per peril)
        self.n_seeds = n_seeds   # GBM only: average this many boosters per stage (different bagging seeds)
        self.params_freq, self.params_sev = params_freq or {}, params_sev or {}
        self.threads = threads   # xgb / cat
        assert cat_encoding in ("native", "label", "count", "both", "target", "target_peril")
        self.cat_te = cat_te   # kind=nn: also feed the smoothed out-of-fold claim-rate (target) encoding of every categorical as a numeric input
        assert num_encoding in ("native", "lowcard", "count", "both")
        self.num_encoding = num_encoding   # tree kinds: native = numeric as is; lowcard = numerics with <= 20 distinct values also as categorical; count = log count of each exact value added; both
        self.cat_encoding = cat_encoding   # tree kinds: how categoricals enter (native = LightGBM categorical splits; label = frequency-ranked integer codes; count = level counts; both; target = smoothed out-of-fold claim-rate)
        self.recency, self.adv_strength, self.tail_lambda, self.entity_cols = recency, adv_strength, tail_lambda, tuple(entity_cols)
        self.uses_target = adv_strength > 0
        self.tail_min_claims, self.tail_min_events = 400, 40   # a peril needs this many claims / large claims to get a tail propensity model
        assert tail_lambda == 0 or kind == "gbm", "tail model is implemented for the LightGBM kind only"
        self.nn_params = nn_params or {}   # kind='nn': epochs, hidden, dropout, lr, n_seeds... (see models/nn.py)
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
        elif self.kind == "nn":
            self.enc_ = _nn().Encoder(self.num_, self.cat_, entity_cols=self.entity_cols, entity_min_count=self.nn_params.get("entity_min", 2),
                                      entity_dim=self.nn_params.get("entity_dim", 4), ple_bins=self.nn_params.get("ple_bins", 0) or 0,
                                      num_transform=self.nn_params.get("num_transform", "standard") or "standard").fit(train)
        else:
            self.levels_ = {c: sorted(train[c].dropna().astype(str).unique()) for c in self.cat_}
            vc = {c: train[c].dropna().astype(str).value_counts() for c in self.cat_}
            self.rank_ = {c: {lv: i for i, lv in enumerate(v.index)} for c, v in vc.items()}   # most frequent level = 0
            self.count_ = {c: v.to_dict() for c, v in vc.items()}
            nums = [c for c in self.num_ if not pd.api.types.is_bool_dtype(train[c])]
            self.lowcard_ = {c: sorted(train[c].dropna().unique().tolist()) for c in nums if train[c].nunique() <= 20}
            self.numcnt_ = {c: train[c].round(6).value_counts().to_dict() for c in nums}

    def _te_stats(self, train: pd.DataFrame, any_claim: np.ndarray, expo: np.ndarray, rows: np.ndarray | None = None):
        """Smoothed relative claim rate per level (exposure-weighted, prior weight 200 exposure-years); `rows` restricts the statistics to those training rows."""
        r0 = any_claim.sum() / expo.sum()
        out = {}
        for c in self.cat_:
            k = train[c].astype(object).where(train[c].notna(), "missing").astype(str).to_numpy()
            d = pd.DataFrame({"k": k, "n": any_claim, "e": expo})
            if rows is not None:
                d = d[rows]
            g = d.groupby("k").sum()
            out[c] = ((g["n"] + 200.0 * r0) / (g["e"] + 200.0) / r0).to_dict()
        return out

    def _cat(self, df):
        return df[self.cat_].astype(object).where(df[self.cat_].notna(), "missing").astype(str)

    def _x(self, df: pd.DataFrame):
        if self.kind == "nn":
            if self.cat_te:
                df = df.assign(**{f"te_{c}": df[c].astype(object).where(df[c].notna(), "missing").astype(str).map(self.te_[c]).fillna(1.0).to_numpy(dtype="float32")
                                  for c in self.cat_})
            return self.enc_.transform(df)
        if self.kind == "glm":
            num = df[self.num_].astype(float).clip(self.lo_, self.hi_, axis=1)
            num = np.clip(self.scaler_.transform(self.imp_.transform(num)), -6.0, 6.0)
            blocks = [sparse.csr_matrix(num), self.ohe_.transform(self._cat(df))]
            if self.band_idx_:
                blocks.append(self.band_.transform(num[:, self.band_idx_]).tocsr())
            return sparse.hstack(blocks).tocsr()
        x = df[self.cols_].copy()
        for c, lv in self.levels_.items():
            sv = x[c].astype(object).where(x[c].notna(), None).map(lambda v: None if v is None else str(v))
            if self.cat_encoding == "native":
                x[c] = pd.Categorical(sv, categories=lv)
            elif self.cat_encoding in ("target", "target_peril"):
                key = sv.fillna("missing")
                x[c] = key.map(self.te_[c]).fillna(1.0).astype("float32")
            else:
                if self.cat_encoding in ("count", "both"):
                    x[f"{c}_cnt"] = np.log1p(sv.map(self.count_[c]).fillna(0.0)).astype("float32")
                if self.cat_encoding in ("label", "both"):
                    x[c] = sv.map(self.rank_[c]).astype("float32")
                else:
                    x = x.drop(columns=c)
        for c in self.num_:
            if pd.api.types.is_bool_dtype(x[c]):
                x[c] = x[c].astype("int8")
        if self.num_encoding in ("lowcard", "both"):
            for c, lv in self.lowcard_.items():
                x[f"{c}_cat"] = pd.Categorical(x[c].astype("float64"), categories=lv)
        if self.num_encoding in ("count", "both"):
            for c, cnt in self.numcnt_.items():
                x[f"{c}_ncnt"] = np.log1p(df[c].round(6).map(cnt).fillna(0.0)).astype("float32")
        return x

    # ---- fitting ------------------------------------------------------------------------------
    def _adv_weights(self, train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
        """Domain-shift weights: odds that a training row looks like a row of `target` (features only), cross-fitted so a row is never scored by a model that saw it."""
        cols = self.cols_
        both = pd.concat([train[cols], target[cols]], ignore_index=True)
        for c in self.cat_:
            both[c] = both[c].astype("category").cat.codes
        Z = both.to_numpy(dtype="float32")
        zt, zg = Z[:len(train)], Z[len(train):]
        rng = np.random.default_rng(SEED)
        fold = pd.util.hash_array(train["row_id"].astype(str).to_numpy()) % 3
        w = np.ones(len(train))
        params = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
        for f in range(3):
            tr_i = np.nonzero(fold != f)[0]
            tr_i = rng.choice(tr_i, min(len(tr_i), 150_000), replace=False)
            tg_i = rng.choice(len(zg), min(len(zg), 150_000), replace=False)
            ds = lgb.Dataset(np.vstack([zt[tr_i], zg[tg_i]]), label=np.r_[np.zeros(len(tr_i)), np.ones(len(tg_i))])
            b = lgb.train(params, ds, num_boost_round=120)
            pr = np.clip(b.predict(zt[fold == f]), 1e-4, 1 - 1e-4)
            w[fold == f] = pr / (1 - pr) * (len(tr_i) / len(tg_i))
        w = np.clip(w, 0.2, 5.0) ** self.adv_strength
        return w / w.mean()

    def _row_weights(self, train: pd.DataFrame, target: pd.DataFrame | None) -> np.ndarray | None:
        w = np.ones(len(train))
        if self.recency:
            w = w * np.exp(self.recency * (train["year"].to_numpy(dtype=float) - float(train["year"].max())))
        if self.adv_strength:
            assert target is not None, "adv_strength needs fit(train, target=rows to be predicted)"
            w = w * self._adv_weights(train, target)
        return None if (not self.recency and not self.adv_strength) else w / w.mean()

    def fit(self, train: pd.DataFrame, target: pd.DataFrame | None = None):
        self._fit_features(train)
        w_row = self._row_weights(train, target)
        expo = train["exposure_calc"].to_numpy()
        pos = pd.Index(train["row_id"])
        years = pd.Series(train["year"].to_numpy(), index=pos)
        ref = int(train["year"].max()) + 1
        cl = self.claims[self.claims["row_id"].isin(pos)].copy()
        cl["year"] = cl["row_id"].map(years).to_numpy()
        counts = cl.groupby(["row_id", "peril"]).size().unstack(fill_value=0).reindex(pos, fill_value=0)
        if self.cat_encoding == "target_peril":   # one out-of-fold encoding per peril (claim rate of that peril); the base columns hold the any-claim encoding
            self.tep_, self.oofp_ = {}, {}
            fold_p = pd.util.hash_array(train["row_id"].astype(str).to_numpy()) % 5
            for p in counts.columns:
                npc = counts[p].to_numpy(dtype=float)
                if npc.sum() < 20:
                    continue
                self.tep_[p] = self._te_stats(train, npc, expo)
                o = {c: np.ones(len(train), dtype="float32") for c in self.cat_}
                for f in range(5):
                    st = self._te_stats(train, npc, expo, rows=(fold_p != f))
                    for c in self.cat_:
                        k = train[c].astype(object).where(train[c].notna(), "missing").astype(str)
                        o[c][fold_p == f] = k[fold_p == f].map(st[c]).fillna(1.0).to_numpy(dtype="float32")
                self.oofp_[p] = o
        if self.cat_encoding in ("target", "target_peril") or (self.kind == "nn" and self.cat_te):   # out-of-fold encodings for the training rows (5 folds), full-training statistics for any other rows
            any_claim = (counts.to_numpy().sum(axis=1) > 0).astype(float)
            self.te_ = self._te_stats(train, any_claim, expo)
            fold = pd.util.hash_array(train["row_id"].astype(str).to_numpy()) % 5
            oof = {c: np.ones(len(train), dtype="float32") for c in self.cat_}
            for f in range(5):
                st = self._te_stats(train, any_claim, expo, rows=(fold != f))
                for c in self.cat_:
                    k = train[c].astype(object).where(train[c].notna(), "missing").astype(str)
                    oof[c][fold == f] = k[fold == f].map(st[c]).fillna(1.0).to_numpy(dtype="float32")
        if self.kind == "nn" and self.cat_te:
            te_names = [f"te_{c}" for c in self.cat_]
            train_aug = train.assign(**{f"te_{c}": oof[c] for c in self.cat_})
            self.enc_ = _nn().Encoder(self.num_ + te_names, self.cat_, entity_cols=self.entity_cols, entity_min_count=self.nn_params.get("entity_min", 2),
                                      entity_dim=self.nn_params.get("entity_dim", 4), ple_bins=self.nn_params.get("ple_bins", 0) or 0,
                                      num_transform=self.nn_params.get("num_transform", "standard") or "standard").fit(train_aug)
            X = self.enc_.transform(train_aug)
        else:
            X = self._x(train)
        if self.cat_encoding in ("target", "target_peril"):
            for c in self.cat_:
                X[c] = oof[c]
        # A) caps and excess loads per peril
        self.freq_, self.sev_, self.excess_, self.cap_ = {}, {}, {}, {}
        self.tail_ = {}
        for p, cp in cl.groupby("peril"):
            cap = float(cp["net_cost"].quantile(self.cap_quantile)) if len(cp) >= self.min_sev_claims else float(cp["net_cost"].max())
            capped = cp["net_cost"].clip(upper=cap)
            xs_cost = cp["net_cost"] if self.outlier_mult is None else cp["net_cost"].clip(upper=self.outlier_mult * cap)
            self.cap_[p], self.excess_[p] = cap, float((xs_cost - capped).sum() / capped.sum())
        cl["cost"] = cl["net_cost"].clip(upper=cl["peril"].map(self.cap_))
        # B) severity trend: projects every claim to the cost level of year `ref`
        self.sev_trend_ = self._estimate_trend(cl) if self.trend else 0.0
        self.ref_year_ = ref
        cl["tf"] = np.exp(self.sev_trend_ * (ref - cl["year"]))
        # C) models per peril
        level = 0.0
        self.level_by_peril_ = {}
        kept = [p for p in counts.columns if counts[p].sum() >= 20]
        if self.kind == "nn":   # one multi-head network per stage instead of one model per peril
            val = (pd.util.hash_array(train["row_id" if self.entity_cols else "customer_id"].astype(str).to_numpy()) % 20 == 0)   # entity embeddings need the hold-out customers seen in training
            cm = np.column_stack([self._c_vec(p, train["year"].to_numpy()) if self.completeness else np.ones(len(train)) for p in kept])
            self.fnet_ = _nn().FreqNet(self.enc_, **self._nn_kw(_nn().FreqNet)).fit(X, counts[kept].to_numpy(dtype=float), expo[:, None] * cm, val, w=w_row)
            sev_idx, sev_p, sev_y = [], [], []
        for j, p in enumerate(kept):
            n = counts[p].to_numpy(dtype=float)
            cvec = self._c_vec(p, train["year"].to_numpy()) if self.completeness else np.ones(len(train))
            cp = cl[cl["peril"] == p]
            idx = pos.get_indexer(cp["row_id"])
            if self.kind == "nn":
                self.freq_[p] = self.sev_[p] = j
                sev_idx.append(idx)
                sev_p.append(np.full(len(idx), j))
                sev_y.append((cp["cost"] * cp["tf"]).clip(lower=1.0).to_numpy())
            else:
                Xp = X
                if self.cat_encoding == "target_peril" and p in self.oofp_:
                    Xp = X.copy()
                    for c in self.cat_:
                        Xp[c] = self.oofp_[p][c]
                self.freq_[p] = self._fit_freq(Xp, n, expo * cvec, w_row)
                self.sev_[p] = self._fit_sev(Xp, idx, (cp["cost"] * cp["tf"]).clip(lower=1.0).to_numpy(), None if w_row is None else w_row[idx])
                if self.tail_lambda:
                    self._fit_tail(p, Xp, idx, cp["net_cost"].to_numpy())
            cc = self._c_vec(p, cp["year"].to_numpy()) if self.completeness else 1.0
            lp_cost = cp["net_cost"] if self.outlier_mult is None else cp["net_cost"].clip(upper=self.outlier_mult * self.cap_[p])
            lp = float((lp_cost * cp["tf"] / cc).sum())   # ultimate-equivalent, trended training cost of this peril
            self.level_by_peril_[p] = lp
            level += lp
        if self.kind == "nn":
            idx = np.concatenate(sev_idx)
            self.snet_ = _nn().SevNet(self.enc_, **self._nn_kw(_nn().SevNet)).fit(X[idx], np.concatenate(sev_p), np.concatenate(sev_y), len(kept), val[idx],
                                                                                                                w=None if w_row is None else w_row[idx])
            self.kept_ = kept
        self.k_ = 1.0
        self.kp_ = {}
        if self.rebalance:
            if self.rebalance_level == "peril":
                by = self.expected_by_peril(train)
                self.kp_ = {p: float(self.level_by_peril_[p] / by[p].sum()) for p in by.columns}
            else:
                self.k_ = float(level / self._predict_raw(train).sum())
        return self

    def _sub(self, X, idx):
        return X.iloc[idx] if self.kind in ("gbm", "xgb", "cat") else X[idx]

    def _nn_kw(self, cls):
        import inspect
        ok = set(inspect.signature(cls.__init__).parameters) - {"self", "enc"}
        return {k: v for k, v in self.nn_params.items() if k in ok}

    def _fit_freq(self, X, n, expo, w=None):
        if self.kind == "glm":
            m = PoissonRegressor(alpha=self.alpha_f, max_iter=200).fit(X, n / expo, sample_weight=expo if w is None else expo * w)
            m.base_ = float(n.sum() / expo.sum())
            return m
        if self.kind in ("xgb", "cat"):
            return _boosters().fit_freq(self.kind, X, n, expo, self.params_freq, self.freq_rounds, self.n_seeds, self.threads)
        base = n.sum() / expo.sum()
        params = dict(objective="poisson", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
        params.update(self.params_freq)
        ds = lgb.Dataset(X, label=n, weight=w, init_score=np.log(expo * base), free_raw_data=False)
        boosters = [lgb.train({**params, "seed": SEED + i}, ds, num_boost_round=self.freq_rounds) for i in range(self.n_seeds)]
        return boosters, base

    def _fit_sev(self, X, idx, y, w=None):
        Xs = self._sub(X, idx)
        if self.kind == "glm":
            m = GammaRegressor(alpha=self.alpha_s, max_iter=200).fit(Xs, y, sample_weight=w)
            m.mean_ = float(np.mean(y))
            return m
        if self.kind in ("xgb", "cat"):
            return _boosters().fit_sev(self.kind, Xs, y, self.params_sev, self.sev_rounds, self.n_seeds, self.threads)
        params = dict(objective="gamma", learning_rate=0.05, num_leaves=15, min_data_in_leaf=60, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=0)
        params.update(self.params_sev)
        ds = lgb.Dataset(Xs, label=y, weight=w, free_raw_data=False)
        return [lgb.train({**params, "seed": SEED + i}, ds, num_boost_round=self.sev_rounds) for i in range(self.n_seeds)]

    def _fit_tail(self, p, X, idx, cost):
        """Propensity of a claim to be large (> the peril's 99th percentile of claim cost) given the risk features; used to make the excess load risk-dependent."""
        if len(cost) < self.tail_min_claims:
            return
        thr = float(np.quantile(cost, 0.99))
        y = (cost > thr).astype(float)
        if y.sum() < self.tail_min_events:
            return
        params = dict(objective="binary", learning_rate=0.05, num_leaves=4, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=20.0, verbose=-1, seed=SEED, num_threads=0)
        b = lgb.train(params, lgb.Dataset(self._sub(X, idx), label=y), num_boost_round=80)
        self.tail_[p] = (b, float(b.predict(self._sub(X, idx)).mean()))

    def _tail_ratio(self, p, X) -> np.ndarray | float:
        if self.tail_lambda == 0 or p not in getattr(self, "tail_", {}):
            return 1.0
        b, m = self.tail_[p]
        return (np.clip(b.predict(X), 1e-6, None) / m) ** self.tail_lambda

    # ---- prediction ---------------------------------------------------------------------------
    def _nn_cache(self, X, key, fn):
        if not hasattr(X, "cache"):
            X.cache = {}
        if key not in X.cache:
            X.cache[key] = fn(X)
        return X.cache[key]

    def _freq_rate(self, p, X):
        if self.kind == "nn":
            return self._nn_cache(X, "f", self.fnet_.rate)[:, self.freq_[p]]
        if self.kind == "glm":
            m = self.freq_[p]
            return np.clip(m.predict(X), 0.0, 20.0 * m.base_)   # bounded: a GLM can explode on extrapolation
        boosters, base = self.freq_[p]
        if self.kind in ("xgb", "cat"):
            return np.mean([np.exp(b.predict_raw(X)) for b in boosters], axis=0) * base
        return np.mean([np.exp(b.predict(X, raw_score=True)) for b in boosters], axis=0) * base

    def _sev(self, p, X):
        if self.kind == "nn":
            return self._nn_cache(X, "s", self.snet_.mean)[:, self.sev_[p]]
        m = self.sev_[p]
        if self.kind == "glm":
            return np.clip(m.predict(X), 0.1 * m.mean_, 10.0 * m.mean_)
        if self.kind in ("xgb", "cat"):
            return np.mean([np.exp(b.predict_raw(X)) for b in m], axis=0)
        return np.mean([b.predict(X) for b in m], axis=0)

    def _xp(self, p, X, rows: pd.DataFrame):
        """Design matrix with the categoricals replaced by peril p's claim-rate encoding (cat_encoding='target_peril')."""
        if self.cat_encoding != "target_peril" or p not in self.tep_:
            return X
        Xp = X.copy()
        for c in self.cat_:
            Xp[c] = rows[c].astype(object).where(rows[c].notna(), "missing").astype(str).map(self.tep_[p][c]).fillna(1.0).to_numpy(dtype="float32")
        return Xp

    def expected_by_peril(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = self._x(rows)
        expo = rows["exposure_calc"].to_numpy()
        kp = getattr(self, "kp_", {})
        cols = {}
        for p in self.freq_:
            Xp = self._xp(p, X, rows)
            cols[p] = kp.get(p, 1.0) * self._freq_rate(p, Xp) * expo * self._sev(p, Xp) * (1.0 + self.excess_[p] * self._tail_ratio(p, Xp))
        return pd.DataFrame(cols, index=rows.index)

    def _predict_raw(self, rows: pd.DataFrame) -> np.ndarray:
        return self.expected_by_peril(rows).sum(axis=1).to_numpy()

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.k_ * self._predict_raw(rows)

    def predict_observed(self, rows: pd.DataFrame) -> np.ndarray:
        """Prediction scaled by the expected reported share of each row's accident year: comparable with an immature (as-valued) actual."""
        by = self.expected_by_peril(rows)
        c = np.column_stack([self._c_vec(p, rows["year"].to_numpy()) for p in by.columns])
        return self.k_ * (by.to_numpy() * c).sum(axis=1)
