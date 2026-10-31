"""Optional engineered features (label-free: nothing here uses a claim or a cost, so computing them on train, validation and test rows together is safe).

* ratios: premium per sum insured / vehicle value, licence age (driver_age - years_licensed), mileage per vehicle age, history frequency and severity
* weather anomalies: each weather variable minus its region-and-year mean, and minus its previous-year value
* prem_resid: log written premium minus a cross-fitted regression of log premium on the other rating features = the underwriter's load beyond the rating factors

add_engineered(frames) returns new frames (the inputs are not modified) and caches the result under artifacts/ keyed by a hash of this file and the frames.
"""
import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, LINES, SEED
from .features import feature_columns

WEATHER = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm"]


def _ratios(d: pd.DataFrame) -> pd.DataFrame:
    out = {}
    if {"driver_age", "years_licensed"} <= set(d):
        out["licence_age"] = d["driver_age"] - d["years_licensed"]
    if {"annual_mileage", "vehicle_age"} <= set(d):
        out["mileage_per_vehicle_age"] = d["annual_mileage"] / (d["vehicle_age"] + 1.0)
    if "sum_insured" in d:
        out["prem_rate"] = d["written_premium"] / d["sum_insured"].clip(lower=1.0)
    if "vehicle_value_group" in d:
        out["prem_per_value_group"] = d["written_premium"] / (d["vehicle_value_group"] + 1.0)
    out["hist_rate"] = d["hist_n_claims"] / (d["renewal_count"] + 1.0)
    out["hist_sev"] = d["hist_cost"] / (d["hist_n_claims"] + 1.0)
    for w in WEATHER + ["mean_temperature_c"]:
        if w in d:
            out[f"{w}_anom"] = d[w] - d.groupby(["region", "year"])[w].transform("mean")
        if f"{w}_lag1" in d:
            out[f"{w}_chg"] = d[w] - d[f"{w}_lag1"]
    return pd.DataFrame(out, index=d.index)


def _prem_resid(d: pd.DataFrame) -> pd.Series:
    cols = [c for c in feature_columns(d, True, ()) if c not in ("written_premium", "discount_applied")]
    X = d[cols].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
        elif pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("int8")
    y = np.log(d["written_premium"].clip(lower=1.0).to_numpy())
    fold = pd.util.hash_array(d["row_id"].astype(str).to_numpy()) % 3
    pred = np.zeros(len(d))
    params = dict(objective="regression", learning_rate=0.1, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, verbose=-1, seed=SEED, num_threads=0)
    rng = np.random.default_rng(SEED)
    for f in range(3):
        tr = np.nonzero(fold != f)[0]
        tr = rng.choice(tr, min(len(tr), 300_000), replace=False)
        b = lgb.train(params, lgb.Dataset(X.iloc[tr], label=y[tr]), num_boost_round=150)
        pred[fold == f] = b.predict(X[fold == f])
    return pd.Series(y - pred, index=d.index)


def _source_hash(frames: dict) -> str:
    h = hashlib.sha1(Path(__file__).read_bytes())
    for l in LINES:
        h.update(str(frames[l].shape).encode())
        h.update(",".join(frames[l].columns).encode())
    return h.hexdigest()[:10]


def add_engineered(frames: dict) -> dict:
    key = _source_hash(frames)
    out = {}
    for line in LINES:
        path = ARTIFACTS_DIR / f"engineered_{key}_{line}.parquet"
        d = frames[line]
        if path.exists():
            extra = pd.read_parquet(path)
        else:
            extra = _ratios(d).assign(prem_resid=_prem_resid(d)).astype("float32")
            extra.insert(0, "row_id", d["row_id"].to_numpy())
            extra.to_parquet(path)
        out[line] = pd.concat([d.reset_index(drop=True), extra.drop(columns="row_id").reset_index(drop=True)], axis=1).set_index(d.index)
    return out
