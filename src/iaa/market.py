"""Part 2 harmonisation: put the developed and the emerging portfolio into one feature space.

The two portfolios differ in what they record (docs/part2/DATA_REVIEW.md):
  * age: developed has the exact age (driver_age / policyholder_age, customer date of birth); emerging only the band
    17-24 / 25-34 / 35-49 / 50-64 / 65+  -> a shared `age_band` (developed ages are cut at the emerging band edges);
  * occupation: emerging is free text (case, trailing blanks, "Self-employed <occupation>") -> `occupation` is normalised to the
    developed vocabulary and `self_employed` is a separate flag (False for developed, where the information does not exist);
  * regions are different code sets (R.. vs E..), so `region` is market specific;
  * developed-only fields (annual_mileage, telematics_*, credit_band, exact ages) and emerging-only `gender` cannot be shared.
`to_common` returns the frame restricted to the columns both markets have after these steps, so that a model trained on one market
can score the other. Claim-history columns are excluded for Part 2 modelling: the emerging data starts in 2023, so the history of every
2023 row is empty by construction and the inputs would mean different things in training and validation (`HISTORY_COLS`).
"""
import numpy as np
import pandas as pd

AGE_EDGES = (17, 25, 35, 50, 65, 200)
AGE_LABELS = ("17-24", "25-34", "35-49", "50-64", "65+")
_EXACT_AGE = {"motor": "driver_age", "buildings": "policyholder_age", "contents": "policyholder_age"}
_BAND_COL = {"motor": "driver_age_band", "buildings": "policyholder_age_band", "contents": "policyholder_age_band"}
MARKET_ONLY = ["annual_mileage", "telematics_opt_in", "telematics_score", "credit_band", "customer_age", "date_of_birth", "driver_age",
               "policyholder_age", "driver_age_band", "policyholder_age_band", "gender", "region"]
HISTORY_COLS = ["hist_n_claims", "hist_cost", "hist_n_last_year", "cust_n_motor", "cust_n_buildings", "cust_n_contents", "cust_n_all",
                "addr_n_claims", "addr_n_last_year"] + [f"{c}_lag1" for c in ("hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm")]


def normalise_occupation(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(occupation, self_employed): trimmed, lower-cased, without the 'self-employed ' prefix; missing stays missing."""
    t = s.astype("string").str.strip().str.lower()
    se = t.str.startswith("self-employed ").fillna(False).astype(bool)
    base = t.str.replace(r"^self-employed ", "", regex=True)
    return base.astype(object).where(base.notna(), np.nan), se


def age_band(df: pd.DataFrame, line: str, market: str) -> pd.Series:
    if market == "emerging":
        band = df[_BAND_COL[line]]
        assert band.dropna().isin(AGE_LABELS).all(), f"unknown age band in {line}: {sorted(set(band.dropna()) - set(AGE_LABELS))}"
        return band
    return pd.cut(df[_EXACT_AGE[line]], bins=list(AGE_EDGES), right=False, labels=list(AGE_LABELS)).astype(object)


def to_common(df: pd.DataFrame, line: str, market: str) -> pd.DataFrame:
    """Same columns, same meaning in both markets (see module docstring). Pure function: returns a new frame."""
    assert market in ("developed", "emerging")
    occ, se = normalise_occupation(df["occupation"])
    out = df.assign(age_band=age_band(df, line, market), occupation=occ, self_employed=se)
    return out.drop(columns=[c for c in MARKET_ONLY + HISTORY_COLS if c in out.columns])
