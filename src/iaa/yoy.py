"""Year-on-year change of a policy's own inputs (label-free).

A claim in year t shows up in the policy's inputs of year t+1: the no-claims bonus drops by 1-2 years (motor ncb_years: given no recorded claim in
the previous row, a drop goes with a claim frequency of 0.23-0.44 against 0.12), premium and excess are re-rated. The test year 2025 is built the same way
from the 2024 row, so the change is available for every row that has a previous-year row of the same policy (NaN otherwise).
"""
import numpy as np
import pandas as pd

DELTA_COLS = ("ncb_years", "voluntary_excess", "missed_payments_12m", "discount_applied", "annual_mileage", "vehicle_age")
RATIO_COLS = ("written_premium", "sum_insured")


def add_yoy(frames: dict, delta_cols=DELTA_COLS, ratio_cols=RATIO_COLS) -> dict:
    out = {}
    for line, df in frames.items():
        d = df.sort_values(["policy_id", "year", "period_start"]).copy()
        first = ~d.duplicated(["policy_id", "year"], keep="first")
        base = d.loc[first, ["policy_id", "year"] + [c for c in (*delta_cols, *ratio_cols) if c in d.columns]]
        prev = base.assign(year=base["year"] + 1)
        m = d[["row_id", "policy_id", "year"]].merge(prev, on=["policy_id", "year"], how="left", suffixes=("", "_prev"))
        res = d.set_index("row_id")
        mm = m.set_index("row_id")
        feats = pd.DataFrame(index=res.index)
        for c in delta_cols:
            if c in res.columns:
                feats[f"yoy_d_{c}"] = res[c].astype(float) - mm[c].astype(float)
        for c in ratio_cols:
            if c in res.columns:
                feats[f"yoy_r_{c}"] = np.log(res[c].astype(float) / mm[c].astype(float))
        out[line] = df.join(feats.astype("float32"), on="row_id")
    return out
