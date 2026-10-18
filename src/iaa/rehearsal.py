"""Rehearsal of the real task one year earlier, to validate how we handle claim maturity.

Pretend today is `snapshot` (31 Dec of year S). Training rows are the policy rows of years <= S with claims known
at the snapshot only (report_date <= snapshot, at their current amounts). The 'test' rows are year S+1, predicted
without any of their claims, exactly as the 2025 rows are. Truth is year S+1 as valued at the real valuation date
(2024-12-31), which for S = 2022 is 24 months old, i.e. nearly ultimate (motor bodily injury still develops ~7%).
"""
import pandas as pd

from . import features
from .config import VALUATION_DATE


def build_rehearsal_frame(line: str, snapshot: pd.Timestamp, customers, addresses, weather, claims, truth_claims=None) -> pd.DataFrame:
    """Frame with split train/test as of `snapshot`, `y_true` for the test year, and `maturity_days`."""
    known = claims[claims["report_date"] <= snapshot]
    asof = features.build_line_frame(line, customers=customers, addresses=addresses, weather=weather, claims=known)
    asof = asof[asof["split"] == "train"].copy()            # policy rows 2021-2024 with y as known at the snapshot
    full = features.build_line_frame(line, customers=customers, addresses=addresses, weather=weather, claims=claims if truth_claims is None else truth_claims)
    truth = full[full["split"] == "train"].set_index("row_id")["y"]
    s = snapshot.year
    out = asof[asof["year"] <= s + 1].copy()
    out["split"] = out["year"].map(lambda y: "train" if y <= s else "test")
    out["y_true"] = out["row_id"].map(truth)
    out["maturity_days"] = (snapshot - out["period_end"]).dt.days.clip(lower=0)
    out.loc[out["split"] == "test", "maturity_days"] = out.loc[out["split"] == "train", "maturity_days"].max()  # ask for the mature view
    return out
