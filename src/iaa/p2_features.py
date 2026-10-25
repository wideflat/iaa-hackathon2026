"""Part 2 feature engineering for the transfer from the developed to the emerging market. Pure functions on frames already in the shared feature space (`market.to_common`).

Sets (each switchable, evaluated one at a time on fold E in both directions):
  mktnorm  written_premium and sum_insured divided by the median of their line, market and policy year (scale differences between markets vanish; raw columns removed);
  pct      percentile rank of written_premium and sum_insured within line, market and policy year (raw columns removed);
  xline    number of lines the customer holds in the same year (motor / buildings / contents) and a bundle flag (known for 2025 rows too: the test files list the policies);
  ratios   ncb_years / years_licensed and the no-claims-bonus-exceeds-licence flag (motor), sum_insured per bedroom and premium rate = written_premium / sum_insured (property),
           written_premium per exposure year, age band as an ordinal number.
Every feature is knowable when the 2025 row is priced and uses no claim information.
"""
import numpy as np
import pandas as pd

SETS = ("mktnorm", "pct", "xline", "ratios")
_SCALE_COLS = ("written_premium", "sum_insured")
_AGE_ORD = {"17-24": 0, "25-34": 1, "35-49": 2, "50-64": 3, "65+": 4}


def lines_held(frames: dict) -> pd.DataFrame:
    """(customer_id, year) -> number of lines held and one flag per line, from the policy rows (train and test) of all lines of ONE market."""
    parts = [f[["customer_id", "year"]].drop_duplicates().assign(line=l) for l, f in frames.items()]
    both = pd.concat(parts, ignore_index=True)
    flags = both.assign(v=1).pivot_table(index=["customer_id", "year"], columns="line", values="v", fill_value=0).add_prefix("holds_")
    flags["n_lines_held"] = flags.sum(axis=1)
    return flags.reset_index()


def add_features(df: pd.DataFrame, line: str, sets: tuple, held: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns a new frame with the engineered columns added (and the raw scale columns removed for mktnorm / pct)."""
    assert set(sets) <= set(SETS), f"unknown feature sets {set(sets) - set(SETS)}"
    out = df.copy()
    present = [c for c in _SCALE_COLS if c in out.columns]
    if "ratios" in sets:
        out["premium_per_exposure"] = out["written_premium"] / out["exposure_calc"].clip(lower=0.05)
        out["age_ordinal"] = out["age_band"].map(_AGE_ORD).astype("float32")
        if line == "motor":
            out["ncb_over_licence"] = out["ncb_years"] / out["years_licensed"].clip(lower=1)
            out["ncb_exceeds_licence"] = (out["ncb_years"] > out["years_licensed"]).astype("int8")
        else:
            out["premium_rate"] = out["written_premium"] / out["sum_insured"]
            if "bedrooms" in out.columns:
                out["sum_insured_per_bedroom"] = out["sum_insured"] / out["bedrooms"].clip(lower=1)
    if "xline" in sets:
        assert held is not None, "xline needs the lines-held table of the market"
        out = out.merge(held, on=["customer_id", "year"], how="left")
        out[["n_lines_held"] + [c for c in held.columns if c.startswith("holds_")]] = out[["n_lines_held"] + [c for c in held.columns if c.startswith("holds_")]].fillna(0).astype("int8")
    if "mktnorm" in sets:
        for c in present:
            out[f"{c}_rel"] = out[c] / out.groupby("year")[c].transform("median")
        out = out.drop(columns=present)
    elif "pct" in sets:
        for c in present:
            out[f"{c}_pct"] = out.groupby("year")[c].rank(pct=True).astype("float32")
        out = out.drop(columns=present)
    return out
