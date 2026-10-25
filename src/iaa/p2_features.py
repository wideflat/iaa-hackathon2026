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


def customer_prev_year_claims(claims: pd.DataFrame, rows_all: pd.DataFrame, by_line: bool = False) -> pd.DataFrame:
    """(customer_id, year) -> claims of the customer over ALL lines with a loss in year-1 that were reported by 31 December of year-1.

    rows_all: policy_id, customer_id, period_start, period_end of every policy row of every line of ONE market (a claim is attached to the customer of the row whose
    period contains its loss date). A window of exactly one year is used because the emerging book starts in 2023: all-time counts would mean a different thing in the two markets."""
    c = claims[claims["status"] != "repudiated"][["claim_id", "policy_id", "line", "loss_date", "report_date"]].merge(rows_all[["policy_id", "customer_id", "period_start", "period_end"]], on="policy_id")
    c = c[(c["loss_date"] >= c["period_start"]) & (c["loss_date"] <= c["period_end"])].drop_duplicates("claim_id")
    c = c.assign(loss_year=c["loss_date"].dt.year)
    c = c[c["report_date"] <= pd.to_datetime(c["loss_year"].astype(str) + "-12-31")]
    n = c.groupby(["customer_id", "loss_year"]).size().rename("cust_prev_year_claims").reset_index()
    if by_line:   # the customer's claims of the previous year on each line separately
        per = c.groupby(["customer_id", "loss_year", "line"]).size().unstack(fill_value=0).reindex(columns=["motor", "buildings", "contents"], fill_value=0).add_prefix("cust_prev_").add_suffix("_claims").reset_index()
        n = n.merge(per, on=["customer_id", "loss_year"], how="left")
    return n.assign(year=n["loss_year"] + 1).drop(columns="loss_year")[["customer_id", "year"] + [c_ for c_ in n.columns if c_.startswith("cust_prev")]]


HIST1_COLS = ("hist_n_last_year", "addr_n_last_year", "cust_prev_year_claims")


def add_history1(df: pd.DataFrame, raw: pd.DataFrame, cust_prev: pd.DataFrame, first_year: int) -> pd.DataFrame:
    """One-year claim-history inputs, defined identically in both markets: the policy's own claims of the previous year (this line), the claims at the same address, the customer's claims over all lines.

    Rows of `first_year` (the first year of the market's data) have an UNKNOWN history, not an empty one: their values are set to missing, which the developed model has seen on its own first year (2021)."""
    out = df.merge(raw[["row_id", "hist_n_last_year", "addr_n_last_year"]], on="row_id", how="left").merge(cust_prev, on=["customer_id", "year"], how="left")
    extra = [c for c in cust_prev.columns if c.startswith("cust_prev_") and c != "cust_prev_year_claims"]
    out[["cust_prev_year_claims"] + extra] = out[["cust_prev_year_claims"] + extra].fillna(0.0)
    first = out["year"] == first_year
    out.loc[first, list(HIST1_COLS) + extra] = np.nan
    return out
