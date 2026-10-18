"""Build the modelling target from raw claims.

Target definition (from the competition Description/Submissions pages):
  ultimate cost of all claims with a loss date in the row's period,
  net of policy excess and net of recoveries, excluding declined claims.

Decisions, each verified against the data (see docs/data/DATA_QUALITY.md):
  * `paid_to_date` is already net of excess, so net cost = incurred_to_date - recoveries.
  * Recoveries only appear on closed motor own_damage/theft claims (about 80% of incurred).
  * Repudiated claims carry zero amounts and are excluded.
  * A claim belongs to the policy row of the same policy_id whose [period_start, period_end]
    contains its loss_date. Claims that match no row (mostly 2020 losses) are dropped and counted.

NOTE: this is the *as-valued* cost at 2024-12-31. It is not yet developed to ultimate; 2024
claims are immature (20% still open). Development is a modelling step, not done here.
"""
import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR


def recompute_exposure(df: pd.DataFrame) -> pd.Series:
    """Exposure in years from the period dates (days / 365, capped at 1).

    The supplied `exposure` column is corrupted on ~2-3% of training rows (values of 365 or
    1.0139 on full-year rows), so it is not trusted.
    """
    days = (df["period_end"] - df["period_start"]).dt.days + 1
    return (days / DAYS_PER_YEAR).clip(upper=1.0)


def exposure_mismatch(df: pd.DataFrame, tol: float = 0.002) -> pd.Series:
    """Boolean mask of rows whose supplied exposure disagrees with the dates."""
    return (df["exposure"] - recompute_exposure(df)).abs() > tol


def claim_net_cost(claims: pd.DataFrame) -> pd.Series:
    """Net cost of each claim as valued: incurred - recoveries, zero if repudiated, floored at 0."""
    net = (claims["incurred_to_date"] - claims["recoveries"]).clip(lower=0.0)
    return net.where(claims["status"] != "repudiated", 0.0)


def match_claims_to_rows(claims: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    """Return claims (one line) with the `row_id` of the policy row covering `loss_date`.

    Unmatched claims are returned with row_id NaN. If a claim falls in overlapping rows the
    earliest-starting row is used (never observed in the competition data).
    """
    rows = policies[["row_id", "policy_id", "period_start", "period_end"]]
    m = claims.merge(rows, on="policy_id", how="left")
    inside = (m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])
    matched = m[inside].sort_values("period_start").drop_duplicates("claim_id", keep="first")
    out = claims.merge(matched[["claim_id", "row_id"]], on="claim_id", how="left")
    return out


def build_row_target(policies: pd.DataFrame, claims: pd.DataFrame, line: str):
    """Attach `exposure_calc`, `y` (net claim cost) and `n_claims` to the policy rows of one line.

    Returns (rows, report) where report holds counts for the data-quality log.
    """
    rows = policies.copy()
    rows["exposure_calc"] = recompute_exposure(rows)
    c = match_claims_to_rows(claims[claims["line"] == line], rows)
    c["net_cost"] = claim_net_cost(c)
    paid = c[c["row_id"].notna() & (c["status"] != "repudiated")]
    agg = paid.groupby("row_id").agg(y=("net_cost", "sum"), n_claims=("claim_id", "size"))
    rows = rows.merge(agg, left_on="row_id", right_index=True, how="left")
    rows["y"] = rows["y"].fillna(0.0)
    rows["n_claims"] = rows["n_claims"].fillna(0).astype(int)
    report = {
        "line": line,
        "rows": len(rows),
        "claims": len(c),
        "claims_unmatched": int(c["row_id"].isna().sum()),
        "claims_repudiated": int((c["status"] == "repudiated").sum()),
        "exposure_mismatch_rows": int(exposure_mismatch(rows).sum()),
        "total_net_cost": float(rows["y"].sum()),
    }
    return rows, report


def matched_claims(policies: pd.DataFrame, claims: pd.DataFrame, line: str) -> pd.DataFrame:
    """Claims of one line with row_id and net_cost, for feature engineering (claim history)."""
    c = match_claims_to_rows(claims[claims["line"] == line], policies)
    c["net_cost"] = claim_net_cost(c)
    return c
