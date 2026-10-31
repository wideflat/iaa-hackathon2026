"""Data-quality repairs applied once, in the loaders (data.py), so every model sees the same cleaned tables.

Each repair was found by the target-construction review (docs/data/DATA_QUALITY.md items 11-14) and is
deterministic, label-free for features, and fails fast if the data no longer looks the way the repair assumes.

  * clean_claims:   (11) exact duplicate claims (same policy, peril, dates, status and amounts under a new
                         claim_id) are dropped; (12) claims whose loss_date is the placeholder 2020-12-01
                         (before the policy exists, reported all through 2021) get a loss date imputed from
                         the report date and the peril's typical report lag so that they land in a policy row.
  * clean_policies: (13) sum_insured entered 1000x too large on ~0.2% of buildings/contents rows is divided
                         by 1000; (14) construction_type is lower-cased and the code T (2021-22) is merged
                         with W (2023+), which is the same construction recoded.
"""
import os

import numpy as np
import pandas as pd

from .config import MARKET

ALL_FIXES = ("duplicates", "placeholders", "sum_insured", "construction")


def active_fixes() -> frozenset:
    """All repairs, or the comma-separated subset named in IAA_CLEAN_FIXES (ablation runs only; "none" disables every repair)."""
    raw = os.environ.get("IAA_CLEAN_FIXES")
    if raw is None:
        return frozenset(ALL_FIXES)
    names = [x for x in raw.split(",") if x and x != "none"]
    assert set(names) <= set(ALL_FIXES), f"unknown repair in IAA_CLEAN_FIXES: {set(names) - set(ALL_FIXES)}"
    return frozenset(names)

PLACEHOLDER_LOSS_DATE = pd.Timestamp("2020-12-01")
_CLAIM_CONTENT = ["policy_id", "line", "peril", "loss_date", "report_date", "status", "paid_to_date", "incurred_to_date",
                  "recoveries", "repairer_id", "third_party_id"]
_UNIT_ERROR_FACTOR = 1000.0
_UNIT_ERROR_MEDIAN_MULTIPLE = 20.0   # clean sum_insured never exceeds ~3x the median; corrupted values are >100x
# Sanity bands for the number of repairs. Developed market: 604 duplicates, 252 placeholder loss dates (the planted defects).
# Emerging market: neither defect is present (0 and 0 measured, docs/part2/DATA_REVIEW.md), so any appearance fails fast.
_BANDS = {"developed": {"duplicates": (550, 650), "placeholders": (200, 300)}, "emerging": {"duplicates": (0, 0), "placeholders": (0, 0)}}
# Emerging construction codes -> the developed vocabulary (BRK/STN are certain; WOD = timber = developed w; MIX is a guess, tested in docs/part2).
_EMERGING_CONSTRUCTION = {"brk": "brick", "stn": "stone", "wod": "w", "mix": "non_standard"}


def drop_duplicate_claims(claims: pd.DataFrame, expected: tuple[int, int]) -> pd.DataFrame:
    """Drop claims that repeat every field except claim_id; keep the first occurrence (the shorter, original id)."""
    ordered = claims.assign(_idlen=claims["claim_id"].str.len()).sort_values(["_idlen", "claim_id"], kind="stable")
    keep = ~ordered.duplicated(_CLAIM_CONTENT, keep="first")
    out = ordered[keep].drop(columns="_idlen").sort_index()
    n = len(claims) - len(out)
    assert expected[0] <= n <= expected[1], f"{n} duplicate claims dropped, expected {expected}: the data changed, review the rule"
    return out


def fix_placeholder_loss_dates(claims: pd.DataFrame, first_start: pd.Series, expected: tuple[int, int]) -> pd.DataFrame:
    """Claims dated PLACEHOLDER_LOSS_DATE: loss_date := report_date - median report lag of the peril, never before the policy's first period start.

    first_start: first period_start of each policy_id (index = policy_id)."""
    ph = claims["loss_date"] == PLACEHOLDER_LOSS_DATE
    assert expected[0] <= ph.sum() <= expected[1], f"{int(ph.sum())} placeholder loss dates, expected {expected}"
    lag = (claims.loc[~ph, "report_date"] - claims.loc[~ph, "loss_date"]).dt.days.groupby([claims.loc[~ph, "line"], claims.loc[~ph, "peril"]]).median()
    key = pd.MultiIndex.from_frame(claims.loc[ph, ["line", "peril"]])
    imputed = claims.loc[ph, "report_date"] - pd.to_timedelta(lag.reindex(key).to_numpy(), unit="D")
    floor = claims.loc[ph, "policy_id"].map(first_start)
    assert floor.notna().all(), "placeholder claim on a policy without rows"
    out = claims.copy()
    out.loc[ph, "loss_date"] = np.maximum(imputed.to_numpy(), floor.to_numpy())
    return out


def clean_claims(claims: pd.DataFrame, first_start: pd.Series, market: str = MARKET) -> pd.DataFrame:
    on, bands = active_fixes(), _BANDS[market]
    out = drop_duplicate_claims(claims, bands["duplicates"]) if "duplicates" in on else claims
    return fix_placeholder_loss_dates(out, first_start, bands["placeholders"]) if "placeholders" in on else out


def clean_policies(df: pd.DataFrame, line: str, market: str = MARKET) -> pd.DataFrame:
    out, on = df.copy(), active_fixes()
    if "sum_insured" in out and "sum_insured" in on:
        bad = out["sum_insured"] > _UNIT_ERROR_MEDIAN_MULTIPLE * out["sum_insured"].median()
        assert bad.mean() < 0.01, f"{bad.mean():.3%} of {line} rows look like a sum_insured unit error"
        out["sum_insured"] = out["sum_insured"].where(~bad, out["sum_insured"] / _UNIT_ERROR_FACTOR)
        assert out["sum_insured"].max() < 10.0 * out["sum_insured"].median(), "sum_insured still has outliers after the unit fix"
    if "construction_type" in out and "construction" in on:
        out["construction_type"] = out["construction_type"].str.lower().replace({"t": "w"})
        if market == "emerging":
            known = out["construction_type"].dropna()
            assert set(known) <= set(_EMERGING_CONSTRUCTION), f"unknown emerging construction codes: {set(known) - set(_EMERGING_CONSTRUCTION)}"
            out["construction_type"] = out["construction_type"].map(_EMERGING_CONSTRUCTION)
    return out
