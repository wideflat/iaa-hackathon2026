"""Regional claim surges ("events") in the training claims, thinned back to the cell's normal monthly level.

A peril x region x loss-month cell whose claim count is far above that cell's usual month is a one-off event (the buildings storm surge in region R07,
October-December 2023: 231 excess claims, about four times normal, unrelated to the wind index). The models see only the region and the year's weather, so
they pin such a surge on that region and on that year's weather values. thin_event_claims keeps, in every flagged cell-month, only as many claims as the cell's
baseline month (the claims with the smallest hash of claim_id, so the choice is deterministic); the thinned claims are used for fitting only, never for scoring.
"""
import numpy as np
import pandas as pd

MIN_RATIO = 2.5      # count / baseline for a month to count as an event
MIN_EXCESS = 25      # and at least this many claims above the baseline


def event_cells(matched: pd.DataFrame) -> pd.DataFrame:
    """Flagged (peril, region, loss month) cells with count, baseline and claims to keep. `matched` needs peril, region, loss_date."""
    m = matched.assign(month=matched["loss_date"].dt.to_period("M"))
    counts = m.groupby(["peril", "region", "month"]).size().rename("n").reset_index()
    full = pd.period_range(m["month"].min(), m["month"].max(), freq="M")
    cells = counts[["peril", "region"]].drop_duplicates()
    grid = cells.merge(pd.DataFrame({"month": full}), how="cross").merge(counts, on=["peril", "region", "month"], how="left").fillna({"n": 0})
    base = grid.groupby(["peril", "region"])["n"].median().rename("baseline").reset_index()
    flagged = grid.merge(base, on=["peril", "region"])
    flagged = flagged[(flagged["n"] >= MIN_RATIO * flagged["baseline"]) & (flagged["n"] - flagged["baseline"] >= MIN_EXCESS)]
    return flagged.assign(keep=np.ceil(flagged["baseline"]).astype(int))


def thin_event_claims(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(claims with event surges thinned to baseline, the flagged cells)."""
    cells = event_cells(matched)
    if cells.empty:
        return matched, cells
    m = matched.assign(month=matched["loss_date"].dt.to_period("M"), h=pd.util.hash_array(matched["claim_id"].astype(str).to_numpy()))
    m = m.merge(cells[["peril", "region", "month", "keep"]], on=["peril", "region", "month"], how="left")
    m["rank"] = m.groupby(["peril", "region", "month"])["h"].rank(method="first")
    drop = m["keep"].notna() & (m["rank"] > m["keep"])
    return matched.loc[~drop.to_numpy()], cells


def thin_for_fitting(matched: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Thin event surges in `matched` (claims with a row_id); `frame` gives each row's region. Unmatched claims (no row) are returned unchanged."""
    region = frame.set_index("row_id")["region"]
    m = matched.assign(region=matched["row_id"].map(region))
    has_row = m["region"].notna()
    thinned, _ = thin_event_claims(m[has_row])
    return pd.concat([thinned, m[~has_row]]).drop(columns="region").sort_index()
