"""Reported-claim development triangles (the loss-triangle method of Werner & Modlin, ch. 6).

Only the *reporting* dimension of development can be rebuilt from this data: report_date tells us when each
claim became known, so a count triangle by accident quarter x report quarter is exact. Paid and case-incurred
triangles cannot be rebuilt (one valuation snapshot, no payment history), so IBNER is not estimable here.
"""
import numpy as np
import pandas as pd

from .config import VALUATION_DATE


def count_triangle(claims: pd.DataFrame, max_age: int = 12, last=None) -> pd.DataFrame:
    """Cumulative reported claim counts. Rows: accident quarter. Columns: age k = report quarter - accident quarter.

    Cells that would be after the valuation date are NaN (the unobserved lower-right of the triangle).
    """
    last = pd.Period(VALUATION_DATE, "Q") if last is None else last
    aq = claims["loss_date"].dt.to_period("Q")
    k = (claims["report_date"].dt.to_period("Q") - aq).apply(lambda x: x.n)
    counts = pd.crosstab(aq, k).reindex(columns=range(max_age + 1), fill_value=0).cumsum(axis=1).astype(float)
    for a in counts.index:
        for age in counts.columns:
            if a + age > last:
                counts.loc[a, age] = np.nan
    return counts


def link_ratios(tri: pd.DataFrame) -> np.ndarray:
    """Volume-weighted age-to-age factors: sum of column k+1 over sum of column k, over accident periods with both observed."""
    out = []
    for k in range(tri.shape[1] - 1):
        both = tri[k].notna() & tri[k + 1].notna()
        out.append(tri.loc[both, k + 1].sum() / tri.loc[both, k].sum())
    return np.array(out)


def factors_to_ultimate(links: np.ndarray, tail: float = 1.0) -> np.ndarray:
    """Cumulative development factor from age k to ultimate (index k), with an optional tail beyond the last age."""
    return np.append(np.cumprod(links[::-1])[::-1] * tail, tail)
