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
        out.append(tri.loc[both, k + 1].sum() / tri.loc[both, k].sum() if both.any() else np.nan)
    return np.array(out)


def factors_to_ultimate(links: np.ndarray, tail: float = 1.0) -> np.ndarray:
    """Cumulative development factor from age k to ultimate (index k), with an optional tail beyond the last age."""
    return np.append(np.cumprod(links[::-1])[::-1] * tail, tail)


def annual_triangle(claims: pd.DataFrame, value: str | None = None, years=(2021, 2022, 2023, 2024), n_ages: int = 4,
                    valuation=VALUATION_DATE) -> pd.DataFrame:
    """Accident-year triangle at annual valuations (31 Dec of each year): textbook layout.

    Rows: accident year. Columns: age in months since 1 Jan of the accident year (12, 24, 36, 48).
    Cell = cumulative count (value=None) or cumulative `value` of claims with report_date on or before that
    valuation date. This is an *as-if* triangle: each reported claim carries its current amount, so only reporting
    emergence is captured, not later movements in case estimates (IBNER). Unobserved cells are NaN.
    """
    ay = claims["loss_date"].dt.year
    out = pd.DataFrame(np.nan, index=list(years), columns=[12 * (j + 1) for j in range(n_ages)])
    for y in years:
        d = claims[ay == y]
        for j in range(n_ages):
            v = pd.Timestamp(y + j, 12, 31)
            if v > valuation:
                continue
            sel = d[d["report_date"] <= v]
            out.loc[y, 12 * (j + 1)] = float(len(sel)) if value is None else float(sel[value].sum())
    return out


def individual_links(tri: pd.DataFrame) -> pd.DataFrame:
    """Age-to-age ratios per accident year (rows) like the table on p.119 of Werner & Modlin ch. 6."""
    ages = list(tri.columns)
    return pd.DataFrame({f"{a}-{b}": tri[b] / tri[a] for a, b in zip(ages[:-1], ages[1:])})


def average_links(tri: pd.DataFrame, latest: int | None = None) -> pd.DataFrame:
    """Simple, volume-weighted and (optionally) latest-n volume-weighted averages of the age-to-age ratios."""
    ages = list(tri.columns)
    ind = individual_links(tri)
    res = {"simple average": ind.mean(), "volume-weighted": pd.Series(link_ratios(tri.set_axis(range(len(ages)), axis=1)), index=ind.columns)}
    if latest:
        vals = {}
        for a, b in zip(ages[:-1], ages[1:]):
            both = tri[[a, b]].dropna().tail(latest)
            vals[f"{a}-{b}"] = both[b].sum() / both[a].sum() if len(both) else np.nan
        res[f"volume-weighted, latest {latest}"] = pd.Series(vals)
    return pd.DataFrame(res).T


def tail_exp_decay(links, min_excess: float = 1e-4, horizon: int = 40) -> float:
    """Tail factor beyond the last observed age by exponential decay of (link - 1) across development periods.

    Fits ln(link - 1) linearly on the observed links that exceed 1 + min_excess and sums the geometric run-off.
    Returns 1.0 when there are fewer than two usable links or the fitted decay is not decreasing: no tail is
    claimed without evidence. This is an extrapolation and must be flagged as such.
    """
    links = np.asarray(links, dtype=float)
    j = np.arange(len(links))
    ok = links - 1.0 > min_excess
    if ok.sum() < 2:
        return 1.0
    b, a = np.polyfit(j[ok], np.log(links[ok] - 1.0), 1)
    if b >= 0:
        return 1.0
    future = np.arange(len(links), len(links) + horizon)
    return float(np.prod(1.0 + np.exp(a + b * future)))
