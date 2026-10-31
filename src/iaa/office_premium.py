"""Office premium from a risk premium, using the `expenses` table of the emerging folder (report exhibit for the 20 reference policies).

    office = (risk + cat_reinsurance_rate x risk [property lines only] + fixed_expense x exposure) / (1 - variable - commission[channel] - tax - margin)

* variable expense, commission, premium tax and the target profit margin are shares of the *office* premium, hence the gross-up by one minus their sum;
* catastrophe reinsurance is a share of the property (buildings, contents) *risk* premium; motor carries none;
* the fixed expense is per policy-year and pro-rated by the row's exposure (assumption: a 25-day row does not carry a full year of fixed cost);
* a channel without a commission in the table (emerging comparison site) is an error, not a silent zero: the caller must state the assumption.
"""
import pandas as pd

PROPERTY_LINES = ("buildings", "contents")
_COMMISSION_ITEM = {"direct": "commission_direct", "comparison_site": "commission_comparison_site", "broker": "commission_broker"}


def assumptions(expenses: pd.DataFrame, market: str, comparison_site_commission: float | None = None) -> dict:
    """{item: value} of one market. comparison_site_commission fills a blank comparison-site commission (an explicit, documented assumption)."""
    s = expenses.set_index("item")[market]
    out = s.to_dict()
    if pd.isna(out["commission_comparison_site"]):
        out["commission_comparison_site"] = comparison_site_commission
    return out


def office_premium(risk: pd.Series, line: pd.Series, channel: pd.Series, exposure: pd.Series, a: dict) -> pd.DataFrame:
    """Build-up table, one row per policy: every component in CU, and the office premium."""
    commission = channel.map({ch: a[item] for ch, item in _COMMISSION_ITEM.items()})
    assert commission.notna().all(), f"no commission assumption for channels {sorted(set(channel[commission.isna()]))}"
    share = a["variable_expense"] + commission + a["premium_tax"] + a["target_profit_margin"]
    assert (share < 1.0).all(), "loadings that are shares of premium must sum to less than 1"
    cat = risk * a["cat_reinsurance"] * line.isin(PROPERTY_LINES)
    fixed = a["fixed_expense_per_policy"] * exposure
    office = (risk + cat + fixed) / (1.0 - share)
    return pd.DataFrame({"risk_premium": risk, "cat_reinsurance": cat, "fixed_expense": fixed, "loading_share": share,
                         "variable_expense": office * a["variable_expense"], "commission": office * commission, "premium_tax": office * a["premium_tax"],
                         "profit_margin": office * a["target_profit_margin"], "office_premium": office})
