"""Back-test of the development step and of the 'closed claims only' idea.

1. Open share by accident year: how much of each year's claims/cost is still open at the valuation date
   (closed-only data would drop it, and it is not random: it is the long-running claims).
2. Leave-one-accident-year-out: predict a past year's ultimate from its 12-month reported cost times the 12m->ult factor
   estimated from the OTHER years, and compare with what the year actually became.
Amounts are current amounts (as-if triangle), so this tests reporting development only, not case-estimate development.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import data, development as dev, target  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 230)
    c = data.load_claims()
    c = c[(c.status != "repudiated") & (c.loss_date >= "2021-01-01")].copy()
    c["net"] = target.claim_net_cost(c).clip(upper=250_000)
    c["ay"] = c.loss_date.dt.year
    c["grp"] = np.where(c.line == "motor", np.where(c.peril == "bodily_injury", "motor BI (capped)", "motor excl BI"), c.line)

    op = c.assign(open=c.status == "open").groupby(["grp", "ay"]).apply(
        lambda d: pd.Series({"open % claims": 100 * d.open.mean(), "open % cost": 100 * d.net[d.open].sum() / d.net.sum()}), include_groups=False
    ).round(1).unstack("ay")
    print("Share of each accident year still open at the valuation date:\n", op.to_string(), "\n")

    rows = []
    for g, d in c.groupby("grp"):
        tri = dev.annual_triangle(d, "net")
        full = pd.Series(dev.link_ratios(tri.set_axis(range(4), axis=1))).fillna(1.0)
        fullcdf = dev.factors_to_ultimate(full.values)
        for a in (2021, 2022, 2023):
            lk = pd.Series(dev.link_ratios(tri.drop(index=a).set_axis(range(4), axis=1))).fillna(1.0).values
            pred = tri.loc[a, 12] * np.prod(lk)
            actual = tri.loc[a].dropna().iloc[-1] * fullcdf[2024 - a]
            rows.append(dict(group=g, AY=a, CDF12_loo=round(float(np.prod(lk)), 3), pred_ult_M=round(pred / 1e6, 2),
                             actual_ult_M=round(actual / 1e6, 2), error_pct=round(100 * (pred / actual - 1), 1)))
    print("Leave-one-year-out back-test (12-month view x CDF12 from other years vs actual):\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
