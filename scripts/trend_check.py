"""Severity trend ('inflation factor') check: the pooled trend the model estimates per training window vs the year-by-year changes in the capped claim severity.

  python scripts/trend_check.py
Reads claims matched to policy rows; log cost is capped per peril at the 99.9th percentile; peril fixed effects are removed (same definition as FreqSev._estimate_trend).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cache, data, target  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402


def main() -> None:
    claims = data.load_claims()
    frames = cache.load_frames()
    for line in LINES:
        tr = frames[line][frames[line]["split"] == "train"]
        m = target.matched_claims(tr, claims, line)
        m = m[m["row_id"].notna() & (m["status"] != "repudiated")]
        m = m.assign(year=m["row_id"].map(pd.Series(tr["year"].to_numpy(), index=tr["row_id"])).to_numpy())
        cap = m.groupby("peril")["net_cost"].transform(lambda s: s.quantile(0.999))
        m = m.assign(cost=m["net_cost"].clip(upper=cap).clip(lower=1.0))
        print(f"\n== {line}: claims {len(m):,}")
        for last in (2022, 2023, 2024):
            w = m[m["year"] <= last]
            print(f"  estimated pooled trend, training years 2021-{last}: {FreqSev._estimate_trend(w):+.4f} per year (model clips to [0, 0.12])")
        lc = np.log(m["cost"])
        dev = lc - lc.groupby(m["peril"]).transform("mean")
        by = dev.groupby(m["year"]).mean()
        n = m.groupby("year").size()
        print("  mean log severity by accident year (peril effects removed):", {int(y): round(float(v), 4) for y, v in by.items()}, "claims", n.to_dict())
        print("  year-on-year change:", {f"{int(a)}->{int(b)}": round(float(by[b] - by[a]), 4) for a, b in zip(by.index[:-1], by.index[1:])})


if __name__ == "__main__":
    main()
