"""Optional richer claim-history features per policy row (add_history_extra): what is known about the policy's own earlier claims at the end of the previous calendar year.

For a row of year Y only claims with loss_date AND report_date on or before 31 Dec of Y-1 are used (same cut-off as features.add_claim_history, so validation rows see the
information structure of the 2025 rows):
* histp_<peril>: number of earlier claims of each peril on the policy (a customer with earlier theft claims is a different theft risk from one with earlier glass claims)
* hist_gap_days: days from the last earlier loss to the cut-off (NaN without history)
* hist_max_cost: largest earlier net cost; hist_open_n: earlier claims still open
Cached under artifacts/ keyed by a hash of this file and the frames.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import target
from .config import ARTIFACTS_DIR, LINES


def _hist(rows: pd.DataFrame, claims_line: pd.DataFrame) -> pd.DataFrame:
    c = claims_line[claims_line["status"] != "repudiated"].copy()
    c["net_cost"] = target.claim_net_cost(c)
    perils = sorted(c["peril"].dropna().unique())
    out = []
    for year in sorted(rows["year"].unique()):
        cutoff = pd.Timestamp(year - 1, 12, 31)
        k = c[(c["loss_date"] <= cutoff) & (c["report_date"] <= cutoff)]
        g = k.groupby("policy_id")
        feat = pd.DataFrame({
            "hist_gap_days": (cutoff - g["loss_date"].max()).dt.days,
            "hist_max_cost": g["net_cost"].max(),
            "hist_open_n": k[k["status"] == "open"].groupby("policy_id").size(),
        })
        by_peril = k.groupby(["policy_id", "peril"]).size().unstack(fill_value=0).reindex(columns=perils, fill_value=0).add_prefix("histp_")
        feat = feat.join(by_peril, how="outer")
        r = rows.loc[rows["year"] == year, ["row_id", "policy_id"]].merge(feat, left_on="policy_id", right_index=True, how="left")
        out.append(r.drop(columns="policy_id"))
    h = pd.concat(out)
    for col in [f"histp_{p}" for p in perils] + ["hist_open_n"]:
        h[col] = h[col].fillna(0)
    return h


def add_history_extra(frames: dict, claims: pd.DataFrame) -> dict:
    h = hashlib.sha1(Path(__file__).read_bytes())
    for line in LINES:
        h.update(str(frames[line].shape).encode())
    key = h.hexdigest()[:10]
    out = {}
    for line in LINES:
        path = ARTIFACTS_DIR / f"history_extra_{key}_{line}.parquet"
        d = frames[line]
        if path.exists():
            extra = pd.read_parquet(path)
        else:
            extra = _hist(d, claims[claims["line"] == line]).astype("float32", errors="ignore")
            extra.to_parquet(path)
        out[line] = d.merge(extra, on="row_id", how="left")
        out[line].index = d.index if len(out[line]) == len(d) else out[line].index
    return out
