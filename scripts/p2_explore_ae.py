"""Where does the transferred developed model disagree with the emerging claims? Actual / expected (A/E) by level of every shared feature.

  python scripts/p2_explore_ae.py

Expected = developed GBM (tag A for 2023 rows, tag B for 2024 rows) x completeness, claim COUNTS (frequency) and COST. Both are scaled by one constant per line so the
line total matches (the level difference is a separate, known effect); a level of a feature is flagged when its A/E departs from 1 by more than chance (Poisson z on counts).
Per feature: chi-square over its levels (numeric features cut into 8 quantile bins). Output: docs/part2/ae_by_feature.csv (every level) and a ranked summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from iaa import cache, data, market, target  # noqa: E402
from iaa.config import LINES  # noqa: E402
import p2_calib as pc  # noqa: E402

SKIP = {"row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure", "exposure_calc", "y", "n_claims", "split", "line", "renewal_decision",
        "customer_service_contacts", "surname", "email_domain", "vehicle_colour", "branch_id", "campaign_code", "document_preference", "marketing_consent", "marketing_segment",
        "previous_insurer", "period_days", "start_month", "end_month", "year", "region_x"}


def levels(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(str)
    if pd.api.types.is_numeric_dtype(s):
        if s.nunique() <= 12:
            return s.astype("float64").astype(str).where(s.notna(), "missing")
        q = pd.qcut(s, 8, duplicates="drop")
        return q.astype(str).where(s.notna(), "missing")
    return s.astype(object).where(s.notna(), "missing").astype(str)


def main() -> None:
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    out = []
    for line in LINES:
        e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"], region=em[line]["region"])
        rows = e[e["split"] == "train"].reset_index(drop=True)
        exp_n, exp_c = [], []
        for yr, tag in ((2023, "A"), (2024, "B")):
            cost, cnt = pc.load_dev(tag, line)
            r = rows[rows["year"] == yr]
            c = pc.completeness(line, list(cnt.columns), r["year"].reset_index(drop=True)).set_axis(r.index)
            exp_n.append((cnt.loc[r["row_id"]].set_axis(r.index) * c).sum(axis=1))
            exp_c.append((cost.loc[r["row_id"]].set_axis(r.index) * c).sum(axis=1))
        rows["exp_n"], rows["exp_c"] = pd.concat(exp_n), pd.concat(exp_c)
        rows["exp_n"] *= rows["n_claims"].sum() / rows["exp_n"].sum()
        rows["exp_c"] *= rows["y"].sum() / rows["exp_c"].sum()
        for col in [c for c in rows.columns if c not in SKIP and not c.startswith(("exp_", "hail", "wind", "freez", "dry", "wild", "rain", "mean_temp"))]:
            if rows[col].isna().all():
                continue
            lv = levels(rows[col])
            g = rows.groupby(lv).agg(rows_=("row_id", "size"), n=("n_claims", "sum"), en=("exp_n", "sum"), y=("y", "sum"), ec=("exp_c", "sum"))
            g = g[g["en"] > 5]
            if len(g) < 2:
                continue
            chi = float((((g["n"] - g["en"]) ** 2) / g["en"]).sum())
            p = float(stats.chi2.sf(chi, len(g) - 1))
            for lev, r in g.iterrows():
                out.append({"line": line, "feature": col, "level": lev, "rows": int(r["rows_"]), "claims": int(r["n"]), "freq_AE": r["n"] / r["en"], "z": (r["n"] - r["en"]) / np.sqrt(r["en"]),
                            "cost_AE": r["y"] / r["ec"], "feature_chi2": chi, "feature_p": p, "levels": len(g)})
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "docs" / "part2" / "ae_by_feature.csv", index=False)
    summ = df.groupby(["line", "feature"]).agg(chi2=("feature_chi2", "first"), p=("feature_p", "first"), levels=("levels", "first"), max_abs_z=("z", lambda s: s.abs().max())).reset_index()
    summ["chi2_per_df"] = summ["chi2"] / (summ["levels"] - 1)
    print(summ.sort_values("p").groupby("line").head(12).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
