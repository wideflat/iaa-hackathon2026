"""Residual-structure search on the final base: which feature levels have an actual / expected claim count that is off in BOTH emerging years?

  python scripts/p2_explore_ae2.py

Expected = the t7 base (GBM + two nets, hist1 inputs) counts x completeness, scaled per line and year to the actual total (the level is handled elsewhere). For every level of every
shared feature (numbers cut into 8 quantile bins): z per year (Poisson); a level is 'consistent' if both years have the same sign and |z| >= 2 each and the combined |z| >= 3.5.
Under no structure about 1 level in 800 is expected to pass the two-year test by chance; the output also prints the number of levels tested. Writes docs/experiments/raw/p2/ae2_levels.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from iaa import cache, market  # noqa: E402
from iaa.config import LINES  # noqa: E402
import p2_calib as pc  # noqa: E402
import p2_make_submission as ms  # noqa: E402
from p2_explore_ae import SKIP, levels  # noqa: E402

MODELS = {"hist1": 0.5, "nn_tabm_hist1": 0.25, "nn_tabm_ple_hist1": 0.25}
TAG = {2023: "A", 2024: "B"}


def main() -> None:
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    out = []
    for line in LINES:
        e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"], region=em[line]["region"])
        rows = e[e["split"] == "train"].reset_index(drop=True)
        parts = []
        for yr in (2023, 2024):
            r = rows[rows["year"] == yr].copy()
            cnt = ms.mixed_costs(MODELS, TAG[yr], line, counts=True)
            c = pc.completeness(line, list(cnt.columns), r["year"].reset_index(drop=True)).set_axis(r.index)
            r["exp_n"] = (cnt.loc[r["row_id"]].set_axis(r.index) * c).sum(axis=1)
            r["exp_n"] *= r["n_claims"].sum() / r["exp_n"].sum()
            parts.append(r)
        rows = pd.concat(parts)
        for col in [c for c in rows.columns if c not in SKIP and not c.startswith(("exp_", "hail", "wind", "freez", "dry", "wild", "rain", "mean_temp"))]:
            if rows[col].isna().all():
                continue
            lv = levels(rows[col])
            g = rows.groupby([lv, rows["year"]]).agg(n=("n_claims", "sum"), en=("exp_n", "sum")).unstack("year")
            for lev, r in g.iterrows():
                if r[("en", 2023)] < 5 or r[("en", 2024)] < 5 or np.isnan(r[("en", 2023)]) or np.isnan(r[("en", 2024)]):
                    continue
                z23 = (r[("n", 2023)] - r[("en", 2023)]) / np.sqrt(r[("en", 2023)])
                z24 = (r[("n", 2024)] - r[("en", 2024)]) / np.sqrt(r[("en", 2024)])
                ae = (r[("n", 2023)] + r[("n", 2024)]) / (r[("en", 2023)] + r[("en", 2024)])
                out.append({"line": line, "feature": col, "level": lev, "claims": int(r[("n", 2023)] + r[("n", 2024)]), "AE": ae, "z2023": z23, "z2024": z24, "z_comb": (z23 + z24) / np.sqrt(2)})
    df = pd.DataFrame(out)
    df["consistent"] = (np.sign(df["z2023"]) == np.sign(df["z2024"])) & (df["z2023"].abs() >= 2) & (df["z2024"].abs() >= 2) & (df["z_comb"].abs() >= 3.5)
    df.to_csv(ROOT / "docs" / "experiments" / "raw" / "p2" / "ae2_levels.csv", index=False)
    print(f"levels tested {len(df)}; consistent in both years {int(df['consistent'].sum())}; combined |z| >= 3.5 in total {(df['z_comb'].abs() >= 3.5).sum()}")
    pd.set_option("display.width", 200)
    print(df[df["consistent"]].sort_values("z_comb", key=abs, ascending=False).head(25).round(3).to_string(index=False))
    print("\nLargest combined |z| (any):")
    print(df.reindex(df["z_comb"].abs().sort_values(ascending=False).index).head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
