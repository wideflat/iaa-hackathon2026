"""Do the emerging/developed claim-frequency ratios follow a few shared multipliers? Per (line, peril, year): actual emerging claim count / expected count of the developed model (t7 base, ultimate basis x completeness of the year), with a Poisson interval.

  python scripts/p2_explore_shared.py            -> prints the table, writes artifacts/p2_shared_ratios.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, data, market, target  # noqa: E402
from iaa.config import LINES  # noqa: E402
import p2_calib as pc  # noqa: E402
import p2_make_submission as ms  # noqa: E402

TAG = {2023: "A", 2024: "B"}
MODELS = {"hist1": 0.5, "nn_tabm_hist1": 0.25, "nn_tabm_ple_hist1": 0.25}


def main() -> None:
    em_claims = data.load_claims(pc.EM_DIR, "emerging")
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    out = []
    for line in LINES:
        e = market.to_common(em[line], line, "emerging")
        rows = e[e["split"] == "train"]
        m = target.matched_claims(rows, em_claims, line)
        m = m[m["row_id"].notna() & (m["status"] != "repudiated")]
        for year in (2023, 2024):
            cal = rows[rows["year"] == year]
            cnt = ms.mixed_costs(MODELS, TAG[year], line, counts=True).loc[cal["row_id"]].set_axis(cal.index)
            perils = list(cnt.columns)
            comp = pc.completeness(line, perils, cal["year"].reset_index(drop=True)).set_axis(cal.index)
            exp = (cnt * comp).sum()
            act = m[m["row_id"].isin(cal["row_id"])].groupby("peril").size().reindex(perils, fill_value=0)
            for p in perils:
                out.append({"line": line, "peril": p, "year": year, "actual": int(act[p]), "expected": float(exp[p])})
    df = pd.DataFrame(out)
    df["ratio"] = df["actual"] / df["expected"]
    df["se_log"] = 1.0 / np.sqrt(df["actual"].clip(lower=1))
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    df.to_csv(ROOT / "artifacts" / "p2_shared_ratios.csv", index=False)


if __name__ == "__main__":
    main()
