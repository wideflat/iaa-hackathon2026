"""Uncertainty scenarios for the Part 2 report: how much do the main assumptions move the validation score and the predicted total?

  python scripts/p2_scenarios.py [--candidate submissions/p2_t3_robust_gbm_tabm.csv]

Two parts. (1) On the validation rows of fold E in both directions (`p2_calib2.py` output) the level of one line, or of the new-business rows, is scaled and the pooled D2 recomputed:
shows how costly a level mistake is. (2) On the 2025 test predictions of the candidate file the same scalings are applied and the change of the predicted total is reported.
Every scenario is a documented switch, not a refit.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache  # noqa: E402
from iaa.metrics import d2_tweedie  # noqa: E402
import p2_calib as pc  # noqa: E402

SCALES = (0.8, 0.9, 1.1, 1.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="submissions/p2_t3_robust_gbm_tabm.csv")
    ap.add_argument("--nn-weight", default="0.5")
    a = ap.parse_args()
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    meta = pd.concat([f[["row_id", "new_business", "exposure_calc"]] for f in em.values()])
    rows = []
    for direction in ("forward", "reverse"):
        o = pd.read_parquet(pc.P2 / f"p2_calib2_{direction}_nn{a.nn_weight}_oof.parquet").merge(meta, on="row_id")
        base = d2_tweedie(o["y"], o["pred"])
        for line in ("motor", "buildings", "contents"):
            for sc in SCALES:
                p = o["pred"].where(o["line"] != line, o["pred"] * sc)
                rows.append({"direction": direction, "scenario": f"{line} level x{sc}", "pooled_D2": d2_tweedie(o["y"], p), "change": d2_tweedie(o["y"], p) - base})
        for sc in SCALES:
            p = o["pred"].where(~o["new_business"], o["pred"] * sc)
            rows.append({"direction": direction, "scenario": f"new-business rows x{sc}", "pooled_D2": d2_tweedie(o["y"], p), "change": d2_tweedie(o["y"], p) - base})
        rows.append({"direction": direction, "scenario": "base", "pooled_D2": base, "change": 0.0})
    t = pd.DataFrame(rows)
    print(t.pivot_table(index="scenario", columns="direction", values="change").round(4).to_string())
    print("\nbase pooled D2:", t[t.scenario == "base"].set_index("direction")["pooled_D2"].round(4).to_dict())
    sub = pd.read_csv(ROOT / a.candidate, dtype={"row_id": str}).merge(meta, on="row_id")
    sub["line"] = sub["row_id"].str[0].map({"M": "motor", "B": "buildings", "C": "contents"})
    total = sub["prediction"].sum()
    out = []
    for line in ("motor", "buildings", "contents"):
        for sc in (0.8, 1.2):
            out.append((f"{line} level x{sc}", (sub["prediction"].where(sub["line"] != line, sub["prediction"] * sc).sum() / total - 1) * 100))
    for sc in (0.9, 1.1):
        out.append((f"new-business rows x{sc}", (sub["prediction"].where(~sub["new_business"], sub["prediction"] * sc).sum() / total - 1) * 100))
    print("\nChange of the predicted 2025 total (%):")
    for k, v in out:
        print(f"  {k:28s} {v:+.1f}")
    print("Share of the predicted total: " + ", ".join(f"{l} {sub.loc[sub.line == l, 'prediction'].sum() / total:.1%}" for l in ("motor", "buildings", "contents")) + f"; new business {sub.loc[sub.new_business, 'prediction'].sum() / total:.1%}")
    t.to_csv(ROOT / "docs" / "part2" / "scenarios_foldE.csv", index=False)


if __name__ == "__main__":
    main()
