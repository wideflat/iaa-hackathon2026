"""Run the one-year-earlier rehearsal: snapshot 2022-12-31, train on 2021-22 as known then, predict 2023, score vs 2023 today.

  python scripts/rehearsal.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cv, data  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.metrics import assert_not_too_good, score_report  # noqa: E402
from iaa.models.baselines import LGBMTweedie, MeanRateModel  # noqa: E402
from iaa.rehearsal import build_rehearsal_frame  # noqa: E402


def main() -> None:
    closed_only = "--closed-only" in sys.argv
    snapshot = pd.Timestamp("2022-12-31")
    claims, cu, ad, we = data.load_claims(), data.load_customers(), data.load_addresses(), data.load_weather()
    # --closed-only: training labels use only claims that are closed (final amounts); the truth always uses all claims
    train_claims = claims[claims["status"] == "closed"] if closed_only else claims
    frames = {l: build_rehearsal_frame(l, snapshot, cu, ad, we, train_claims, truth_claims=claims) for l in LINES}
    variants = {
        "constant rate (as-known)": lambda: MeanRateModel(),
        "GBM, no maturity feature": lambda: LGBMTweedie(n_rounds=300, extra_drop=("maturity_days",)),
        "GBM + maturity_days": lambda: LGBMTweedie(n_rounds=300),
    }
    res = {}
    for name, mk in variants.items():
        parts = []
        for l in LINES:
            f = frames[l]
            tr, te = f[f.split == "train"], f[f.split == "test"]
            p = mk().fit(tr).predict(te)
            parts.append(te[["row_id", "customer_id", "line", "y_true"]].rename(columns={"y_true": "y"}).assign(pred=p))
        oof = pd.concat(parts, ignore_index=True)
        res[name] = score_report(oof, n_boot=150)
        assert_not_too_good(res[name]["d2_pooled"], f"for variant '{name}'")
        r = res[name]; b = r["d2_pooled_boot"]
        print(f"{name:28s} D2={r['d2_pooled']:.4f} (CI {b['lo']:.4f}..{b['hi']:.4f}) | " + " ".join(f"{l} D2={r[f'd2_{l}']:.4f}/cal{r[f'calibration_{l}']:.2f}" for l in LINES))
    cv.log_experiment("rehearsal_snapshot2022" + ("_closed_only" if closed_only else ""), {"snapshot": "2022-12-31", "closed_only": closed_only}, res, note="train 2021-22 as known at snapshot, predict 2023 at mature maturity, truth = 2023 at 2024-12-31")


if __name__ == "__main__":
    main()
