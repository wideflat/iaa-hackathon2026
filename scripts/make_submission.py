"""Fit on all training years (2021-2024), predict the 2025 test rows, write a validated CSV.

  python scripts/make_submission.py --model lgbm --out submissions/lgbm_v1.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cache, data, features, target  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.models.baselines import LGBMTweedie, MeanRateModel  # noqa: E402
from iaa.submission import write_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mean_rate", "lgbm", "fs_gbm", "fs_glm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-noise", action="store_true")
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()

    claims = data.load_claims()
    frames = cache.load_frames()
    parts = []
    for line in LINES:
        df = frames[line]
        tr = df[df["split"] == "train"]
        if a.model.startswith("fs_"):
            model = FreqSev(target.matched_claims(tr, claims, line), kind=a.model[3:], drop_noise=not a.keep_noise)
        else:
            model = MeanRateModel() if a.model == "mean_rate" else LGBMTweedie(drop_noise=not a.keep_noise, n_rounds=a.rounds)
        model.fit(tr)
        test = df[df["split"] == "test"]
        pred = model.predict(test)
        parts.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred}))
        hist = tr[tr["year"] == 2024]  # sanity: predicted 2025 cost per exposure vs 2024 as valued and vs the 2021-24 average
        print(f"{line:10s} predicted cost/exposure 2025 {pred.sum()/test['exposure_calc'].sum():8.1f} | actual 2024 (as valued) {hist['y'].sum()/hist['exposure_calc'].sum():8.1f} | "
              f"2021-24 avg {tr['y'].sum()/tr['exposure_calc'].sum():8.1f} | min {pred.min():.3f} max {pred.max():,.0f}", flush=True)
    sub = write_submission(pd.concat(parts, ignore_index=True), data.load_sample_submission(), Path(a.out))
    print(f"wrote {a.out}: {len(sub)} rows, mean prediction {sub['prediction'].mean():.2f}")


if __name__ == "__main__":
    main()
