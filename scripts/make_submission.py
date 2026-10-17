"""Fit on all training years (2021-2024), predict the 2025 test rows, write a validated CSV.

  python scripts/make_submission.py --model lgbm --out submissions/lgbm_v1.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import data, features  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.models.baselines import LGBMTweedie, MeanRateModel  # noqa: E402
from iaa.submission import write_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mean_rate", "lgbm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-noise", action="store_true")
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()

    claims, customers = data.load_claims(), data.load_customers()
    addresses, weather = data.load_addresses(), data.load_weather()
    parts = []
    for line in LINES:
        df = features.build_line_frame(line, customers=customers, addresses=addresses, weather=weather, claims=claims)
        model = MeanRateModel() if a.model == "mean_rate" else LGBMTweedie(drop_noise=not a.keep_noise, n_rounds=a.rounds)
        model.fit(df[df["split"] == "train"])
        test = df[df["split"] == "test"]
        parts.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": model.predict(test)}))
    sub = write_submission(pd.concat(parts, ignore_index=True), data.load_sample_submission(), Path(a.out))
    print(f"wrote {a.out}: {len(sub)} rows, mean prediction {sub['prediction'].mean():.2f}")


if __name__ == "__main__":
    main()
