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
    ap.add_argument("--model", choices=["mean_rate", "lgbm", "fs_gbm", "fs_glm", "fs_nn"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-noise", action="store_true")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--leaves", type=int, default=None, help="fs: num_leaves for both stages")
    ap.add_argument("--freq-rounds", type=int, default=250)
    ap.add_argument("--sev-rounds", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--nn-tabm-k", type=int, default=0, help="fs_nn: TabM members per MLP (0 = plain MLP)")
    ap.add_argument("--engineered", action="store_true")
    ap.add_argument("--num-encoding", default="native", choices=["native", "lowcard", "count", "both"])
    ap.add_argument("--nn-numtf", default=None, choices=["standard", "log", "quantile"])
    ap.add_argument("--cat-encoding", default="native", choices=["native", "label", "count", "both", "target"])
    ap.add_argument("--nn-ple", type=int, default=0)
    ap.add_argument("--rebalance-level", choices=["line", "peril"], default="line")
    ap.add_argument("--outlier-mult", type=float, default=None, help="winsorise single claims above k x cap in excess load and level")
    a = ap.parse_args()

    claims = data.load_claims()
    frames = cache.load_frames()
    if a.engineered:
        from iaa.engineered import add_engineered
        frames = add_engineered(frames)
    parts = []
    for line in LINES:
        df = frames[line]
        tr = df[df["split"] == "train"]
        if a.model.startswith("fs_"):
            p = {"num_leaves": a.leaves} if a.leaves else {}
            model = FreqSev(target.matched_claims(tr, claims, line), kind=a.model[3:], drop_noise=not a.keep_noise, freq_rounds=a.freq_rounds,
                            sev_rounds=a.sev_rounds, params_freq=p, params_sev=p, n_seeds=a.seeds, nn_params={"n_seeds": a.seeds, "threads": 4, **({"tabm_k": a.nn_tabm_k} if a.nn_tabm_k else {}), **({"ple_bins": a.nn_ple} if a.nn_ple else {}), **({"num_transform": a.nn_numtf} if a.nn_numtf else {})},
                            rebalance_level=a.rebalance_level, outlier_mult=a.outlier_mult, cat_encoding=a.cat_encoding, num_encoding=a.num_encoding)
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
