"""Run a model through the fixed validation folds and log the result.

  python scripts/run_cv.py --model mean_rate
  python scripts/run_cv.py --model lgbm --keep-noise --note "ablation: noise columns kept"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cv, data, features, target  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402
from iaa.config import ARTIFACTS_DIR, LINES  # noqa: E402
from iaa.models.baselines import LGBMTweedie, MeanRateModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mean_rate", "lgbm", "fs_glm", "fs_gbm"], required=True)
    ap.add_argument("--folds", default="A,B")
    ap.add_argument("--keep-noise", action="store_true", help="keep suspected noise columns as features")
    ap.add_argument("--drop", default="", help="comma-separated extra columns to drop")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--form", choices=["rate", "offset"], default="rate")
    ap.add_argument("--closed-only", action="store_true", help="train on labels built from closed claims only; score on all claims")
    ap.add_argument("--level-all", action="store_true", help="with --closed-only: rebalance predictions to the all-claims training total")
    ap.add_argument("--no-rebalance", action="store_true", help="skip the per-line total rebalancing of predictions")
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    claims, customers = data.load_claims(), data.load_customers()
    addresses, weather = data.load_addresses(), data.load_weather()
    frames = {l: features.build_line_frame(l, customers=customers, addresses=addresses, weather=weather, claims=claims)
              for l in LINES}
    train_frames = None
    if a.closed_only:
        closed = claims[claims["status"] == "closed"]
        train_frames = {l: features.build_line_frame(l, customers=customers, addresses=addresses, weather=weather, claims=closed) for l in LINES}
        for l in LINES:  # keep the all-claims label available for level calibration
            train_frames[l]["y_all"] = train_frames[l]["row_id"].map(frames[l].set_index("row_id")["y"]).fillna(0.0)
    extra = tuple(c for c in a.drop.split(",") if c)
    matched = {}
    if a.model.startswith("fs_"):
        for l in LINES:  # claims matched to policy rows (all training years), repudiated excluded inside the model
            tr_rows = frames[l][frames[l]["split"] == "train"]
            matched[l] = target.matched_claims(tr_rows, claims, l)
    if a.model.startswith("fs_"):
        kind = a.model[3:]
        factory = lambda line: FreqSev(matched[line], kind=kind, rebalance=not a.no_rebalance, drop_noise=not a.keep_noise, extra_drop=extra)  # noqa: E731
    else:
        factory = ((lambda line: MeanRateModel()) if a.model == "mean_rate" else
                   (lambda line: LGBMTweedie(drop_noise=not a.keep_noise, extra_drop=extra, n_rounds=a.rounds, form=a.form,
                                             rebalance=not a.no_rebalance,
                                             level_col="y_all" if (a.closed_only and a.level_all) else "y")))
    results, oof = cv.run_cv(frames, factory, fold_keys=a.folds.split(","), train_frames=train_frames)
    name = a.name or f"{a.model}{'_keepnoise' if a.keep_noise else ''}{'_drop-' + '-'.join(extra) if extra else ''}"
    for fold, r in results.items():
        b = r.get("d2_pooled_boot", {})
        print(f"{name} | {fold}: D2={r['d2_pooled']:.4f} (95% CI {b.get('lo', float('nan')):.4f}..{b.get('hi', float('nan')):.4f}) "
              + " ".join(f"{l}={r[f'd2_{l}']:.4f}/cal{r[f'calibration_{l}']:.2f}" for l in LINES))
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    oof.to_parquet(ARTIFACTS_DIR / f"oof_{name}.parquet")
    cv.log_experiment(name, vars(a), results, note=a.note)


if __name__ == "__main__":
    main()
