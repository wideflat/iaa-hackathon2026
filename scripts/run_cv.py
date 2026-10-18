"""Run a model through the fixed validation folds and log the result.

  python scripts/run_cv.py --model fs_gbm --folds A,B --name fs_trend --note "..."
  python scripts/run_cv.py --model fs_gbm --folds A --no-trend --threads 2     # ablation, 2 CPU threads (for parallel runs)

Feature frames are cached in artifacts/ (see iaa.cache); the first run builds them.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cache, cv, data, features, target  # noqa: E402
from iaa.config import ARTIFACTS_DIR, LINES  # noqa: E402
from iaa.models.baselines import LGBMTweedie, MeanRateModel  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mean_rate", "lgbm", "fs_glm", "fs_gbm", "fs_nn"], required=True)
    ap.add_argument("--folds", default="A,B")
    ap.add_argument("--keep-noise", action="store_true", help="keep suspected noise columns as features")
    ap.add_argument("--drop", default="", help="comma-separated extra columns to drop")
    ap.add_argument("--rounds", type=int, default=400, help="direct GBM rounds")
    ap.add_argument("--form", choices=["rate", "offset"], default="rate")
    ap.add_argument("--closed-only", action="store_true", help="train on labels built from closed claims only; score on all claims")
    ap.add_argument("--level-all", action="store_true", help="with --closed-only: rebalance predictions to the all-claims training total")
    ap.add_argument("--no-rebalance", action="store_true", help="skip the per-line total rebalancing of predictions")
    # frequency x severity options
    ap.add_argument("--no-trend", action="store_true", help="fs: no severity trend projection")
    ap.add_argument("--no-completeness", action="store_true", help="fs: no completeness offset for immature years")
    ap.add_argument("--alpha-freq", type=float, default=None, help="fs_glm: L2 penalty of the frequency GLM")
    ap.add_argument("--alpha-sev", type=float, default=None, help="fs_glm: L2 penalty of the severity GLM")
    ap.add_argument("--bands", action="store_true", help="fs_glm: add quantile-band (sparse one-hot) terms for continuous inputs")
    ap.add_argument("--rebalance-level", choices=["line", "peril"], default="line", help="fs: rescale each line or each peril to its trended training total")
    ap.add_argument("--outlier-mult", type=float, default=None, help="fs: winsorise single claims above this multiple of the peril cap in excess load and level targets")
    ap.add_argument("--seeds", type=int, default=1, help="fs_gbm: average this many boosters (bagging seeds) per stage")
    ap.add_argument("--freq-rounds", type=int, default=250)
    ap.add_argument("--sev-rounds", type=int, default=200)
    ap.add_argument("--lr", type=float, default=None, help="fs: learning rate for both stages")
    ap.add_argument("--leaves", type=int, default=None, help="fs: num_leaves for both stages")
    ap.add_argument("--min-leaf-freq", type=int, default=None)
    ap.add_argument("--cap-q", type=float, default=0.999, help="fs: severity cap quantile")
    ap.add_argument("--nn-epochs", type=int, default=None, help="fs_nn: max epochs of the frequency net")
    ap.add_argument("--nn-hidden", default=None, help="fs_nn: hidden layer widths, e.g. 192,96")
    ap.add_argument("--nn-dropout", type=float, default=None)
    ap.add_argument("--nn-lr", type=float, default=None)
    ap.add_argument("--nn-wd", type=float, default=None)
    ap.add_argument("--nn-seeds", type=int, default=1, help="fs_nn: average this many nets per stage")
    ap.add_argument("--threads", type=int, default=0, help="LightGBM threads (0 = all); use 2 to run two experiments in parallel")
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    claims = data.load_claims()
    frames = cache.load_frames()
    train_frames = None
    if a.closed_only:
        cu, ad, we = data.load_customers(), data.load_addresses(), data.load_weather()
        closed = claims[claims["status"] == "closed"]
        train_frames = {l: features.build_line_frame(l, customers=cu, addresses=ad, weather=we, claims=closed) for l in LINES}
        for l in LINES:  # keep the all-claims label available for level calibration
            train_frames[l]["y_all"] = train_frames[l]["row_id"].map(frames[l].set_index("row_id")["y"]).fillna(0.0)
    extra = tuple(c for c in a.drop.split(",") if c)

    if a.model.startswith("fs_"):
        matched = {l: target.matched_claims(frames[l][frames[l]["split"] == "train"], claims, l) for l in LINES}
        pf, ps = {"num_threads": a.threads}, {"num_threads": a.threads}
        if a.lr:
            pf["learning_rate"] = ps["learning_rate"] = a.lr
        if a.leaves:
            pf["num_leaves"] = ps["num_leaves"] = a.leaves
        if a.min_leaf_freq:
            pf["min_data_in_leaf"] = a.min_leaf_freq
        nnp = {k: v for k, v in {"epochs": a.nn_epochs, "hidden": tuple(int(x) for x in a.nn_hidden.split(",")) if a.nn_hidden else None,
                                 "dropout": a.nn_dropout, "lr": a.nn_lr, "wd": a.nn_wd, "n_seeds": a.nn_seeds,
                                 "threads": a.threads or 4}.items() if v is not None}
        factory = lambda line: FreqSev(matched[line], kind=a.model[3:], rebalance=not a.no_rebalance, drop_noise=not a.keep_noise,  # noqa: E731
                                       extra_drop=extra, trend=not a.no_trend, completeness=not a.no_completeness,
                                       freq_rounds=a.freq_rounds, sev_rounds=a.sev_rounds, cap_quantile=a.cap_q,
                                       params_freq=pf, params_sev=ps, nn_params=nnp, bands=a.bands, n_seeds=a.seeds, rebalance_level=a.rebalance_level, outlier_mult=a.outlier_mult,
                                       **({"glm_alpha_freq": a.alpha_freq} if a.alpha_freq else {}), **({"glm_alpha_sev": a.alpha_sev} if a.alpha_sev else {}))
    elif a.model == "mean_rate":
        factory = lambda line: MeanRateModel()  # noqa: E731
    else:
        factory = lambda line: LGBMTweedie(drop_noise=not a.keep_noise, extra_drop=extra, n_rounds=a.rounds, form=a.form,  # noqa: E731
                                           rebalance=not a.no_rebalance, level_col="y_all" if (a.closed_only and a.level_all) else "y")
    results, oof = cv.run_cv(frames, factory, fold_keys=a.folds.split(","), train_frames=train_frames)
    name = a.name or f"{a.model}{'_keepnoise' if a.keep_noise else ''}{'_drop-' + '-'.join(extra) if extra else ''}"
    for fold, r in results.items():
        b = r.get("d2_pooled_boot", {})
        print(f"{name} | {fold}: D2={r['d2_pooled']:.4f} [cap250k {r['d2_pooled_cap250k']:.4f}] (95% CI {b.get('lo', float('nan')):.4f}..{b.get('hi', float('nan')):.4f}) "
              + " ".join(f"{l}={r[f'd2_{l}']:.4f}/cal{r[f'calibration_{l}']:.2f}" for l in LINES), flush=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    oof.to_parquet(ARTIFACTS_DIR / f"oof_{name}.parquet")
    cv.log_experiment(name, vars(a), results, note=a.note)


if __name__ == "__main__":
    main()
