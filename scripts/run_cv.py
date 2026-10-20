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
    ap.add_argument("--model", choices=["mean_rate", "lgbm", "fs_glm", "fs_gbm", "fs_nn", "fs_xgb", "fs_cat", "nn_tweedie"], required=True)
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
    ap.add_argument("--depth", type=int, default=None, help="fs_xgb / fs_cat: tree depth for both stages")
    ap.add_argument("--min-leaf-freq", type=int, default=None)
    ap.add_argument("--cap-q", type=float, default=0.999, help="fs: severity cap quantile")
    ap.add_argument("--nn-epochs", type=int, default=None, help="fs_nn: max epochs of the frequency net")
    ap.add_argument("--nn-hidden", default=None, help="fs_nn: hidden layer widths, e.g. 192,96")
    ap.add_argument("--nn-dropout", type=float, default=None)
    ap.add_argument("--nn-lr", type=float, default=None)
    ap.add_argument("--nn-wd", type=float, default=None)
    ap.add_argument("--nn-seeds", type=int, default=1, help="fs_nn: average this many nets per stage")
    ap.add_argument("--recency", type=float, default=0.0, help="fs: row/claim weight exp(rho x (year - last year))")
    ap.add_argument("--adv-weights", type=float, default=0.0, help="fs: domain-shift weight strength s (odds^s of looking like the rows to predict)")
    ap.add_argument("--tail-lambda", type=float, default=0.0, help="fs_gbm: risk-dependent excess load exponent")
    ap.add_argument("--entity", action="store_true", help="fs_nn: customer_id and address_id as learned embeddings")
    ap.add_argument("--nn-tabm-k", type=int, default=None, help="fs_nn: TabM BatchEnsemble members per MLP")
    ap.add_argument("--history-extra", action="store_true", help="add per-peril / gap / max-cost / open-claim history features (iaa.engineered_history)")
    ap.add_argument("--engineered", action="store_true", help="add the label-free engineered features (iaa.engineered)")
    ap.add_argument("--nn-ple", type=int, default=None, help="fs_nn: piecewise-linear encoding of numeric inputs with this many quantile bins")
    ap.add_argument("--cat-encoding", default="native", choices=["native", "label", "count", "both", "target", "target_peril"], help="tree models: categorical encoding")
    ap.add_argument("--num-encoding", default="native", choices=["native", "lowcard", "count", "both"], help="tree models: extra encodings of numeric inputs")
    ap.add_argument("--nn-numtf", default=None, choices=["standard", "log", "quantile"], help="fs_nn: numeric input transform")
    ap.add_argument("--max-bin", type=int, default=None, help="fs_gbm: LightGBM max_bin for both stages")
    ap.add_argument("--nn-gbm-offset", action="store_true", help="fs_nn: the net learns a multiplicative correction on top of the cross-fitted target-encoded GBM (leaves 4, 350/300 rounds, outlier 5)")
    ap.add_argument("--level-halflife", type=float, default=None, help="fs: rebalance to the recent-weighted trended cost (half-life in years) instead of all training years equally")
    ap.add_argument("--deflate", default="", help="comma-separated nominal columns to divide by their median in the row's own year (e.g. written_premium,sum_insured)")
    ap.add_argument("--year-feature", action="store_true", help="fs_gbm: add the policy year as a numeric input")
    ap.add_argument("--trend-mode", default="pooled", choices=["pooled", "peril"], help="fs: severity trend pooled per line or per peril")
    ap.add_argument("--nn-cat-te", action="store_true", help="fs_nn: also feed target-encoded categoricals (out-of-fold claim rate) as numeric inputs")
    ap.add_argument("--trend-scale", type=float, default=1.0, help="fs: multiply the estimated severity trend")
    ap.add_argument("--trend-halflife", type=float, default=None, help="fs: weight recent claim years more when estimating the severity trend (half-life in years)")
    ap.add_argument("--lgb-params", default="", help="fs_gbm: extra LightGBM parameters for both stages, e.g. linear_tree=true,linear_lambda=1.0")
    ap.add_argument("--threads", type=int, default=0, help="LightGBM threads (0 = all); use 2 to run two experiments in parallel")
    ap.add_argument("--resume", action="store_true", help="save/reuse a checkpoint per (fold, line) under artifacts/ckpt/<name>; change --name when settings change")
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    claims = data.load_claims()
    frames = cache.load_frames()
    if a.deflate:
        from iaa.deflate import deflate_frames
        frames = deflate_frames(frames, [c for c in a.deflate.split(",") if c])
    if a.history_extra:
        from iaa.engineered_history import add_history_extra
        frames = add_history_extra(frames, claims)
    if a.engineered:
        from iaa.engineered import add_engineered
        frames = add_engineered(frames)
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
        for kv in [x for x in a.lgb_params.split(",") if x]:
            k, v = kv.split("=")
            val = {"true": True, "false": False}.get(v.lower(), float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v)
            pf[k] = ps[k] = val
        if a.max_bin:
            pf["max_bin"] = ps["max_bin"] = a.max_bin
        if a.min_leaf_freq:
            pf["min_data_in_leaf"] = a.min_leaf_freq
        if a.model in ("fs_xgb", "fs_cat"):   # LightGBM parameter names do not apply
            pf, ps = {}, {}
            if a.lr:
                pf["eta" if a.model == "fs_xgb" else "learning_rate"] = ps["eta" if a.model == "fs_xgb" else "learning_rate"] = a.lr
            if a.depth:
                pf["max_depth" if a.model == "fs_xgb" else "depth"] = ps["max_depth" if a.model == "fs_xgb" else "depth"] = a.depth
        nnp = {k: v for k, v in {"epochs": a.nn_epochs, "hidden": tuple(int(x) for x in a.nn_hidden.split(",")) if a.nn_hidden else None,
                                 "dropout": a.nn_dropout, "lr": a.nn_lr, "wd": a.nn_wd, "n_seeds": a.nn_seeds,
                                 "tabm_k": a.nn_tabm_k, "ple_bins": a.nn_ple, "num_transform": a.nn_numtf, "threads": a.threads or 4}.items() if v is not None}
        GBM_OFFSET = dict(freq_rounds=350, sev_rounds=300, outlier_mult=5.0, cat_encoding="target",
                          params_freq={"num_threads": a.threads or 4, "num_leaves": 4}, params_sev={"num_threads": a.threads or 4, "num_leaves": 4})
        factory = lambda line: FreqSev(matched[line], kind=a.model[3:], rebalance=not a.no_rebalance, drop_noise=not a.keep_noise,  # noqa: E731
                                       extra_drop=extra, trend=not a.no_trend, completeness=not a.no_completeness,
                                       freq_rounds=a.freq_rounds, sev_rounds=a.sev_rounds, cap_quantile=a.cap_q,
                                       params_freq=pf, params_sev=ps, nn_params=nnp, threads=a.threads or 4, recency=a.recency, adv_strength=a.adv_weights, tail_lambda=a.tail_lambda,
                                       entity_cols=("customer_id", "address_id") if a.entity else (), cat_encoding=a.cat_encoding, num_encoding=a.num_encoding, cat_te=a.nn_cat_te, trend_scale=a.trend_scale, trend_halflife=a.trend_halflife, gbm_offset=GBM_OFFSET if a.nn_gbm_offset else None, level_halflife=a.level_halflife, year_feature=a.year_feature, trend_mode=a.trend_mode, bands=a.bands, n_seeds=a.seeds, rebalance_level=a.rebalance_level, outlier_mult=a.outlier_mult,
                                       **({"glm_alpha_freq": a.alpha_freq} if a.alpha_freq else {}), **({"glm_alpha_sev": a.alpha_sev} if a.alpha_sev else {}))
    elif a.model == "nn_tweedie":
        from iaa.models.nn_tweedie import NNTweedie
        kw = {k: v for k, v in {"epochs": a.nn_epochs, "hidden": tuple(int(x) for x in a.nn_hidden.split(",")) if a.nn_hidden else None,
                                "dropout": a.nn_dropout, "lr": a.nn_lr, "wd": a.nn_wd}.items() if v is not None}
        factory = lambda line: NNTweedie(drop_noise=not a.keep_noise, extra_drop=extra, n_seeds=a.nn_seeds, threads=a.threads or 4,  # noqa: E731
                                         rebalance=not a.no_rebalance, **kw)
    elif a.model == "mean_rate":
        factory = lambda line: MeanRateModel()  # noqa: E731
    else:
        factory = lambda line: LGBMTweedie(drop_noise=not a.keep_noise, extra_drop=extra, n_rounds=a.rounds, form=a.form,  # noqa: E731
                                           rebalance=not a.no_rebalance, level_col="y_all" if (a.closed_only and a.level_all) else "y")
    results, oof = cv.run_cv(frames, factory, fold_keys=a.folds.split(","), train_frames=train_frames,
                                ckpt_name=(a.name if a.resume else None))
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
