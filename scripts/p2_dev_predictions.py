"""Train the developed-market model on the shared feature space and score every emerging row with it (zero-shot transfer base).

  python scripts/p2_dev_predictions.py --tags A,B,C

Tag A = developed 2021-22 (level of 2023), B = 2021-23 (level of 2024), C = 2021-24 (level of 2025). Each (tag, line) is saved once to
artifacts/p2/dev_pred2_<tag>_<line>.parquet: per-peril expected cost (dev_<peril>) and expected claim count (devn_<peril>) of every emerging row on the ultimate basis at the tag's level, incl. the
model's rebalancing factor. Resumable: finished files are skipped. The completeness table of the developed claims (to correct the immature
emerging 2024 year) is saved per line as artifacts/p2/dev_completeness_<line>.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, data, market, p2_features, target  # noqa: E402
from iaa.config import ARTIFACTS_DIR, LINES  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402

DIRS = {"developed": ROOT / "data" / "part2" / "developed", "emerging": ROOT / "data" / "part2" / "emerging"}
LAST_TRAIN_YEAR = {"A": 2022, "B": 2023, "C": 2024}
POOL = {"PF": (2023, (2023,)), "PR": (2023, (2024,)), "PC": (2024, (2023, 2024))}   # pooled tag -> (last developed year, labelled emerging years): forward / reverse validation / final
THREADS = 4


def score(model, e: pd.DataFrame):
    """(frame with per-peril cost dev_<p> and count devn_<p> for every row of e, cost frame). Counts x severity must reproduce the cost."""
    by = model.k_ * model.expected_by_peril(e)
    if model.kind == "tw":   # no peril split and no claim count: one cost column
        return pd.concat([e[["row_id", "year", "split"]], by.add_prefix("dev_")], axis=1), by
    cnt = model.expected_counts(e)
    sev_ratio = (by / cnt.where(cnt > 0)).fillna(0.0)   # mean cost per claim incl. excess load and rebalancing
    assert np.allclose((cnt * sev_ratio).to_numpy(), by.to_numpy(), rtol=1e-6), "counts x severity must reproduce the cost"
    return pd.concat([e[["row_id", "year", "split"]], by.add_prefix("dev_"), cnt.add_prefix("devn_")], axis=1), by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="A,B,C")
    ap.add_argument("--lines", default=",".join(LINES))
    ap.add_argument("--drop", default="", help="comma-separated shared features to leave out of the developed model (e.g. written_premium,discount_applied)")
    ap.add_argument("--features", default="", help="comma-separated engineered feature sets of iaa.p2_features (mktnorm, pct, xline, ratios), applied to both markets")
    ap.add_argument("--leaves", type=int, default=4, help="gbm: num_leaves of both stages")
    ap.add_argument("--freq-rounds", type=int, default=350)
    ap.add_argument("--sev-rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-leaf-freq", type=int, default=300)
    ap.add_argument("--adv", type=float, default=0.0, help="gbm: domain-shift weights of strength s (odds^s that a developed row looks like an emerging row; features only, no labels)")
    ap.add_argument("--seeds", type=int, default=1, help="gbm / tw: bagging seeds averaged per stage; nn_*: nets averaged per stage (Part 1 final: 3)")
    ap.add_argument("--variant", default="", help="suffix of the output files when --drop or --mix-as is used")
    ap.add_argument("--mix-as", default="", help="buildings only: comma-separated developed levels (w, brick, stone) to read the emerging code MIX as; one extra file per level, the model is trained once")
    ap.add_argument("--model", default="gbm", choices=["gbm", "tw", "glm", "xgb", "nn_tabm", "nn_tabm_ple", "nn_tabm_q", "nn_tabm_w"], help="gbm = LightGBM frequency x severity (Part 1 tree half); tw = direct pure-premium Tweedie LightGBM (--leaves 4 --freq-rounds 350, or 8 / 300; one column dev_all); glm = regularised frequency x severity GLM with quantile bands (blend challenger); nn_* = the TabM nets of the Part 1 NN half")
    ap.add_argument("--feature-fraction", type=float, default=1.0, help="gbm: LightGBM feature_fraction of both stages (column subsampling, a diversity member for the ensemble)")
    ap.add_argument("--depth", type=int, default=3, help="xgb: tree depth")
    ap.add_argument("--pool", default="", choices=["", "PF", "PR", "PC"], help="gbm only: also train on the labelled emerging rows of POOL[tag] with a market_emerging input; the tag replaces --tags")
    ap.add_argument("--pool-weight", type=float, default=1.0, help="row weight of the emerging rows relative to a developed row")
    a = ap.parse_args()
    out_dir = ARTIFACTS_DIR / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = data.load_claims(DIRS["developed"], "developed")
    dev, em = cache.load_frames(market="developed", data_dir=DIRS["developed"]), cache.load_frames(market="emerging", data_dir=DIRS["emerging"])
    fsets_all = tuple(x for x in a.features.split(",") if x)
    fsets = tuple(x for x in fsets_all if x not in ("hist1", "hist2"))
    hist1 = {}
    if "hist1" in fsets_all or "hist2" in fsets_all:   # one-year claim history, first year of each market unknown (iaa.p2_features.add_history1)
        em_claims = data.load_claims(DIRS["emerging"], "emerging")
        for mk, fr, cl, first in (("developed", dev, claims, 2021), ("emerging", em, em_claims, 2023)):
            rows_all = pd.concat([f[["policy_id", "customer_id", "period_start", "period_end"]] for f in fr.values()], ignore_index=True)
            hist1[mk] = (p2_features.customer_prev_year_claims(cl, rows_all, by_line="hist2" in fsets_all), first)
    em_claims_all = data.load_claims(DIRS["emerging"], "emerging")
    held = {"developed": p2_features.lines_held(dev), "emerging": p2_features.lines_held(em)} if "xline" in fsets else {}
    for line in a.lines.split(","):
        d = market.to_common(dev[line], line, "developed")
        e = market.to_common(em[line], line, "emerging")
        if hist1:
            d = p2_features.add_history1(d, dev[line], hist1["developed"][0], hist1["developed"][1])
            e = p2_features.add_history1(e, em[line], hist1["emerging"][0], hist1["emerging"][1])
        if fsets:
            d, e = p2_features.add_features(d, line, fsets, held.get("developed")), p2_features.add_features(e, line, fsets, held.get("emerging"))
        d_train = d[d["split"] == "train"]
        matched = target.matched_claims(d_train, claims, line)
        for tag in ([a.pool] if a.pool else a.tags.split(",")):
            stem = ("dev_pred2" if a.model == "gbm" else f"dev_pred2_{a.model}") + (f"_{a.variant}" if a.variant else "")
            path = out_dir / f"{stem}_{tag}_{line}.parquet"   # costs and counts; dev_pred_ = first run of the GBM (costs only)
            old_path = out_dir / f"dev_pred_{tag}_{line}.parquet" if (a.model == "gbm" and not a.variant) else out_dir / "does_not_exist.parquet"
            if path.exists():
                continue
            if a.pool:
                assert a.model == "gbm", "--pool is implemented for the gbm only"
                last_dev, em_years = POOL[tag]
                dev_tr = d_train[d_train["year"] <= last_dev]
                em_tr = e[(e["split"] == "train") & e["year"].isin(em_years)]
                matched_dev = matched[matched["row_id"].isin(dev_tr["row_id"])]
                table = FreqSev(matched, kind="gbm").c_by_year   # completeness from the developed claims only (all training years), as in the unpooled model
                m_em = target.matched_claims(em_tr, em_claims_all, line)
                m_em = m_em[m_em["row_id"].notna()]
                prefix = lambda df: df.assign(row_id="E" + df["row_id"].astype(str))   # noqa: E731  (row ids restart at M00000000 in each market)
                train = pd.concat([dev_tr.assign(market_emerging=0, pool_w=1.0), prefix(em_tr).assign(market_emerging=1, pool_w=a.pool_weight)], ignore_index=True)
                matched_p = pd.concat([matched_dev, prefix(m_em)], ignore_index=True)
                model = FreqSev(matched_p, kind="gbm", cat_encoding="target", outlier_mult=5.0, freq_rounds=a.freq_rounds, sev_rounds=a.sev_rounds, threads=THREADS, n_seeds=a.seeds,
                                params_freq={"num_threads": THREADS, "num_leaves": a.leaves, "learning_rate": a.lr, "min_data_in_leaf": a.min_leaf_freq}, params_sev={"num_threads": THREADS, "num_leaves": a.leaves, "learning_rate": a.lr},
                                extra_drop=tuple(c for c in a.drop.split(",") if c) + ("pool_w",), completeness_table=table).fit(train)
                out, by = score(model, e.assign(market_emerging=1))
                out.to_parquet(path)
                print(f"{line} tag {tag} W={a.pool_weight}: trained on {len(dev_tr)} developed + {len(em_tr)} emerging rows, k={model.k_:.3f}, trend={model.sev_trend_:.3f}; emerging mean cost/row {by.sum(axis=1).mean():.1f}", flush=True)
                continue
            train = d_train[d_train["year"] <= LAST_TRAIN_YEAR[tag]]
            if a.model == "gbm":
                model = FreqSev(matched, kind="gbm", cat_encoding="target", outlier_mult=5.0, freq_rounds=a.freq_rounds, sev_rounds=a.sev_rounds, threads=THREADS, n_seeds=a.seeds,
                                params_freq={"num_threads": THREADS, "num_leaves": a.leaves, "learning_rate": a.lr, "min_data_in_leaf": a.min_leaf_freq, "feature_fraction": a.feature_fraction}, params_sev={"num_threads": THREADS, "num_leaves": a.leaves, "learning_rate": a.lr, "feature_fraction": a.feature_fraction}, extra_drop=tuple(c for c in a.drop.split(",") if c), adv_strength=a.adv).fit(train, target=e if a.adv else None)
            elif a.model == "tw":   # as scripts/make_submission.py --model fs_tw --leaves L --freq-rounds R --outlier-mult 5 --cat-encoding target
                model = FreqSev(matched, kind="tw", cat_encoding="target", outlier_mult=5.0, freq_rounds=a.freq_rounds, sev_rounds=a.sev_rounds, threads=THREADS,
                                params_freq={"num_threads": THREADS, "num_leaves": a.leaves}, params_sev={"num_threads": THREADS, "num_leaves": a.leaves}, extra_drop=tuple(c for c in a.drop.split(",") if c)).fit(train)
            elif a.model == "xgb":   # as scripts/run_cv.py --model fs_xgb --depth 3 --freq-rounds 350 --sev-rounds 300 --outlier-mult 5 --cat-encoding target
                model = FreqSev(matched, kind="xgb", cat_encoding="target", outlier_mult=5.0, freq_rounds=a.freq_rounds, sev_rounds=a.sev_rounds, threads=THREADS,
                                params_freq={"num_threads": THREADS, "max_depth": a.depth}, params_sev={"num_threads": THREADS, "max_depth": a.depth}, extra_drop=tuple(c for c in a.drop.split(",") if c)).fit(train)
            elif a.model == "glm":   # as scripts/run_cv.py --model fs_glm --bands --outlier-mult 5
                model = FreqSev(matched, kind="glm", bands=True, outlier_mult=5.0, threads=THREADS, extra_drop=tuple(c for c in a.drop.split(",") if c)).fit(train)
            else:   # as scripts/run_cv.py --model fs_nn --outlier-mult 5 --nn-tabm-k 4 [--nn-ple 8]
                nnp = {"tabm_k": 4, "threads": THREADS, "n_seeds": a.seeds} | ({"ple_bins": 8} if a.model == "nn_tabm_ple" else {}) | ({"num_transform": "quantile"} if a.model == "nn_tabm_q" else {}) | ({"hidden": (384, 192), "dropout": 0.2} if a.model == "nn_tabm_w" else {})
                model = FreqSev(matched, kind="nn", outlier_mult=5.0, threads=THREADS, nn_params=nnp, extra_drop=tuple(c for c in a.drop.split(",") if c)).fit(train)
            out, by = score(model, e)
            if old_path.exists():   # reproducibility check against the first run (same seeds, same data): costs must be identical
                old = pd.read_parquet(old_path)
                assert np.allclose(old[by.add_prefix("dev_").columns].to_numpy(), out[by.add_prefix("dev_").columns].to_numpy(), rtol=1e-6), f"{tag} {line}: rerun differs from the first run"
                print(f"{line} tag {tag}: reproduces the first run", flush=True)
            out.to_parquet(path)
            for v in [x for x in a.mix_as.split(",") if x]:   # emerging code MIX is the only source of "non_standard" in the emerging frame (cleaning maps MIX -> non_standard)
                assert line == "buildings", "--mix-as is for buildings"
                out_v, _ = score(model, e.assign(construction_type=e["construction_type"].replace({"non_standard": v})))
                out_v.to_parquet(out_dir / f"{stem}_mix-{v}_{tag}_{line}.parquet")
            print(f"{line} tag {tag}: trained on {len(train)} rows, k={model.k_:.3f}, trend={model.sev_trend_:.3f}; emerging mean dev cost/row {by.sum(axis=1).mean():.1f}", flush=True)
            (out_dir / f"dev_completeness_{line}.json").write_text(json.dumps({p: {str(y): v for y, v in t.items()} for p, t in model.c_by_year.items()}))


if __name__ == "__main__":
    main()
