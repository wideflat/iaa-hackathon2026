"""Part 2 (emerging market) prediction file: developed-market model + level calibration + gender factor.

  python scripts/p2_make_submission.py --base gbm --out submissions/p2_t1_gbm.csv

Per line: (1) the developed-market model (trained on 2021-24, shared features, scripts/p2_dev_predictions.py tag C) scores the 2025 emerging test rows;
(2) per-peril level factors, a line closure and a gender factor are fitted on the emerging 2023-24 rows (the developed model trained up to the previous year
scores them: tags A for 2023, B for 2024; as-valued basis through the developed completeness) and applied to the test predictions (iaa.transfer.calibrate_and_apply).
--base takes one or more `name:weight` bases (gbm, nn_tabm, nn_tabm_ple), blended arithmetically in per-peril cost per line weights given as --nn-share.
Writes the CSV (validated against the sample submission), <out>.factors.json (all calibration factors) and <out>.summary.csv (level checks).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, data, market, target, transfer  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.submission import write_submission  # noqa: E402
import p2_calib as pc  # noqa: E402


def base_costs(model: str, tag: str, line: str) -> pd.DataFrame:
    stem = "dev_pred2" if model == "gbm" else f"dev_pred2_{model}"
    d = pd.read_parquet(pc.P2 / f"{stem}_{tag}_{line}.parquet").set_index("row_id")
    return d[[c for c in d.columns if c.startswith("dev_") and c != "dev_year"]].rename(columns=lambda c: c.removeprefix("dev_"))


def base_counts(model: str, tag: str, line: str) -> pd.DataFrame:
    stem = "dev_pred2" if model == "gbm" else f"dev_pred2_{model}"
    d = pd.read_parquet(pc.P2 / f"{stem}_{tag}_{line}.parquet").set_index("row_id")
    return d[[c for c in d.columns if c.startswith("devn_")]].rename(columns=lambda c: c.removeprefix("devn_"))


def mixed_costs(models: dict, tag: str, line: str, counts: bool = False) -> pd.DataFrame:
    """Weighted arithmetic mean of the per-peril costs (or expected claim counts) of several bases (weights of the line, summing to 1)."""
    out = None
    for m, w in models.items():
        c = (base_counts if counts else base_costs)(m, tag, line) * w
        out = c if out is None else out + c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-share", default="", help="NN half weight per line, e.g. motor=0.7,buildings=0.4,contents=0.3 (empty = GBM only)")
    ap.add_argument("--nn-models", default="nn_tabm,nn_tabm_ple", help="nets averaged inside the NN half")
    ap.add_argument("--k-peril", type=float, default=300.0)
    ap.add_argument("--k-gender", type=float, default=300.0)
    ap.add_argument("--no-gender", action="store_true")
    ap.add_argument("--tw-names", default="", help="direct Tweedie GBM variants (averaged) blended into the base, e.g. twl4,twl8")
    ap.add_argument("--tw-share", type=float, default=0.25, help="weight of the Tweedie part in the total base (Part 1: tree half 50/50 x tree weight 50% = 25%)")
    ap.add_argument("--gbm-name", default="gbm", help="name of the tree base: gbm, or a variant stem such as hist1")
    ap.add_argument("--level", default="closure", choices=["closure", "robust"], help="closure: per-peril cost factors + line closure (p2_t1); robust: count x median factors, no closure")
    ap.add_argument("--sev-location", default="w99", choices=["median", "w99"], help="severity scale of a peril: ratio of medians, or of the claim means winsorised at the 99% quantile of each market")
    ap.add_argument("--k-freq", default="20", help="frequency credibility constant: 20, or motor=60,buildings=400,contents=20")
    ap.add_argument("--k-sev", default="30")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    share = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.nn_share.split(",") if kv}
    nets = [m for m in a.nn_models.split(",") if m]
    claims = data.load_claims(pc.EM_DIR, "emerging")
    dev_claims = data.load_claims(ROOT / "data" / "part2" / "developed", "developed")
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    sample = data.load_sample_submission(pc.EM_DIR)
    parts, factors, summary = [], {}, []
    for line in LINES:
        e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"])
        rows = e[e["split"] == "train"].reset_index(drop=True)
        test = e[e["split"] == "test"].reset_index(drop=True)
        w_nn = share.get(line, 0.0)
        models = ({a.gbm_name: 1.0 - w_nn} | {m: w_nn / len(nets) for m in nets}) if w_nn > 0 else {a.gbm_name: 1.0}
        cost = {tag: mixed_costs(models, tag, line) for tag in ("A", "B", "C")}
        perils = list(cost["C"].columns)
        # as-valued predictions of the training rows: 2023 rows from tag A, 2024 rows from tag B, times the completeness of their accident year
        pred_train = pd.concat([cost["A"].loc[rows.loc[rows["year"] == 2023, "row_id"]], cost["B"].loc[rows.loc[rows["year"] == 2024, "row_id"]]]).loc[rows["row_id"]].set_axis(rows.index)
        pred_train = pred_train * pc.completeness(line, perils, rows["year"]).to_numpy()
        matched = target.matched_claims(rows, claims, line)
        matched = matched[matched["row_id"].notna() & (matched["status"] != "repudiated")]
        act = matched.groupby(["row_id", "peril"])["net_cost"].sum().unstack(fill_value=0.0).reindex(index=rows["row_id"], columns=perils, fill_value=0.0).set_axis(rows.index)
        n_by = matched.groupby("peril").size().reindex(perils, fill_value=0).astype(float)
        g_train = rows["gender"] if not a.no_gender else pd.Series("all", index=rows.index)
        g_test = test["gender"] if not a.no_gender else pd.Series("all", index=test.index)
        pred_test = cost["C"].loc[test["row_id"]].set_axis(test.index)
        if a.level == "closure":
            pred, fac = transfer.calibrate_and_apply(pred_train, act, n_by, rows["y"], rows["n_claims"], g_train, pred_test, g_test, a.k_peril, a.k_gender)
        else:
            cnt = {tag: mixed_costs(models, tag, line, counts=True) for tag in ("A", "B")}
            cnt_train = pd.concat([cnt["A"].loc[rows.loc[rows["year"] == 2023, "row_id"]], cnt["B"].loc[rows.loc[rows["year"] == 2024, "row_id"]]]).loc[rows["row_id"]].set_axis(rows.index)
            cnt_train = cnt_train * pc.completeness(line, perils, rows["year"]).to_numpy()
            dc = dev_claims[(dev_claims["line"] == line) & (dev_claims["status"] != "repudiated") & dev_claims["loss_date"].dt.year.isin([2023, 2024])]
            dnet = target.claim_net_cost(dc)
            if a.sev_location == "median":
                dmed = dnet.groupby(dc["peril"]).median().reindex(perils)
                emed = matched.groupby("peril")["net_cost"].median().reindex(perils).fillna(dmed)
            else:
                dmed = dnet.groupby(dc["peril"]).apply(lambda x: x.clip(upper=x.quantile(0.99)).mean()).reindex(perils)
                emed = matched.groupby("peril")["net_cost"].apply(lambda x: x.clip(upper=x.quantile(0.99)).mean()).reindex(perils).fillna(dmed)
            tw_kw = {}
            if a.tw_names:
                tws = [x for x in a.tw_names.split(",") if x]
                raw_train = pd.concat([cost["A"].loc[rows.loc[rows["year"] == 2023, "row_id"]], cost["B"].loc[rows.loc[rows["year"] == 2024, "row_id"]]]).loc[rows["row_id"]].set_axis(rows.index)
                crow = pred_train.sum(axis=1) / raw_train.sum(axis=1)   # share reported by the valuation date, row specific
                twc = lambda tag: sum(base_costs(f"tw_{t}", tag, line)["all"] for t in tws) / len(tws)   # noqa: E731
                tw_raw = pd.concat([twc("A").loc[rows.loc[rows["year"] == 2023, "row_id"]], twc("B").loc[rows.loc[rows["year"] == 2024, "row_id"]]]).loc[rows["row_id"]].set_axis(rows.index)
                tw_kw = dict(tw_train=tw_raw * crow, tw_test=twc("C").loc[test["row_id"]].set_axis(test.index), tw_share=a.tw_share)
            pred, fac = transfer.calibrate_robust(pred_train, cnt_train, n_by, rows["y"], rows["n_claims"], g_train, emed, dmed, pred_test, g_test, transfer.line_value(a.k_freq, line), transfer.line_value(a.k_sev, line), a.k_gender, **tw_kw)
            fac["closure"] = fac["train_actual_over_predicted"]
        factors[line] = fac
        parts.append(pd.DataFrame({"row_id": test["row_id"], "prediction": pred}))
        summary.append({"line": line, "test_rows": len(test), "train_actual_per_exposure": rows["y"].sum() / rows["exposure_calc"].sum(),
                        "test_pred_per_exposure": pred.sum() / test["exposure_calc"].sum(), "test_pred_per_row": pred.mean(),
                        "written_premium_per_row": em[line].loc[em[line]["split"] == "test", "written_premium"].mean(),
                        "dev_uncalibrated_per_row": pred_test.sum(axis=1).mean(), "closure": fac["closure"]})
        print(line, json.dumps({k: (round(float(v), 3) if k != "line" else v) for k, v in summary[-1].items()}), flush=True)
    sub = write_submission(pd.concat(parts, ignore_index=True), sample, Path(a.out))
    Path(a.out).with_suffix(".factors.json").write_text(json.dumps(factors, indent=1, default=float))
    pd.DataFrame(summary).to_csv(Path(a.out).with_suffix(".summary.csv"), index=False)
    print(f"wrote {a.out}: {len(sub)} rows, mean {sub['prediction'].mean():.2f}, min {sub['prediction'].min():.3f}, max {sub['prediction'].max():.1f}")


if __name__ == "__main__":
    main()
