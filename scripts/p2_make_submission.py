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


def mixed_costs(models: dict, tag: str, line: str) -> pd.DataFrame:
    """Weighted arithmetic mean of the per-peril costs of several bases (weights of the line, summing to 1)."""
    out = None
    for m, w in models.items():
        c = base_costs(m, tag, line) * w
        out = c if out is None else out + c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-share", default="", help="NN half weight per line, e.g. motor=0.7,buildings=0.4,contents=0.3 (empty = GBM only)")
    ap.add_argument("--nn-models", default="nn_tabm,nn_tabm_ple", help="nets averaged inside the NN half")
    ap.add_argument("--k-peril", type=float, default=300.0)
    ap.add_argument("--k-gender", type=float, default=300.0)
    ap.add_argument("--no-gender", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    share = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.nn_share.split(",") if kv}
    nets = [m for m in a.nn_models.split(",") if m]
    claims = data.load_claims(pc.EM_DIR, "emerging")
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    sample = data.load_sample_submission(pc.EM_DIR)
    parts, factors, summary = [], {}, []
    for line in LINES:
        e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"])
        rows = e[e["split"] == "train"].reset_index(drop=True)
        test = e[e["split"] == "test"].reset_index(drop=True)
        w_nn = share.get(line, 0.0)
        models = ({"gbm": 1.0 - w_nn} | {m: w_nn / len(nets) for m in nets}) if w_nn > 0 else {"gbm": 1.0}
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
        pred, fac = transfer.calibrate_and_apply(pred_train, act, n_by, rows["y"], rows["n_claims"], g_train, pred_test, g_test, a.k_peril, a.k_gender)
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
