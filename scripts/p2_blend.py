"""Fold E: blend the LightGBM and the neural-net bases of the transferred developed model (final recipe on top of each blend).

  python scripts/p2_blend.py --nn-models nn_tabm,nn_tabm_ple --lines motor,buildings,contents

Per line the base cost per peril is (1 - w) x GBM + w x mean(nets); the final recipe (per-peril factors k = 300, line closure, gender factor k = 300, fitted on the
emerging 2023 rows) is applied to every blend and the 2024 rows are scored. Weights: 0 (GBM only), 1 (nets only), the Part 1 per-line weights, 0.5.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, cv, data, market, target, transfer  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, score_report  # noqa: E402
from iaa.validation import FOLDS, split  # noqa: E402
import p2_calib as pc  # noqa: E402
import p2_make_submission as ms  # noqa: E402

PART1_W = {"motor": 0.7, "buildings": 0.4, "contents": 0.3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-models", default="nn_tabm")
    ap.add_argument("--lines", default="motor,buildings,contents")
    ap.add_argument("--name", default="p2_blend_foldE")
    a = ap.parse_args()
    nets = a.nn_models.split(",")
    lines = a.lines.split(",")
    claims = data.load_claims(pc.EM_DIR, "emerging")
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    fold = FOLDS["E"]
    schemes = {"gbm": lambda l: 0.0, "nn": lambda l: 1.0, "part1_w": lambda l: PART1_W[l], "half": lambda l: 0.5}
    preds, parts = {k: [] for k in schemes}, []
    for line in lines:
        e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"])
        rows = e[e["split"] == "train"]
        train, valid = split(rows, fold)
        matched = target.matched_claims(rows, claims, line)
        matched = matched[matched["row_id"].notna() & (matched["status"] != "repudiated")]
        ca = matched[matched["row_id"].isin(train["row_id"])]
        parts.append(valid[["row_id", "customer_id", "line", "y", "year"]])
        for name, wfun in schemes.items():
            w = wfun(line)
            models = {"gbm": 1.0 - w} | {m: w / len(nets) for m in nets} if w > 0 else {"gbm": 1.0}
            models = {m: v for m, v in models.items() if v > 0}
            ca_, cb_ = ms.mixed_costs(models, "A", line), ms.mixed_costs(models, "B", line)
            perils = list(ca_.columns)
            c_tr = pc.completeness(line, perils, train["year"].reset_index(drop=True)).set_axis(train.index)
            c_va = pc.completeness(line, perils, valid["year"].reset_index(drop=True)).set_axis(valid.index)
            ptr = ca_.loc[train["row_id"]].set_axis(train.index) * c_tr
            pva = cb_.loc[valid["row_id"]].set_axis(valid.index) * c_va
            act = ca.groupby(["row_id", "peril"])["net_cost"].sum().unstack(fill_value=0.0).reindex(index=train["row_id"], columns=perils, fill_value=0.0).set_axis(train.index)
            n_by = ca.groupby("peril").size().reindex(perils, fill_value=0).astype(float)
            p, _ = transfer.calibrate_and_apply(ptr, act, n_by, train["y"], train["n_claims"], train["gender"], pva, valid["gender"], 300.0, 300.0)
            preds[name].append(p)
    V = pd.concat(parts, ignore_index=True)
    P = {k: np.concatenate(v) for k, v in preds.items()}
    results = {}
    for k, p in P.items():
        r = score_report(V.assign(pred=p), n_boot=200)
        results[k] = r
        print(f"{k:8s} D2={r['d2_pooled']:.4f} (se {r['d2_pooled_boot']['se']:.4f}) " + " ".join(f"{l}={r[f'd2_{l}']:.4f}" for l in lines), flush=True)
    for k in ("nn", "part1_w", "half"):
        d = bootstrap_d2_diff(V["y"], P[k], P["gbm"], V["customer_id"])
        print(f"  {k} - gbm: {d['diff']:+.4f} (95% {d['lo']:+.4f}..{d['hi']:+.4f}, P(better)={d['p_a_better']:.2f})")
    cv.log_experiment(a.name, vars(a), results, note="emerging fold E: GBM vs NN bases of the transferred model, final recipe on top")


if __name__ == "__main__":
    main()
