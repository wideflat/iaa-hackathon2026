"""Fold E (emerging: train 2023 -> validate 2024): emerging-only models vs transfer from the developed market.

  python scripts/p2_cv.py --variants em_gbm,em_glm,dev_raw,dev_cal,em_stack,blend

Needs artifacts/p2/dev_pred_{A,B}_<line>.parquet and dev_completeness_<line>.json from scripts/p2_dev_predictions.py.
Validation rows: emerging 2024 (claims immature at the valuation date; predictions are scaled by the developed market's
completeness of the accident year, which matches the emerging report-lag pattern: docs/part2/DATA_REVIEW.md).
Prints pooled and per-line D2 with cluster-bootstrap SE, then the paired difference of every variant against the reference (em_gbm).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, cv, data, market, target, transfer  # noqa: E402
from iaa.config import ARTIFACTS_DIR, LINES  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, score_report  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402
from iaa.validation import FOLDS, split  # noqa: E402

EM_DIR = ROOT / "data" / "part2" / "emerging"
P2 = ARTIFACTS_DIR / "p2"
THREADS = 4


def completeness(line: str) -> dict:
    raw = json.loads((P2 / f"dev_completeness_{line}.json").read_text())
    return {p: {int(y): v for y, v in t.items()} for p, t in raw.items()}


def dev_pred(tag: str, line: str) -> pd.DataFrame:
    d = pd.read_parquet(P2 / f"dev_pred_{tag}_{line}.parquet").set_index("row_id")
    return d[[c for c in d.columns if c.startswith("dev_") and c != "dev_year"]]


def gbm(matched, tbl, freq_rounds, sev_rounds, kind="gbm", leaves=4):
    kw = dict(params_freq={"num_threads": THREADS, "num_leaves": leaves}, params_sev={"num_threads": THREADS, "num_leaves": leaves}) if kind == "gbm" else {}
    return FreqSev(matched, kind=kind, cat_encoding="target" if kind == "gbm" else "native", outlier_mult=5.0, freq_rounds=freq_rounds, sev_rounds=sev_rounds,
                   threads=THREADS, completeness_table=tbl, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="em_gbm,em_glm,dev_raw,dev_cal,em_stack,blend")
    ap.add_argument("--freq-rounds", type=int, default=200)
    ap.add_argument("--sev-rounds", type=int, default=150)
    ap.add_argument("--k", type=float, default=100.0, help="credibility constant of the peril-level calibration")
    ap.add_argument("--lines", default=",".join(LINES))
    ap.add_argument("--name", default="p2_foldE")
    a = ap.parse_args()
    variants = a.variants.split(",")
    claims = data.load_claims(EM_DIR, "emerging")
    em = cache.load_frames(market="emerging", data_dir=EM_DIR)
    fold = FOLDS["E"]
    preds: dict[str, list] = {v: [] for v in variants}
    valid_parts, factor_rows = [], []
    for line in a.lines.split(","):
        e = market.to_common(em[line], line, "emerging")
        rows = e[e["split"] == "train"]
        matched = target.matched_claims(rows, claims, line)
        tbl = completeness(line)
        train, valid = split(rows, fold)
        valid_parts.append(valid[["row_id", "customer_id", "line", "y", "year"]])
        c24 = pd.DataFrame({p: valid["year"].map(t).fillna(1.0) for p, t in tbl.items()})   # completeness by peril, valid rows
        if "em_gbm" in variants:
            m = gbm(matched, tbl, a.freq_rounds, a.sev_rounds).fit(train)
            preds["em_gbm"].append(m.predict_observed(valid))
        if "em_glm" in variants:
            m = gbm(matched, tbl, 0, 0, kind="glm").fit(train)
            preds["em_glm"].append(m.predict_observed(valid))
        if {"dev_raw", "dev_cal", "em_stack", "blend"} & set(variants):
            pb, pa = dev_pred("B", line), dev_pred("A", line)
            perils = [c.removeprefix("dev_") for c in pb.columns]
            vb = pb.loc[valid["row_id"]].set_axis(perils, axis=1)
            vb_obs = vb * c24.reindex(columns=perils).fillna(1.0).to_numpy()
            if "dev_raw" in variants:
                preds["dev_raw"].append(vb_obs.sum(axis=1).to_numpy())
            # calibration on the 2023 emerging rows, scored by the developed model trained on 2021-22 (never saw emerging data)
            t23 = train
            ca = matched[matched["row_id"].isin(t23["row_id"]) & matched["row_id"].notna()]
            act = ca.groupby(["row_id", "peril"])["net_cost"].sum().unstack(fill_value=0.0).reindex(index=t23["row_id"], columns=perils, fill_value=0.0)
            pa23 = pa.loc[t23["row_id"]].set_axis(perils, axis=1) * pd.DataFrame({p: t23["year"].map(tbl[p]).fillna(1.0) for p in perils}).to_numpy()
            ncl = ca.groupby("peril").size()
            f_peril = transfer.calibration_factors(act, pa23, ncl, k=a.k)
            f_line = transfer.calibration_factors(act, pa23, ncl, k=1e12)
            factor_rows += [{"line": line, "peril": p, "n_claims": int(ncl.get(p, 0)), "f_peril": f_peril[p], "f_line": f_line[p]} for p in perils]
            if "dev_cal" in variants:
                preds["dev_cal"].append(transfer.apply_factors(vb_obs, f_peril))
            if {"em_stack", "blend"} & set(variants):
                devf = pb.set_axis(perils, axis=1)
                extra = pd.DataFrame({f"devr_{p}": devf[p] for p in perils} | {"devr_total": devf.sum(axis=1)}, index=devf.index)
                extra = extra.div(rows.set_index("row_id")["exposure_calc"].reindex(extra.index), axis=0)
                rows_s = rows.join(extra, on="row_id")
                tr_s, va_s = split(rows_s, fold)
                m = gbm(matched, tbl, a.freq_rounds, a.sev_rounds).fit(tr_s)
                preds.setdefault("em_stack", [])
                if len(preds["em_stack"]) < len(valid_parts):
                    preds["em_stack"].append(m.predict_observed(va_s))
        print(f"{line} done", flush=True)
    valid_all = pd.concat(valid_parts, ignore_index=True)
    P = {v: np.concatenate(p) for v, p in preds.items() if p}
    if {"dev_cal", "em_gbm"} <= set(P):
        for w in (0.3, 0.5, 0.7):
            P[f"blend_cal_gbm_{w}"] = transfer.geometric_blend(P["dev_cal"], P["em_gbm"], w)
    if {"dev_cal", "em_stack"} <= set(P):
        P["blend_cal_stack_0.5"] = transfer.geometric_blend(P["dev_cal"], P["em_stack"], 0.5)
    results = {}
    for v, p in P.items():
        rep = score_report(valid_all.assign(pred=p), n_boot=200)
        results[v] = rep
        print(f"{v:22s} D2={rep['d2_pooled']:.4f} (se {rep['d2_pooled_boot']['se']:.4f}) cap250k={rep['d2_pooled_cap250k']:.4f} "
              + " ".join(f"{l}={rep[f'd2_{l}']:.4f}/cal{rep[f'calibration_{l}']:.2f}" for l in a.lines.split(",")), flush=True)
    ref = "em_gbm" if "em_gbm" in P else next(iter(P))
    for v, p in P.items():
        if v != ref:
            d = bootstrap_d2_diff(valid_all["y"], p, P[ref], valid_all["customer_id"])
            print(f"  {v} - {ref}: {d['diff']:+.4f} (95% {d['lo']:+.4f}..{d['hi']:+.4f}, P(better)={d['p_a_better']:.2f})")
    pd.DataFrame(factor_rows).to_csv(P2 / f"{a.name}_factors.csv", index=False)
    valid_all.assign(**{v: p for v, p in P.items()}).to_parquet(P2 / f"{a.name}_oof.parquet")
    cv.log_experiment(a.name, vars(a), results, note="emerging fold E: train 2023 -> validate 2024")
    print(pd.DataFrame(factor_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
