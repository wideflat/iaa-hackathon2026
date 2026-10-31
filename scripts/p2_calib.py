"""Calibration variants of the transferred developed-market model on fold E (emerging: calibrate on 2023, validate on 2024).

  python scripts/p2_calib.py

Needs artifacts/p2/dev_pred2_{A,B}_<line>.parquet (scripts/p2_dev_predictions.py) and dev_completeness_<line>.json.
Variants: cost-level factor per peril (credibility k), frequency / severity split factors, and a new-business multiplier;
every variant is renormalised so that the calibration rows' predicted total equals their actual total.
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
from iaa.validation import FOLDS, split  # noqa: E402

EM_DIR = ROOT / "data" / "part2" / "emerging"
P2 = ARTIFACTS_DIR / "p2"


def load_dev(tag: str, line: str):
    d = pd.read_parquet(P2 / f"dev_pred2_{tag}_{line}.parquet").set_index("row_id")
    cost = d[[c for c in d.columns if c.startswith("dev_") and c != "dev_year"]].rename(columns=lambda c: c.removeprefix("dev_"))
    cnt = d[[c for c in d.columns if c.startswith("devn_")]].rename(columns=lambda c: c.removeprefix("devn_"))
    return cost, cnt


def completeness(line: str, perils, years: pd.Series) -> pd.DataFrame:
    raw = json.loads((P2 / f"dev_completeness_{line}.json").read_text())
    return pd.DataFrame({p: years.map({int(y): v for y, v in raw[p].items()}).fillna(1.0) for p in perils}, index=years.index)


def nb_multiplier(rows: pd.DataFrame, pred: np.ndarray, k: float) -> float:
    """Shrunk ratio (actual / predicted) of new-business rows over renewal rows, as a multiplier for new-business rows."""
    nb = rows["new_business"].to_numpy()
    r_nb, r_ren = rows["y"].to_numpy()[nb].sum() / pred[nb].sum(), rows["y"].to_numpy()[~nb].sum() / pred[~nb].sum()
    n_nb = rows["n_claims"].to_numpy()[nb].sum()
    z = transfer.credibility(float(n_nb), k)
    return float(np.exp(z * np.log(r_nb / r_ren)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="p2_calib_foldE")
    a = ap.parse_args()
    claims = data.load_claims(EM_DIR, "emerging")
    em = cache.load_frames(market="emerging", data_dir=EM_DIR)
    fold = FOLDS["E"]
    parts, preds = [], {}
    for line in LINES:
        e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"])   # gender: emerging-only, allowed for pricing there (competition data page)
        rows = e[e["split"] == "train"]
        train, valid = split(rows, fold)
        matched = target.matched_claims(rows, claims, line)
        matched = matched[matched["row_id"].notna() & (matched["status"] != "repudiated")]
        ca = matched[matched["row_id"].isin(train["row_id"])]
        ca = ca.assign(year=ca["row_id"].map(rows.set_index("row_id")["year"]))
        cost_a, cnt_a = load_dev("A", line)
        cost_b, cnt_b = load_dev("B", line)
        perils = list(cost_a.columns)
        c_tr = completeness(line, perils, train["year"].reset_index(drop=True)).set_axis(train.index)
        c_va = completeness(line, perils, valid["year"].reset_index(drop=True)).set_axis(valid.index)
        ea_c, ea_n = cost_a.loc[train["row_id"]].set_axis(train.index) * c_tr, cnt_a.loc[train["row_id"]].set_axis(train.index) * c_tr
        ev_c, ev_n = cost_b.loc[valid["row_id"]].set_axis(valid.index) * c_va, cnt_b.loc[valid["row_id"]].set_axis(valid.index) * c_va
        act_c = ca.groupby("peril")["net_cost"].sum().reindex(perils, fill_value=0.0)
        act_n = ca.groupby("peril").size().reindex(perils, fill_value=0).astype(float)
        act_rows = ca.groupby(["row_id", "peril"])["net_cost"].sum().unstack(fill_value=0.0).reindex(index=train["row_id"], columns=perils, fill_value=0.0).set_axis(train.index)
        valid_all = valid[["row_id", "customer_id", "line", "y", "year"]]
        parts.append(valid_all)

        def add(name, pred_valid, pred_train):
            c = float(train["y"].sum() / pred_train.sum())   # closes the level on the calibration rows
            preds.setdefault(name, []).append(pred_valid * c)

        add("dev_raw", ev_c.sum(axis=1).to_numpy(), ea_c.sum(axis=1).to_numpy())
        for k in (30, 100, 300, 1000, 1e12):
            f = transfer.calibration_factors(act_rows, ea_c, act_n, k=k)
            add(f"cost_k{int(min(k, 99999))}", transfer.apply_factors(ev_c, f), transfer.apply_factors(ea_c, f))
        for kf, ks in ((20, 100), (20, 300), (50, 300), (20, 1e12)):
            sf = transfer.split_factors(act_n, ea_n.sum(), act_c, ea_c.sum(), kf, ks)
            fac = sf["freq"] * sf["sev"]
            add(f"split_kf{kf}_ks{int(min(ks, 99999))}", transfer.apply_factors(ev_c, fac), transfer.apply_factors(ea_c, fac))
        # gender (emerging-only rating factor) on top of a line-level calibration
        f1 = transfer.calibration_factors(act_rows, ea_c, act_n, k=300)
        q_tr, q_va = transfer.apply_factors(ea_c, f1), transfer.apply_factors(ev_c, f1)
        q_tr = q_tr * float(train["y"].sum() / q_tr.sum())
        for k_g in (100, 300):
            gf = transfer.group_factors(train["y"], pd.Series(q_tr, index=train.index), train["gender"], train["n_claims"], k=k_g)
            add(f"cost_k300_gender{k_g}", q_va * valid["gender"].map(gf).to_numpy(), q_tr * train["gender"].map(gf).to_numpy())
            print(f"{line} gender factors k={k_g}: {gf.round(3).to_dict()}", flush=True)
        # new-business multiplier on top of the reference calibration
        f = transfer.calibration_factors(act_rows, ea_c, act_n, k=100)
        p_tr, p_va = transfer.apply_factors(ea_c, f), transfer.apply_factors(ev_c, f)
        for k_nb in (100, 400):
            m = nb_multiplier(train, p_tr, k_nb)
            add(f"cost_k100_nb{k_nb}", p_va * np.where(valid["new_business"].to_numpy(), m, 1.0), p_tr * np.where(train["new_business"].to_numpy(), m, 1.0))
            print(f"{line} new-business multiplier (k_nb={k_nb}): {m:.3f}", flush=True)
    valid_all = pd.concat(parts, ignore_index=True)
    P = {v: np.concatenate(p) for v, p in preds.items()}
    results = {}
    for v, p in P.items():
        rep = score_report(valid_all.assign(pred=p), n_boot=200)
        results[v] = rep
        print(f"{v:22s} D2={rep['d2_pooled']:.4f} (se {rep['d2_pooled_boot']['se']:.4f}) " + " ".join(f"{l}={rep[f'd2_{l}']:.4f}/cal{rep[f'calibration_{l}']:.2f}" for l in LINES), flush=True)
    ref = "cost_k100"
    for v, p in P.items():
        if v != ref:
            d = bootstrap_d2_diff(valid_all["y"], p, P[ref], valid_all["customer_id"])
            print(f"  {v} - {ref}: {d['diff']:+.4f} (95% {d['lo']:+.4f}..{d['hi']:+.4f})")
    cv.log_experiment(a.name, vars(a), results, note="emerging fold E calibration variants of the transferred developed model")


if __name__ == "__main__":
    main()
