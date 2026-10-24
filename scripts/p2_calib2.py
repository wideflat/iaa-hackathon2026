"""Level calibration of the transferred developed model, tested in both directions on the two emerging years.

  python scripts/p2_calib2.py

Forward: calibrate on 2023 (developed model A), score 2024 (model B). Reverse: calibrate on 2024 (B), score 2023 (A). Pooled D2 over the three lines per direction.
Schemes: line closure only; per-peril cost factors (k = 300); robust frequency x severity factors (counts and medians, iaa.transfer.robust_factors), each with and without
the gender factor, and the robust scheme with/without the final line closure. The 2024 year is immature, the developed completeness is applied to both.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, data, market, target, transfer  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.metrics import bootstrap_d2_diff, score_report  # noqa: E402
import p2_calib as pc  # noqa: E402
import p2_make_submission as ms  # noqa: E402
import argparse  # noqa: E402

DEV_DIR = ROOT / "data" / "part2" / "developed"
TAG = {2023: "A", 2024: "B"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-weight", type=float, default=0.0, help="weight of the nets in the base (0 = GBM only)")
    ap.add_argument("--nn-models", default="nn_tabm")
    a = ap.parse_args()
    nets = [m for m in a.nn_models.split(",") if m]
    models = ({"gbm": 1.0 - a.nn_weight} | {m: a.nn_weight / len(nets) for m in nets}) if a.nn_weight > 0 else {"gbm": 1.0}

    def load(tag, line):
        return ms.mixed_costs(models, tag, line), ms.mixed_costs(models, tag, line, counts=True)

    em_claims, dev_claims = data.load_claims(pc.EM_DIR, "emerging"), data.load_claims(DEV_DIR, "developed")
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    for name, cal_y, val_y in (("forward 2023->2024", 2023, 2024), ("reverse 2024->2023", 2024, 2023)):
        parts, preds = [], {}
        for line in LINES:
            e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"])
            rows = e[e["split"] == "train"]
            cal, val = rows[rows["year"] == cal_y], rows[rows["year"] == val_y]
            cost_c, cnt_c = load(TAG[cal_y], line)
            cost_v, _ = load(TAG[val_y], line)
            perils = list(cost_c.columns)
            comp = lambda d: pc.completeness(line, perils, d["year"].reset_index(drop=True)).set_axis(d.index)  # noqa: E731
            ec, en = cost_c.loc[cal["row_id"]].set_axis(cal.index) * comp(cal), cnt_c.loc[cal["row_id"]].set_axis(cal.index) * comp(cal)
            ev = cost_v.loc[val["row_id"]].set_axis(val.index) * comp(val)
            m = target.matched_claims(rows, em_claims, line)
            m = m[m["row_id"].notna() & (m["status"] != "repudiated") & m["row_id"].isin(cal["row_id"])]
            act_rows = m.groupby(["row_id", "peril"])["net_cost"].sum().unstack(fill_value=0.0).reindex(index=cal["row_id"], columns=perils, fill_value=0.0).set_axis(cal.index)
            n_by = m.groupby("peril").size().reindex(perils, fill_value=0).astype(float)
            # medians: emerging claims of the calibration year vs developed claims of the same year
            dc = dev_claims[(dev_claims["line"] == line) & (dev_claims["status"] != "repudiated") & (dev_claims["loss_date"].dt.year == cal_y)]
            dmed = target.claim_net_cost(dc).groupby(dc["peril"]).median().reindex(perils)
            emed = m.groupby("peril")["net_cost"].median().reindex(perils).fillna(dmed)
            rf = transfer.robust_factors(n_by, en.sum(), emed, dmed)
            dcost = target.claim_net_cost(dc)
            loc = {}
            for q in (0.95, 0.99):   # winsorised mean: every claim capped at the quantile of its own market and peril
                dw = dcost.groupby(dc["peril"]).apply(lambda x: x.clip(upper=x.quantile(q)).mean()).reindex(perils)
                ew = m.groupby("peril")["net_cost"].apply(lambda x: x.clip(upper=x.quantile(q)).mean()).reindex(perils).fillna(dw)
                loc[q] = transfer.robust_factors(n_by, en.sum(), ew, dw)
            f_cost = transfer.calibration_factors(act_rows, ec, n_by, k=300)
            variants = {
                "closure_only": (ec.sum(axis=1), ev.sum(axis=1)),
                "cost_k300": (pd.Series(transfer.apply_factors(ec, f_cost), index=cal.index), pd.Series(transfer.apply_factors(ev, f_cost), index=val.index)),
                "robust_fs": (pd.Series(transfer.apply_factors(ec, rf["cost_factor"]), index=cal.index), pd.Series(transfer.apply_factors(ev, rf["cost_factor"]), index=val.index)),
            }
            for q, rfq in loc.items():
                variants[f"robustw{int(q * 100)}"] = (pd.Series(transfer.apply_factors(ec, rfq["cost_factor"]), index=cal.index), pd.Series(transfer.apply_factors(ev, rfq["cost_factor"]), index=val.index))
            for v, (ptr, pva) in variants.items():
                c = float(cal["y"].sum() / ptr.sum())
                for tag, closure in (("", True), ("_noclose", False)):
                    if v == "closure_only" and not closure:
                        continue
                    k = c if closure else 1.0
                    gf = transfer.group_factors(cal["y"], ptr * k, cal["gender"], cal["n_claims"], 300.0)
                    for g_tag, g in (("", None), ("+gender", gf)):
                        p = pva.to_numpy() * k * (1.0 if g is None else val["gender"].map(g).to_numpy())
                        preds.setdefault(f"{v}{tag}{g_tag}", []).append(p)
            print(name, line, "actual / predicted on the calibration rows: median", round(float(cal["y"].sum() / variants["robust_fs"][0].sum()), 3), " w95", round(float(cal["y"].sum() / variants["robustw95"][0].sum()), 3), " w99", round(float(cal["y"].sum() / variants["robustw99"][0].sum()), 3), flush=True)
            parts.append(val[["row_id", "customer_id", "line", "y"]])
        V = pd.concat(parts, ignore_index=True)
        P = {k: np.concatenate(v) for k, v in preds.items()}
        print(f"=== {name}: {len(V)} validation rows")
        for k, p in P.items():
            r = score_report(V.assign(pred=p), n_boot=0)
            print(f"  {k:28s} D2={r['d2_pooled']:.4f} " + " ".join(f"{l}={r[f'd2_{l}']:.4f}/cal{r[f'calibration_{l}']:.2f}" for l in LINES), flush=True)
        for k in ("cost_k300+gender", "robust_fs_noclose+gender", "robustw95_noclose+gender", "robustw99_noclose+gender"):
            d = bootstrap_d2_diff(V["y"], P[k], P["closure_only+gender"], V["customer_id"])
            print(f"  {k} - closure_only+gender: {d['diff']:+.4f} (95% {d['lo']:+.4f}..{d['hi']:+.4f})")


if __name__ == "__main__":
    main()
