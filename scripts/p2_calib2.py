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
from iaa.metrics import bootstrap_d2_diff, d2_tweedie, score_report  # noqa: E402
import p2_calib as pc  # noqa: E402
import p2_make_submission as ms  # noqa: E402
import argparse  # noqa: E402

DEV_DIR = ROOT / "data" / "part2" / "developed"
TAG = {2023: "A", 2024: "B"}


class _Tee:
    """Mirror stdout into docs/experiments/raw/p2/calib2_runs.log (committed): every run keeps its full printed table next to its command line."""
    def __init__(self, path: Path, header: str):
        self.out, self.f = sys.stdout, open(path, "a")
        self.f.write(f"\n##### {header}\n")

    def write(self, text):
        self.out.write(text)
        self.f.write(text)

    def flush(self):
        self.out.flush()
        self.f.flush()


def main() -> None:
    import datetime
    sys.stdout = _Tee(ROOT / "docs" / "experiments" / "raw" / "p2" / "calib2_runs.log", f"{datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} python scripts/p2_calib2.py {' '.join(sys.argv[1:])}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-weight", type=float, default=0.0, help="weight of the nets in the base (0 = GBM only)")
    ap.add_argument("--nn-models", default="nn_tabm")
    ap.add_argument("--extra", default="", help="semicolon-separated extra factor specs applied after the robust level: 'gender+age_band' (replaces the gender factor), or 'age_band', 'ncb', 'voluntary_excess', 'occupation', ... added on top of gender")
    ap.add_argument("--tw-names", default="", help="direct Tweedie GBM variants (stems written by p2_dev_predictions --model tw --variant), averaged; blended into the tree base")
    ap.add_argument("--tw-share", type=float, default=0.5, help="weight of the Tweedie part in the blended tree base (Part 1 tree half: 0.5)")
    ap.add_argument("--weather-slope", action="store_true", help="per-peril hail slope correction (glass_hail in motor, storm in buildings): the response of the developed model to hail days is re-estimated on the calibration year, centred within each year")
    ap.add_argument("--k-freq", default="20", help="frequency credibility constant: 20, or motor=60,buildings=400,contents=20")
    ap.add_argument("--k-sev", default="30")
    ap.add_argument("--gbm-name", default="gbm", help="name of the tree base: gbm, or a variant stem written by p2_dev_predictions --variant")
    ap.add_argument("--glm-share", type=float, default=0.0, help="weight of the developed-market GLM (stem glm_hist1) in the frequency x severity base; the other bases are scaled by 1 - share")
    ap.add_argument("--pool-stem", default="", help="pooled GBM variant (p2_dev_predictions --pool): forward uses tag PF for both years, reverse tag PR; replaces the base models")
    ap.add_argument("--force-tag", default="", help="score calibration and validation rows with this developed cut-off tag (e.g. B) in both directions: the control of the pooled runs, which also see developed 2023")
    ap.add_argument("--shared-freq", action="store_true", help="frequency prior of a peril name present in several lines = its pooled actual / expected ratio over the lines of the calibration year (transfer.shared_peril_prior)")
    ap.add_argument("--oracle", action="store_true", help="ceiling analysis: calibrate on the validation year itself (in-sample level and gender factors), i.e. what a perfect level estimate of that year would score")
    a = ap.parse_args()
    nets = [m for m in a.nn_models.split(",") if m]
    gbms = [g for g in a.gbm_name.split(",") if g]   # tree bases averaged with equal weights
    models = ({g: (1.0 - a.nn_weight) / len(gbms) for g in gbms} | {m: a.nn_weight / len(nets) for m in nets}) if a.nn_weight > 0 else {g: 1.0 / len(gbms) for g in gbms}

    if a.pool_stem:
        models = {a.pool_stem: 1.0}
    if a.glm_share > 0:
        models = {m: w * (1.0 - a.glm_share) for m, w in models.items()} | {"glm_hist1": a.glm_share}

    def load(tag, line):
        return ms.mixed_costs(models, tag, line), ms.mixed_costs(models, tag, line, counts=True)

    em_claims, dev_claims = data.load_claims(pc.EM_DIR, "emerging"), data.load_claims(DEV_DIR, "developed")
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    for name, cal_y, val_y in ((("oracle 2024", 2024, 2024), ("oracle 2023", 2023, 2023)) if a.oracle else (("forward 2023->2024", 2023, 2024), ("reverse 2024->2023", 2024, 2023))):
        parts, preds = [], {}
        prior = None
        if a.shared_freq:   # pre-pass over the lines: calibration-year claim counts and developed expected counts by peril
            act_l, exp_l = {}, {}
            for line_ in LINES:
                e_ = market.to_common(em[line_], line_, "emerging")
                rows_ = e_[e_["split"] == "train"]
                cal_ = rows_[rows_["year"] == cal_y]
                _, cnt_ = load(("PF" if cal_y == 2023 else "PR") if a.pool_stem else (a.force_tag or TAG[cal_y]), line_)
                comp_ = pc.completeness(line_, list(cnt_.columns), cal_["year"].reset_index(drop=True)).set_axis(cal_.index)
                en_ = (cnt_.loc[cal_["row_id"]].set_axis(cal_.index) * comp_).sum()
                m_ = target.matched_claims(rows_, em_claims, line_)
                m_ = m_[m_["row_id"].notna() & (m_["status"] != "repudiated") & m_["row_id"].isin(cal_["row_id"])]
                act_l[line_], exp_l[line_] = m_.groupby("peril").size().reindex(en_.index, fill_value=0).astype(float), en_
            prior = transfer.shared_peril_prior(act_l, exp_l)
        tag_of = (lambda y: "PF" if cal_y == 2023 else "PR") if a.pool_stem else ((lambda y: a.force_tag) if a.force_tag else (lambda y: TAG[y]))   # noqa: E731
        for line in LINES:
            e = market.to_common(em[line], line, "emerging").assign(gender=em[line]["gender"])
            rows = e[e["split"] == "train"]
            cal, val = rows[rows["year"] == cal_y], rows[rows["year"] == val_y]
            cost_c, cnt_c = load(tag_of(cal_y), line)
            cost_v, _ = load(tag_of(val_y), line)
            perils = list(cost_c.columns)
            comp = lambda d: pc.completeness(line, perils, d["year"].reset_index(drop=True)).set_axis(d.index)  # noqa: E731
            ec, en = cost_c.loc[cal["row_id"]].set_axis(cal.index) * comp(cal), cnt_c.loc[cal["row_id"]].set_axis(cal.index) * comp(cal)
            ev = cost_v.loc[val["row_id"]].set_axis(val.index) * comp(val)
            m = target.matched_claims(rows, em_claims, line)
            m = m[m["row_id"].notna() & (m["status"] != "repudiated") & m["row_id"].isin(cal["row_id"])]
            if a.weather_slope and line in ("motor", "buildings"):
                wp = "glass_hail" if line == "motor" else "storm"
                nrow = m[m["peril"] == wp].groupby("row_id").size().reindex(cal["row_id"], fill_value=0).to_numpy()
                h_c = cal["hail_days"].to_numpy() - cal["hail_days"].mean()
                e_c = ec[wp].to_numpy()
                from scipy.optimize import minimize_scalar
                def nll(b):   # Poisson likelihood of the peril's counts with the expected cost as relative rate, profile over the level
                    mu = e_c * np.exp(b * h_c)
                    mu = mu * nrow.sum() / mu.sum()
                    return float((mu - nrow * np.log(mu + 1e-12)).sum())
                beta = float(minimize_scalar(nll, bounds=(0.0, 0.06), method="bounded").x)
                print(f"{name} {line}: {wp} hail slope {beta:+.4f} per day (clipped to 0..0.06)", flush=True)
                for frame, dfs in ((cal, (ec, en)), (val, (ev,))):
                    f_ = np.exp(beta * (frame["hail_days"].to_numpy() - frame["hail_days"].mean()))
                    f_ = f_ / np.average(f_, weights=dfs[0][wp].to_numpy())   # keeps the peril's total of the year
                    for df_ in dfs:
                        df_[wp] = df_[wp] * f_
            act_rows = m.groupby(["row_id", "peril"])["net_cost"].sum().unstack(fill_value=0.0).reindex(index=cal["row_id"], columns=perils, fill_value=0.0).set_axis(cal.index)
            n_by = m.groupby("peril").size().reindex(perils, fill_value=0).astype(float)
            # medians: emerging claims of the calibration year vs developed claims of the same year
            dc = dev_claims[(dev_claims["line"] == line) & (dev_claims["status"] != "repudiated") & (dev_claims["loss_date"].dt.year == cal_y)]
            dmed = target.claim_net_cost(dc).groupby(dc["peril"]).median().reindex(perils)
            emed = m.groupby("peril")["net_cost"].median().reindex(perils).fillna(dmed)
            rf = transfer.robust_factors(n_by, en.sum(), emed, dmed, transfer.line_value(a.k_freq, line), transfer.line_value(a.k_sev, line), None if prior is None else prior[line])
            dcost = target.claim_net_cost(dc)
            loc = {}
            for q in (0.95, 0.99):   # winsorised mean: every claim capped at the quantile of its own market and peril
                dw = dcost.groupby(dc["peril"]).apply(lambda x: x.clip(upper=x.quantile(q)).mean()).reindex(perils)
                ew = m.groupby("peril")["net_cost"].apply(lambda x: x.clip(upper=x.quantile(q)).mean()).reindex(perils).fillna(dw)
                loc[q] = transfer.robust_factors(n_by, en.sum(), ew, dw, transfer.line_value(a.k_freq, line), transfer.line_value(a.k_sev, line), None if prior is None else prior[line])
            f_cost = transfer.calibration_factors(act_rows, ec, n_by, k=300)
            variants = {
                "closure_only": (ec.sum(axis=1), ev.sum(axis=1)),
                "cost_k300": (pd.Series(transfer.apply_factors(ec, f_cost), index=cal.index), pd.Series(transfer.apply_factors(ev, f_cost), index=val.index)),
                "robust_fs": (pd.Series(transfer.apply_factors(ec, rf["cost_factor"]), index=cal.index), pd.Series(transfer.apply_factors(ev, rf["cost_factor"]), index=val.index)),
            }
            def labels(d, cols):
                parts = [(pd.cut(d["ncb_years"], [-1, 0, 1, 2, 5, 8, 11, 100]).astype(object) if c == "ncb" else d[c].astype(object)).where(lambda x: x.notna(), "NA").astype(str) for c in cols]
                out = parts[0]
                for q in parts[1:]:
                    out = out + "|" + q
                return out
            for spec in [x for x in a.extra.split(";") if x]:
                rfw = loc[0.99]
                ptr0 = pd.Series(transfer.apply_factors(ec, rfw["cost_factor"]), index=cal.index)
                pva0 = pd.Series(transfer.apply_factors(ev, rfw["cost_factor"]), index=val.index)
                base_cols = spec.split("+") if "gender" in spec.split("+") else ["gender"] + spec.split("+")
                base_cols = [c for c in base_cols if (c == "ncb" and "ncb_years" in cal.columns) or c in cal.columns]   # a factor that does not exist in this line is left out (gender only)
                g_cal, g_val = labels(cal, base_cols), labels(val, base_cols)
                gf = transfer.group_factors(cal["y"], ptr0, g_cal, cal["n_claims"], 300.0)
                preds.setdefault(f"extra[{spec}]", []).append(pva0.to_numpy() * g_val.map(gf).fillna(1.0).to_numpy())
            for q, rfq in loc.items():
                variants[f"robustw{int(q * 100)}"] = (pd.Series(transfer.apply_factors(ec, rfq["cost_factor"]), index=cal.index), pd.Series(transfer.apply_factors(ev, rfq["cost_factor"]), index=val.index))
            if a.tw_names:   # direct Tweedie GBM: relativity only; its level follows the robust frequency x severity level on the calibration rows
                tws = [x for x in a.tw_names.split(",") if x]
                def tw_cost(tag, rows_):
                    return sum(ms.base_costs(f"tw_{t}", tag, line)["all"] for t in tws).loc[rows_["row_id"]].set_axis(rows_.index) / len(tws)
                crow_c = ec.sum(axis=1) / cost_c.loc[cal["row_id"]].set_axis(cal.index).sum(axis=1)   # share reported by the valuation date, row specific (from the frequency x severity model)
                crow_v = ev.sum(axis=1) / cost_v.loc[val["row_id"]].set_axis(val.index).sum(axis=1)
                tw_c, tw_v = tw_cost(TAG[cal_y], cal) * crow_c, tw_cost(TAG[val_y], val) * crow_v
                fs_c, fs_v = variants["robustw99"]
                scale = float(fs_c.sum() / tw_c.sum())
                variants["robustw99_tw"] = ((1 - a.tw_share) * fs_c + a.tw_share * scale * tw_c, (1 - a.tw_share) * fs_v + a.tw_share * scale * tw_v)
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
        main_key = "robustw99_tw_noclose+gender" if "robustw99_tw_noclose+gender" in P else "robustw99_noclose+gender"
        pd.DataFrame({"row_id": V["row_id"], "customer_id": V["customer_id"], "line": V["line"], "y": V["y"], "pred": P[main_key], "pred_closure": P["closure_only+gender"]}).to_parquet(
            pc.P2 / f"p2_calib2_{name.split()[0]}_{a.gbm_name}_nn{a.nn_weight}{'_tw' if a.tw_names else ''}{'_' + a.nn_models if a.nn_weight > 0 and a.nn_models != 'nn_tabm' else ''}_oof.parquet")   # for the scenario analysis
        if "robustw99_tw_noclose+gender" in P:
            d = bootstrap_d2_diff(V["y"], P["robustw99_tw_noclose+gender"], P["robustw99_noclose+gender"], V["customer_id"])
            print(f"  with Tweedie part - without: {d['diff']:+.4f} (95% {d['lo']:+.4f}..{d['hi']:+.4f})")
        for k in [x for x in P if x.startswith("extra[")]:
            d = bootstrap_d2_diff(V["y"], P[k], P["robustw99_noclose+gender"], V["customer_id"])
            print(f"  {k} - robustw99_noclose+gender: {d['diff']:+.4f} (95% {d['lo']:+.4f}..{d['hi']:+.4f})  D2 {d2_tweedie(V['y'], P[k]):.4f}")
        for k in ("cost_k300+gender", "robust_fs_noclose+gender", "robustw95_noclose+gender", "robustw99_noclose+gender"):
            d = bootstrap_d2_diff(V["y"], P[k], P["closure_only+gender"], V["customer_id"])
            print(f"  {k} - closure_only+gender: {d['diff']:+.4f} (95% {d['lo']:+.4f}..{d['hi']:+.4f})")


if __name__ == "__main__":
    main()
