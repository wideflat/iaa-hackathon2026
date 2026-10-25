"""Simulation test of the Part 2 level method (iaa.transfer.robust_factors) against a known truth.

  python scripts/p2_simulate_level.py [--sims 400]

Per line and direction (calibrate 2023 / score 2024 and calibrate 2024 / score 2023, the A/B tags of the real validation): the t7 base's expected counts and
costs of the calibration year define a simulated emerging book: claim counts ~ Poisson(expected count x true frequency multiplier), claim costs bootstrapped from the
developed claims of the peril x true severity multiplier. Level estimators are applied to the simulated book and judged on the validation rows against the truth
mu_i = sum_p cost_ip x multiplier_p by the Tweedie(1.5) Bregman loss d(mu_i, m_i) (= the expected excess deviance of predicting m instead of mu), in D2 points of the real
validation year's null deviance. Scenarios: A one multiplier per line, B the observed per-peril values, C line value x lognormal(0, 0.25) per peril redrawn per book.
Writes artifacts/p2_simulation_level.csv. Truth is known only inside this simulator, so the test can reject an estimator, not prove one right.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, data, market, target, transfer  # noqa: E402
from iaa.config import LINES  # noqa: E402
import p2_calib as pc  # noqa: E402
import p2_make_submission as ms  # noqa: E402

DEV_DIR = ROOT / "data" / "part2" / "developed"
TAG = {2023: "A", 2024: "B"}
MODELS = {"hist1": 0.5, "nn_tabm_hist1": 0.25, "nn_tabm_ple_hist1": 0.25}   # the t7 base (tree half without the Tweedie part: that part only reshapes relativities)
ESTIMATORS = [("closure", None, None)] + [(f"robust kf={kf:g} ks={ks:g}", kf, ks) for kf in (20, 100, 400, 5000) for ks in (30, 300)]


def bregman(mu: np.ndarray, m: np.ndarray) -> float:
    """Tweedie(1.5) unit deviance d(y=mu, m): expected excess deviance of predicting m when the true mean is mu."""
    return float((-8.0 * np.sqrt(mu) + 4.0 * mu / np.sqrt(m) + 4.0 * np.sqrt(m)).sum())


def winsor_mean(x: np.ndarray, q: float = 0.99) -> float:
    return float(np.minimum(x, np.quantile(x, q)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261025)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    em_claims, dev_claims = data.load_claims(pc.EM_DIR, "emerging"), data.load_claims(DEV_DIR, "developed")
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    out = []
    for line in LINES:
        e = market.to_common(em[line], line, "emerging")
        rows = e[e["split"] == "train"]
        m_all = target.matched_claims(rows, em_claims, line)
        m_all = m_all[m_all["row_id"].notna() & (m_all["status"] != "repudiated")]
        yr = rows.set_index("row_id")["year"]
        m_all = m_all.assign(year=m_all["row_id"].map(yr))
        # observed per-peril values (both years) for scenario B, and the developed claim pools per peril
        obs_f, obs_s, pools, dw_all = {}, {}, {}, {}
        n_exp = {}
        for y in (2023, 2024):
            cal = rows[rows["year"] == y]
            cnt = ms.mixed_costs(MODELS, TAG[y], line, counts=True).loc[cal["row_id"]].set_axis(cal.index)
            comp = pc.completeness(line, list(cnt.columns), cal["year"].reset_index(drop=True)).set_axis(cal.index)
            n_exp[y] = (cnt * comp).sum()
        perils = list(n_exp[2023].index)
        dc_all = dev_claims[(dev_claims["line"] == line) & (dev_claims["status"] != "repudiated")]
        dcost_all = target.claim_net_cost(dc_all)
        for p in perils:
            pools[p] = dcost_all[dc_all["peril"] == p].to_numpy()
            dw_all[p] = winsor_mean(pools[p])
            ma = m_all[m_all["peril"] == p]["net_cost"].to_numpy()
            obs_f[p] = float(np.clip(len(ma) / (n_exp[2023][p] + n_exp[2024][p]), 0.2, 5.0))
            obs_s[p] = float(np.clip(winsor_mean(ma) / dw_all[p], 0.3, 3.0)) if len(ma) >= 5 else 1.0
        line_f = float(sum(len(m_all[m_all["peril"] == p]) for p in perils) / (n_exp[2023].sum() + n_exp[2024].sum()))
        line_s = float(np.average([obs_s[p] for p in perils], weights=[obs_f[p] * (n_exp[2023][p] + n_exp[2024][p]) for p in perils]))
        for cal_y, val_y in ((2023, 2024), (2024, 2023)):
            cal, val = rows[rows["year"] == cal_y], rows[rows["year"] == val_y]
            cost_c, cnt_c = ms.mixed_costs(MODELS, TAG[cal_y], line), ms.mixed_costs(MODELS, TAG[cal_y], line, counts=True)
            cost_v = ms.mixed_costs(MODELS, TAG[val_y], line)
            comp_c = pc.completeness(line, perils, cal["year"].reset_index(drop=True)).set_axis(cal.index)
            comp_v = pc.completeness(line, perils, val["year"].reset_index(drop=True)).set_axis(val.index)
            ec = cost_c.loc[cal["row_id"]].set_axis(cal.index)[perils] * comp_c
            en = (cnt_c.loc[cal["row_id"]].set_axis(cal.index)[perils] * comp_c).sum()
            ev = (cost_v.loc[val["row_id"]].set_axis(val.index)[perils] * comp_v).to_numpy()
            # null deviance of the real validation year (D2 denominator)
            y_val = val["y"].to_numpy()
            ybar = y_val.mean()
            null_dev = float((2.0 * (np.where(y_val > 0, -4.0 * np.sqrt(y_val), 0.0) + 2.0 * y_val / np.sqrt(ybar) + 2.0 * np.sqrt(ybar))).sum())
            # developed winsorised means of the calibration year's pools are the denominator of the severity factor (as in p2_calib2)
            dcy = dev_claims[(dev_claims["line"] == line) & (dev_claims["status"] != "repudiated") & (dev_claims["loss_date"].dt.year == cal_y)]
            dcost_y = target.claim_net_cost(dcy)
            dw = pd.Series({p: winsor_mean(dcost_y[dcy["peril"] == p].to_numpy()) for p in perils})
            for scen in ("A", "B", "C"):
                reg = {name: [] for name, _, _ in ESTIMATORS}
                bias = {name: [] for name, _, _ in ESTIMATORS}
                for _ in range(a.sims):
                    if scen == "A":
                        tf = pd.Series(line_f, index=perils); ts = pd.Series(line_s, index=perils)
                    elif scen == "B":
                        tf = pd.Series(obs_f); ts = pd.Series(obs_s)
                    else:
                        tf = pd.Series({p: line_f * np.exp(rng.normal(0, 0.25)) for p in perils}); ts = pd.Series({p: line_s * np.exp(rng.normal(0, 0.25)) for p in perils})
                    n_by = pd.Series({p: float(rng.poisson(en[p] * tf[p])) for p in perils})
                    costs = {p: rng.choice(pools[p], int(n_by[p])) * ts[p] for p in perils}
                    ew = pd.Series({p: (winsor_mean(costs[p]) if n_by[p] > 0 else dw[p]) for p in perils})
                    total_cost = float(sum(c.sum() for c in costs.values()))
                    mu = ev @ (tf * ts).to_numpy()
                    for name, kf, ks in ESTIMATORS:
                        if kf is None:
                            f = pd.Series(total_cost / float(ec.to_numpy().sum()), index=perils)
                        else:
                            f = transfer.robust_factors(n_by, en, ew, dw, kf, ks)["cost_factor"]
                        m = ev @ f.to_numpy()
                        reg[name].append(bregman(mu, m) / null_dev)
                        bias[name].append(float(np.log(m.sum() / mu.sum())))
                for name, _, _ in ESTIMATORS:
                    out.append({"line": line, "direction": f"{cal_y}->{val_y}", "scenario": scen, "estimator": name, "regret_d2": float(np.mean(reg[name])), "regret_sd": float(np.std(reg[name])),
                                "log_bias": float(np.mean(bias[name])), "log_bias_sd": float(np.std(bias[name])), "null_dev": null_dev})
            print(line, cal_y, "->", val_y, "done", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "artifacts" / "p2_simulation_level.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nMean regret in D2 points by scenario, estimator and line (average of the two directions); pooled = sum of the lines' regret over the sum of their null deviances")
    for scen in ("A", "B", "C"):
        d = df[df["scenario"] == scen]
        t = d.pivot_table(index="estimator", columns="line", values="regret_d2", aggfunc="mean")
        w = d.groupby("line")["null_dev"].mean()
        t["pooled"] = (d.assign(r=d["regret_d2"] * d["null_dev"]).groupby("estimator")["r"].sum() / d.groupby("estimator")["null_dev"].sum())
        t["bias_buildings"] = d[d["line"] == "buildings"].groupby("estimator")["log_bias"].mean()
        print(f"\n== scenario {scen}\n", (t * 1).round(4).to_string())


if __name__ == "__main__":
    main()
