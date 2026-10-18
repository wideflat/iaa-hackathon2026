import numpy as np
import pandas as pd
import pytest

from iaa.models.freqsev import FreqSev


def _data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], n)
    expo = rng.uniform(0.2, 1.0, n)
    rows = pd.DataFrame({"row_id": [f"R{i}" for i in range(n)], "x": x, "g": g, "exposure_calc": expo, "year": 2022})
    f_a = np.exp(-2.2 + 0.7 * x) * expo           # peril A frequency depends on x
    f_b = np.exp(-3.0 + (g == "c")) * expo         # peril B frequency depends on g
    cl = []
    for p, f, scale in (("A", f_a, 800.0), ("B", f_b, 3000.0)):
        k = rng.poisson(f)
        for i in np.nonzero(k)[0]:
            for _ in range(k[i]):
                cl.append({"row_id": rows.row_id[i], "peril": p, "net_cost": rng.gamma(2.0, scale / 2.0), "status": "closed"})
    claims = pd.DataFrame(cl)
    claims["loss_date"], claims["report_date"] = pd.Timestamp("2022-06-01"), pd.Timestamp("2022-06-10")
    y = claims.groupby("row_id").net_cost.sum().reindex(rows.row_id).fillna(0.0).to_numpy()
    rows["y"], rows["n_claims"] = y, 0
    return rows, claims


@pytest.mark.parametrize("kind", ["glm", "gbm"])
def test_freqsev_balances_total_ranks_risk_and_is_nonnegative(kind):
    rows, claims = _data()
    m = FreqSev(claims, kind=kind, freq_rounds=60, sev_rounds=40).fit(rows)
    p = m.predict(rows)
    assert (p >= 0).all() and np.isclose(p.sum(), rows.y.sum(), rtol=1e-9)
    hi, lo = rows.assign(x=2.0, g="c", exposure_calc=1.0), rows.assign(x=-2.0, g="a", exposure_calc=1.0)
    assert m.predict(hi).mean() > m.predict(lo).mean()


def test_peril_decomposition_adds_up_and_excess_load_is_nonnegative():
    rows, claims = _data()
    m = FreqSev(claims, kind="glm", rebalance=False).fit(rows)
    by = m.expected_by_peril(rows)
    assert set(by.columns) == {"A", "B"} and np.allclose(by.sum(axis=1), m.predict(rows))
    assert all(v >= 0 for v in m.excess_.values())


@pytest.mark.parametrize("kind", ["glm", "gbm"])
def test_zero_net_cost_claims_do_not_break_the_gamma_fit(kind):
    rows, claims = _data()
    claims.loc[claims.index[:5], "net_cost"] = 0.0  # e.g. a claim whose recoveries equal its incurred amount
    m = FreqSev(claims, kind=kind, freq_rounds=30, sev_rounds=30).fit(rows)
    assert np.isfinite(m.predict(rows)).all()


def test_trend_is_recovered_and_projects_to_the_year_after_training():
    rng = np.random.default_rng(3)
    n = 12000
    year = rng.choice([2021, 2022, 2023], n)
    rows = pd.DataFrame({"row_id": [f"R{i}" for i in range(n)], "x": rng.normal(size=n), "exposure_calc": 1.0, "year": year})
    k = rng.poisson(0.08, n)
    cl = [{"row_id": rows.row_id[i], "peril": "A", "net_cost": 1000.0 * 1.06 ** (year[i] - 2021) * rng.gamma(20, 1 / 20), "status": "closed"}
          for i in np.nonzero(k)[0] for _ in range(k[i])]
    claims = pd.DataFrame(cl)
    rows["y"] = claims.groupby("row_id").net_cost.sum().reindex(rows.row_id).fillna(0.0).to_numpy()
    m = FreqSev(claims, kind="glm", completeness=False, rebalance=False).fit(rows)
    assert abs(m.sev_trend_ - np.log(1.06)) < 0.02 and m.ref_year_ == 2024
    flat = FreqSev(claims, kind="glm", completeness=False, trend=False, rebalance=False).fit(rows)
    assert m.predict(rows).sum() > flat.predict(rows).sum() * 1.05  # projecting to 2024 level lifts the level by ~2 years of trend


def test_completeness_offset_raises_ultimate_frequency_for_immature_years_and_observed_prediction_scales_back():
    rows, claims = _data()
    rows["year"] = 2022
    claims["loss_date"] = pd.Timestamp("2022-06-01")
    claims["report_date"] = pd.Timestamp("2022-06-10")
    m = FreqSev(claims, kind="glm", trend=False, rebalance=False)
    m.c_by_year = {"A": {2022: 0.5}, "B": {2022: 0.5}}   # pretend half of each year's claims are not yet reported
    m.fit(rows)
    obs, ult = m.predict_observed(rows), m.predict(rows)
    assert np.allclose(obs, 0.5 * ult) and ult.sum() > 1.6 * rows.y.sum()


def test_glm_with_splines_fits_and_captures_a_non_linear_effect():
    rng = np.random.default_rng(5)
    n = 8000
    x = rng.uniform(-3, 3, n)
    rows = pd.DataFrame({"row_id": [f"R{i}" for i in range(n)], "x": x, "exposure_calc": 1.0, "year": 2022})
    lam = 0.05 + 0.25 * (np.abs(x) > 2)            # U-shaped risk
    k = rng.poisson(lam)
    claims = pd.DataFrame([{"row_id": rows.row_id[i], "peril": "A", "net_cost": 1000.0, "status": "closed",
                            "loss_date": pd.Timestamp("2022-06-01"), "report_date": pd.Timestamp("2022-06-02")} for i in np.nonzero(k)[0] for _ in range(k[i])])
    rows["y"] = claims.groupby("row_id").net_cost.sum().reindex(rows.row_id).fillna(0.0).to_numpy()
    plain = FreqSev(claims, kind="glm", trend=False, completeness=False, rebalance=False).fit(rows)
    spl = FreqSev(claims, kind="glm", trend=False, completeness=False, rebalance=False, splines=True).fit(rows)
    edge, mid = rows.assign(x=2.8), rows.assign(x=0.0)
    ratio = lambda m: m.predict(edge).mean() / m.predict(mid).mean()  # noqa: E731
    assert ratio(spl) > 2.5 and ratio(plain) < 2.0                    # a linear GLM cannot see the U shape
