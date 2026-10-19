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


def test_glm_with_bands_fits_and_captures_a_non_linear_effect():
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
    bnd = FreqSev(claims, kind="glm", trend=False, completeness=False, rebalance=False, bands=True).fit(rows)
    edge, mid = rows.assign(x=2.8), rows.assign(x=0.0)
    ratio = lambda m: m.predict(edge).mean() / m.predict(mid).mean()  # noqa: E731
    assert ratio(bnd) > 2.5 and ratio(plain) < 2.0                    # a linear GLM cannot see the U shape


def test_seed_averaging_keeps_the_model_working_and_changes_predictions_slightly():
    rows, claims = _data()
    one = FreqSev(claims, kind="gbm", freq_rounds=40, sev_rounds=30, rebalance=False).fit(rows)
    three = FreqSev(claims, kind="gbm", freq_rounds=40, sev_rounds=30, rebalance=False, n_seeds=3).fit(rows)
    p1, p3 = one.predict(rows), three.predict(rows)
    assert len(three.freq_["A"][0]) == 3 and np.isfinite(p3).all() and (p3 >= 0).all()
    assert abs(p3.sum() / p1.sum() - 1) < 0.1 and np.corrcoef(p1, p3)[0, 1] > 0.9


def test_per_peril_rebalancing_matches_each_perils_training_total():
    rows, claims = _data()
    m = FreqSev(claims, kind="glm", cap_quantile=0.99, trend=False, completeness=False, rebalance_level="peril").fit(rows)
    by = m.expected_by_peril(rows)
    for p in by.columns:
        assert np.isclose(by[p].sum(), m.level_by_peril_[p], rtol=1e-9)
    assert np.isclose(m.predict(rows).sum(), sum(m.level_by_peril_.values()), rtol=1e-9)


def test_outlier_winsorisation_removes_a_single_giant_claim_from_the_excess_load_and_level():
    rows, claims = _data()
    claims = claims.copy()
    first = claims.index[claims.peril == "A"][0]
    claims.loc[first, "net_cost"] = 5_000_000.0           # one absurd claim
    plain = FreqSev(claims, kind="glm", cap_quantile=0.99, trend=False, completeness=False, rebalance_level="peril").fit(rows)
    robust = FreqSev(claims, kind="glm", cap_quantile=0.99, trend=False, completeness=False, rebalance_level="peril", outlier_mult=5.0).fit(rows)
    assert robust.excess_["A"] < 0.2 * plain.excess_["A"]
    assert robust.predict(rows).sum() < 0.5 * plain.predict(rows).sum()   # level no longer carries the giant claim


def test_nn_freqsev_balances_total_ranks_risk_and_is_nonnegative():
    pytest.importorskip("torch")
    rows, claims = _data()
    rows["customer_id"] = [f"C{i // 3}" for i in range(len(rows))]
    m = FreqSev(claims, kind="nn", nn_params={"epochs": 4, "hidden": (16, 8), "threads": 2}).fit(rows)
    p = m.predict(rows)
    assert (p >= 0).all() and np.isclose(p.sum(), rows.y.sum(), rtol=1e-9)
    hi, lo = rows.assign(x=2.0, g="c", exposure_calc=1.0), rows.assign(x=-2.0, g="a", exposure_calc=1.0)
    assert m.predict(hi).mean() > m.predict(lo).mean()


@pytest.mark.parametrize("kind", ["xgb", "cat"])
def test_alternative_boosters_balance_total_rank_risk_and_are_nonnegative(kind):
    pytest.importorskip("xgboost" if kind == "xgb" else "catboost")
    rows, claims = _data()
    m = FreqSev(claims, kind=kind, freq_rounds=60, sev_rounds=40, threads=2, params_freq={"depth": 3} if kind == "cat" else {}).fit(rows)
    p = m.predict(rows)
    assert (p >= 0).all() and np.isclose(p.sum(), rows.y.sum(), rtol=1e-9)
    hi, lo = rows.assign(x=2.0, g="c", exposure_calc=1.0), rows.assign(x=-2.0, g="a", exposure_calc=1.0)
    assert m.predict(hi).mean() > m.predict(lo).mean()


def test_nn_tweedie_balances_total_and_ranks_risk():
    pytest.importorskip("torch")
    from iaa.models.nn_tweedie import NNTweedie
    rows, _ = _data()
    rows["customer_id"] = [f"C{i // 3}" for i in range(len(rows))]
    m = NNTweedie(epochs=4, hidden=(16, 8), threads=2).fit(rows)
    p = m.predict(rows)
    assert (p >= 0).all() and np.isclose(p.sum(), rows.y.sum(), rtol=1e-9)
    hi, lo = rows.assign(x=2.0, g="c", exposure_calc=1.0), rows.assign(x=-2.0, g="a", exposure_calc=1.0)
    assert m.predict(hi).mean() > m.predict(lo).mean()


def _with_ids(rows):
    return rows.assign(customer_id=[f"C{i // 3}" for i in range(len(rows))], address_id=[f"A{i // 5}" for i in range(len(rows))])


def _checks(m, rows):
    p = m.predict(rows)
    assert (p >= 0).all() and np.isfinite(p).all() and np.isclose(p.sum(), rows.y.sum(), rtol=1e-9)
    hi, lo = rows.assign(x=2.0, g="c", exposure_calc=1.0), rows.assign(x=-2.0, g="a", exposure_calc=1.0)
    assert m.predict(hi).mean() > m.predict(lo).mean()


def test_tabm_and_entity_embeddings_train_and_rank_risk():
    pytest.importorskip("torch")
    rows, claims = _data()
    rows = _with_ids(rows)
    m = FreqSev(claims, kind="nn", nn_params={"epochs": 4, "hidden": (16, 8), "threads": 2, "tabm_k": 3, "entity_min": 2}, entity_cols=("customer_id", "address_id")).fit(rows)
    _checks(m, rows)
    assert m.fnet_.nets_[0].k == 3 and len(m.enc_.cards_) == len(m.enc_.cat_cols)


def test_recency_and_domain_shift_weights_fit_and_rank_risk():
    rows, claims = _data()
    rows = rows.assign(year=np.where(np.arange(len(rows)) % 2 == 0, 2021, 2022))
    target = rows.sample(1000, random_state=0).assign(x=lambda d: d.x + 0.5)
    m = FreqSev(claims, kind="gbm", freq_rounds=40, sev_rounds=30, recency=0.3, adv_strength=0.5)
    assert m.uses_target
    m.fit(rows, target=target)
    _checks(m, rows)
    w = m._row_weights(rows, target)
    assert np.isclose(w.mean(), 1.0) and w[rows.year.to_numpy() == 2022].mean() > w[rows.year.to_numpy() == 2021].mean() * 0.9


def test_risk_dependent_tail_load_keeps_total_and_ranking():
    rows, claims = _data()
    claims = claims.copy()
    big = claims.index[claims.peril == "A"][::40]          # a few large claims, more of them for high x rows
    claims.loc[big, "net_cost"] *= 15
    rows = rows.assign(y=claims.groupby("row_id").net_cost.sum().reindex(rows.row_id).fillna(0.0).to_numpy())
    m = FreqSev(claims, kind="gbm", freq_rounds=40, sev_rounds=30, tail_lambda=0.5)
    m.tail_min_claims, m.tail_min_events = 100, 4
    m.fit(rows)
    assert "A" in m.tail_
    _checks(m, rows)


def test_piecewise_linear_encoding_adds_inputs_and_trains():
    pytest.importorskip("torch")
    rows, claims = _data()
    rows["customer_id"] = [f"C{i // 3}" for i in range(len(rows))]
    m = FreqSev(claims, kind="nn", nn_params={"epochs": 3, "hidden": (16, 8), "threads": 2, "ple_bins": 6}).fit(rows)
    assert m.enc_.ple_ and m.enc_.n_num_ > 1
    _checks(m, rows)


@pytest.mark.parametrize("enc", ["label", "count", "both", "target"])
def test_alternative_categorical_encodings_train_and_rank_risk(enc):
    rows, claims = _data()
    rows = rows.assign(row_id=rows["row_id"])
    m = FreqSev(claims, kind="gbm", freq_rounds=40, sev_rounds=30, cat_encoding=enc).fit(rows)
    _checks(m, rows)
    X = m._x(rows)
    assert not any(isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns)
    if enc in ("count", "both"):
        assert "g_cnt" in X.columns


@pytest.mark.parametrize("enc", ["lowcard", "count", "both"])
def test_numeric_encodings_for_trees_add_columns_and_rank_risk(enc):
    rows, claims = _data()
    rows = rows.assign(seats=(np.arange(len(rows)) % 4).astype(float))
    m = FreqSev(claims, kind="gbm", freq_rounds=40, sev_rounds=30, cat_encoding="label", num_encoding=enc).fit(rows)
    _checks(m, rows)
    X = m._x(rows)
    assert (("seats_cat" in X.columns) == (enc in ("lowcard", "both"))) and (("x_ncnt" in X.columns) == (enc in ("count", "both")))


@pytest.mark.parametrize("tf", ["log", "quantile"])
def test_nn_numeric_transforms_train_and_rank_risk(tf):
    pytest.importorskip("torch")
    rows, claims = _data()
    rows["customer_id"] = [f"C{i // 3}" for i in range(len(rows))]
    rows["skewed"] = np.exp(np.random.default_rng(1).normal(size=len(rows)) * 1.5)
    m = FreqSev(claims, kind="nn", nn_params={"epochs": 3, "hidden": (16, 8), "threads": 2, "num_transform": tf}).fit(rows)
    _checks(m, rows)
    assert (tf != "log") or "skewed" in m.enc_.logcols_
