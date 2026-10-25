import numpy as np
import pandas as pd
import pytest

from iaa import market, transfer


def _frame(vals):
    return pd.DataFrame(vals, index=[10, 11, 12], columns=["a", "b"])


def test_factors_recover_a_known_scale_when_credibility_is_full():
    pred = _frame([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    act = pred * np.array([2.0, 0.5])
    f = transfer.calibration_factors(act, pred, pd.Series({"a": 1e9, "b": 1e9}), k=1.0)
    assert f["a"] == pytest.approx(2.0, rel=1e-6) and f["b"] == pytest.approx(0.5, rel=1e-6)


def test_factors_shrink_to_line_ratio_without_claims():
    pred = _frame([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    act = pred * np.array([2.0, 0.5])
    f = transfer.calibration_factors(act, pred, pd.Series({"a": 0.0, "b": 0.0}))
    line = act.to_numpy().sum() / pred.to_numpy().sum()
    assert f["a"] == pytest.approx(line) and f["b"] == pytest.approx(line)


def test_factors_reject_misaligned_inputs():
    pred = _frame([[1.0, 2.0]] * 3)
    with pytest.raises(AssertionError):
        transfer.calibration_factors(pred.rename(columns={"b": "c"}), pred, pd.Series({"a": 5, "c": 5}))


def test_apply_factors_sums_scaled_perils():
    pred = _frame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = transfer.apply_factors(pred, pd.Series({"a": 2.0, "b": 0.5}))
    assert out == pytest.approx([3.0, 8.0, 13.0])


def test_geometric_blend_endpoints():
    a, b = np.array([1.0, 4.0]), np.array([9.0, 1.0])
    assert transfer.geometric_blend(a, b, 1.0) == pytest.approx(a)
    assert transfer.geometric_blend(a, b, 0.0) == pytest.approx(b)
    assert transfer.geometric_blend(a, b, 0.5) == pytest.approx([3.0, 2.0])


def test_occupation_normalisation_flags_self_employed_and_trims():
    occ, se = market.normalise_occupation(pd.Series(["Nurse ", "nurse", "Self-employed nurse", None, "Software developer"]))
    assert list(occ.fillna("NA")) == ["nurse", "nurse", "nurse", "NA", "software developer"]
    assert list(se) == [False, False, True, False, False]


def test_age_band_developed_cut_points_match_emerging_labels():
    df = pd.DataFrame({"driver_age": [17.0, 24.9, 25.0, 49.9, 50.0, 64.9, 65.0, 92.0]})
    assert list(market.age_band(df, "motor", "developed")) == ["17-24", "17-24", "25-34", "35-49", "50-64", "50-64", "65+", "65+"]


def test_emerging_age_band_rejects_unknown_labels():
    with pytest.raises(AssertionError):
        market.age_band(pd.DataFrame({"driver_age_band": ["17-24", "99+"]}), "motor", "emerging")


def test_credibility_weights():
    assert transfer.credibility(100.0, 100.0) == pytest.approx(0.5)
    assert transfer.credibility(0.0, 50.0) == 0.0


def test_split_factors_recover_known_freq_and_sev_scale_with_ample_claims():
    exp_n = pd.Series({"a": 1000.0, "b": 2000.0})
    exp_c = pd.Series({"a": 1000.0 * 50.0, "b": 2000.0 * 10.0})
    act_n = exp_n * pd.Series({"a": 1.5, "b": 0.8})
    act_c = exp_c * pd.Series({"a": 1.5 * 0.7, "b": 0.8 * 0.7})   # severity 0.7 everywhere
    f = transfer.split_factors(act_n, exp_n, act_c, exp_c, k_freq=1.0, k_sev=1.0)
    assert f["freq"]["a"] == pytest.approx(1.5, rel=0.01) and f["freq"]["b"] == pytest.approx(0.8, rel=0.01)
    assert f["sev"]["a"] == pytest.approx(0.7, rel=0.01) and f["sev"]["b"] == pytest.approx(0.7, rel=0.01)


def test_split_factors_keep_the_observed_line_total():
    exp_n = pd.Series({"a": 100.0, "b": 30.0, "c": 5.0})
    exp_c = pd.Series({"a": 5000.0, "b": 9000.0, "c": 2000.0})
    act_n = pd.Series({"a": 90.0, "b": 45.0, "c": 2.0})
    act_c = pd.Series({"a": 3500.0, "b": 8000.0, "c": 900.0})
    f = transfer.split_factors(act_n, exp_n, act_c, exp_c, k_freq=20.0, k_sev=100.0)
    total = (f["freq"] * f["sev"] * exp_c).sum()
    assert total == pytest.approx(act_c.sum(), rel=0.15)   # shrinkage moves perils, the line stays close to the observed total
    assert (f[["freq", "sev"]] > 0).all().all()


def test_group_factors_recover_ratio_and_preserve_total():
    idx = range(6)
    pred = pd.Series([1.0] * 6, index=idx)
    actual = pd.Series([1.5, 1.5, 1.5, 0.5, 0.5, 0.5], index=idx)
    g = pd.Series(["M"] * 3 + ["F"] * 3, index=idx)
    n = pd.Series([1e6] * 6, index=idx)
    f = transfer.group_factors(actual, pred, g, n, k=1.0)
    assert f["M"] / f["F"] == pytest.approx(3.0, rel=1e-3)
    assert (pred.groupby(g).sum() * f).sum() == pytest.approx(pred.sum())


def test_group_factors_without_claims_are_all_one():
    idx = range(4)
    f = transfer.group_factors(pd.Series([3.0, 3.0, 1.0, 1.0], index=idx), pd.Series([1.0] * 4, index=idx), pd.Series(list("MMFF"), index=idx), pd.Series([0.0] * 4, index=idx), k=100.0)
    assert f.tolist() == pytest.approx([1.0, 1.0])


def test_calibrate_and_apply_recovers_a_known_level_and_group_effect():
    rng = np.random.default_rng(0)
    n = 4000
    idx = pd.RangeIndex(n)
    pred = pd.DataFrame({"a": rng.uniform(50, 150, n), "b": rng.uniform(5, 15, n)}, index=idx)
    gender = pd.Series(rng.choice(["M", "F"], n), index=idx)
    truth = pred * 0.6 * np.where(gender == "M", 1.2, 0.8)[:, None]
    act = truth.copy()
    out, fac = transfer.calibrate_and_apply(pred, act, pd.Series({"a": 1e6, "b": 1e6}), truth.sum(axis=1), pd.Series(1e3, index=idx), gender, pred, gender, k_peril=1.0, k_group=1.0)
    assert out == pytest.approx(truth.sum(axis=1).to_numpy(), rel=0.02)
    assert fac["group"]["M"] / fac["group"]["F"] == pytest.approx(1.5, rel=0.02)


def test_robust_factors_ignore_one_huge_claim_and_shrink_small_perils():
    act_n = pd.Series({"a": 400.0, "b": 5.0})
    exp_n = pd.Series({"a": 200.0, "b": 5.0})
    med_em = pd.Series({"a": 70.0, "b": 10.0})
    med_dev = pd.Series({"a": 100.0, "b": 100.0})
    f = transfer.robust_factors(act_n, exp_n, med_em, med_dev, k_freq=20.0, k_sev=30.0)
    assert f["freq"]["a"] == pytest.approx(2.0, rel=0.05)
    assert f["sev"]["b"] == pytest.approx((5 / 35) * 0.1 + (30 / 35) * ((0.7 * 400 + 0.1 * 5) / 405), rel=1e-9)   # own ratio 0.1 with Z = 5/35, the rest the claim-weighted line ratio
    assert f["cost_factor"].to_numpy() == pytest.approx((f["freq"] * f["sev"]).to_numpy())


def test_calibrate_robust_scales_by_counts_and_medians_and_ignores_a_huge_claim():
    n = 2000
    idx = pd.RangeIndex(n)
    cost = pd.DataFrame({"a": np.full(n, 100.0), "b": np.full(n, 10.0)}, index=idx)
    cnt = pd.DataFrame({"a": np.full(n, 0.1), "b": np.full(n, 0.05)}, index=idx)
    gender = pd.Series(["M", "F"] * (n // 2), index=idx)
    n_by = pd.Series({"a": 400.0, "b": 200.0})   # frequency factor 2.0 for a, 2.0 for b
    y = pd.Series(np.full(n, 1e9 / n), index=idx)  # a single absurd total must not drive the level
    pred, fac = transfer.calibrate_robust(cost, cnt, n_by, y, pd.Series(1.0, index=idx), gender, pd.Series({"a": 70.0, "b": 70.0}), pd.Series({"a": 100.0, "b": 100.0}), cost, gender,
                                          k_freq=1.0, k_sev=1e-9, k_group=1e12)
    assert fac["peril"]["a"]["cost_factor"] == pytest.approx(2.0 * 0.7, rel=0.02)
    assert pred[0] == pytest.approx(100.0 * 2.0 * 0.7 + 10.0 * 2.0 * 0.7, rel=0.02)


def test_calibrate_robust_with_a_tweedie_part_keeps_the_robust_level_and_reorders_by_the_tweedie_relativity():
    n = 1000
    idx = pd.RangeIndex(n)
    cost = pd.DataFrame({"a": np.full(n, 100.0)}, index=idx)
    cnt = pd.DataFrame({"a": np.full(n, 0.1)}, index=idx)
    gender = pd.Series("M", index=idx)
    tw = pd.Series(np.where(idx < n // 2, 50.0, 150.0), index=idx)   # same total as the cost column, different relativities
    args = (cost, cnt, pd.Series({"a": 100.0}), pd.Series(100.0, index=idx), pd.Series(1.0, index=idx), gender, pd.Series({"a": 100.0}), pd.Series({"a": 100.0}), cost, gender)
    base, _ = transfer.calibrate_robust(*args, k_freq=1.0, k_sev=1.0, k_group=1e12)
    mixed, fac = transfer.calibrate_robust(*args, k_freq=1.0, k_sev=1.0, k_group=1e12, tw_train=tw * 3.0, tw_test=tw * 3.0, tw_share=0.5)
    assert mixed.sum() == pytest.approx(base.sum(), rel=1e-9)            # the level is the robust one
    assert mixed[0] < mixed[-1] and fac["tw_scale"] == pytest.approx(1.0 / 3.0)
