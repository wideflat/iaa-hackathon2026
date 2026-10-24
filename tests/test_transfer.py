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
