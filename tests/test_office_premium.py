import pandas as pd
import pytest

from iaa import office_premium as op

EXP = pd.DataFrame({"item": ["fixed_expense_per_policy", "variable_expense", "commission_direct", "commission_comparison_site", "commission_broker",
                             "premium_tax", "cat_reinsurance", "target_profit_margin"],
                    "emerging": [85.0, 0.15, 0.0, float("nan"), 0.20, 0.08, 0.12, 0.08]})


def _build(line, channel, exposure=1.0, risk=300.0, cs=None):
    a = op.assumptions(EXP, "emerging", cs)
    return op.office_premium(pd.Series([risk]), pd.Series([line]), pd.Series([channel]), pd.Series([exposure]), a)


def test_components_sum_to_office_premium():
    t = _build("buildings", "broker")
    parts = t[["risk_premium", "cat_reinsurance", "fixed_expense", "variable_expense", "commission", "premium_tax", "profit_margin"]].sum(axis=1)
    assert parts.iloc[0] == pytest.approx(t["office_premium"].iloc[0], rel=1e-12)


def test_motor_has_no_cat_reinsurance_and_property_has():
    assert _build("motor", "direct")["cat_reinsurance"].iloc[0] == 0.0
    assert _build("contents", "direct")["cat_reinsurance"].iloc[0] == pytest.approx(300.0 * 0.12)


def test_broker_example_by_hand():
    # (300 + 36 + 85) / (1 - (0.15 + 0.20 + 0.08 + 0.08)) = 421 / 0.49
    assert _build("buildings", "broker")["office_premium"].iloc[0] == pytest.approx(421.0 / 0.49)


def test_fixed_expense_is_pro_rated_by_exposure():
    assert _build("motor", "direct", exposure=0.5)["fixed_expense"].iloc[0] == pytest.approx(42.5)


def test_blank_comparison_site_commission_is_an_error_until_assumed():
    with pytest.raises(AssertionError):
        _build("motor", "comparison_site")
    assert _build("motor", "comparison_site", cs=0.10)["commission"].iloc[0] > 0
