import numpy as np
import pandas as pd
import pytest

from iaa import p2_features as pf


def _frame():
    return pd.DataFrame({"customer_id": ["A", "A", "B", "C"], "year": [2024, 2025, 2024, 2024], "written_premium": [100.0, 120.0, 300.0, 200.0], "sum_insured": [10.0, 12.0, 30.0, 20.0],
                         "exposure_calc": [1.0, 1.0, 0.5, 1.0], "age_band": ["17-24", "65+", "35-49", "25-34"], "ncb_years": [3, 5, 1, 0], "years_licensed": [2, 10, 0, 4], "bedrooms": [2, 3, 0, 1]})


def test_mktnorm_divides_by_the_median_of_the_year_and_drops_the_raw_columns():
    out = pf.add_features(_frame(), "motor", ("mktnorm",))
    assert "written_premium" not in out and "sum_insured" not in out
    m24 = out[out.year == 2024]
    assert m24["written_premium_rel"].tolist() == pytest.approx([0.5, 1.5, 1.0])   # median of 100, 300, 200 is 200
    assert out[out.year == 2025]["written_premium_rel"].iloc[0] == pytest.approx(1.0)   # a single row is its own median


def test_pct_ranks_within_year():
    out = pf.add_features(_frame(), "motor", ("pct",))
    assert out[out.year == 2024]["written_premium_pct"].tolist() == pytest.approx([1 / 3, 1.0, 2 / 3])


def test_lines_held_counts_distinct_lines_per_customer_year():
    fr = {"motor": pd.DataFrame({"customer_id": ["A", "B"], "year": [2024, 2024]}), "buildings": pd.DataFrame({"customer_id": ["A", "A"], "year": [2024, 2024]}),
          "contents": pd.DataFrame({"customer_id": ["A"], "year": [2025]})}
    h = pf.lines_held(fr).set_index(["customer_id", "year"])
    assert h.loc[("A", 2024), "n_lines_held"] == 2 and h.loc[("B", 2024), "n_lines_held"] == 1 and h.loc[("A", 2025), "n_lines_held"] == 1
    out = pf.add_features(_frame(), "motor", ("xline",), held=pf.lines_held(fr))
    assert out["n_lines_held"].tolist() == [2, 1, 1, 0]   # customer C is in no table of the toy market: 0


def test_ratio_features_motor_and_property():
    m = pf.add_features(_frame(), "motor", ("ratios",))
    assert m["ncb_over_licence"].tolist() == pytest.approx([1.5, 0.5, 1.0, 0.0]) and m["ncb_exceeds_licence"].tolist() == [1, 0, 1, 0]
    assert m["age_ordinal"].tolist() == [0, 4, 2, 1]
    b = pf.add_features(_frame(), "buildings", ("ratios",))
    assert b["premium_rate"].tolist() == pytest.approx([10.0, 10.0, 10.0, 10.0]) and b["sum_insured_per_bedroom"].tolist() == pytest.approx([5.0, 4.0, 30.0, 20.0])


def test_unknown_set_and_missing_held_table_fail():
    with pytest.raises(AssertionError):
        pf.add_features(_frame(), "motor", ("bogus",))
    with pytest.raises(AssertionError):
        pf.add_features(_frame(), "motor", ("xline",))
