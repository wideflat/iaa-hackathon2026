import numpy as np
import pandas as pd
import pytest

from iaa.yoy import add_yoy


def test_year_on_year_change_uses_the_previous_year_row_of_the_same_policy_only():
    df = pd.DataFrame({"row_id": ["a21", "a22", "a23", "b22"], "policy_id": ["A", "A", "A", "B"], "year": [2021, 2022, 2023, 2022],
                       "period_start": pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01", "2022-01-01"]),
                       "ncb_years": [5, 6, 4, 3], "written_premium": [100.0, 110.0, 99.0, 50.0]})
    out = add_yoy({"motor": df})["motor"].set_index("row_id")
    assert np.isnan(out.loc["a21", "yoy_d_ncb_years"]) and np.isnan(out.loc["b22", "yoy_d_ncb_years"])
    assert out.loc["a22", "yoy_d_ncb_years"] == 1 and out.loc["a23", "yoy_d_ncb_years"] == -2
    assert out.loc["a23", "yoy_r_written_premium"] == pytest.approx(np.log(99.0 / 110.0), abs=1e-6)
    assert "yoy_d_annual_mileage" not in out.columns
