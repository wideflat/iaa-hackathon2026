import pandas as pd
import pytest

from iaa.deflate import deflate_frames


def test_deflate_divides_by_the_year_median_per_line_and_skips_absent_columns():
    f = {"buildings": pd.DataFrame({"year": [2021, 2021, 2021, 2022, 2022], "sum_insured": [100.0, 200.0, 300.0, 206.0, 412.0], "written_premium": [1.0, 2.0, 3.0, 4.0, 5.0]}),
         "motor": pd.DataFrame({"year": [2021, 2022], "written_premium": [10.0, 20.0]})}
    out = deflate_frames(f, ["sum_insured", "written_premium"])
    assert out["buildings"]["sum_insured"].tolist() == pytest.approx([0.5, 1.0, 1.5, 206 / 309, 412 / 309])
    assert out["motor"]["written_premium"].tolist() == [1.0, 1.0] and "sum_insured" not in out["motor"]
    assert f["buildings"]["sum_insured"].tolist()[0] == 100.0   # input untouched
