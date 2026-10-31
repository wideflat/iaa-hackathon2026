import pandas as pd
import pytest

from iaa.crossline import add_crossline


def _frames():
    motor = pd.DataFrame({"customer_id": ["c1", "c1", "c2"], "year": [2023, 2023, 2023], "written_premium": [500.0, 700.0, 400.0], "vehicle_value_group": [3, 5, 2], "row_id": ["m1", "m2", "m3"]})
    buildings = pd.DataFrame({"customer_id": ["c1"], "year": [2023], "written_premium": [300.0], "sum_insured": [250000.0], "row_id": ["b1"]})
    contents = pd.DataFrame({"customer_id": ["c3"], "year": [2023], "written_premium": [100.0], "sum_insured": [40000.0], "row_id": ["k1"]})
    return {"motor": motor, "buildings": buildings, "contents": contents}


def test_crossline_adds_other_line_columns_without_changing_rows():
    out = add_crossline(_frames())
    assert [len(out[l]) for l in ("motor", "buildings", "contents")] == [3, 1, 1]
    m = out["motor"].set_index("row_id")
    assert m.loc["m1", "x_prem_buildings"] == pytest.approx(300.0) and m.loc["m1", "x_si_buildings"] == pytest.approx(250000.0)
    assert m.loc["m1", "x_prem_motor"] == pytest.approx(700.0) and m.loc["m1", "x_vvg_motor"] == 5   # larger of the two motor policies of c1
    assert m.loc["m1", "x_lines_held"] == 2 and m.loc["m3", "x_lines_held"] == 1
    assert pd.isna(m.loc["m3", "x_prem_buildings"])
    assert out["contents"].loc[0, "x_lines_held"] == 1


def test_crossline_inputs_are_not_modified():
    f = _frames()
    cols = list(f["motor"].columns)
    add_crossline(f)
    assert list(f["motor"].columns) == cols
