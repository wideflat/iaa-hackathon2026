import pandas as pd
import pytest

from iaa import validation


def _rows():
    years = [2021, 2022, 2023, 2024]
    return pd.DataFrame({
        "year": years,
        "period_start": pd.to_datetime([f"{y}-01-01" for y in years]),
        "period_end": pd.to_datetime([f"{y}-12-31" for y in years]),
    })


def test_folds_are_strictly_forward_in_time():
    rows = _rows()
    for fold in validation.FOLDS.values():
        train, valid = validation.split(rows, fold)
        assert train["year"].max() < valid["year"].min()
        assert valid["year"].unique().tolist() == [fold.valid_year]


def test_leak_guard_rejects_overlap():
    rows = _rows()
    bad = validation.Fold("bad", (2021, 2023), 2022, 1.0, True)
    with pytest.raises(AssertionError):
        validation.split(rows, bad)
