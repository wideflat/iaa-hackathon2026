import numpy as np
import pandas as pd
import pytest

from iaa import submission


def _sample():
    return pd.DataFrame({"row_id": ["a", "b", "c"], "prediction": 0.0})


def test_write_submission_orders_floors_and_validates(tmp_path):
    pred = pd.DataFrame({"row_id": ["c", "a", "b"], "prediction": [5.0, 0.0, 2.0]})
    sub = submission.write_submission(pred, _sample(), tmp_path / "s.csv")
    assert sub["row_id"].tolist() == ["a", "b", "c"]
    assert sub["prediction"].min() >= submission.MIN_PREDICTION
    assert pd.read_csv(tmp_path / "s.csv").columns.tolist() == ["row_id", "prediction"]


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"row_id": ["a", "b"], "prediction": [1.0, 1.0]}),                   # missing row
    pd.DataFrame({"row_id": ["a", "a", "c"], "prediction": [1.0, 1.0, 1.0]}),         # duplicate
    pd.DataFrame({"row_id": ["a", "b", "c"], "prediction": [1.0, np.nan, 1.0]}),      # NaN
    pd.DataFrame({"row_id": ["a", "b", "c"], "prediction": [1.0, -1.0, 1.0]}),        # negative
])
def test_validate_rejects_bad_files(bad):
    with pytest.raises(AssertionError):
        submission.validate_submission(bad, _sample())
