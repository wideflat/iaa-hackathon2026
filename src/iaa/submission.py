"""Submission writing with hard validation: a malformed file costs a daily submission."""
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PREDICTION = 0.01  # strictly positive: Tweedie deviance explodes at pred ~ 0 when a claim occurs


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise AssertionError unless `sub` is a valid submission against the sample file."""
    assert list(sub.columns) == ["row_id", "prediction"], f"columns must be row_id,prediction, got {list(sub.columns)}"
    assert sub["row_id"].is_unique, "duplicate row_id"
    assert set(sub["row_id"]) == set(sample["row_id"]), "row_id set differs from sample_submission"
    assert len(sub) == len(sample), "row count differs from sample_submission"
    assert np.isfinite(sub["prediction"]).all(), "non-finite predictions"
    assert (sub["prediction"] >= 0).all(), "negative predictions"


def write_submission(pred: pd.DataFrame, sample: pd.DataFrame, path: Path, floor: float = MIN_PREDICTION) -> pd.DataFrame:
    """pred has columns row_id and prediction. Floors, orders like the sample, validates, writes CSV."""
    sub = pred[["row_id", "prediction"]].copy()
    sub["prediction"] = sub["prediction"].clip(lower=floor)
    sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
    validate_submission(sub, sample)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub
