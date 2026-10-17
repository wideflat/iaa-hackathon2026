import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def policies():
    """Three rows: two for policy P1 (mid-term change in 2022), one full-year for P2."""
    return pd.DataFrame({
        "row_id": ["R1", "R2", "R3"],
        "policy_id": ["P1", "P1", "P2"],
        "customer_id": ["C1", "C1", "C2"],
        "period_start": pd.to_datetime(["2022-01-01", "2022-07-01", "2022-01-01"]),
        "period_end": pd.to_datetime(["2022-06-30", "2022-12-31", "2022-12-31"]),
        "exposure": [181 / 365, 1.0139, 365.0],  # last two are the corruptions seen in the real data
        "year": [2022, 2022, 2022],
    })


@pytest.fixture
def claims():
    d = pd.to_datetime
    return pd.DataFrame({
        "claim_id": ["K1", "K2", "K3", "K4", "K5"],
        "policy_id": ["P1", "P1", "P2", "P2", "P1"],
        "line": ["motor"] * 5,
        "loss_date": d(["2022-03-01", "2022-08-01", "2022-05-05", "2022-06-06", "2020-05-01"]),
        "report_date": d(["2022-03-05", "2022-08-09", "2022-05-06", "2022-06-10", "2020-05-02"]),
        "status": ["closed", "open", "closed", "repudiated", "closed"],
        "incurred_to_date": [1000.0, 500.0, 2000.0, 700.0, 300.0],
        "paid_to_date": [1000.0, 100.0, 2000.0, 0.0, 300.0],
        "recoveries": [800.0, 0.0, 0.0, 0.0, 0.0],
    })
