import numpy as np
import pandas as pd

from iaa import development


def _claims():
    # 100 claims per accident quarter 2023Q1-2023Q4; 70 reported in-quarter, 20 next quarter, 10 two quarters later
    rows = []
    for q in pd.period_range("2023Q1", "2023Q4", freq="Q"):
        start = q.start_time
        for i in range(100):
            lag_q = 0 if i < 70 else (1 if i < 90 else 2)
            rep = (q + lag_q).start_time + pd.Timedelta(days=5)
            rows.append({"loss_date": start + pd.Timedelta(days=10), "report_date": rep})
    return pd.DataFrame(rows)


def test_triangle_is_cumulative_and_masks_unobserved_cells():
    tri = development.count_triangle(_claims(), max_age=3, last=pd.Period("2023Q4"))
    assert tri.loc[pd.Period("2023Q1")].tolist()[:3] == [70, 90, 100]
    assert tri.loc[pd.Period("2023Q4")].isna().tolist() == [False, True, True, True]  # only age 0 observed


def test_link_ratios_and_factors_to_ultimate():
    tri = development.count_triangle(_claims(), max_age=3, last=pd.Period("2024Q4"))
    links = development.link_ratios(tri)
    assert np.allclose(links[:2], [90 / 70, 100 / 90])
    ult = development.factors_to_ultimate(links)
    assert np.isclose(ult[0], 100 / 70) and np.isclose(ult[2], 1.0)
