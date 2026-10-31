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


def _annual_claims():
    rows = []
    for y in (2021, 2022, 2023, 2024):  # 100 claims per accident year, each worth 10
        for i in range(100):
            lag_years = 0 if i < 80 else (1 if i < 95 else 2)  # 80% reported in the accident year, 15% next, 5% two years on
            rows.append({"loss_date": pd.Timestamp(y, 6, 15), "report_date": pd.Timestamp(y + lag_years, 3, 1) if lag_years else pd.Timestamp(y, 9, 1), "v": 10.0})
    return pd.DataFrame(rows)


def test_annual_triangle_counts_values_and_masks_unobserved():
    c = _annual_claims()
    tri = development.annual_triangle(c, None, valuation=pd.Timestamp("2024-12-31"))
    assert tri.loc[2021].tolist() == [80, 95, 100, 100]
    assert tri.loc[2023].tolist()[:2] == [80, 95] and tri.loc[2023].isna().tolist()[2:] == [True, True]
    assert tri.loc[2024].isna().tolist() == [False, True, True, True]
    vt = development.annual_triangle(c, "v", valuation=pd.Timestamp("2024-12-31"))
    assert vt.loc[2022, 24] == 950.0


def test_links_averages_and_tail():
    tri = development.annual_triangle(_annual_claims(), None, valuation=pd.Timestamp("2024-12-31"))
    ind = development.individual_links(tri)
    assert np.isclose(ind.loc[2021, "12-24"], 95 / 80) and np.isclose(ind.loc[2022, "24-36"], 100 / 95)
    avg = development.average_links(tri, latest=2)
    assert np.isclose(avg.loc["volume-weighted", "12-24"], 95 / 80) and avg.shape[0] == 3
    # geometric run-off of (link-1): 0.5, 0.25, 0.125 ... -> tail beyond is close to prod(1 + 0.5^k, k>=3)
    assert 1.2 < development.tail_exp_decay(np.array([2.0, 1.5, 1.25])) < 1.3
    assert development.tail_exp_decay(np.array([1.0, 1.00001])) == 1.0  # nothing to extrapolate
    assert development.tail_exp_decay(np.array([1.1, 1.2])) == 1.0  # increasing excess: no tail claimed
