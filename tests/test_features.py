import pandas as pd

from iaa import features


def test_claim_history_uses_only_claims_known_before_the_row_year(claims):
    rows = pd.DataFrame({"row_id": ["R_2022", "R_2023", "R_2024"], "policy_id": ["P1", "P1", "P1"], "year": [2022, 2023, 2024]})
    # extra claim: lost in 2022 but only reported in 2023 -> unknown at the start of 2023
    late = claims.iloc[[0]].assign(claim_id="K9", loss_date=pd.Timestamp("2022-12-20"),
                                   report_date=pd.Timestamp("2023-02-01"), recoveries=0.0, incurred_to_date=900.0)
    out = features.add_claim_history(rows, pd.concat([claims, late])).set_index("row_id")
    # 2022 row: only the 2020 claim (K5, 300) is history
    assert out.loc["R_2022", "hist_n_claims"] == 1 and out.loc["R_2022", "hist_cost"] == 300.0
    # 2023 row: K1(200) K2(500) K5(300) known; K9 (reported Feb 2023) must NOT be
    assert out.loc["R_2023", "hist_n_claims"] == 3 and out.loc["R_2023", "hist_cost"] == 1000.0
    assert out.loc["R_2023", "hist_n_last_year"] == 2  # K1, K2 lost in 2022; K9 still unreported
    # 2024 row: K9 now known; "last year" is 2023 and nothing was lost in 2023
    assert out.loc["R_2024", "hist_n_claims"] == 4 and out.loc["R_2024", "hist_cost"] == 1900.0
    assert out.loc["R_2024", "hist_n_last_year"] == 0


def test_feature_columns_exclude_leakage_and_ids():
    df = pd.DataFrame({"row_id": [1], "renewal_decision": ["renewed"], "y": [1.0], "exposure": [1.0],
                       "exposure_calc": [1.0], "surname": ["x"], "driver_age": [30], "year": [2022],
                       "customer_service_contacts": [3]})
    assert features.feature_columns(df) == ["driver_age"]
    assert "surname" in features.feature_columns(df, drop_noise=False)


def test_outcome_generated_column_is_never_a_feature_even_when_noise_is_kept():
    df = pd.DataFrame({"customer_service_contacts": [3], "driver_age": [30]})
    assert "customer_service_contacts" not in features.feature_columns(df, drop_noise=False)


def test_every_target_like_column_is_excluded_from_features():
    df = pd.DataFrame({"y": [1.0], "y_true": [1.0], "y_all": [1.0], "n_claims": [1], "pred": [1.0], "driver_age": [30]})
    assert features.feature_columns(df) == ["driver_age"]


def test_network_history_counts_other_lines_and_address_using_only_claims_known_before_the_year():
    d = pd.Timestamp
    rows = pd.DataFrame({"row_id": ["R22", "R23"], "customer_id": ["C1", "C1"], "address_id": ["A1", "A1"], "year": [2022, 2023]})
    all_rows = pd.DataFrame({"policy_id": ["M1", "B1"], "customer_id": ["C1", "C1"], "address_id": ["A1", "A1"],
                             "period_start": [d("2021-01-01")] * 2, "period_end": [d("2023-12-31")] * 2})
    claims = pd.DataFrame({
        "claim_id": ["K1", "K2", "K3"], "policy_id": ["M1", "B1", "B1"], "line": ["motor", "buildings", "buildings"], "status": ["closed"] * 3,
        "loss_date": [d("2021-05-01"), d("2022-03-01"), d("2022-12-20")], "report_date": [d("2021-05-05"), d("2022-03-02"), d("2023-02-01")]})
    out = features.add_network_history(rows, claims, all_rows).set_index("row_id")
    assert out.loc["R22", "cust_n_all"] == 1 and out.loc["R22", "cust_n_motor"] == 1 and out.loc["R22", "cust_n_buildings"] == 0
    # 2023 row: K1 (motor) and K2 (buildings) known; K3 was only reported in Feb 2023 so it must not count
    assert out.loc["R23", "cust_n_all"] == 2 and out.loc["R23", "cust_n_buildings"] == 1
    assert out.loc["R23", "addr_n_claims"] == 2 and out.loc["R23", "addr_n_last_year"] == 1
