import pandas as pd

from iaa import target


def test_recompute_exposure_fixes_corrupted_values(policies):
    e = target.recompute_exposure(policies)
    assert e.round(4).tolist() == [round(181 / 365, 4), round(184 / 365, 4), 1.0]


def test_exposure_mismatch_flags_only_bad_rows(policies):
    assert target.exposure_mismatch(policies).tolist() == [False, True, True]


def test_net_cost_is_incurred_minus_recoveries_and_zero_if_repudiated(claims):
    assert target.claim_net_cost(claims).tolist() == [200.0, 500.0, 2000.0, 0.0, 300.0]


def test_claims_matched_by_policy_and_loss_date(policies, claims):
    m = target.match_claims_to_rows(claims, policies).set_index("claim_id")["row_id"]
    assert m["K1"] == "R1" and m["K2"] == "R2" and m["K3"] == "R3"
    assert pd.isna(m["K5"])  # loss in 2020 matches no row


def test_row_target_sums_net_cost_and_excludes_repudiated(policies, claims):
    rows, report = target.build_row_target(policies, claims, "motor")
    y = rows.set_index("row_id")["y"]
    assert y.to_dict() == {"R1": 200.0, "R2": 500.0, "R3": 2000.0}
    assert rows.set_index("row_id")["n_claims"].to_dict() == {"R1": 1, "R2": 1, "R3": 1}
    assert report["claims_unmatched"] == 1 and report["claims_repudiated"] == 1
    assert rows["exposure_calc"].max() <= 1.0
