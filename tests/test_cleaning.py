import numpy as np
import pandas as pd
import pytest

from iaa import cleaning


def _claims():
    base = dict(policy_id="M1", line="motor", peril="own_damage", status="closed", paid_to_date=100.0, incurred_to_date=100.0, recoveries=0.0, repairer_id="R1", third_party_id=None)
    rows = [
        dict(base, claim_id="CL00000001", loss_date="2022-03-01", report_date="2022-03-08"),
        dict(base, claim_id="CL000000002", loss_date="2022-03-01", report_date="2022-03-08"),            # exact copy under a new, longer id
        dict(base, claim_id="CL00000003", loss_date="2022-03-01", report_date="2022-03-08", incurred_to_date=250.0, paid_to_date=250.0),   # same day, different amount: a real second claim
        dict(base, claim_id="CL00000004", loss_date="2021-02-01", report_date="2021-02-08"),
        dict(base, claim_id="CL00000005", loss_date="2021-03-01", report_date="2021-03-08"),
        dict(base, claim_id="CL00000006", loss_date="2020-12-01", report_date="2021-06-20"),             # placeholder loss date
    ]
    df = pd.DataFrame(rows)
    df[["loss_date", "report_date"]] = df[["loss_date", "report_date"]].apply(pd.to_datetime)
    return df


def test_duplicates_are_dropped_keeping_the_original_id_and_real_repeats_stay():
    out = cleaning.drop_duplicate_claims(_claims(), expected=(1, 1))
    assert len(out) == 5 and "CL000000002" not in set(out["claim_id"]) and "CL00000003" in set(out["claim_id"])


def test_duplicate_count_outside_the_expected_band_fails_fast():
    with pytest.raises(AssertionError):
        cleaning.drop_duplicate_claims(_claims(), expected=(100, 200))


def test_placeholder_loss_date_is_imputed_from_report_date_and_peril_lag_and_floored_at_policy_start():
    first_start = pd.Series({"M1": pd.Timestamp("2021-01-01")})
    out = cleaning.fix_placeholder_loss_dates(cleaning.drop_duplicate_claims(_claims(), expected=(1, 1)), first_start, expected=(1, 1))
    fixed = out.set_index("claim_id").loc["CL00000006", "loss_date"]
    assert fixed == pd.Timestamp("2021-06-13")   # report 2021-06-20 minus the median lag of 7 days
    late_start = pd.Series({"M1": pd.Timestamp("2021-06-17")})
    out2 = cleaning.fix_placeholder_loss_dates(cleaning.drop_duplicate_claims(_claims(), expected=(1, 1)), late_start, expected=(1, 1))
    assert out2.set_index("claim_id").loc["CL00000006", "loss_date"] == pd.Timestamp("2021-06-17")
    assert (out["loss_date"] != cleaning.PLACEHOLDER_LOSS_DATE).all()


def test_policy_unit_error_and_construction_codes_are_repaired():
    n = 1000
    si = np.full(n, 200_000.0)
    si[:3] = 200_000_000.0
    df = pd.DataFrame({"sum_insured": si, "construction_type": ["Brick", "BRICK", "T", "t", "W", "stone"] * 166 + ["w"] * 4})
    out = cleaning.clean_policies(df, "buildings")
    assert out["sum_insured"].max() == 200_000.0
    assert set(out["construction_type"]) == {"brick", "w", "stone"}
    assert df["sum_insured"].max() == 200_000_000.0   # the input is not mutated


def test_too_many_unit_errors_fail_fast():
    df = pd.DataFrame({"sum_insured": [1.0e5] * 90 + [1.0e8] * 10})
    with pytest.raises(AssertionError):
        cleaning.clean_policies(df, "buildings")
