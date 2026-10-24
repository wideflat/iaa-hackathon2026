"""Data contracts for the emerging market and the harmonisation layer (small synthetic fixtures; the real tables are never read)."""
import numpy as np
import pandas as pd
import pytest

from iaa import cleaning, features, market

KEY_BANDS = {"developed": ((550, 650), (200, 300)), "emerging": ((0, 0), (0, 0))}


def _claim(i, loss="2023-03-01", **kw):
    base = dict(claim_id=f"CL{i:08d}", policy_id="M1", line="motor", peril="own_damage", loss_date=pd.Timestamp(loss), report_date=pd.Timestamp(loss) + pd.Timedelta(days=7), status="closed",
                paid_to_date=100.0, incurred_to_date=100.0, recoveries=0.0, repairer_id="R1", third_party_id=None)
    return {**base, **kw}


def test_emerging_market_expects_no_duplicates_and_no_placeholder_dates():
    assert cleaning._BANDS["emerging"] == {"duplicates": (0, 0), "placeholders": (0, 0)}
    clean = pd.DataFrame([_claim(1), _claim(2, loss="2023-05-01", incurred_to_date=300.0, paid_to_date=300.0)])
    first_start = pd.Series({"M1": pd.Timestamp("2023-01-01")})
    out = cleaning.clean_claims(clean, first_start, market="emerging")
    assert len(out) == 2


def test_a_duplicate_claim_in_the_emerging_market_fails_fast():
    dup = pd.DataFrame([_claim(1), _claim(2, claim_id="CL000000002")])
    with pytest.raises(AssertionError):
        cleaning.clean_claims(dup, pd.Series({"M1": pd.Timestamp("2023-01-01")}), market="emerging")


def test_a_placeholder_loss_date_in_the_emerging_market_fails_fast():
    ph = pd.DataFrame([_claim(1), _claim(2, loss="2020-12-01")])
    with pytest.raises(AssertionError):
        cleaning.clean_claims(ph, pd.Series({"M1": pd.Timestamp("2023-01-01")}), market="emerging")


def test_emerging_construction_codes_are_mapped_and_unknown_codes_fail():
    df = pd.DataFrame({"construction_type": ["BRK", "STN", "WOD", "MIX", None], "sum_insured": [200.0, 210.0, 220.0, 230.0, 240.0]})
    out = cleaning.clean_policies(df, "buildings", market="emerging")
    assert out["construction_type"].tolist()[:4] == ["brick", "stone", "w", "non_standard"] and pd.isna(out["construction_type"].iloc[4])
    with pytest.raises(AssertionError):
        cleaning.clean_policies(df.assign(construction_type=["BRK", "XXX", "WOD", "MIX", None]), "buildings", market="emerging")


def test_sum_insured_unit_error_is_repaired_in_the_emerging_market_too():
    si = np.full(2000, 200_000.0)
    si[:3] = 200_000_000.0   # 0.15% of rows, x1000
    out = cleaning.clean_policies(pd.DataFrame({"sum_insured": si}), "buildings", market="emerging")
    assert out["sum_insured"].max() == pytest.approx(200_000.0)


def _policy_frame(market_name, line="motor"):
    base = {"row_id": ["R1", "R2"], "occupation": ["Self-employed Builder ", "nurse"], "region": ["R01" if market_name == "developed" else "E01"] * 2, "customer_service_contacts": [1, 2]}
    if market_name == "developed":
        base |= {"driver_age": [24.0, 66.0], "policyholder_age": [24.0, 66.0], "annual_mileage": [8000.0, 9000.0], "credit_band": ["A", "B"], "date_of_birth": pd.to_datetime(["2000-01-01", "1958-01-01"]), "customer_age": [24.0, 66.0]}
    else:
        base |= {"driver_age_band": ["17-24", "65+"], "policyholder_age_band": ["17-24", "65+"], "gender": ["M", "F"]}
    return pd.DataFrame(base)


def test_to_common_gives_both_markets_the_same_columns_and_the_same_age_bands():
    d = market.to_common(_policy_frame("developed"), "motor", "developed")
    e = market.to_common(_policy_frame("emerging"), "motor", "emerging")
    assert set(d.columns) == set(e.columns)
    assert d["age_band"].tolist() == e["age_band"].tolist() == ["17-24", "65+"]
    assert d["occupation"].tolist() == e["occupation"].tolist()[:1] + ["nurse"] and d["self_employed"].tolist() == [True, False]
    assert not set(market.MARKET_ONLY) & set(d.columns)


def test_customer_service_contacts_is_never_a_model_input_in_either_market():
    for m in ("developed", "emerging"):
        cols = features.feature_columns(market.to_common(_policy_frame(m), "motor", m).assign(customer_service_contacts=[1, 2]))
        assert "customer_service_contacts" not in cols
