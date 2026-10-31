import numpy as np
import pandas as pd

from iaa import events


def _claims():
    rng = np.random.default_rng(0)
    days = pd.date_range("2021-01-01", "2023-12-31")
    base = pd.DataFrame({"loss_date": rng.choice(days, 30 * 36), "region": "R1", "peril": "storm"})   # about 30 claims a month
    surge = pd.DataFrame({"loss_date": pd.to_datetime("2023-11-01") + pd.to_timedelta(rng.integers(0, 30, 120), unit="D"), "region": "R1", "peril": "storm"})
    other = pd.DataFrame({"loss_date": rng.choice(days, 30 * 36), "region": "R2", "peril": "storm"})
    c = pd.concat([base, surge, other], ignore_index=True)
    return c.assign(claim_id=[f"CL{i:08d}" for i in range(len(c))], row_id="x", net_cost=1.0)


def test_surge_month_is_flagged_and_thinned_to_baseline_and_other_cells_are_untouched():
    c = _claims()
    cells = events.event_cells(c)
    assert len(cells) == 1 and cells.iloc[0]["region"] == "R1" and str(cells.iloc[0]["month"]) == "2023-11"
    thinned, _ = events.thin_event_claims(c)
    nov = thinned[(thinned.region == "R1") & (thinned.loss_date.dt.to_period("M").astype(str) == "2023-11")]
    assert len(nov) == cells.iloc[0]["keep"] and len(thinned) == len(c) - (cells.iloc[0]["n"] - cells.iloc[0]["keep"])
    assert (thinned.region == "R2").sum() == (c.region == "R2").sum()


def test_thinning_is_deterministic_and_a_quiet_history_is_unchanged():
    c = _claims()
    a, _ = events.thin_event_claims(c)
    b, _ = events.thin_event_claims(c.sample(frac=1.0, random_state=1))
    assert set(a.claim_id) == set(b.claim_id)
    quiet = c[~((c.region == "R1") & (c.loss_date >= "2023-11-01") & (c.loss_date < "2023-12-01") & (c.index >= 30 * 36))]
    kept, cells = events.thin_event_claims(quiet)
    assert cells.empty and len(kept) == len(quiet)
