import numpy as np
import pandas as pd
import pytest

from iaa.models.freqsev import FreqSev


def _data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], n)
    expo = rng.uniform(0.2, 1.0, n)
    rows = pd.DataFrame({"row_id": [f"R{i}" for i in range(n)], "x": x, "g": g, "exposure_calc": expo})
    f_a = np.exp(-2.2 + 0.7 * x) * expo           # peril A frequency depends on x
    f_b = np.exp(-3.0 + (g == "c")) * expo         # peril B frequency depends on g
    cl = []
    for p, f, scale in (("A", f_a, 800.0), ("B", f_b, 3000.0)):
        k = rng.poisson(f)
        for i in np.nonzero(k)[0]:
            for _ in range(k[i]):
                cl.append({"row_id": rows.row_id[i], "peril": p, "net_cost": rng.gamma(2.0, scale / 2.0), "status": "closed"})
    claims = pd.DataFrame(cl)
    y = claims.groupby("row_id").net_cost.sum().reindex(rows.row_id).fillna(0.0).to_numpy()
    rows["y"], rows["n_claims"] = y, 0
    return rows, claims


@pytest.mark.parametrize("kind", ["glm", "gbm"])
def test_freqsev_balances_total_ranks_risk_and_is_nonnegative(kind):
    rows, claims = _data()
    m = FreqSev(claims, kind=kind, freq_rounds=60, sev_rounds=40).fit(rows)
    p = m.predict(rows)
    assert (p >= 0).all() and np.isclose(p.sum(), rows.y.sum(), rtol=1e-9)
    hi, lo = rows.assign(x=2.0, g="c", exposure_calc=1.0), rows.assign(x=-2.0, g="a", exposure_calc=1.0)
    assert m.predict(hi).mean() > m.predict(lo).mean()


def test_peril_decomposition_adds_up_and_excess_load_is_nonnegative():
    rows, claims = _data()
    m = FreqSev(claims, kind="glm", rebalance=False).fit(rows)
    by = m.expected_by_peril(rows)
    assert set(by.columns) == {"A", "B"} and np.allclose(by.sum(axis=1), m.predict(rows))
    assert all(v >= 0 for v in m.excess_.values())
