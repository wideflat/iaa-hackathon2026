import numpy as np
import pandas as pd

from iaa import drift


def test_shift_flags_a_column_whose_distribution_changes_and_ignores_a_stable_one():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"stable": rng.normal(size=5000), "leaky": rng.poisson(2.0, 5000), "cat": rng.choice(list("ab"), 5000)})
    new = pd.DataFrame({"stable": rng.normal(size=5000), "leaky": rng.poisson(0.3, 5000), "cat": rng.choice(list("ab"), 5000)})
    t = drift.feature_shift_table(ref, new, ["stable", "leaky", "cat"]).set_index("feature")["shift"]
    assert t["leaky"] > 0.4 and t["stable"] < 0.05 and t["cat"] < 0.05


def test_target_signal_ranks_outcome_driven_feature_first():
    rng = np.random.default_rng(1)
    n = rng.poisson(0.1, 20000)
    rows = pd.DataFrame({"n_claims": n, "contacts": n * 3 + rng.poisson(0.4, 20000), "age": rng.integers(18, 80, 20000)})
    s = drift.target_signal(rows, ["contacts", "age"])
    assert s.index[0] == "contacts" and s["contacts"] > 0.5 and s["age"] < 0.05
