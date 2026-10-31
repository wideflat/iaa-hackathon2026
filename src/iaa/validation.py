"""Fixed, time-based validation folds. The test set is a future year (2025), so we validate forward.

Fold A  train 2021-22 -> validate 2023   mature claims (3% open): the primary, trusted fold.
Fold B  train 2021-23 -> validate 2024   claims immature (20% open): sanity check only until the
                                         2024 target is developed to ultimate; lower weight.

Rows are NOT grouped by policy or customer: ~85% of 2025 test policies also appear in training,
and a time split reproduces exactly that. Random K-fold would leak the same policy's future.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_years: tuple
    valid_year: int
    weight: float
    mature: bool


FOLDS = {
    "A": Fold("A_valid2023", (2021, 2022), 2023, weight=1.0, mature=True),
    "B": Fold("B_valid2024", (2021, 2022, 2023), 2024, weight=0.5, mature=False),
    # Emerging market (2023-24 only): train 2023 -> validate 2024; the claim-history inputs of the 2023 rows are empty by construction
    "E": Fold("E_valid2024", (2023,), 2024, weight=1.0, mature=False),
}


def split(rows: pd.DataFrame, fold: Fold):
    """Return (train, valid) views of `rows` (needs a `year` column)."""
    train = rows[rows["year"].isin(fold.train_years)]
    valid = rows[rows["year"] == fold.valid_year]
    assert_no_leak(train, valid, fold)
    return train, valid


def assert_no_leak(train: pd.DataFrame, valid: pd.DataFrame, fold: Fold) -> None:
    """Guard: training must lie entirely before validation, and neither may be empty."""
    if train.empty or valid.empty:
        raise AssertionError(f"{fold.name}: empty train or valid set")
    if train["period_end"].max() >= valid["period_start"].min():
        raise AssertionError(f"{fold.name}: training periods overlap the validation period")
    if fold.valid_year <= max(fold.train_years):
        raise AssertionError(f"{fold.name}: validation year must be after all training years")
