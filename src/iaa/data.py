"""Loading helpers. Identifiers are always read as text and dates are parsed once, here."""
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, LINES

_POLICY_FILES = {line: f"policies_{line}.parquet" for line in LINES}
_TEST_FILES = {line: f"test_{line}.parquet" for line in LINES}


def _read(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir or DATA_DIR) / name)


def _parse_policy_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["year"] = df["period_start"].dt.year
    return df


def load_policies(line: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Training policy rows (2021-2024) for one line, with `year` and parsed dates."""
    return _parse_policy_dates(_read(_POLICY_FILES[line], data_dir))


def load_test(line: str, data_dir: Path | None = None) -> pd.DataFrame:
    """2025 rows to predict for one line (no target, no renewal_decision)."""
    return _parse_policy_dates(_read(_TEST_FILES[line], data_dir))


def load_claims(data_dir: Path | None = None) -> pd.DataFrame:
    df = _read("claims.parquet", data_dir)
    df["loss_date"] = pd.to_datetime(df["loss_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def load_customers(data_dir: Path | None = None) -> pd.DataFrame:
    df = _read("customers.parquet", data_dir)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    return df


def load_addresses(data_dir: Path | None = None) -> pd.DataFrame:
    return _read("addresses.parquet", data_dir)


def load_weather(data_dir: Path | None = None) -> pd.DataFrame:
    return _read("weather.parquet", data_dir)


def load_sample_submission(data_dir: Path | None = None) -> pd.DataFrame:
    return _read("sample_submission.parquet", data_dir)
