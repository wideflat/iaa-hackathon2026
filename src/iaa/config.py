"""Competition constants and paths. Everything that encodes a rule of the competition lives here."""
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("IAA_DATA_DIR", REPO_ROOT / "data" / "raw" / "public"))
ARTIFACTS_DIR = Path(os.environ.get("IAA_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))

COMPETITION = "iaa-ai-assisted-hackathon-2026-part-1"

LINES = ("motor", "buildings", "contents")

# Claims in claims.parquet are valued at this date; the test target is *ultimate* cost.
VALUATION_DATE = pd.Timestamp("2024-12-31")
TEST_YEAR = 2025
TRAIN_YEARS = (2021, 2022, 2023, 2024)

# Official metric: sklearn.metrics.d2_tweedie_score(power=1.5), pooled over all test rows.
TWEEDIE_POWER = 1.5

# Exposure convention observed on clean rows: days in period / 365, capped at 1.
DAYS_PER_YEAR = 365

SEED = 2026
