"""Competition constants and paths. Everything that encodes a rule of the competition lives here."""
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("IAA_DATA_DIR", REPO_ROOT / "data" / "raw" / "public"))
ARTIFACTS_DIR = Path(os.environ.get("IAA_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
# Part 2 has two portfolios: the developed market (identical to Part 1) and the emerging market (2023-24 train, 2025 test).
# IAA_MARKET selects the cleaning bands and the validation folds; IAA_DATA_DIR selects the tables.
MARKET = os.environ.get("IAA_MARKET", "developed")
assert MARKET in ("developed", "emerging"), f"IAA_MARKET must be developed or emerging, got {MARKET}"

COMPETITION = "iaa-ai-assisted-hackathon-2026-part-1"
PART2_COMPETITION = "iaa-ai-assisted-hackathon-2026-part-2"   # emerging-market test rows only; closes 2026-10-31 03:59 UTC

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
