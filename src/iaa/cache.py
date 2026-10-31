"""Cache of the per-line feature frames so experiments start in seconds, not minutes.

The cache key is a hash of the source files that define the frames (data, target, features): changing any of them
invalidates the cache automatically. Frames live in artifacts/ (gitignored).
"""
import hashlib
from pathlib import Path

import pandas as pd

from . import cleaning, data, features
from .config import ARTIFACTS_DIR, DATA_DIR, LINES, MARKET, REPO_ROOT

_KEEP_FLOAT64 = {"y", "y_all", "y_true", "exposure", "exposure_calc"}
_SOURCES = ("src/iaa/data.py", "src/iaa/cleaning.py", "src/iaa/target.py", "src/iaa/features.py")


def cache_key(market: str = MARKET, data_dir=DATA_DIR) -> str:
    h = hashlib.md5()
    for p in _SOURCES:
        h.update((REPO_ROOT / p).read_bytes())
    h.update(",".join(sorted(cleaning.active_fixes())).encode())
    h.update(f"{market}:{Path(data_dir).resolve()}".encode())   # frames differ by market and by the tables they were built from   # an ablation run must not reuse the frames of another set of repairs
    return h.hexdigest()[:10]


def load_frames(force: bool = False, market: str = MARKET, data_dir=DATA_DIR) -> dict:
    """{line: frame} with train+test rows, targets and features. Built once per code version, market and data directory."""
    key, out = cache_key(market, data_dir), {}
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    missing = [l for l in LINES if force or not (ARTIFACTS_DIR / f"frames_{key}_{l}.parquet").exists()]
    if missing:
        claims, cu, ad, we = data.load_claims(data_dir, market), data.load_customers(data_dir), data.load_addresses(data_dir), data.load_weather(data_dir)
        for l in missing:
            df = features.build_line_frame(l, data_dir, customers=cu, addresses=ad, weather=we, claims=claims, market=market)
            df.to_parquet(ARTIFACTS_DIR / f"frames_{key}_{l}.parquet")
    for l in LINES:
        df = pd.read_parquet(ARTIFACTS_DIR / f"frames_{key}_{l}.parquet")
        for c in df.select_dtypes("float64").columns:      # halve the memory of feature columns; keep targets and exposure exact
            if c not in _KEEP_FLOAT64:
                df[c] = df[c].astype("float32")
        out[l] = df
    return out
