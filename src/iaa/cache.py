"""Cache of the per-line feature frames so experiments start in seconds, not minutes.

The cache key is a hash of the source files that define the frames (data, target, features): changing any of them
invalidates the cache automatically. Frames live in artifacts/ (gitignored).
"""
import hashlib

import pandas as pd

from . import data, features
from .config import ARTIFACTS_DIR, LINES, REPO_ROOT

_KEEP_FLOAT64 = {"y", "y_all", "y_true", "exposure", "exposure_calc"}
_SOURCES = ("src/iaa/data.py", "src/iaa/target.py", "src/iaa/features.py")


def cache_key() -> str:
    h = hashlib.md5()
    for p in _SOURCES:
        h.update((REPO_ROOT / p).read_bytes())
    return h.hexdigest()[:10]


def load_frames(force: bool = False) -> dict:
    """{line: frame} with train+test rows, targets and features. Built once per code version."""
    key, out = cache_key(), {}
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    missing = [l for l in LINES if force or not (ARTIFACTS_DIR / f"frames_{key}_{l}.parquet").exists()]
    if missing:
        claims, cu, ad, we = data.load_claims(), data.load_customers(), data.load_addresses(), data.load_weather()
        for l in missing:
            df = features.build_line_frame(l, customers=cu, addresses=ad, weather=we, claims=claims)
            df.to_parquet(ARTIFACTS_DIR / f"frames_{key}_{l}.parquet")
    for l in LINES:
        df = pd.read_parquet(ARTIFACTS_DIR / f"frames_{key}_{l}.parquet")
        for c in df.select_dtypes("float64").columns:      # halve the memory of feature columns; keep targets and exposure exact
            if c not in _KEEP_FLOAT64:
                df[c] = df[c].astype("float32")
        out[l] = df
    return out
