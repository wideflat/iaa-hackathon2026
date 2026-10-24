"""Build (or reuse) the cached feature frames of both Part 2 portfolios.

  python scripts/p2_build_frames.py            # developed (= Part 1 data) and emerging

Frames land in artifacts/ keyed by market, data directory and code version (iaa.cache).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iaa import cache  # noqa: E402

DIRS = {"developed": ROOT / "data" / "part2" / "developed", "emerging": ROOT / "data" / "part2" / "emerging"}

if __name__ == "__main__":
    for market, d in DIRS.items():
        frames = cache.load_frames(market=market, data_dir=d)
        for line, df in frames.items():
            print(market, line, df.shape, "train rows", int((df["split"] == "train").sum()), "test rows", int((df["split"] == "test").sum()), flush=True)
