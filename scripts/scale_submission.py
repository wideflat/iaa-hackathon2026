"""Multiply a submission's predictions by per-line factors (line of each row taken from the cached test frames).

  python scripts/scale_submission.py --in submissions/a.csv --factors motor=1.0,buildings=1.0757,contents=1.0312 --out submissions/b.csv
Used for forward-looking level assumptions (e.g. a 2025 severity trend above the historical average); the factor is documented in docs/experiments/RESULTS.md.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cache, data  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.submission import validate_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--factors", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    f = {k: float(v) for k, v in (p.split("=") for p in x.factors.split(","))}
    assert set(f) == set(LINES), f
    frames = cache.load_frames()
    line_of = pd.concat([frames[l].loc[frames[l]["split"] == "test", ["row_id"]].assign(line=l) for l in LINES], ignore_index=True)
    s = pd.read_csv(x.src, dtype={"row_id": str})
    m = s.merge(line_of, on="row_id", how="left")
    assert m["line"].notna().all() and len(m) == len(s) and m["row_id"].is_unique
    out = s.assign(prediction=s["prediction"].to_numpy() * m["line"].map(f).to_numpy())
    validate_submission(out, data.load_sample_submission())
    out.to_csv(x.out, index=False)
    print(f"wrote {x.out}: sum {out['prediction'].sum():,.0f} vs input {s['prediction'].sum():,.0f} (x{out['prediction'].sum() / s['prediction'].sum():.4f})")


if __name__ == "__main__":
    main()
