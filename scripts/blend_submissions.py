"""Blend submission CSVs with per-line weights (line of each row taken from the cached test frames).

  python scripts/blend_submissions.py --a submissions/fs_gbm_v4_l4_out5.csv --b submissions/fs_nn_tabm4.csv --wb motor=0.7,buildings=0.4,contents=0.3 --out submissions/x.csv
Weight wb is the weight on file b (a gets 1 - wb). A single number (e.g. --wb 0.5) applies to every line.
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
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--wb", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    wb = {l: float(x.wb) for l in LINES} if "=" not in x.wb else {k: float(v) for k, v in (p.split("=") for p in x.wb.split(","))}
    assert set(wb) == set(LINES), wb
    frames = cache.load_frames()
    line_of = pd.concat([frames[l].loc[frames[l]["split"] == "test", ["row_id"]].assign(line=l) for l in LINES], ignore_index=True)
    a = pd.read_csv(x.a, dtype={"row_id": str})
    b = pd.read_csv(x.b, dtype={"row_id": str})
    assert (a["row_id"] == b["row_id"]).all()
    m = a.merge(line_of, on="row_id", how="left")
    assert m["line"].notna().all() and len(m) == len(a) and m["row_id"].is_unique
    w = m["line"].map(wb).to_numpy()
    out = a.assign(prediction=(1 - w) * a["prediction"].to_numpy() + w * b["prediction"].to_numpy())
    validate_submission(out, data.load_sample_submission())
    out.to_csv(x.out, index=False)
    print(f"wrote {x.out}: {len(out)} rows, sum {out['prediction'].sum():,.0f} (a {a['prediction'].sum():,.0f}, b {b['prediction'].sum():,.0f})")


if __name__ == "__main__":
    main()
