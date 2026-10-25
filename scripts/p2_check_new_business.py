"""Does the history-input base work on new business? D2 of saved forward validation predictions (calibrate 2023, score 2024) split into rows whose policy is new in 2024 and renewals.

  python scripts/p2_check_new_business.py     (needs the oof files written by p2_calib2.py for the named runs; prints and appends to docs/experiments/raw/p2/new_business_check.txt)
The files hold different level methods, so only compare rows within a split. 2025 test rows are 76% new business, the 2024 validation year 15%.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache  # noqa: E402
import p2_calib as pc  # noqa: E402

FILES = {"t7 (history, nets, Tweedie)": "p2_calib2_forward_hist1_nn0.5_tw_nn_tabm_hist1,nn_tabm_ple_hist1_oof.parquet",
         "t4 (no history, two nets)": "p2_calib2_forward_gbm_nn0.5_nn_tabm,nn_tabm_ple_oof.parquet"}


def d2(y, p, pw=1.5):
    p = np.maximum(p, 1e-9)
    dev = lambda m: 2 * (np.where(y > 0, y ** (2 - pw) / ((1 - pw) * (2 - pw)), 0) - y * m ** (1 - pw) / (1 - pw) + m ** (2 - pw) / (2 - pw))   # noqa: E731
    return 1 - dev(p).sum() / dev(np.full_like(p, y.mean())).sum()


def main() -> None:
    em = cache.load_frames(market="emerging", data_dir=pc.EM_DIR)
    nb = {}
    for line, e in em.items():
        v = e[(e["year"] == 2024) & (e["split"] == "train")]
        nb[line] = pd.Series(~v["policy_id"].isin(set(e.loc[e["year"] == 2023, "policy_id"])).to_numpy(), index=v["row_id"].to_numpy())
    lines = []
    for name, f in FILES.items():
        d = pd.read_parquet(pc.P2 / f)
        d["nb"] = [nb[l].get(r, np.nan) for l, r in zip(d["line"], d["row_id"])]
        r = {s: round(float(d2(d[d["nb"] == flag]["y"].to_numpy(), d[d["nb"] == flag]["pred"].to_numpy())), 4) for flag, s in ((True, "new"), (False, "renewal"))}
        lines.append(f"{name}: new business {r['new']} (rows {int((d['nb'] == True).sum())}), renewals {r['renewal']} (rows {int((d['nb'] == False).sum())})")  # noqa: E712
    print("\n".join(lines))
    with open(ROOT / "docs" / "experiments" / "raw" / "p2" / "new_business_check.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
