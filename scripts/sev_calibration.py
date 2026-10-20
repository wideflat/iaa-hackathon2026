"""Severity calibration of the production target-encoded GBM per peril on a validation fold (motor by default): predicted mean severity of the rows' expected claims vs the actual capped mean severity of the validation year.

  python scripts/sev_calibration.py B motor
Ratios are predicted / actual; 1.00 is calibrated.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cache, data, target  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402
from iaa.validation import FOLDS, split  # noqa: E402


def main() -> None:
    key, line = sys.argv[1], sys.argv[2]
    fold = FOLDS[key]
    claims, frames = data.load_claims(), cache.load_frames()
    fr = frames[line][frames[line]["split"] == "train"]
    mc = target.matched_claims(fr, claims, line)
    train, valid = split(fr, fold)
    pf = {"num_threads": 4, "num_leaves": 4}
    m = FreqSev(mc, kind="gbm", freq_rounds=350, sev_rounds=300, outlier_mult=5.0, cat_encoding="target", params_freq=pf, params_sev=pf).fit(train)
    X, expo = m._x(valid), valid["exposure_calc"].to_numpy()
    act = mc[mc["row_id"].isin(valid["row_id"]) & (mc["status"] != "repudiated")]
    out = []
    for p in m.freq_:
        c = m._c_vec(p, valid["year"].to_numpy()) if not fold.mature else np.ones(len(valid))
        w = m._freq_rate(p, X) * expo * c
        a = act[act["peril"] == p]
        out.append(dict(peril=p, trend=round(m.sev_trend_, 3), pred_sev=float((w * m._sev(p, X)).sum() / w.sum()), act_sev=float(a["net_cost"].clip(upper=m.cap_[p]).mean()),
                        pred_n=float(w.sum()), act_n=len(a)))
    t = pd.DataFrame(out).assign(sev_ratio=lambda d: d.pred_sev / d.act_sev, n_ratio=lambda d: d.pred_n / d.act_n)
    pd.set_option("display.width", 200)
    print(f"fold {key} {line}"); print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
