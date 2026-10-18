"""Per-peril calibration of the frequency x severity model on a validation fold: counts, capped severity and capped cost.

  python scripts/peril_calibration.py A      # train 2021-22, validate 2023
  python scripts/peril_calibration.py B      # train 2021-23, validate 2024 (observed space: predictions scaled by completeness)

Ratios are predicted / actual; 1.00 is calibrated. Costs are capped at the peril cap used by the model (so one giant claim cannot dominate).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import cache, data, target  # noqa: E402
from iaa.config import LINES  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402
from iaa.validation import FOLDS, split  # noqa: E402


def main() -> None:
    key = sys.argv[1]
    fold = FOLDS[key]
    claims, frames = data.load_claims(), cache.load_frames()
    rows_out = []
    for line in LINES:
        fr = frames[line][frames[line]["split"] == "train"]
        mc = target.matched_claims(fr, claims, line)
        train, valid = split(fr, fold)
        m = FreqSev(mc, kind="gbm", params_freq={"num_leaves": 15}, params_sev={"num_leaves": 15}).fit(train)
        X, expo = m._x(valid), valid["exposure_calc"].to_numpy()
        act = mc[mc["row_id"].isin(valid["row_id"]) & (mc["status"] != "repudiated")]
        for p in m.freq_:
            c = m._c_vec(p, valid["year"].to_numpy()) if not fold.mature else np.ones(len(valid))
            n_pred = float((m._freq_rate(p, X) * expo * c).sum())
            sev_pred = m._sev(p, X)
            cost_pred = float((m._freq_rate(p, X) * expo * c * sev_pred).sum())
            a = act[act["peril"] == p]
            cap = m.cap_[p]
            n_act, cost_act = len(a), float(a["net_cost"].clip(upper=cap).sum())
            sev_pred_mean = cost_pred / n_pred if n_pred else np.nan
            sev_act_mean = cost_act / n_act if n_act else np.nan
            rows_out.append(dict(line=line, peril=p, n_actual=n_act, count_ratio=n_pred / n_act if n_act else np.nan,
                                 sev_ratio=sev_pred_mean / sev_act_mean if n_act else np.nan, cost_ratio=cost_pred / cost_act if cost_act else np.nan,
                                 trend=m.sev_trend_))
    t = pd.DataFrame(rows_out)
    pd.set_option("display.width", 200)
    print(f"Fold {key} ({fold.name}); predictions are at {'ultimate level' if fold.mature else 'observed level (x completeness)'}")
    print(t.round(3).to_string(index=False))
    t.to_csv(Path(__file__).resolve().parents[1] / "artifacts" / f"peril_calibration_{key}.csv", index=False)


if __name__ == "__main__":
    main()
