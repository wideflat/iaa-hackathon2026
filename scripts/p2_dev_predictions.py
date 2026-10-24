"""Train the developed-market model on the shared feature space and score every emerging row with it (zero-shot transfer base).

  python scripts/p2_dev_predictions.py --tags A,B,C

Tag A = developed 2021-22 (level of 2023), B = 2021-23 (level of 2024), C = 2021-24 (level of 2025). Each (tag, line) is saved once to
artifacts/p2/dev_pred2_<tag>_<line>.parquet: per-peril expected cost (dev_<peril>) and expected claim count (devn_<peril>) of every emerging row on the ultimate basis at the tag's level, incl. the
model's rebalancing factor. Resumable: finished files are skipped. The completeness table of the developed claims (to correct the immature
emerging 2024 year) is saved per line as artifacts/p2/dev_completeness_<line>.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from iaa import cache, data, market, target  # noqa: E402
from iaa.config import ARTIFACTS_DIR, LINES  # noqa: E402
from iaa.models.freqsev import FreqSev  # noqa: E402

DIRS = {"developed": ROOT / "data" / "part2" / "developed", "emerging": ROOT / "data" / "part2" / "emerging"}
LAST_TRAIN_YEAR = {"A": 2022, "B": 2023, "C": 2024}
THREADS = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="A,B,C")
    ap.add_argument("--lines", default=",".join(LINES))
    ap.add_argument("--model", default="gbm", choices=["gbm", "nn_tabm", "nn_tabm_ple"], help="gbm = LightGBM (Part 1 tree half); nn_* = the TabM nets of the Part 1 NN half")
    a = ap.parse_args()
    out_dir = ARTIFACTS_DIR / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = data.load_claims(DIRS["developed"], "developed")
    dev, em = cache.load_frames(market="developed", data_dir=DIRS["developed"]), cache.load_frames(market="emerging", data_dir=DIRS["emerging"])
    for line in a.lines.split(","):
        d = market.to_common(dev[line], line, "developed")
        e = market.to_common(em[line], line, "emerging")
        d_train = d[d["split"] == "train"]
        matched = target.matched_claims(d_train, claims, line)
        for tag in a.tags.split(","):
            stem = "dev_pred2" if a.model == "gbm" else f"dev_pred2_{a.model}"
            path = out_dir / f"{stem}_{tag}_{line}.parquet"   # costs and counts; dev_pred_ = first run of the GBM (costs only)
            old_path = out_dir / f"dev_pred_{tag}_{line}.parquet" if a.model == "gbm" else out_dir / "does_not_exist.parquet"
            if path.exists():
                continue
            train = d_train[d_train["year"] <= LAST_TRAIN_YEAR[tag]]
            if a.model == "gbm":
                model = FreqSev(matched, kind="gbm", cat_encoding="target", outlier_mult=5.0, freq_rounds=350, sev_rounds=300, threads=THREADS,
                                params_freq={"num_threads": THREADS, "num_leaves": 4}, params_sev={"num_threads": THREADS, "num_leaves": 4}).fit(train)
            else:   # as scripts/run_cv.py --model fs_nn --outlier-mult 5 --nn-tabm-k 4 [--nn-ple 8]
                nnp = {"tabm_k": 4, "threads": THREADS} | ({"ple_bins": 8} if a.model == "nn_tabm_ple" else {})
                model = FreqSev(matched, kind="nn", outlier_mult=5.0, threads=THREADS, nn_params=nnp).fit(train)
            by = model.k_ * model.expected_by_peril(e)
            cnt = model.expected_counts(e)
            sev_ratio = (by / cnt.where(cnt > 0)).fillna(0.0)   # mean cost per claim incl. excess load and rebalancing
            assert np.allclose((cnt * sev_ratio).to_numpy(), by.to_numpy(), rtol=1e-6), "counts x severity must reproduce the cost"
            out = pd.concat([e[["row_id", "year", "split"]], by.add_prefix("dev_"), cnt.add_prefix("devn_")], axis=1)
            if old_path.exists():   # reproducibility check against the first run (same seeds, same data): costs must be identical
                old = pd.read_parquet(old_path)
                assert np.allclose(old[by.add_prefix("dev_").columns].to_numpy(), out[by.add_prefix("dev_").columns].to_numpy(), rtol=1e-6), f"{tag} {line}: rerun differs from the first run"
                print(f"{line} tag {tag}: reproduces the first run", flush=True)
            out.to_parquet(path)
            print(f"{line} tag {tag}: trained on {len(train)} rows, k={model.k_:.3f}, trend={model.sev_trend_:.3f}; emerging mean dev cost/row {by.sum(axis=1).mean():.1f}", flush=True)
            (out_dir / f"dev_completeness_{line}.json").write_text(json.dumps({p: {str(y): v for y, v in t.items()} for p, t in model.c_by_year.items()}))


if __name__ == "__main__":
    main()
