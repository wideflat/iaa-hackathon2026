"""Cross-validation harness: one place that turns a model factory into comparable scores."""
import json
import time
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS_DIR, LINES, REPO_ROOT
from .metrics import assert_not_too_good, score_report
from .validation import FOLDS, split

EXPERIMENT_LOG = REPO_ROOT / "experiments" / "log.jsonl"


def run_cv(frames: dict, model_factory, fold_keys=("A", "B"), n_boot: int = 200, train_frames: dict | None = None, ckpt_name: str | None = None):
    """frames: {line: frame from features.build_line_frame}. model_factory(line) -> model.

    train_frames: optional frames used for the TRAINING rows (e.g. closed-claims-only labels); validation rows and their
    truth always come from `frames` (all claims), because that is what the competition scores.
    ckpt_name: if given, each finished (fold, line) is saved to artifacts/ckpt/<ckpt_name>/ and reused on a re-run, so a run killed by a sandbox
    restart resumes where it stopped (the settings are part of the name: change the name when the model or its parameters change).
    Returns (results, oof) where oof holds out-of-fold predictions for every validation row.
    """
    results, oof_parts = {}, []
    for key in fold_keys:
        fold = FOLDS[key]
        parts = []
        for line in LINES:
            rows = frames[line][frames[line]["split"] == "train"]
            _, valid = split(rows, fold)
            ck = ARTIFACTS_DIR / "ckpt" / ckpt_name / f"{key}_{line}.parquet" if ckpt_name else None
            if ck is not None and ck.exists():
                parts.append(pd.read_parquet(ck))
                continue
            tr_src = (train_frames or frames)[line]
            train, _ = split(tr_src[tr_src["split"] == "train"], fold)
            model = model_factory(line)
            model = model.fit(train, target=valid) if getattr(model, "uses_target", False) else model.fit(train)   # domain-shift weights see the features of the rows to be predicted
            observed = not fold.mature and hasattr(model, "predict_observed")
            pred = model.predict_observed(valid) if observed else model.predict(valid)
            part = valid[["row_id", "customer_id", "line", "y", "year"]].assign(pred=pred)
            if hasattr(model, "predict_by_peril"):   # per-peril expected cost (sums to pred), for blends with peril-level weights
                part = part.join(model.predict_by_peril(valid, observed).add_prefix("pp_").set_axis(valid.index))
            if ck is not None:
                ck.parent.mkdir(parents=True, exist_ok=True)
                part.to_parquet(ck)
            parts.append(part)
        oof = pd.concat(parts, ignore_index=True)
        oof["fold"] = fold.name
        results[fold.name] = score_report(oof, n_boot=n_boot)
        assert_not_too_good(results[fold.name]["d2_pooled"], f"in fold {fold.name}")
        oof_parts.append(oof)
    return results, pd.concat(oof_parts, ignore_index=True)


def log_experiment(name: str, config: dict, results: dict, public_score: float | None = None, note: str = "") -> None:
    """Append one line to the tracked experiment log (CV mean/SE, public score, what changed)."""
    EXPERIMENT_LOG.parent.mkdir(exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": name, "config": config,
           "cv": results, "public_score": public_score, "note": note}
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
