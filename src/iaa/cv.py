"""Cross-validation harness: one place that turns a model factory into comparable scores."""
import json
import time
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS_DIR, LINES, REPO_ROOT
from .metrics import score_report
from .validation import FOLDS, split

EXPERIMENT_LOG = REPO_ROOT / "experiments" / "log.jsonl"


def run_cv(frames: dict, model_factory, fold_keys=("A", "B"), n_boot: int = 200):
    """frames: {line: frame from features.build_line_frame}. model_factory(line) -> model.

    Returns (results, oof) where oof holds out-of-fold predictions for every validation row.
    """
    results, oof_parts = {}, []
    for key in fold_keys:
        fold = FOLDS[key]
        parts = []
        for line in LINES:
            rows = frames[line][frames[line]["split"] == "train"]
            train, valid = split(rows, fold)
            model = model_factory(line).fit(train)
            parts.append(valid[["row_id", "customer_id", "line", "y", "year"]].assign(pred=model.predict(valid)))
        oof = pd.concat(parts, ignore_index=True)
        oof["fold"] = fold.name
        results[fold.name] = score_report(oof, n_boot=n_boot)
        oof_parts.append(oof)
    return results, pd.concat(oof_parts, ignore_index=True)


def log_experiment(name: str, config: dict, results: dict, public_score: float | None = None, note: str = "") -> None:
    """Append one line to the tracked experiment log (CV mean/SE, public score, what changed)."""
    EXPERIMENT_LOG.parent.mkdir(exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": name, "config": config,
           "cv": results, "public_score": public_score, "note": note}
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
