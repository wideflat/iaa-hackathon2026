"""Render experiments/log.jsonl as a readable table: docs/experiments/EXPERIMENT_LOG.md.

  python scripts/render_experiments.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import REPO_ROOT  # noqa: E402


def fmt_cv(cv: dict) -> str:
    parts = []
    for fold, r in cv.items():
        lines = " ".join(f"{k[3:5]}{r[k]:.3f}" for k in ("d2_motor", "d2_buildings", "d2_contents") if k in r)
        ci = r.get("d2_pooled_boot")
        ci_txt = f" ({ci['lo']:.3f}-{ci['hi']:.3f})" if ci else ""
        parts.append(f"**{fold}** {r['d2_pooled']:.4f}{ci_txt} [{lines}]")
    return "<br>".join(parts) if parts else "-"


def main() -> None:
    recs = [json.loads(l) for l in open(REPO_ROOT / "experiments" / "log.jsonl")]
    out = ["# Experiment log (generated from experiments/log.jsonl by scripts/render_experiments.py)\n",
           "Pooled Tweedie D2 (power 1.5) with 95% customer-clustered bootstrap CI where available; in brackets the per-line D2 "
           "(mo = motor, bu = buildings, co = contents). Fold A = train 2021-22, validate 2023; fold B = train 2021-23, validate 2024 (immature). "
           "Rehearsal rows are keyed by variant. Entries are appended chronologically; the INVALID entry is kept on purpose as evidence.\n",
           "| # | Time (UTC) | Name | Results | Public LB | Note |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(recs, 1):
        pub = r.get("public_score")
        out.append(f"| {i} | {r['ts'][:16].replace('T', ' ')} | `{r['name']}` | {fmt_cv(r.get('cv', {}))} | {pub if pub is not None else '-'} | {(r.get('note') or '').replace('|', '/')} |")
    p = REPO_ROOT / "docs" / "experiments" / "EXPERIMENT_LOG.md"
    p.write_text("\n".join(out) + "\n")
    print(f"wrote {p} ({len(recs)} entries)")


if __name__ == "__main__":
    main()
