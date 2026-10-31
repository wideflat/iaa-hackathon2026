#!/usr/bin/env python3
"""PreCompact hook: save a factual snapshot of the working state before the context is compacted.

Writes artifacts/session_snapshot.md (gitignored, overwritten each time) with what a fresh context cannot recover from the summary alone:
branch / uncommitted files, running jobs, the last experiment-log lines, the newest submission files and the transcript path.
It cannot write judgement notes (hooks cannot call the model); those live in docs/workflow/HANDOFF.md, which post_compact.py points to.
Never fails the compaction: every step is wrapped, and the exit code is always 0.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "session_snapshot.md"


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(unavailable: {e})"


def experiments(n: int = 8) -> list[str]:
    path = ROOT / "experiments" / "log.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines()[-n:]:
        try:
            r = json.loads(line)
            cv = " ".join(f"{k.split('_')[0]}={v.get('d2_pooled', float('nan')):.4f}" for k, v in r.get("cv", {}).items())
            out.append(f"- {r['ts'][:16]} {r['name']}: {cv} public={r.get('public_score')} | {str(r.get('note', ''))[:90]}")
        except Exception:  # noqa: BLE001
            out.append("- (unparsable log line)")
    return out


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        payload = {}
    subs = sorted((ROOT / "submissions").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:6]
    status = sh("git", "status", "--short").splitlines()
    lines = [
        f"# Session snapshot (auto, PreCompact {payload.get('trigger', '?')}) {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "Written by scripts/hooks/pre_compact.py. Facts only; the judgement notes are in docs/workflow/HANDOFF.md.",
        "",
        f"- branch: {sh('git', 'branch', '--show-current')}  HEAD: {sh('git', 'log', '-1', '--format=%h %s')}",
        f"- uncommitted files: {len(status)}" + ("".join(f"\n    {s}" for s in status[:15]) + ("\n    ..." if len(status) > 15 else "")),
        f"- unpushed commits: {sh('git', 'log', '--oneline', '@{u}..HEAD') or 'none (or no upstream)'}",
        f"- full transcript: {payload.get('transcript_path', '(not given)')}",
        "",
        "## Running jobs (python scripts/*)",
        sh("pgrep", "-af", "^python scripts/") or "none",
        "",
        "## Last experiments (experiments/log.jsonl)",
        *experiments(),
        "",
        "## Newest submission files (gitignored; not necessarily submitted - check scripts/check_submissions.py)",
        *[f"- {p.name} ({time.strftime('%m-%d %H:%M', time.gmtime(p.stat().st_mtime))} UTC)" for p in subs],
        "",
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"pre_compact snapshot failed: {e}", file=sys.stderr)
    sys.exit(0)
