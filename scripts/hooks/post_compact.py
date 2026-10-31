#!/usr/bin/env python3
"""SessionStart hook (matcher 'compact'): after a compaction, print a short pointer + digest that is added to the new context.

Kept under ~2 KB on purpose: in parc2026 a 24 KB injection was cut to a 2 KB excerpt and moved to a file, so only headings, line numbers and
the live facts are printed; the model reads the sections it needs.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    out = ["[context was compacted] Recover state before acting:",
           "1. docs/workflow/HANDOFF.md - 'START HERE' (current state, open decisions, next steps); read the sections you need.",
           "2. artifacts/session_snapshot.md - facts saved just before compaction (git state, running jobs, last experiments).",
           "3. CLAUDE.md rules still apply (no forum posts, no leaderboard probing, one heavy job at a time, work on `dev`)."]
    hand = ROOT / "docs" / "workflow" / "HANDOFF.md"
    if hand.exists():
        heads = [f"   L{i}: {l.strip()}" for i, l in enumerate(hand.read_text().splitlines(), 1) if re.match(r"#{2,3} ", l)][:12]
        out += ["", "HANDOFF.md headings:"] + heads
    snap = ROOT / "artifacts" / "session_snapshot.md"
    if snap.exists():
        keep = [l for l in snap.read_text().splitlines() if l.startswith(("- branch", "- uncommitted", "- unpushed")) or l.startswith("# Session")]
        out += ["", "Snapshot:"] + [f"   {l}" for l in keep]
    text = "\n".join(out)
    print(text[:1900])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"post_compact hook failed: {e}", file=sys.stderr)
    sys.exit(0)
