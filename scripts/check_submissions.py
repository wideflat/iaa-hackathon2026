"""List this team's submissions with status and public score (read-only). Exit code 0 when none are pending.

Uses the public REST endpoint through the sandbox proxy (which injects the Kaggle credential). The SDK's list call returned 'Authentication required'
for hours on 2026-10-18 while submitting through the SDK worked; the REST call does not have that problem.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import COMPETITION, PART2_COMPETITION  # noqa: E402


def main() -> int:
    competition = PART2_COMPETITION if "--part2" in sys.argv else COMPETITION
    r = requests.get(f"https://www.kaggle.com/api/v1/competitions/submissions/list/{competition}", params={"pageSize": 50}, timeout=60)
    r.raise_for_status()
    pending = 0
    for s in r.json():
        status = str(s.get("status"))
        pending += status.lower() == "pending"
        print(f"{s['date'][:16]} | {status:9s} | public score {s.get('publicScoreNullable') or '-':>8} | {(s.get('description') or '')[:90]}")
    return 1 if pending else 0


if __name__ == "__main__":
    sys.exit(main())
