"""List this team's submissions with status and public score (read-only). Exit code 0 when none are pending."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import COMPETITION  # noqa: E402


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest

    with KaggleApi.build_kaggle_client_with_params(args=[], api_token="placeholder-proxy-injects-real-header") as k:
        r = ApiListSubmissionsRequest()
        r.competition_name = COMPETITION
        subs = k.competitions.competition_api_client.list_submissions(r).submissions
    pending = 0
    for s in subs:
        status = str(s.status).split(".")[-1]
        pending += status == "PENDING"
        print(f"{s.date} | {status:9s} | public score {s.public_score or '-':>8} | {(s.description or '')[:90]}")
    return 1 if pending else 0


if __name__ == "__main__":
    sys.exit(main())
