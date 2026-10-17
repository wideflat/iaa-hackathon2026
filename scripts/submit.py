"""Validate a submission CSV and (only with --confirm) submit it to the competition via the Kaggle SDK.

  python scripts/submit.py submissions/lgbm.csv -m "lgbm_v1 CV A=0.098"            # dry run
  python scripts/submit.py submissions/lgbm.csv -m "lgbm_v1 CV A=0.098" --confirm   # really submit

Dry run = format validation + request an upload slot (no submission is created, no quota used).
Auth: KAGGLE_API_TOKEN if set; in the cloud sandbox the proxy injects the credential, so a placeholder is used.
The Kaggle CLI itself cannot be used in the sandbox because it introspects the token locally first.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa import data  # noqa: E402
from iaa.config import COMPETITION  # noqa: E402
from iaa.submission import validate_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-m", "--message", required=True, help="submission description (what it is, CV score)")
    ap.add_argument("--confirm", action="store_true", help="actually submit (uses one of the daily submissions)")
    a = ap.parse_args()

    sub = pd.read_csv(a.csv, dtype={"row_id": str})
    validate_submission(sub, data.load_sample_submission())
    print(f"format OK: {len(sub)} rows, mean prediction {sub['prediction'].mean():.2f}, min {sub['prediction'].min():.4f}")

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiStartSubmissionUploadRequest

    token = os.environ.get("KAGGLE_API_TOKEN", "placeholder-proxy-injects-real-header")
    if not a.confirm:
        with KaggleApi.build_kaggle_client_with_params(args=[], api_token=token) as k:
            r = ApiStartSubmissionUploadRequest()
            r.competition_name, r.file_name = COMPETITION, os.path.basename(a.csv)
            r.content_length, r.last_modified_epoch_seconds = os.path.getsize(a.csv), int(os.path.getmtime(a.csv))
            resp = k.competitions.competition_api_client.start_submission_upload(r)
        print("dry run OK: upload slot granted =", bool(resp.create_url), "- NOT submitted (add --confirm)")
        return

    api = KaggleApi()
    api.config_values[api.CONFIG_NAME_TOKEN] = token
    api._authenticated = True
    resp = api.competition_submit(a.csv, a.message, COMPETITION)
    print("submitted:", resp.message)


if __name__ == "__main__":
    main()
