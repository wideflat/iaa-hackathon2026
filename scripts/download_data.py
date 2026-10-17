"""Download and unzip the competition data into data/raw/ (gitignored).

Auth: uses KAGGLE_API_TOKEN as a Bearer token if set. In the cloud sandbox the proxy injects the
credential, so no variable is needed there. You must have accepted the competition rules on kaggle.com.
"""
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import COMPETITION, REPO_ROOT  # noqa: E402

URL = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}"


def main() -> None:
    raw = REPO_ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    zpath = raw / "all.zip"
    with urllib.request.urlopen(req) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw)
    zpath.unlink()
    print("extracted to", raw / "public")


if __name__ == "__main__":
    main()
