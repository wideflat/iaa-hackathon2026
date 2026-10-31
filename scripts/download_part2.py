"""Download and unzip the Part 2 data (developed + emerging markets) into data/part2/ (gitignored). Same auth as download_data.py."""
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iaa.config import REPO_ROOT  # noqa: E402

SLUG = "iaa-ai-assisted-hackathon-2026-part-2"
URL = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{SLUG}"


def main() -> None:
    out = REPO_ROOT / "data" / "part2"
    out.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    zpath = out / "all.zip"
    with urllib.request.urlopen(req) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    zpath.unlink()
    print("extracted to", out, sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
