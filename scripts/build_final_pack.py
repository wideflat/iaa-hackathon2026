"""Build the files for the final submission form into artifacts/final_submission/ (and small copies of the text files into docs/final/).

  python scripts/build_final_pack.py [--team-name NAME] [--members "A, B"] [--w2-file submissions/p2_t11_ensemble_e3.csv]

* week1_prediction.csv / week2_prediction.csv: copies of the chosen prediction files (checked against the sample submissions; sha256 printed)
* writeup.docx / writeup.pdf: docs/part2/REPORT.md with the team line filled in (pandoc, then LibreOffice for the PDF)
* the designed slide deck and the demo video come from scripts/video/ (build_slides.py, build_video.py)
Names and results that only the team knows are arguments; without them the files carry a visible [..] placeholder.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

OUT = ROOT / "artifacts" / "final_submission"
DOCS = ROOT / "docs" / "final"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_prediction_file(path: Path, sample: pd.DataFrame, label: str) -> None:
    d = pd.read_csv(path)
    assert list(d.columns) == list(sample.columns), f"{label}: columns {list(d.columns)}"
    assert len(d) == len(sample) and set(d["row_id"]) == set(sample["row_id"]), f"{label}: row ids differ from the sample submission"
    assert d["prediction"].notna().all() and (d["prediction"] >= 0).all(), f"{label}: missing or negative predictions"
    print(f"{label}: {len(d)} rows, mean {d['prediction'].mean():.2f}, sha256 {sha(path)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", default="[TEAM NAME]")
    ap.add_argument("--members", default="[TEAM MEMBERS' NAMES]")
    ap.add_argument("--w1-file", default="submissions/blend_v10_perline.csv")
    ap.add_argument("--w2-file", default="submissions/p2_t11_ensemble_e3.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    from iaa import data
    s1 = data.load_sample_submission()
    s2 = data.load_sample_submission(ROOT / "data" / "part2" / "emerging")
    check_prediction_file(ROOT / a.w1_file, s1, "week 1"); check_prediction_file(ROOT / a.w2_file, s2, "week 2")
    shutil.copy(ROOT / a.w1_file, OUT / "week1_prediction.csv"); shutil.copy(ROOT / a.w2_file, OUT / "week2_prediction.csv")
    md = (ROOT / "docs" / "part2" / "REPORT.md").read_text()
    lines = md.split("\n")
    lines[2] = f"Team: {a.team_name}; members: {a.members}. Repository: `wideflat/iaa-hackathon2026`, branch `dev`. Final version of 2026-10-31."
    src = OUT / "writeup.md"; src.write_text("\n".join(lines))
    subprocess.run(["pandoc", str(src), "-o", str(OUT / "writeup.docx")], check=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(OUT / "writeup.docx")], check=True, capture_output=True)
    for f in ("writeup.docx", "writeup.pdf"):
        shutil.copy(OUT / f, DOCS / f)
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
