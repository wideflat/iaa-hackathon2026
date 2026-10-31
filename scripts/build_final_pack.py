"""Build the files for the final submission form into artifacts/final_submission/ (and small copies of the text files into docs/final/).

  python scripts/build_final_pack.py [--team-name NAME] [--members "A, B"] [--w2-private "text"] [--w2-file submissions/p2_t11_ensemble_e3.csv]

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
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

OUT = ROOT / "artifacts" / "final_submission"
DOCS = ROOT / "docs" / "final"
NAVY, TEAL, GREY, LIGHT = RGBColor(0x14, 0x2B, 0x4D), RGBColor(0x1F, 0x8A, 0x8A), RGBColor(0x55, 0x5B, 0x66), RGBColor(0xEA, 0xF1, 0xF6)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_prediction_file(path: Path, sample: pd.DataFrame, label: str) -> None:
    d = pd.read_csv(path)
    assert list(d.columns) == list(sample.columns), f"{label}: columns {list(d.columns)}"
    assert len(d) == len(sample) and set(d["row_id"]) == set(sample["row_id"]), f"{label}: row ids differ from the sample submission"
    assert d["prediction"].notna().all() and (d["prediction"] >= 0).all(), f"{label}: missing or negative predictions"
    print(f"{label}: {len(d)} rows, mean {d['prediction'].mean():.2f}, sha256 {sha(path)}")


def text_box(slide, x, y, w, h, text, size=18, bold=False, color=GREY, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        p.space_after = Pt(6)
    return tb


def card(slide, x, y, w, h, big, small, fill=LIGHT):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill; s.line.fill.background()
    text_box(slide, x + 0.1, y + 0.1, w - 0.2, h * 0.5, big, 30, True, NAVY, PP_ALIGN.CENTER)
    text_box(slide, x + 0.1, y + h * 0.5, w - 0.2, h * 0.5, small, 14, False, GREY, PP_ALIGN.CENTER)


def table(slide, x, y, w, rows, col_w, size=13, header=True):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.38 * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size); p.font.bold = header and i == 0
    return t


def slide_base(prs, title, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    text_box(s, 0.5, 0.18, 12.3, 0.7, title, 28, True, RGBColor(255, 255, 255))
    s.notes_slide.notes_text_frame.text = notes
    return s


def build_deck(path: Path, team: str, w2_private: str) -> None:
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    s = slide_base(prs, "One engine, two markets: pricing with an AI-assisted workflow", f"Result. Week 1: private rank 3 of 54, D2 0.13864. Week 2: the same engine, adapted to a market with two years of data; public 0.16458 for the final ensemble. Private: {w2_private}.")
    text_box(s, 0.6, 1.3, 12, 0.6, f"Team: {team}", 20, True, TEAL)
    card(s, 0.6, 2.2, 3.9, 2.0, "Rank 3 of 54", "Week 1 private leaderboard, pooled Tweedie D2 0.13864")
    card(s, 4.7, 2.2, 3.9, 2.0, "0.16458", "Week 2 public score of the final ensemble (first file 0.16077)")
    card(s, 8.8, 2.2, 3.9, 2.0, w2_private, "Week 2 private leaderboard", fill=RGBColor(0xFF, 0xF4, 0xD6))
    text_box(s, 0.6, 4.6, 12, 2.2, ["Week 1: developed market, 2.4 million policy rows.", "Week 2: emerging market, two years of data, 76% new business in the test book.", "The emerging data decides how much; the developed data decides who."], 20)

    s = slide_base(prs, "What changed in the new market", "Three numbers: motor frequency +34% and cost -42%; 76% of the test book is new business; no history before 2023. The new market is the old market times stable peril multipliers: we measured that, we did not assume it.")
    card(s, 0.6, 1.3, 3.9, 1.7, "+34% / -42%", "motor claim frequency / mean claim cost")
    card(s, 4.7, 1.3, 3.9, 1.7, "76%", "of the 2025 test book is new business (14% in training)")
    card(s, 8.8, 1.3, 3.9, 1.7, "2 years", "of emerging history, none before 2023")
    table(s, 0.6, 3.4, 12.1, [["Peril (frequency ratio emerging / developed)", "Ratio"], ["glass-hail", "1.5 - 1.8"], ["third-party property, bodily injury", "1.4 - 1.5"], ["storm", "1.3 - 1.7"], ["escape of water", "0.65 - 0.7"], ["severity scale (median ratio)", "motor 0.6, buildings 0.7, contents 0.75"]], [7.6, 4.5])
    text_box(s, 0.6, 6.0, 12, 1, "Ratios are stable between the two emerging years, except for the rare buildings perils (fire 2.3x then 0.8x).", 16)

    s = slide_base(prs, "Method", "Developed model on shared variables, add last-year claim history, blend three tree models, a direct Tweedie GBM and six nets, take the level from the emerging book with counts and winsorised severity per peril, credibility-weighted, no closing to the training total, then a gender factor. The emerging data decides how much, the developed data decides who.")
    steps = [("Developed model", "2021-24, 2.4M rows, variables both markets record"), ("+ claim history", "last year's claims of the policy, address and customer"), ("Ensemble", "5 tree models, 7 nets, 4 Tweedie GBMs, equal weights in groups"), ("Level from emerging book", "claim counts x winsorised severity per peril, credibility-weighted"), ("Gender factor", "one flat factor per line, credibility 300 claims")]
    for i, (a, b) in enumerate(steps):
        x = 0.4 + i * 2.55
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.8), Inches(2.3), Inches(2.6))
        box.fill.solid(); box.fill.fore_color.rgb = LIGHT if i not in (3,) else RGBColor(0xD6, 0xEE, 0xEC); box.line.color.rgb = TEAL
        text_box(s, x + 0.08, 1.9, 2.15, 0.9, a, 18, True, NAVY, PP_ALIGN.CENTER)
        text_box(s, x + 0.08, 2.8, 2.15, 1.6, b, 14, False, GREY, PP_ALIGN.CENTER)
        if i < 4:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.32), Inches(2.9), Inches(0.2), Inches(0.4)); ar.fill.solid(); ar.fill.fore_color.rgb = TEAL; ar.line.fill.background()
    text_box(s, 0.6, 4.9, 12, 2, ["No emerging-only model: it scores 0.1110 against 0.1405 for the transfer.", "Frequency x severity per peril, trended and completeness-adjusted; the direct Tweedie GBM adds ranking, not level.", "Level from claim counts, not from a few large claims: closing to the training total copied large-loss noise."], 17)

    s = slide_base(prs, "How we trust it", "Both-direction validation, forward and reverse, because with two years a single split mistakes noise for signal. More than thirty ideas were rejected because they helped in one direction only. A simulation of the level method with a known truth and a ceiling analysis bound what more tuning could give. Part 1 shake-up: public and private agree overall (0.997) but not inside the top group (0.20): we select by local validation.")
    table(s, 0.6, 1.3, 7.4, [["Step", "Forward", "Reverse"], ["Emerging-only GBM", "0.1110", ""], ["Developed model + closure + gender", "0.1256", "0.1162"], ["+ robust level (no closure)", "0.1281", "0.1176"], ["+ nets", "0.1289", "0.1220"], ["+ claim history", "0.1313", "0.1229"], ["+ Tweedie part", "0.1336", "0.1237"], ["+ buildings credibility 400", "0.1397", "0.1271"], ["+ ensemble (final)", "0.1405", "0.1268"]], [4.2, 1.6, 1.6], 14)
    text_box(s, 8.4, 1.3, 4.5, 5.5, ["Forward: calibrate 2023, score 2024. Reverse: calibrate 2024, score 2023.", "Adopt only if neither direction is worse.", "Simulation, known truth: robust level beats the line closure in every scenario.", "Ceiling: a perfect level adds at most 0.005 D2.", "Residual search: no feature structure left after the gender factor (973 levels)."], 16)

    s = slide_base(prs, "AI in the workflow", "Claude Code did EDA, code with tests (125), experiments; helper sessions ran refits in parallel; a blind second agent compared the markets from raw tables and found two things we missed. Every number traces to a command.")
    text_box(s, 0.6, 1.3, 12.2, 5.8, ["Claude Code: data review, test-first code (125 automated tests), experiments, documents.", "Helper sessions in separate containers ran the hours-long refits in parallel.", "A blind second agent compared the two markets from the raw tables only and found two items we had missed.", "Errors the checks caught: three unsupported statements in the first data review; a simulator defect that looked like an estimator bias; a net that Part 1 had already rejected.", "Every decision has a numbered row with its evidence (83 so far); every result has a command."], 20)

    s = slide_base(prs, "Guardrails and uncertainty", "Model card, monitoring triggers, leak alarm, the buildings level as the main stated risk (+-20% is about +-0.01 D2), gender rating flagged for regulatory review. Written premiums roughly equal expected claims, so loadings need a 2.5x premium.")
    text_box(s, 0.6, 1.3, 12.2, 5.8, ["Automated checks on every push; leak alarm (pooled D2 above 0.35 stops a run); reproducibility assertion.", "Model card with monitoring triggers (actual / expected by line and peril).", "Main stated risk: the buildings level; two years, one large-loss peril swinging 2.3x then 0.8x (+-20% is about +-0.01 D2).", "Gender is used as a rating factor and flagged for regulatory review; a gender-free file is the stated alternative.", "Office premium: expected claims plus loadings; written premiums roughly equal expected claims, so loadings need about 2.5x the premium."], 20)
    prs.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", default="[TEAM NAME]")
    ap.add_argument("--members", default="[TEAM MEMBERS' NAMES]")
    ap.add_argument("--w2-private", default="[add private result]")
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
