"""Slide deck in the same design as the video: one still per scene (animation finished, no caption) as a full-bleed slide, speaker notes = the narration.

  python scripts/video/build_slides.py     -> artifacts/final_submission/demo_slides.pptx and demo_slides.pdf (+ slides_png/)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_video import H, W, timeline  # noqa: E402
from PIL import Image  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.util import Inches  # noqa: E402

OUT = ROOT / "artifacts" / "final_submission"


def main() -> None:
    tl = timeline()
    png_dir = OUT / "slides_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    shots = []   # (path, notes)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto((HERE / "scene.html").as_uri())
        pg.evaluate("tl => setTimeline(tl)", tl)
        for s in tl["scenes"]:
            notes = " ".join(e["text"] for e in s["sents"])
            times = [(s["end"] - 0.9, notes)]
            if s["id"] == "bus":
                times = [(s["sents"][2]["start"] + 6.0, notes)]
            if s["id"] == "guard":
                times = [(s["sents"][3]["start"] - 0.3, " ".join(e["text"] for e in s["sents"][:3])), (s["end"] - 0.2, s["sents"][3]["text"])]
            for k, (t, n) in enumerate(times):
                pg.evaluate("t => renderAt(t)", t)
                pg.evaluate("() => {document.getElementById('captxt').style.opacity=0;document.getElementById('clock').textContent='Week 1 → Week 2';}")
                path = png_dir / f"slide_{s['id']}_{k}.png"
                pg.screenshot(path=str(path))
                shots.append((path, n))
        b.close()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for path, notes in shots:
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        sl.shapes.add_picture(str(path), 0, 0, prs.slide_width, prs.slide_height)
        sl.notes_slide.notes_text_frame.text = notes
    prs.save(OUT / "demo_slides.pptx")
    ims = [Image.open(p).convert("RGB") for p, _ in shots]
    ims[0].save(OUT / "demo_slides.pdf", save_all=True, append_images=ims[1:], resolution=144)
    print("slides:", len(shots), "->", OUT / "demo_slides.pptx")


if __name__ == "__main__":
    main()
