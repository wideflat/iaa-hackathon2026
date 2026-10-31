"""Render the final demo video of the IAA AI-Assisted Hackathon 2026 entry (frame-exact HTML scenes, no screen recorder).

  python scripts/video/build_video.py --preview 4,20,45     # PNGs of single timestamps into artifacts/video_preview/
  python scripts/video/build_video.py --render              # all frames -> artifacts/video_frames/, then ffmpeg -> artifacts/final_submission/demo_video*.mp4

Timeline: narration.json gives the scenes and sentences; each sentence lasts words / 2.35 + 0.45 s (a relaxed speaking pace) so a voice-over can be recorded on top.
Outputs: demo_video.mp4 (music + captions), demo_video_no_music.mp4 (captions only), captions.srt, voiceover_script.md (timecoded), music.wav.
Needs Playwright with Chromium, ffmpeg, numpy/scipy; no network and no external packages.
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FPS = 30
W, H = 1920, 1080
WPS, GAP, SCENE_PAD, INTRO, OUTRO = 2.35, 0.45, 0.9, 0.6, 3.4


def timeline() -> dict:
    d = json.loads((HERE / "narration.json").read_text())
    t = INTRO
    scenes = []
    for s in d["scenes"]:
        start = t
        sents = []
        for x in s["sents"]:
            dur = len(x.split()) / WPS + GAP
            sents.append({"text": x, "start": round(t, 3), "end": round(t + dur - GAP * 0.35, 3)})
            t += dur
        scenes.append({"id": s["id"], "chapter": s["chapter"], "oneliner": s["oneliner"], "start": round(start, 3), "end": round(t + SCENE_PAD, 3), "sents": sents})
        t += SCENE_PAD
    scenes[-1]["end"] = round(t - SCENE_PAD + OUTRO, 3)
    return {"scenes": scenes, "total": scenes[-1]["end"]}


def fmt(t: float) -> str:
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_text_files(tl: dict, out: Path) -> None:
    srt, md, n = [], ["# Voice-over script with timecodes (relaxed pace, about 141 words per minute)\n"], 1
    for s in tl["scenes"]:
        md.append(f"\n## {s['chapter']}  ({s['start']:.1f}s - {s['end']:.1f}s)\n")
        for e in s["sents"]:
            srt.append(f"{n}\n{fmt(e['start'])} --> {fmt(e['end'])}\n{e['text']}\n")
            md.append(f"* [{int(e['start'] // 60)}:{e['start'] % 60:04.1f}] {e['text']}")
            n += 1
    (out / "captions.srt").write_text("\n".join(srt))
    (out / "voiceover_script.md").write_text("\n".join(md) + "\n")


def phone_images() -> list:
    """Real phone screenshots dropped into scripts/video/assets/phone/ (jpg or png, shown in the phone frame in file-name order); empty list = the re-created UI."""
    d = HERE / "assets" / "phone"
    return [p.as_uri() for p in sorted(d.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")] if d.exists() else []


def render_chunk(args):
    frames, tl, outdir = args
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto((HERE / "scene.html").as_uri())
        pg.evaluate("a => setPhoneImages(a)", phone_images())
        pg.evaluate("tl => setTimeline(tl)", tl)
        for i in frames:
            pg.evaluate("t => renderAt(t)", i / FPS)
            pg.screenshot(path=str(outdir / f"f{i:06d}.jpg"), type="jpeg", quality=93)
        b.close()
    return len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default="")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tl = timeline()
    print(f"total {tl['total']:.1f}s ({int(tl['total'] // 60)}:{tl['total'] % 60:04.1f}); scenes:", [(s['id'], round(s['start'], 1)) for s in tl['scenes']])
    out = ROOT / "artifacts" / "final_submission"
    out.mkdir(parents=True, exist_ok=True)
    write_text_files(tl, out)
    if a.preview:
        pv = ROOT / "artifacts" / "video_preview"
        pv.mkdir(parents=True, exist_ok=True)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
            pg = b.new_page(viewport={"width": W, "height": H})
            pg.goto((HERE / "scene.html").as_uri())
            pg.evaluate("a => setPhoneImages(a)", phone_images())
            pg.evaluate("tl => setTimeline(tl)", tl)
            for t in [float(x) for x in a.preview.split(",")]:
                pg.evaluate("t => renderAt(t)", t)
                pg.screenshot(path=str(pv / f"p{t:06.1f}.png"))
            b.close()
        return
    if a.render:
        fr = ROOT / "artifacts" / "video_frames"
        fr.mkdir(parents=True, exist_ok=True)
        n = int(tl["total"] * FPS)
        todo = [i for i in range(n) if not (fr / f"f{i:06d}.jpg").exists()]
        chunks = [todo[k::a.workers] for k in range(a.workers)]
        with mp.Pool(a.workers) as pool:
            pool.map(render_chunk, [(c, tl, fr) for c in chunks if c])
        subprocess.run([sys.executable, str(HERE / "make_music.py"), "--seconds", str(tl["total"]), "--out", str(out / "music.wav")], check=True)
        base = ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(fr / "f%06d.jpg")]
        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart"]
        subprocess.run(base + ["-i", str(out / "music.wav")] + enc + ["-c:a", "aac", "-b:a", "192k", "-shortest", str(out / "demo_video.mp4")], check=True)
        subprocess.run(base + enc + ["-an", str(out / "demo_video_no_music.mp4")], check=True)
        print("written", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
