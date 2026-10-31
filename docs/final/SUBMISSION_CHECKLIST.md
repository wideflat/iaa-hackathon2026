# Final submission form: checklist (prepared 2026-10-31, after the deadline of 2026-10-30 23:59:59 EDT)

The form (Google Forms, "IAA AI Assisted Hackathon 2026 Final Submission") closed at the stated deadline. Whether a late entry is accepted is the organisers' decision; nothing here can be submitted by the assistant (the form needs the team's Google account and file uploads). The draft message at the end can be sent to the organisers.

All files are in `artifacts/final_submission/` (not in git: prediction files; the text files are copied to `docs/final/`). Rebuild with `python scripts/build_final_pack.py --team-name ... --members "..." --w2-private "..."`.

| Form field | What to give | Status |
|---|---|---|
| Email | the team's Google account email | team |
| Team Name in Kaggle | Kohei Kudo (as given by the team lead; check it is spelled exactly as on the Kaggle leaderboard) | ready |
| Team Members Actual Names | Kohei Kudo (sole member) | ready |
| Week 1 Prediction File | `week1_prediction.csv` = `submissions/blend_v10_perline.csv`, 626,639 rows, sha256 `2223edb5502401f3f5b1833ae4281df5e3d43bdd60452f993d2ae5ec01caa8bd`, 17.8 MB | ready (matches the file selected on Kaggle in Week 1, PART1_FINAL.md) |
| Week 2 Prediction File | `week2_prediction.csv` = `submissions/p2_t11_ensemble_e3.csv` (the file selected on Kaggle, confirmed by the team lead), 209,063 rows, sha256 `5013e6c952154dd8494ee68f52cec3753152416450a39d764a45e73eca57e8cb`, 5.9 MB | ready |
| Video Demo Link | URL of the video (YouTube unlisted / Drive, link viewable without sign-in) | **video built** (4:09, 1080p, original music, captions): `artifacts/final_submission/demo_video.mp4`; upload it and paste the link. Optional: record your own voice-over on `demo_video_no_music.mp4` following `voiceover_script.md` (timecoded) |
| Project Write-Up File | `writeup.pdf` (or `writeup.docx`), 6 pages; covers both parts; names and final file (t11) filled in; Week 1 rank 3 (known at the time), no Week 2 private result | ready |
| Code Repository Link | `https://github.com/wideflat/iaa-hackathon2026` (default branch `main` after the release PR of 2026-10-31; `dev` is the same code) | team to confirm the repository is readable by the judges (visibility or collaborator access) |

## Recording the video (about 45 minutes)
1. The deck and the write-up already carry the name and the ranks. Rebuild after any change: `python scripts/build_final_pack.py --team-name "Kohei Kudo" --members "Kohei Kudo (sole member)"`.
2. Open `artifacts/final_submission/video_slides.pptx`, present with the notes visible (`docs/part2/VIDEO_SCRIPT.md` has the timing: 0:30 result, 1:00 what changed, 1:00 method, 1:00 how we trust it, 0:45 AI in the workflow, 0:15 guardrails). Record screen and voice (Zoom / Loom / QuickTime), 4:30 target, 5:00 maximum.
3. Upload and set the link to viewable by anyone with the link; paste it into the form.

## Draft message to the organisers (to be sent by the team; nothing has been sent)
Subject: Late final-submission request, IAA AI Assisted Hackathon 2026

Hello, our team ([team name], Kaggle) completed both weeks' predictions and the write-up before the deadline of 30 October 2026, 11:59:59 PM EDT, but did not submit the Google form in time because the demo video was not finished. We now have the video, both prediction files, the write-up and the repository ready. Could you let us know whether a late form submission can still be accepted for judging? We understand if not, and we will accept whatever you decide. Thank you, [names].

## What the competition page says about the video (Kaggle, Part 2 'Submissions' page, read 2026-10-31)
* Form: https://forms.gle/8TYHcYXYqdd6ue1X6; deadline Oct 30 2026, 11:59:59 PM EDT.
* Video: "A short video demo explaining your approach, results, AI-assisted workflow, and how your solution changed when the new market data was introduced. Only first 5 minutes will be graded." Nothing is said about voice-over, language, captions or format.
* "AI-assisted review tools will be used as a first layer to screen videos, documentation, and code." The final deliverables "must cover both parts" (rules page).
* Recommendation: record a voice-over (narrate the slides live while screen-recording): a video that must explain approach and results is judged on explanation, and a spoken transcript is also what an AI screening tool can read. Add captions (auto-generated is enough) or paste the narration from `docs/part2/VIDEO_SCRIPT.md` into the video description. Keep it to 4:30; cover Week 1 in the first 30 seconds as well as Week 2.

## Demo video and slide deck (rebuilt 2026-10-31)
* `demo_video.mp4`: 4:44, 1920x1080, 30 fps, original generated music, burned-in captions of the narration; `demo_video_no_music.mp4` is the same without music for your own voice-over; `captions.srt` and `voiceover_script.md` (timecoded, about 153 words per minute) match it. Order: the general approach and result (one engine, two markets; Week 1 rank 3, Week 2 public score 0.16458; keep what transfers, re-measure what does not, test both ways; Claude Code, remote control from the phone and voice stated explicitly), what changed, method, **the final ensemble structure** (16 members in three groups with their weights), validation, AI in the workflow, the phone example with **your two real screen recordings** (Claude app = remote control of a Claude Code session, voice dictation, Claude working), guardrails and uncertainty.
* No private leaderboard result of Week 2 appears anywhere in the video, deck or write-up (it did not exist at submission time); Week 1 rank 3 was final before Week 2 started.
* `demo_slides.pptx` / `demo_slides.pdf`: 9 slides in the same design (speaker notes = narration).
* Everything is generated from code: `python scripts/video/build_video.py --render`, `python scripts/video/build_slides.py`; text in `scripts/video/narration.json`, design in `scripts/video/scene.html`, music in `scripts/video/make_music.py`. The two screen recordings live in `artifacts/video_footage/` (clip_a.mp4 keyboard dictation, clip_b.mp4 in-app voice input; not in git).
* There is no synthetic voice: installing a text-to-speech package was not permitted in this session, so the narration is captions plus the script for your own voice.

## One-liners (added 2026-10-31)
* Every video scene and slide carries a one-line storyline next to the chapter chip (`oneliner` in `scripts/video/narration.json`); every section of the write-up starts with `*In one line: ...*` (docs/part2/REPORT.md). The write-up is now 7 pages (the first 5 are graded: sections 1 to 5 plus the start of section 6).
