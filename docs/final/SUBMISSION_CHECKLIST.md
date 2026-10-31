# Final submission form: checklist (prepared 2026-10-31, after the deadline of 2026-10-30 23:59:59 EDT)

The form (Google Forms, "IAA AI Assisted Hackathon 2026 Final Submission") closed at the stated deadline. Whether a late entry is accepted is the organisers' decision; nothing here can be submitted by the assistant (the form needs the team's Google account and file uploads). The draft message at the end can be sent to the organisers.

All files are in `artifacts/final_submission/` (not in git: prediction files; the text files are copied to `docs/final/`). Rebuild with `python scripts/build_final_pack.py --team-name ... --members "..." --w2-private "..."`.

| Form field | What to give | Status |
|---|---|---|
| Email | the team's Google account email | team |
| Team Name in Kaggle | exactly as on the Kaggle leaderboard | team (the report line currently says `[TEAM NAME]`) |
| Team Members Actual Names | real names of all members | team (placeholders in the write-up and deck) |
| Week 1 Prediction File | `week1_prediction.csv` = `submissions/blend_v10_perline.csv`, 626,639 rows, sha256 `2223edb5502401f3f5b1833ae4281df5e3d43bdd60452f993d2ae5ec01caa8bd`, 17.8 MB | ready (matches the file selected on Kaggle in Week 1, PART1_FINAL.md) |
| Week 2 Prediction File | `week2_prediction.csv` = `submissions/p2_t10_ensemble_e2.csv`, 209,063 rows, sha256 `89b3f660f2343056ab9b98f7d591d9205016d0231aa13c0fb8cca846f3d27962`, 5.9 MB | ready, **but it must be the file that was selected on Kaggle**; alternatives in `submissions/`: t7 (`p2_t7_t6_buildings_k400.csv`), t11 (`p2_t11_ensemble_e3.csv`) |
| Video Demo Link | URL of the recorded video (YouTube unlisted / Drive / Loom, link viewable without sign-in) | **not recorded**: use `video_slides.pptx` (6 slides, speaker notes) and `docs/part2/VIDEO_SCRIPT.md` (4:30) |
| Project Write-Up File | `writeup.pdf` (or `writeup.docx`), 6 pages, 14,500 characters | ready except names and team name |
| Code Repository Link | `https://github.com/wideflat/iaa-hackathon2026` (branch `dev` holds the final code; `main` is the older stable state) | team to confirm the repository is readable by the judges (visibility or collaborator access) and whether `dev` is merged to `main` |

## Recording the video (about 45 minutes)
1. Fill the names and the Week 2 private result: `python scripts/build_final_pack.py --team-name "..." --members "..." --w2-private "..."` (rebuilds the deck and the write-up).
2. Open `artifacts/final_submission/video_slides.pptx`, present with the notes visible (`docs/part2/VIDEO_SCRIPT.md` has the timing: 0:30 result, 1:00 what changed, 1:00 method, 1:00 how we trust it, 0:45 AI in the workflow, 0:15 guardrails). Record screen and voice (Zoom / Loom / QuickTime), 4:30 target, 5:00 maximum.
3. Upload and set the link to viewable by anyone with the link; paste it into the form.

## Draft message to the organisers (to be sent by the team; nothing has been sent)
Subject: Late final-submission request, IAA AI Assisted Hackathon 2026

Hello, our team ([team name], Kaggle) completed both weeks' predictions and the write-up before the deadline of 30 October 2026, 11:59:59 PM EDT, but did not submit the Google form in time because the demo video was not finished. We now have the video, both prediction files, the write-up and the repository ready. Could you let us know whether a late form submission can still be accepted for judging? We understand if not, and we will accept whatever you decide. Thank you, [names].
