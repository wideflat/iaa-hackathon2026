# Handoff (maintained by hand; refresh START HERE at every milestone with the `record-progress` skill)

## START HERE (2026-10-18, ~14:10 UTC)

### State
- Branch `dev` (pushed through 5a9d006 + uncommitted hook/handoff files). Default branch on GitHub should become `main` (only the user can; I offered to fast-forward `main` to `dev`, awaiting OK).
- Task: pricing forecast per 2025 row; per-peril frequency x capped severity (`src/iaa/models/freqsev.py`, kinds gbm / glm / nn), trend + completeness offset, per-line rebalancing, `outlier_mult=5` winsorisation (D22).
- Local validation (folds A 2023 / B 2024, pooled D2, cap250k in brackets):
  - GBM leaves 4 + out5 (`fs_gbm_l4_out5`): A 0.1339 [0.1323], B 0.1323 [0.1333]
  - NN v1 (`fs_nn_v1`, CANN-style, 1 seed): A 0.1365 [0.1319], B 0.1296 [0.1309] (motor better, buildings/contents worse than GBM)
  - 50/50 blend GBM+NN (OOF): A 0.1388 [0.1356], B 0.1338 [0.1349]; paired vs GBM alone +0.0049 (CI +0.0024..+0.0082) on A, +0.0014 (CI +0.0003..+0.0025) on B
- Public LB so far: v1 0.12153, v2 0.12455, GLM 0.12133, blend GBM+NN 0.12587 (organisers' GLM 0.10563, top seen 0.12815). 
- Submitted 2026-10-18 14:08 UTC (at the user's request): `submissions/blend_gbm_nn_v1.csv` = 0.5 x `fs_gbm_v4_l4_out5.csv` + 0.5 x `fs_nn_v1.csv`. **Public score 0.12587** (read by the user from the Kaggle page; best so far; `check_submissions.py` gave auth errors). Submissions today (UTC): 6 of 10 including the blend.
- Not submitted: `fs_gbm_v4_l4_out5.csv`, `fs_gbm_v3_leaves4.csv`, `fs_nn_v1.csv` alone.

### Running / half done
- Helper session `session_01W8BsyyYBjof6NBX6rAiaWP` (started 14:48 UTC, own container): runs `fs_xgb_d3` then `fs_cat_d4` on folds A,B; it pushes compact OOF predictions (row_id, fold, pred) to branch `helper/oof-boosters` and messages me. When they arrive: `git fetch origin helper/oof-boosters`, read the parquet files into artifacts/ as `oof_<name>.parquet` by merging y/customer_id from `oof_fs_gbm_l4_out5.parquet`, then `scripts/blend_oof.py` against GBM/NN. It must not submit or push to dev.
- Local queue (scratchpad queue1.sh then queue2.sh, started 15:12 UTC): `nn_tweedie_v1` (direct Tweedie net, folds A,B) -> `fs_nn_wide` (hidden 384,192, dropout 0.2) -> `fs_nn_s3` (3 seeds). The sandbox restarted three times (14:40, 14:55, 15:09) and each time killed the running job; check `pgrep -af run_cv` and `uptime` first, relaunch what is missing (an OOF file in artifacts/ means a run finished).
- (older note) `fs_nn_s3` (3-seed NN, folds A,B) relaunched 14:44 UTC after a sandbox restart killed it; output `$SCRATCH/nns3.txt`, expect ~50 min.
- (restart command for the NN run if the container restarts again: `python scripts/run_cv.py --model fs_nn --folds A,B --threads 4 --outlier-mult 5 --nn-seeds 3 --name fs_nn_s3`; background jobs do not survive a sandbox restart.)
- torch was installed from PyPI (`pip install torch`; the PyTorch index is blocked by the proxy). Re-install after a new container.

**Time zone: report EDT (user's instruction 2026-10-18 12:20 EDT); timestamps above this line are UTC.**
**Restarts: the sandbox restarts at user turns and kills background jobs; use `scripts/queue_runs.sh` (resumable, checkpoints per fold and line) and relaunch it when `pgrep -af run_cv` is empty. Currently queued: fs_nn_wide, then fs_nn_s3.**

### 19:15 EDT 2026-10-19: target-construction review DONE (RESULTS 35, D44, DATA_QUALITY 11-15). Found and repaired 4 planted data defects (604 duplicate claims; 252 claims with placeholder loss date 2020-12-01; sum_insured x1000 on 0.2% of buildings/contents rows incl. 730 test rows; construction_type T->W recode + case variants) and added the expected recovery on open own_damage/theft claims. Repairs live in `src/iaa/cleaning.py` (loaders apply them; ablation switch `IAA_CLEAN_FIXES`; frame cache key includes them). **Local D2 effect is within noise** (all four: A -0.0007, B -0.0001; recoveries 0.0000), so v5 (built on the raw data, public 0.12790) is not obsolete. Not done: refit of the NN/GBM finals on the cleaned data and a cleaned v8 blend (about 1.5 h of compute, expected change ~0); do it only if a final candidate on the repaired baseline is wanted. `git` note: `docs/data` is partly gitignored; use `git add -f` for new files there.
Also checked and found clean: weather table, customers/addresses, policy ages vs dates of birth, lagged `customer_service_contacts` (no information).

### 17:40 EDT 2026-10-19: structural idea 1 (GBM-offset net) REJECTED (D42, RESULTS 34): alone = target GBM (A 0.1387 / B 0.1339), blend changes within +/-0.001, none positive on both folds. v5 still the best (public 0.12790). Helpers 1 and 3 are still running NN-config screens (wd 1e-3, dropout 0.25, 3 layers, lr 1e-3); collect via branches helper/oof-weights and helper/oof-entity.

### 16:50 EDT 2026-10-19: PLATEAU. v5 0.12790, v7 (3-seed finals) 0.12785, v4 0.12779, v6 0.12761. All blend variants tried since v5 (leaves 7, 3 seeds, history features, trend uplift, re-weighting, encodings for NN) are within +/-0.0003 locally and publicly. Recommended final selections on Kaggle (up to 2 may be selected): v5 and v7 (same recipe, 1 vs 3 seeds). Remaining ideas are all structural (GLM-offset CANN, residual NN on GBM, different architecture).

### 15:20 EDT 2026-10-19 (superseded): running: final 3-seed fits of TabM and TabM-PLE (scripts/final_nn_seeds.sh -> submissions/fs_nn_tabm4_s3.csv, fs_nn_tabm4_ple_s3.csv; variance reduction for the final blend; idempotent, rerun after a restart); helper 1: linear-leaf GBM; helper 3: GBM hyperparameter re-screen with target encoding. Residual scan and richer history: no gain (D38).

### 00:22 EDT: v6 (2025 inflation uplift) scored 0.12761 < v5 0.12790: REJECTED (D37). v5 (blend_gbmtarget_tabm_ple_perline_v5.csv, public 0.12790) remains the final candidate. Re-weighting did not beat v5 on both folds (D36). Public scores so far: v5 0.12790, v4 0.12779, v6 0.12761, target GBM 0.12720, ...

### 22:29 EDT: v5 SUBMITTED, public 0.12790 = NEW BEST (blend_gbmtarget_tabm_ple_perline_v5.csv). Next: NN with target-encoded categorical inputs (`--nn-cat-te`) is running.

### 22:25 EDT (superseded): GBM with target encoding alone scores 0.12720 (public). Candidate NOT yet submitted (awaiting the user's go-ahead): submissions/blend_gbmtarget_tabm_ple_perline_v5.csv (target GBM + mean(TabM, TabM-PLE), NN weights 0.7/0.4/0.3; local A 0.1442, B 0.1361). Numeric encodings tested: only PLE helps (D32). Local queue and helpers are idle.

### 21:13 EDT: previous best public 0.12779 = blend_gbm_tabm_ple_perline_v4.csv (GBM v4 + mean of TabM and TabM-PLE, per-line NN weights 0.7/0.4/0.3). Next idea (user): alternative categorical encodings (label / count) as separate GBMs for the blend.

### 19:56 EDT: previous best public 0.12718 = per-line-weight blend GBM v4 + TabM (blend_gbm_tabm4_perline_v3.csv, TabM weight 0.7 motor / 0.4 buildings / 0.3 contents). Submissions: 10 used on the 2026-10-18 UTC day (the 10th at 23:55 UTC); the daily limit probably resets at 00:00 UTC = 20:00 EDT. Helper 1's TabM k=8 run is still pending (not expected to matter).

### 18:20 EDT: PUBLIC SCORES - blend GBM+TabM 0.12679 (best), blend GBM+NN v1 0.12587, GBM v4 0.12548, TabM alone 0.12346. `scripts/check_submissions.py` now uses the REST endpoint (works). See RESULTS section 16.

### 18:15 EDT: at the user's request two component checks were also submitted: `fs_nn_tabm4.csv` (TabM alone) and `fs_gbm_v4_l4_out5.csv` (GBM v4 alone). 9 of 10 submissions used today (UTC day of 2026-10-18; check the reset time); 1 left, held for the next real candidate. Public scores to be read from the Kaggle page. These are real models fitted on 2021-24, not scaled probes; D21 (no probing) is not affected.

### 18:00 EDT status
IMPORTANT: background jobs only progress while a turn is active (the sandbox sleeps between turns: 2.4 h passed between turns with nothing running); keep a turn open with waiting loops for long jobs. Results of the parallel ideas: tail load, entity embeddings, recency, domain-shift: all rejected (D25-D27). TabM NN (k=4): best single model; GBM + TabM beats submitted blend on both folds (D28). SUBMITTED at the user's go-ahead (about 18:05 EDT; 7 of 10 today); public score pending / read it from the Kaggle page. File: `submissions/blend_gbm_tabm4_v2.csv` = 0.5 x fs_gbm_v4_l4_out5 + 0.5 x fs_nn_tabm4 (both fitted on 2021-24). `scripts/submit.py` returned an authentication error in the dry run at 17:58 EDT (same as check_submissions.py): if it persists, the user can upload the CSV by hand (artifacts are in the sandbox; send the file via SendUserFile).

### Parallel ideas started 14:45 EDT (user request) - code on dev (a6e1339)
Helpers (own containers, no submit/push to dev; predictions pushed to their branches, messages to me): `session_01P3N5TGee8xNV4QmCHwGn3s` recency 0.3/0.7 + domain-shift 0.5/1.0 (branch helper/oof-weights); `session_01T8JKohLCBD1QxVMf88yCM4` risk-dependent tail load lambda 0.5/1.0 (helper/oof-tail); `session_01Et9MuJaioBVYPQPdYYrNzL` entity embeddings NN (helper/oof-entity). Local queue: fs_nn_s3 -> fs_nn_tabm4 (TabM k=4). Compare each with `scripts/blend_multi.py` (GBM-type results against fs_gbm_l4_out5, NN-type results as replacement/addition to R = GBM + NN v1); fetch OOF files with `git show origin/<branch>:artifacts/oofpred_<name>.parquet`, merge y/customer_id from `oof_fs_gbm_l4_out5.parquet` by row_id+fold, save as artifacts/oof_<name>.parquet.

### Results since the blend (RESULTS section 10, D24)
XGBoost / CatBoost / direct Tweedie NN / GLM do not improve GBM + NN on both folds (rejected). Helper session finished (idle; helper/oof-boosters branch holds its predictions; can be archived). The sandbox restarts whenever my working directory changes (16:05 again): relaunch `scratchpad/queue3.sh` (fs_nn_wide then fs_nn_s3) if `pgrep -af run_cv` is empty and no OOF files `oof_fs_nn_wide`/`oof_fs_nn_s3` exist.

### Next steps (in order)
1. Read the blend's public score; record it in RESULTS.md and DECISIONS (new row: NN + blend adopted as candidate, evidence above).
2. NN improvements, judged on both folds: 3 seeds; hidden size / dropout / weight decay; then re-test the blend weight (use simple fixed weights; per-line weights only if both folds agree).
3. Other algorithms for diversity: CatBoost / XGBoost Tweedie or Poisson-Gamma, direct-Tweedie NN. Keep the Allstate lesson: diversity helps, but blend only what is validated on both folds.
4. Document: RESULTS.md section for NN and blend, AI_USAGE ledger, README table; add `--nn-*` flags to make_submission if a tuned config wins.
5. Non-score 60%: reviewer names (user to supply), verification matrix, caught-errors ledger, report/video prep, Part 2 kit.

### Open decisions for the user
- OK to fast-forward `main` to `dev`? (then delete branch `claude/brave-mayer-b3poow`)
- Final-selection rule is D20 (local validation + public score); blend weights and the final candidate are not chosen yet.

### Traps hit (do not repeat)
- Two heavy jobs at once -> OOM. NN fit ~7 GB RSS, GBM ~7 GB, GLM ~5.5 GB. `pgrep -f "^python scripts/run_cv.py"` (never `pkill -f` with a pattern that matches your own shell).
- Rebuild the frame cache once before parallel runs (race corrupted a parquet).
- `customer_service_contacts` leak; `y_true`/`y_all` must never be features.
- Test fixtures: a 99.9th-percentile cap with one 5M claim in ~1000 is itself huge; use cap_quantile 0.99 in such tests.
- Kaggle CLI unusable in the sandbox; use `scripts/submit.py` (dry run by default, `--confirm` submits).

## Pointers
- Experiment results: `docs/experiments/RESULTS.md`, log `experiments/log.jsonl`; decisions `DECISIONS.md` (D1-D22); narrative `DEVELOPMENT_LOG.md`.
- Transcript of earlier contexts: `/root/.claude/projects/-home-user-iaa-hackathon2026/*.jsonl` (path also in `artifacts/session_snapshot.md`).
