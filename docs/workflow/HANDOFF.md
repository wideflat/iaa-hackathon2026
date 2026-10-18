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
