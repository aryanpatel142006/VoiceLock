# AFW static vs stage-2 held-out evaluation (T-398, Muse B)

Status: **PROVISIONAL, no result yet.** The job is queued and has not run. Nothing here is reviewed. Codex B must review independently before anything merges.

## Submission (as relayed by the orchestrator, not re-verified from this host)
- sbatch ID: **450747**, job name `T-398-afw-heldout`, state PENDING at 19:43 EDT.
- Dependency: `afterok:450723` (AFW stage-2 training). It never starts if 450723 fails.
- Time limits: 33-minute self-limit in the script, Slurm `--time=1:00:00` hard cap.
- Baseline: frozen stage-1 static checkpoint `/common/users/jdr317/afw-runs/stage1-static/best.pt` (finished 18:38 EDT).
- Evaluation: static vs stage-2 on 200 held-out streaming scenarios, seed 9999, held-out speakers (every-10th slice). Conditions match stage-2 training: absence 0/1/2/4 s, TMR ±5 dB, channel_shift=-2.
- Remote output (unique, not shared with T-393): `~/jobs/T-398/` containing `run.json` (sha256 provenance), `rows.jsonl`, `summary.json`, `t398-<jid>.out`.
- Evidence branch: `research/run-450747` in the sister research repo.

## Verification gap
This worktree's host (Aryans-MacBook-Pro) got `Permission denied` on `ssh ilab` (BatchMode, no credentials). So I could not confirm the queue state, the output directory, or the job ID. Stage-2 job 450723 was reported as having about 11 h left at 19:43 EDT, so no output exists yet.

## Results
(not measured: job 450747 has not started; waiting on 450723)

| condition | static SI-SNRi | stage-2 SI-SNRi | latency static | latency stage-2 |
|---|---|---|---|---|
| (pending) | | | | |

## Next steps for a later watch run
1. `squeue -j 450723,450747`, then `sacct -j 450747`. On a failed dependency, preserve the logs and report the failure.
2. Check that `~/jobs/T-398/run.json` has non-empty sha256 values, then compare `summary.json` per condition.
3. Commit the result or the failure evidence to `research/run-450747` in the research repo. Do not merge before independent review.
