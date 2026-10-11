# T-420 short-window fine-tunes (Oct 10-11 2026): 0.5 / 0.75 / 1.0 s crops from T-387's checkpoint

For the low-delay live path (shorter enrolled-tier windows): T-374/T-387/T-391 trained on 2 s
crops; the live tier separates 1.0 s windows today and the low-delay task wants 0.5 / 0.75 s.
Code: `experiments/gpu/T-420/` (copied to `~/tmp/T-420/code/` on iLab). Everything runs on iLab
and needs no ssh link once submitted. Data and features stay on iLab. All times EDT.

## What changed against T-391
- `train_sep.py --crop-s S` (default 2.0 = the T-374..T-391 crop). Train and validation renders
  are S seconds long. The turn-taking hand-over point is drawn over the same fraction of the crop
  (0.3-0.7) and its overlap (0.2-0.6 s) is capped at 40 % of the crop (so 0.2 s at 0.5 s,
  0.3 s at 0.75 s). Checked on the iLab login node: at 2.0 s, 150 v3 val renders are bit-identical
  to T-391's `render`; at 0.5 / 0.75 / 1.0 s, 400 renders each have the right shapes, finite
  values, peak <= 1 and every v3 kind present.
- `eval_sep.py --window-s W --hop-s H` (default 1.0 / 0.5 = T-374..T-391). Same 19 rows and
  metrics; each checkpoint is scored at its own window with hop = W / 2.

## Jobs (submitted Oct 10 11:00-11:01 PM EDT; `bash T-420/code/submit.sh j0..j4` on ilab1)
Common: v3 recipe, `--order fixed`, `--sir -10 10`, init `~/tmp/T-374b/run_fixed/best.pt` (T-387's
best), lr 1e-5 cosine to 5 %, `--minutes 120 --steps 1000000 --val-every 400`, `--mem=48G`,
`--time=03:30:00`, no `--cpus-per-task`. Micro-batches hold 2-4 s of audio (T-391 S2 ran 2 x 2 s on
a 24 GB card) and every optimiser step sees about 4-4.5 s of audio.

| Job | Slurm id | GPU (squeue -d) | Crop / seed / batch | Expected finish (EDT) | Outputs (`~/tmp/` on iLab) |
|---|---|---|---|---|---|
| J1 `sh-T420-j1` | 450833 | rlab1, RTX 4500 Ada, IDX 1 | 0.5 s, seed 420, 4 x accum 2 | train about 1:02 AM, eval at 0.5 / 0.25 s about 1:10 AM | `T-420/run_j1/best.pt`, `T-420/run_j1/log.jsonl`, `T-420/eval_j1.json` |
| J2 `sh-T420-j2` | 450834 | ilab2, RTX A4500, IDX 5 | 0.75 s, seed 420, 3 x accum 2 | train about 1:02 AM, eval at 0.75 / 0.375 s about 1:10 AM | `T-420/run_j2/best.pt`, `T-420/eval_j2.json` |
| J3 `sh-T420-j3` | 450838 | rlab1, RTX 4500 Ada, IDX 2 | 0.5 s, seed 2, 4 x accum 2 | train about 1:02 AM, eval at 0.5 / 0.25 s about 1:10 AM | `T-420/run_j3/best.pt`, `T-420/eval_j3.json` |
| J4 `sh-T420-j4` | 450836 | rlab7, RTX A6000, IDX 1 | 1.0 s, seed 420, 4 x accum 1 | train about 1:02 AM, eval at 1.0 / 0.5 s about 1:10 AM | `T-420/run_j4/best.pt`, `T-420/eval_j4.json` |
| J0 `sh-T420-j0` | 450837 | rlab3, RTX A4000, IDX 0 | eval only: T-387 at 0.5 / 0.75 / 1.0 s, T-391 v3 at 0.5 / 0.75 s | about 11:35 PM (0.5 s done 11:07 PM) | `T-420/eval_t387_w{0.5,0.75,1.0}.json`, `T-420/eval_v3_w{0.5,0.75}.json` |

J3 was first submitted as 450835 on an A4500; it sat pending (`Resources`, ilab2's memory was
taken) and was cancelled and resubmitted on the RTX 4500 Ada as 450838. Slurm's hard stop
(`--time`) is 2:31 AM for J1-J4 and 2:01 AM for J0. T-391 v3 at 1.0 s is already in
`~/tmp/T-391/eval_all.json` (`v3/*`). Finish times are estimates (train 120 min from about
11:01 PM plus the step-0 and final validation, then one eval: J0's T-387 eval at 0.5 / 0.25 s took
6.5 min on the A4000, 11:01-11:07 PM). `best.pt` is saved at step 0, so each exists from the first minute; a job whose
training fails never runs its eval (`&&`). Logs: `T-420/run_j*.log`, `T-420/eval_*.log`,
`T-420/slurm-<jobid>.out`.

squeue at 11:01 PM:
```
   JOBID           NAME ST       TIME   TIME_LIMIT NODELIST(REASON) TRES_PER_NODE
  450838     sh-T420-j3  R       0:15      3:30:00 rlab1 gres/gpu:4500_ada:1
  450836     sh-T420-j4  R       0:24      3:30:00 rlab7 gres/gpu:a6000:1
  450837     sh-T420-j0  R       0:24      3:00:00 rlab3 gres/gpu:a4000:1
  450834     sh-T420-j2  R       0:41      3:30:00 ilab2 gres/gpu:a4500:1
  450833     sh-T420-j1  R       0:42      3:30:00 rlab1 gres/gpu:4500_ada:1
```

## Morning fetch (laptop-a)
The ControlMaster expires about 5:00 AM EDT; open a new one first (Duo prompt), then copy:
```
ssh -o ControlPath=~/.ssh/cm-ilab -o ControlMaster=auto -o ControlPersist=10h ap2772@ilab.cs.rutgers.edu true
I() { ssh -o ControlPath=~/.ssh/cm-ilab ap2772@ilab.cs.rutgers.edu "$@"; }
I 'squeue -u ap2772; sacct -j 450833,450834,450836,450837,450838 -X -o JobID,JobName,State,Elapsed,ExitCode; ls -la ~/tmp/T-420/*.json ~/tmp/T-420/run_*/best.pt'
I 'for r in j1 j2 j3 j4; do echo $r; grep done ~/tmp/T-420/run_$r/log.jsonl; done'
R=/Users/aryanpatel/CODE/selective-hearing/eval/results/T-420; M=$HEARING_MODELS_DIR/T-420; mkdir -p $R $M
scp -o ControlPath=~/.ssh/cm-ilab 'ap2772@ilab.cs.rutgers.edu:tmp/T-420/*.json' 'ap2772@ilab.cs.rutgers.edu:tmp/T-420/*.log' $R/
for r in j1 j2 j3 j4; do scp -o ControlPath=~/.ssh/cm-ilab ap2772@ilab.cs.rutgers.edu:tmp/T-420/run_$r/log.jsonl $R/log_$r.jsonl; done
for r in j1 j2 j3 j4; do scp -o ControlPath=~/.ssh/cm-ilab ap2772@ilab.cs.rutgers.edu:tmp/T-420/run_$r/best.pt $M/${r}_best.pt; done
```
Compare each `eval_j*.json` with `eval_t387_w<its window>.json` row by row (same rows, same window,
same hop; `fixed` and `ecapa` picks). The `best.pt` files hold Aryan's and Joseph's voices: keep
them under `$HEARING_MODELS_DIR`, never commit or publish.

## Measured at submission (accuracy only, step-0 validation of T-387's checkpoint)
Each job validates its init on 300 v3 val renders at its own crop (val set drawn from its seed, so
the sets differ). T-387 was trained on 2 s crops; its step-0 numbers fall with the crop:

| Val crop | Run | fixed-slot SI-SDRi | slots swapped | stranger in a silent slot |
|---|---|---|---|---|
| 2.0 s | T-391 C1 step 0 (seed 391) | 18.5 dB | 0.6 % | -34.7 dB |
| 1.0 s | J4 (seed 420) | 16.5 dB | 2.4 % | -30.6 dB |
| 0.75 s | J2 (seed 420) | 14.2 dB | 6.6 % | -29.2 dB |
| 0.5 s | J1 (seed 420) | 14.1 dB | 6.2 % | -32.0 dB |
| 0.5 s | J3 (seed 2) | 14.0 dB | 6.7 % | -25.4 dB |

All four trainers were past step 50 within 2 min with no out-of-memory (train loss -15.3 to -17.2).
About 11 min in, val already moved: J1 (0.5 s) step 800 14.1 -> 15.0 dB, stranger -32.0 -> -48.9 dB;
J3 (0.5 s) step 800 14.0 -> 15.6 dB, swapped 6.7 -> 2.2 %; J2 (0.75 s) step 400 14.2 -> 15.3 dB;
J4 (1.0 s) step 800 16.5 -> 17.1 dB.

## Baseline: T-387 at a 0.5 s window (J0, `eval_t387_w0.5.json`) vs 1.0 s (`T-391/eval_all.json`)
Held-out rows, fixed pick (slot of the target), wrong % / SI-SDRi dB; the four real-pair rows first.

| Row | 0.5 s window, 0.25 s hop | 1.0 s window, 0.5 s hop |
|---|---|---|
| aryan:joseph holdout-sir0-cafe | 12.1 % / 8.2 | 2.9 % / 11.0 |
| joseph:aryan holdout-sir0-cafe | 11.7 % / 8.0 | 2.8 % / 12.5 |
| aryan:joseph holdout-sir0-none | 6.1 % / 12.2 | 0.0 % / 15.8 |
| joseph:aryan holdout-sir0-none | 6.6 % / 10.6 | 4.2 % / 13.7 |
| aryan:joseph holdout-sir-6-cafe | 18.2 % / 9.9 | 7.2 % / 14.3 |
| joseph:aryan holdout-sir-6-cafe | 11.0 % / 11.8 | 2.8 % / 15.9 |
| aryan:joseph holdout-sir0-snr0 | 17.4 % / 5.5 | 11.6 % / 8.1 |
| joseph:aryan holdout-sir0-snr0 | 16.1 % / 5.4 | 2.8 % / 10.8 |

At 0.5 s the ECAPA pick is worse still (wrong 14.6-31.1 % on the team rows; 4.4-23.2 % at 1.0 s).
This is what J1/J3 (0.5 s) and J2 (0.75 s) have to win back; J0 also writes the 0.75 s and 1.0 s rows.

## Results
(not measured: every run was still training when this task ended; read the JSONs above in the morning)
