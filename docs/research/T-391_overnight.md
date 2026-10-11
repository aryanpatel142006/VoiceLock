# T-391 overnight GPU runs (Oct 10-11 2026): fine-tune recipe v3, A100 chain, sbatch replicates

Code: `experiments/gpu/T-391/` (copied to `~/tmp/T-391/code/` on iLab). Everything runs on iLab and
needs no ssh link from the laptop once submitted. Data stays on iLab. All times EDT.

## Recipe v3 (`train_sep.py --recipe v3`), against T-374 / T-387 (`--recipe v1`)
| | v1 (T-374, T-387) | v3 (T-391) |
|---|---|---|
| train takes | profile-02..05 | profile-02..05 + profile-07 (val profile-06; holdout/same/profile-01 never read) |
| noise SNR | 0-15 dB, 10 % clean (`aj_clean` only) | 0-20 dB, 20 % of every non-noise kind clean |
| overlap | partial talker covers 25-100 % | 5-100 %, plus 12 % turn-taking items (hand-over with 0.2-0.6 s overlap) |
| level | -25..-5 dBFS | -35..-5 dBFS |
| stranger | none (babble only in the noise) | 10 %: a LibriSpeech voice 0-10 dB louder than the team voice, next to one team member (8 %) or alone (2 %); its slots' refs stay silent |
| lr | 5e-5 constant | 1e-5 cosine to 5 % over the run (from a fine-tuned init); 5e-5 cosine from base |
| SIR of the 2nd talker | T-374 -6..6, T-387 -10..10 | -10..10 |

Stranger voices are the split's own babble speakers (train speakers for train, val for val);
the test speakers (8297, 84) are kept for the eval's stranger rows, so a stranger heard at
eval was never trained on. Validation adds `stranger_slot_db` (energy in the slot that should
be silent over the mixture, dB).

`eval_sep.py` adds rows `holdout-sir-10-cafe`, `holdout-sir+6-cafe`, `holdout-sir0-snr0`,
`<target>:stranger holdout-sir-6-cafe` (stranger 6 dB louder; `<label>/slot<k>_absent` = the
absent team member's fixed slot energy over the mixture on stranger windows) and
`stranger-only cafe` (both slots absent).

## Jobs (submitted Oct 10 19:11 EDT)
| Run | Where | What | Expected finish (EDT) | Outputs (`~/tmp/` on iLab) |
|---|---|---|---|---|
| S1 `sh-T391-s1` 450733 | ilab3, RTX 4500 Ada | v1 (T-387 recipe), fixed, from base, lr 5e-5, batch 2 x accum 2, 170 min | train 22:03, eval about 22:20 | `T-391/run_s1/best.pt`, `T-391/run_s1/log.jsonl`, `T-391/eval_s1.json` |
| S2 `sh-T391-s2` 450734 | ilab3, RTX 4500 Ada | v3 seed 2 from T-374's fixed best.pt, lr 1e-5 cosine, batch 2 x accum 2, 170 min | train 22:03, eval about 22:20 | `T-391/run_s2/best.pt`, `T-391/eval_s2.json` |
| S3 `sh-T391-s3` 450735 | rlab7, RTX A6000 | v3 seed 4 from base, lr 5e-5 cosine, batch 1 x accum 4, 240 min | train 23:13, eval about 23:30 | `T-391/run_s3/best.pt`, `T-391/eval_s3.json` |
| chain (A100, job 435775) | rlab2 via `srun --overlap`, driver on ilab1 | waits for `T-374b/chain.done` (T-387 train + eval, about 20:30) | see below | `T-391/chain.log`, `T-391/chain.done` |
| C1 | A100 | v3 from the better of T-374 / T-387 (`pick_init.py`: team rows' fixed SI-SDRi - 0.1 x wrong %), lr 1e-5 cosine, batch 4, 110 min | about 22:25 | `T-391/run_v3/best.pt` (chosen init in `T-391/chain.log`) |
| C2 | A100 | eval base / t374 / t387 / v3 in one pass (19 rows each) | about 22:45 | `T-391/eval_all.json`, `T-391/eval_all.log` |
| C3 | A100 | per-window gate features from the v3 separator (2000 train + 400 val 2 s renders x 3 windows, holdout scenes, stranger scenes; about 8000 windows) | about 23:05 | `T-391/gate_features.npz` |
| C4 | A100 | v3 seed 3 from base, lr 5e-5 cosine, batch 4, 240 min, then its eval | train about 03:10, eval about 03:20 | `T-391/run_c4/best.pt`, `T-391/eval_c4.json` |

Finish times are estimates: the chain's start moves with T-387's end (if `T-374b/chain.done` never appears, the chain goes on at 21:45 and `pick_init.py` falls back to T-374's checkpoint). Do not overwrite `~/tmp/T-391/code/chain.sh` while the chain runs: bash reads a script as it goes. Logs: `T-391/run_*.log`,
`T-391/eval_*.log`, `T-391/slurm-<jobid>.out`. Each run's `log.jsonl` has the val curve; `best.pt` is
the lowest val loss (saved at step 0, so it exists from the first minutes).

## Morning fetch (laptop-a)
The ControlMaster expires about 05:00 EDT; open a new one first (Duo prompt), then copy:
```
ssh -o ControlPath=~/.ssh/cm-ilab -o ControlMaster=auto -o ControlPersist=10h ap2772@ilab.cs.rutgers.edu true
I() { ssh -o ControlPath=~/.ssh/cm-ilab ap2772@ilab.cs.rutgers.edu "$@"; }
I 'cat ~/tmp/T-391/chain.log; squeue -u ap2772; ls -la ~/tmp/T-391/*.json ~/tmp/T-391/*.npz ~/tmp/T-391/run_*/best.pt'
I 'for r in s1 s2 s3 v3 c4; do echo $r; grep done ~/tmp/T-391/run_$r/log.jsonl; done'
R=/Users/aryanpatel/CODE/selective-hearing/eval/results/T-391; M=$HEARING_MODELS_DIR/T-391; mkdir -p $R $M
scp -o ControlPath=~/.ssh/cm-ilab 'ap2772@ilab.cs.rutgers.edu:tmp/T-391/{eval_all,eval_s1,eval_s2,eval_s3,eval_c4}.json' 'ap2772@ilab.cs.rutgers.edu:tmp/T-391/gate_features.npz' $R/
scp -o ControlPath=~/.ssh/cm-ilab 'ap2772@ilab.cs.rutgers.edu:tmp/T-391/run_*/log.jsonl' $R/ 2>/dev/null  # same name: copy one at a time if you need every curve
for r in s1 s2 s3 v3 c4; do scp -o ControlPath=~/.ssh/cm-ilab ap2772@ilab.cs.rutgers.edu:tmp/T-391/run_$r/best.pt $M/${r}_best.pt; done
```
The `best.pt` files hold Aryan's and Joseph's voices: keep them under `$HEARING_MODELS_DIR`, never commit or publish.

## Results
(not measured: every run was still training when this task ended; read the JSONs above in the morning)

Measured at submission: S2's step-0 validation on the v3 val set (T-374's fixed best.pt) is
fixed-order SI-SDRi 16.6 dB, slots swapped 0.6 %, stranger in a silent slot -23.8 dB; S1's
(base) is -9.6 dB / 45 % (base has no slot order). Smoke run (450732, A4500, 53 steps from base): train and
the extended eval run end to end
(train 1.6 min, then eval 17 rows in about 13 min on the A4500). That smoke checkpoint is base after 53 steps at
lr 1e-5, so its stranger rows show the base behaviour v3 has to fix: with Aryan's holdout and a stranger 6 dB louder,
fixed slot 0 is the stranger in 88 % of target windows, and with the stranger alone in cafe one slot holds it at
-1.2 dB re the mixture (`slot0_absent`; the other slot -14.8 dB). The ECAPA pick still finds the target there (wrong 13 % / 7 %).
`dump_gate_features.py` wrote 781 windows (6 train + 4 val renders + every holdout scene) in about 6 min on the A4500.
