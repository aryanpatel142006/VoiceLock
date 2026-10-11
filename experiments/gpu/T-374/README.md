# T-374: MossFormer2_SS_16K fine-tuned on Aryan+Joseph mixtures (fixed order vs PIT)

Recipe: `docs/research/RESEARCH_oct10_adapt.md` (#1). Ran on Aryan's iLab job 435775 (rlab2,
A100-PCIE-40GB) next to the hearing-remote servers, under `~/tmp/T-374/` on iLab.

- `train_sep.py`: renders 2 s mixtures on the fly (55 % A+J with 25-100 % overlap at -6..+6 dB,
  15 % A only, 15 % J only, 5 % noise only, 10 % A+J clean; SNR 0-15 dB; -25..-5 dBFS) from the
  team takes split by take (train profile-02..05, val profile-06; holdout-01/02, same-01/02,
  profile-01/07 and the fixture/cafe takes are never read). Noise = DEMAND PCAFETER ch01
  (time-split train/val/test) + 0-2 LibriSpeech voices (speakers split train/val/test), so an
  outside talker belongs to neither slot. Loss: shared inference gain, then MixIT thresholded
  SNR (tau 1e-3) per slot, L0 for silent slots; `--order fixed` (slot 0 = Aryan) or `--order pit`.
  Adam, lr 5e-5, batch 4, bf16 autocast, best checkpoint by val loss.
- `eval_sep.py`: 1.0 s / 0.5 s windows over held-out scenes (see its docstring), ECAPA or
  fixed pick, wrong-speaker %, SI-SDRi, leak with bootstrap CIs and window counts.
- `table.py`: merges the eval JSONs and applies the ship rule (beats base on every row).

Deviation from the recipe: our own ~250-line loop instead of ClearerVoice's `train.sh`
(same network class and init checkpoint); on-the-fly rendering instead of 20 k written wavs.

```
# on iLab, in ~/tmp/T-374 (code/ = this dir, data/ = cafe/ch01.wav, babble/*.wav, grid/)
srun --jobid=435775 --overlap ~/hearing-remote/.venv/bin/python code/train_sep.py --order fixed \
  --data data --init ~/hearing-remote/models/MossFormer2_SS_16K/last_best_checkpoint.pt \
  --out run_fixed --minutes 27 --val-every 400 --batch 4
srun --jobid=435775 --overlap ~/hearing-remote/.venv/bin/python code/eval_sep.py \
  --ckpt fixed=run_fixed/best.pt --data data --grid data/grid \
  --ecapa ~/hearing-remote/models/spkrec-ecapa-voxceleb --out eval_fixed.json
```

Use a checkpoint on the Mac: `HEARING_SEP_CKPT=<path to best.pt>` (default unset = shipped
base). `HEARING_PICK=fixed` (default off) marks a fixed-order checkpoint; `hearing.tse.blind.fixed_slot(profile)`
gives the stream (aryan 0, joseph 1). The enrolled tier does not consume it yet (outside T-374's scope).

## Results (Oct 10 2026, accuracy only, offline 1.0 s / 0.5 s windows)

GPU: A100-PCIE-40GB (rlab2, job 435775). Fixed order: 1894 steps x batch 4, 27.4 GPU min, best val
(profile-06 mixtures) fixed-slot SI-SDRi +16.0 dB, slots swapped 0.5 % (base -9.6 dB / 43.9 %).
PIT: 1005 steps x batch 2 x accum 2 (batch 4 ran out of memory next to the servers), 15.4 GPU min.
Two batch-4 PIT attempts died at step 0 (OOM), about 1 GPU min. Total about 44 GPU min of training
plus about 6 min of eval. Full table with CIs: `$HEARING_RESULTS_DIR/T-374_table.md`; raw JSON
`T-374_eval_{base,fixed,pit}.json`.

Wrong-speaker % (target-active windows) / SI-SDRi dB, means; base picks with ECAPA:

| row | base | fixed ckpt, fixed pick | fixed ckpt, ECAPA pick | PIT ckpt, ECAPA pick |
|---|---|---|---|---|
| aryan:joseph holdout-sir0-cafe | 33.3 / -0.7 | 2.9 / +9.7 | 15.9 / +5.0 | 20.3 / +4.1 |
| joseph:aryan holdout-sir0-cafe | 36.1 / -3.7 | 2.8 / +11.4 | 13.9 / +7.0 | 22.2 / +4.7 |
| aryan:joseph holdout-sir0-none | 14.5 / +6.3 | 1.4 / +14.3 | 8.7 / +12.0 | 11.6 / +10.4 |
| joseph:aryan holdout-sir0-none | 19.4 / +4.3 | 2.8 / +12.6 | 9.7 / +10.7 | 12.5 / +7.8 |
| aryan:joseph holdout-sir-6-cafe | 37.7 / +1.4 | 14.5 / +9.5 | 29.0 / +5.5 | 36.2 / +3.6 |
| joseph:aryan holdout-sir-6-cafe | 41.7 / -0.4 | 5.6 / +13.4 | 18.1 / +9.6 | 16.7 / +8.5 |
| aryan:joseph same-sir0-cafe | 32.3 / -1.2 | 8.4 / +9.9 | 23.2 / +4.5 | 25.8 / +4.1 |
| joseph:aryan same-sir0-cafe | 37.2 / -0.0 | 4.5 / +11.1 | 16.7 / +6.4 | 19.9 / +4.5 |
| grid-s1:grid-s4 sir0-cafe | 17.6 / -0.8 | n/a | 11.8 / +3.1 | 9.8 / +4.4 |
| grid-s4:grid-s1 sir0-cafe | 7.5 / +5.1 | n/a | 62.3 / -16.9 | 50.9 / -11.1 |

Verdict: the fixed-order checkpoint beats base on every team row (with either pick); the PIT
twin beats base on every team row only with the ECAPA pick, and by less. Both fine-tunes break
the non-team pair grid-s4:grid-s1, so this is a **team (Aryan+Joseph) preset only**, behind
`HEARING_SEP_CKPT` (+ `HEARING_PICK=fixed`), default off. Leak windows (only the other talks)
are 2-7 per row because the takes are continuous speech; read leak as indicative only.
