# T-387 · Longer fixed-order fine-tune, SIR -10..+10 dB

Continues T-374's fixed-order MossFormer2_SS_16K fine-tune (`models/T-374/fixed_best.pt`) on the iLab A100
(job 435775), batch 4, lr 5e-5, second talker's level sampled from -10..+10 dB (T-374: -6..+6) on the 55 % overlapped
Aryan+Joseph mixes. Code: `experiments/gpu/T-387/train_sep.py` (`--sir LO HI`). Evaluated with T-374's `eval_sep.py`,
same rows and seeds; label `fixed` = T-374 checkpoint, `fixed2` = this run. Table from `experiments/gpu/T-374/table.py`
(its trailing "SHIP" lines compare against a base run that is not in this comparison, so ignore them).

**GPU minutes: 51.9** (4000 steps, hit train_sep's default `--steps 4000` cap before the planned 90 min) plus ~2 min eval.
Run was not rerun for the full 90 min. Best step = last step (4000): val fixed-slot SI-SDRi on -10..+10 val mixes
15.62 dB (step 0) -> 16.39 (step 800) -> 17.36 (step 4000), slots swapped 0 %; the val curve was still rising.

New best checkpoint: iLab `~/tmp/T-374b/run_fixed/best.pt` (223 MB, never committed; not copied to the laptop).
Eval JSON: iLab `~/tmp/T-374b/eval_fixed2.json`.

| row | checkpoint/pick | wrong-speaker % [95% CI] (windows) | SI-SDRi dB | leak dB |
|---|---|---|---|---|
| aryan:joseph holdout-sir0-cafe | fixed/ecapa | +15.9 [+8.7, +24.6] (69) | +5.0 [+1.3, +8.2] (69) | -1.4 [-2.2, -0.7] (7) |
| aryan:joseph holdout-sir0-cafe | fixed/fixed | +2.9 [+0.0, +7.2] (69) | +9.7 [+7.4, +11.6] (69) | -65.6 [-68.5, -61.7] (7) |
| aryan:joseph holdout-sir0-cafe | fixed2/ecapa | +14.5 [+7.2, +23.2] (69) | +7.6 [+4.1, +10.5] (69) | -6.5 [-13.5, -0.7] (7) |
| aryan:joseph holdout-sir0-cafe | fixed2/fixed | +2.9 [+0.0, +7.2] (69) | +11.0 [+8.6, +13.0] (69) | -43.0 [-54.9, -30.9] (7) |
| joseph:aryan holdout-sir0-cafe | fixed/ecapa | +13.9 [+6.9, +22.2] (72) | +7.0 [+3.5, +10.1] (72) | -5.1 [-12.4, -1.1] (4) |
| joseph:aryan holdout-sir0-cafe | fixed/fixed | +2.8 [+0.0, +6.9] (72) | +11.4 [+10.1, +12.7] (72) | -40.2 [-65.8, -14.3] (4) |
| joseph:aryan holdout-sir0-cafe | fixed2/ecapa | +9.7 [+2.8, +16.7] (72) | +9.6 [+6.7, +12.0] (72) | -6.8 [-18.4, -0.8] (4) |
| joseph:aryan holdout-sir0-cafe | fixed2/fixed | +2.8 [+0.0, +6.9] (72) | +12.5 [+11.2, +13.8] (72) | -51.2 [-65.4, -34.3] (4) |
| aryan:joseph holdout-sir0-none | fixed/ecapa | +8.7 [+2.9, +15.9] (69) | +12.0 [+9.4, +14.5] (69) | -0.0 [-0.0, -0.0] (7) |
| aryan:joseph holdout-sir0-none | fixed/fixed | +1.4 [+0.0, +4.3] (69) | +14.3 [+12.6, +15.8] (69) | -72.5 [-75.5, -69.2] (7) |
| aryan:joseph holdout-sir0-none | fixed2/ecapa | +4.3 [+0.0, +8.7] (69) | +14.8 [+12.5, +16.9] (69) | -1.9 [-5.5, -0.0] (7) |
| aryan:joseph holdout-sir0-none | fixed2/fixed | +0.0 [+0.0, +0.0] (69) | +15.8 [+14.0, +17.5] (69) | -57.8 [-71.2, -37.3] (7) |
| joseph:aryan holdout-sir0-none | fixed/ecapa | +9.7 [+2.8, +16.7] (72) | +10.7 [+7.3, +13.4] (72) | -0.0 [-0.0, -0.0] (4) |
| joseph:aryan holdout-sir0-none | fixed/fixed | +2.8 [+0.0, +6.9] (72) | +12.6 [+10.0, +14.8] (72) | -75.8 [-78.7, -74.0] (4) |
| joseph:aryan holdout-sir0-none | fixed2/ecapa | +11.1 [+4.2, +19.4] (72) | +11.6 [+7.6, +14.7] (72) | -17.0 [-50.4, -0.0] (4) |
| joseph:aryan holdout-sir0-none | fixed2/fixed | +4.2 [+0.0, +9.7] (72) | +13.7 [+11.1, +16.2] (72) | -51.5 [-66.8, -23.0] (4) |
| aryan:joseph holdout-sir-6-cafe | fixed/ecapa | +29.0 [+18.8, +39.1] (69) | +5.5 [+2.0, +8.9] (69) | -9.8 [-28.8, -0.2] (7) |
| aryan:joseph holdout-sir-6-cafe | fixed/fixed | +14.5 [+7.2, +23.2] (69) | +9.5 [+6.4, +12.2] (69) | -71.0 [-74.0, -67.9] (7) |
| aryan:joseph holdout-sir-6-cafe | fixed2/ecapa | +21.7 [+13.0, +31.9] (69) | +10.0 [+6.2, +13.4] (69) | -8.4 [-17.8, -0.2] (7) |
| aryan:joseph holdout-sir-6-cafe | fixed2/fixed | +7.2 [+1.4, +14.5] (69) | +14.3 [+11.5, +16.6] (69) | -58.5 [-71.5, -45.0] (7) |
| joseph:aryan holdout-sir-6-cafe | fixed/ecapa | +18.1 [+9.7, +27.8] (72) | +9.6 [+5.7, +12.8] (72) | -0.3 [-0.4, -0.1] (4) |
| joseph:aryan holdout-sir-6-cafe | fixed/fixed | +5.6 [+1.4, +11.2] (72) | +13.4 [+10.5, +15.6] (72) | -71.4 [-74.8, -66.7] (4) |
| joseph:aryan holdout-sir-6-cafe | fixed2/ecapa | +13.9 [+6.9, +22.2] (72) | +11.9 [+8.0, +14.9] (72) | -18.3 [-54.4, -0.2] (4) |
| joseph:aryan holdout-sir-6-cafe | fixed2/fixed | +2.8 [+0.0, +6.9] (72) | +15.9 [+13.8, +17.8] (72) | -62.9 [-71.0, -53.6] (4) |
| aryan:joseph same-sir0-cafe | fixed/ecapa | +23.2 [+16.8, +29.7] (155) | +4.5 [+1.9, +6.8] (155) | -1.2 [-1.3, -1.0] (3) |
| aryan:joseph same-sir0-cafe | fixed/fixed | +8.4 [+3.9, +12.9] (155) | +9.9 [+8.5, +11.4] (155) | -51.2 [-67.2, -26.7] (3) |
| aryan:joseph same-sir0-cafe | fixed2/ecapa | +13.6 [+8.4, +18.7] (155) | +9.2 [+7.2, +11.1] (155) | -1.2 [-1.3, -1.0] (3) |
| aryan:joseph same-sir0-cafe | fixed2/fixed | +4.5 [+1.3, +8.4] (155) | +12.6 [+11.6, +13.6] (155) | -36.9 [-63.5, -22.3] (3) |
| joseph:aryan same-sir0-cafe | fixed/ecapa | +16.7 [+11.5, +23.1] (156) | +6.4 [+4.1, +8.6] (156) | -0.8 [-1.0, -0.5] (2) |
| joseph:aryan same-sir0-cafe | fixed/fixed | +4.5 [+1.9, +7.7] (156) | +11.1 [+10.1, +12.0] (156) | -52.3 [-70.2, -34.4] (2) |
| joseph:aryan same-sir0-cafe | fixed2/ecapa | +9.6 [+5.8, +14.7] (156) | +9.7 [+7.6, +11.5] (156) | -0.8 [-1.0, -0.6] (2) |
| joseph:aryan same-sir0-cafe | fixed2/fixed | +2.6 [+0.6, +5.1] (156) | +12.7 [+11.9, +13.4] (156) | -49.7 [-68.0, -31.5] (2) |
| grid-s1:grid-s4 sir0-cafe | fixed/ecapa | +11.8 [+3.9, +21.6] (51) | +3.1 [-1.0, +7.1] (51) | -36.6 [-55.7, -17.5] (5) |
| grid-s1:grid-s4 sir0-cafe | fixed2/ecapa | +15.7 [+7.8, +25.5] (51) | -1.1 [-6.3, +3.8] (51) | -42.5 [-54.7, -27.9] (5) |
| grid-s4:grid-s1 sir0-cafe | fixed/ecapa | +62.3 [+49.1, +75.5] (53) | -16.9 [-20.7, -13.2] (53) | -38.3 [-41.0, -34.0] (3) |
| grid-s4:grid-s1 sir0-cafe | fixed2/ecapa | +81.1 [+69.8, +90.6] (53) | -29.2 [-33.1, -25.5] (53) | -45.8 [-64.8, -19.3] (3) |


## Verdict

- **Team rows (Aryan+Joseph, fixed pick):** SI-SDRi is higher in all 8 rows (+0.6 to +4.8 dB; the CIs overlap in several
  rows, the clear gains are sir-6-cafe +4.8 / +2.5 dB and same-sir0-cafe +2.7 / +1.6 dB). Wrong-speaker % is lower or equal in 7 of 8
  rows (sir-6-cafe 14.5 -> 7.2 and 5.6 -> 2.8; same-sir0-cafe 8.4 -> 4.5); the one rise is joseph:aryan holdout-sir0-none
  2.8 -> 4.2 % (inside the CIs). With the ECAPA pick it is mostly better too (same-sir0-cafe 23.2 -> 13.6 and 16.7 -> 9.6).
- **Generality rows (grid speakers, ECAPA pick) drop:** grid-s4:grid-s1 SI-SDRi -16.9 -> -29.2 dB, wrong-speaker 62 -> 81 %;
  grid-s1:grid-s4 +3.1 -> -1.1 dB. T-374's ship rule requires generality rows not to drop, so by the letter it does not pass.
- **Recommendation:** replace T-374's checkpoint for the Aryan+Joseph fixed slot (`HEARING_PICK=fixed`), where it improves
  every team row, but only there: do not use it for other speakers. Leak (-43..-63 dB with the fixed pick) is still strongly
  negative but a few dB higher than T-374's in the none/cafe rows; not checked by ear.
