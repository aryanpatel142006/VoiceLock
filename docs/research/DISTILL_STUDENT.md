# Distilling the fine-tuned MossFormer2 into a causal streaming student (T-424)

**Goal.** The live separator today is the T-387 fixed-order MossFormer2 fine-tune run on 1.0 s
windows (0.5 s hop): about 1.5 s from speech to ear. This overnight experiment trains small
causal separators with the same fixed slot order (slot 0 = Aryan, slot 1 = Joseph). They have
8–32 ms algorithmic latency and should run well under real time on the Mac. The fine-tuned
MossFormer2 is the teacher. **This is not wired into the live app.** The integration plan is at
the end; it only applies if the morning numbers justify it.

## Student
`experiments/gpu/T-424/student.py`: causal Conv-TasNet.
- Learned conv encoder: window `win`, hop `win/2`.
- TCN separator: dilated depthwise-separable blocks, every conv left-padded, a per-frame
  channel LayerNorm (no statistics over time).
- **Three softmax masks**: slot 0 Aryan, slot 1 Joseph, slot 2 the residual (noise and strangers).
- Transposed-conv overlap-add decoder, then the mixture-consistency projection
  (est_k += (mix − Σ est)/3), so the slots always add up to the input.

`net(x)` returns slots 0–1. Output sample n depends only on input up to n + win − 1 (tested),
so the algorithmic latency is one window.

`Streamer` runs the net chunk by chunk with carried conv state, an overlap-add tail and the
aligned input for the projection. Its output equals the whole-signal pass within 1e-5 for any
chunking (tested, `tests/test_distill_student_t424.py`). It computes the dilated depthwise
convs as sums of shifted slices: torch's CPU `slow_conv_dilated2d` makes one call per channel
and took 110–220 ms per 10 ms chunk (RTF 10–22) before this change.

| variant | window / hop | latency | params | TCN | receptive field |
|---|---|---|---|---|---|
| `tcn16` | 256 / 128 samples | 16 ms | 2.68 M | B128 H256, 8 blocks x 3 | 12.3 s |
| `tcn32` | 512 / 256 | 32 ms | 2.60 M | B128 H256, 6 blocks x 3 | 6.1 s |
| `tcn8s` | 128 / 64 | 8 ms | 1.08 M | B96 H192, 8 blocks x 2 | 4.1 s |

## Training (`distill.py`)
- **Data.** T-391's `--recipe v3` renderer (imported from `experiments/gpu/T-391/train_sep.py`),
  split by take exactly like T-374/T-391.
  - Train: profile-02..05 + 07. Val: profile-06.
  - Never read: holdout-01/02, same-01/02, profile-01 and the cafe fixture takes.
  - Mixtures: DEMAND café noise (time-split) and LibriSpeech babble/strangers (split by speaker).
  - SIR −10..+10 dB, 3 s items, batch 8.
  - Data stays on iLab (`~/hearing-data/team`, `~/tmp/T-374/data`).
- **Teacher.** `~/tmp/T-374b/run_fixed/best.pt` (T-387 fixed-order fine-tune), run non-causally
  on the whole 3 s item, bf16, no grad.
- **Loss.** On the student's raw output, fixed order:
  - T-374's per-slot thresholded SNR (τ = 1e-3, with an L0 term against a silent reference),
    over Aryan, Joseph and the residual mix − aryan − joseph;
  - plus α = 1 × a distillation term per team slot, 10·log10(|s_k − t_k|² / |mix|² + 1e-3),
    with the teacher on its shared gain to the mixture.
- **Two failed launches.** From scratch, both 2-slot sigmoid-mask versions of this loss
  collapsed within 1 k steps, so they were cancelled after a few minutes. (The losses work for
  the fine-tunes because those start from a working separator.) The residual slot and mixture
  consistency stop both collapses. The logs are in `~/tmp/T-424/old_gaincollapse/` and
  `old_silentcollapse/`.
  1. T-374's shared-gain loss led to an output orthogonal to the mixture: the gain went to 0
     and every silent slot sat at the −30 dB floor. Val loss froze at −8.0, fixed-slot SI-SDRi
     −58 dB.
  2. The same loss without the gain led to all-zero masks, because 0 dB on active slots beats
     an early crude mask. Fixed-slot SI-SDRi −27 dB.
- **Fixed version check.** At step 1000 the no-teacher canary is at +0.9 dB fixed-slot SI-SDRi
  with 19 % slots swapped on the profile-06 validation mixtures, and still improving.
- **Teacher throughput.** On an A4000 the teacher takes ~3 s per 8 × 3 s batch, against
  ~0.05 s for a student step. Teacher-labelled items therefore go into a FIFO GPU replay buffer
  (2048 items). The student takes `--k 12` steps per teacher batch: the fresh batch plus 11
  draws from the buffer, each with a random ±6 dB gain. Measured: ~5.6 student steps/s with
  the teacher, ~16 steps/s without it.
- **Schedule and checkpoints.** Adam, lr 1e-3, cosine decay to 2 % over `--minutes 270`.
  `best.pt` is chosen by val loss (team slots, 300 profile-06 items). Also saved: `last.pt`
  and `ckpt_<N>m.pt` every 30 min. A restarted job resumes from `last.pt`.

### Runs (submitted Oct 10 2026, 23:15 EDT, iLab rlab3, RTX A4000 16 GB each)
| job | name | variant | teacher | end of training (expected) |
|---|---|---|---|---|
| 450858 | `sh-T424-tcn16` | tcn16 | α = 1, k = 12 | ~03:50 EDT Oct 11 |
| 450859 | `sh-T424-tcn32` | tcn32 | α = 1, k = 12 | ~03:50 |
| 450860 | `sh-T424-tcn8s` | tcn8s | α = 1, k = 12 | ~03:50 |
| 450857 | `sh-T424-tcn16_nokd` | tcn16 | none (α = 0, ablation at equal GPU time: every step a fresh item, ~3x the steps) | ~03:50 |
| 450861 | `sh-T424-final` | eval of all four `best.pt` | starts when all four end (`afterany`) | ~04:10 |
| 450845 | `sh-T424-ref` | teacher @1.0 s and @0.5 s, base @1.0/0.5 s (ECAPA pick), plus an `early` column from a collapsed first-launch checkpoint (ignore it) | started 23:07 | ~23:40 |

Each training job is capped at 270 min of training plus validation, under `--time 05:15:00`.
Expected ends are submission + 270 min + validation overhead (~5 min).

## Eval (`eval_student.py`)
Same scenes and metrics as T-391's `eval_sep.py`:
- Rows: holdout SIR 0 with café / without noise, SIR −6/−10/+6, SNR 0, the same-sentence
  takes, stranger rows and stranger-only.
- Metrics: wrong-speaker % (with a fixed slot this is the slot-swap %), SI-SDRi, leak and
  `slot<k>_absent`, each with a 95 % bootstrap CI.

The student runs **frame-causally at its own latency**: one causal pass over the scene, equal to
`Streamer`. Its output is cut into the teacher's scoring grids (1.0 s / 0.5 s hop and
0.5 s / 0.25 s hop) and put on the mixture scale per window. The teacher and the base run
non-causally per window at 1.0 s and 0.5 s. `eval_*.md` next to the JSON is the means table
(wrong % / SI-SDRi dB / leak dB).

### Reference numbers (ref job 450845, A4000, Oct 10 2026)
Wrong-speaker % / SI-SDRi dB, means over windows:
- The teacher is the T-387 fixed-order fine-tune with fixed-slot pick.
- The base is shipped MossFormer2_SS_16K with ECAPA pick (profile-01).

| row | teacher @1.0 s | teacher @0.5 s | base @1.0 s | base @0.5 s |
|---|---|---|---|---|
| aryan:joseph holdout-sir0-cafe | 2.9 / +11.0 | 12.1 / +8.2 | 31.9 / −0.7 | 32.6 / −3.0 |
| joseph:aryan holdout-sir0-cafe | 2.8 / +12.5 | 11.7 / +8.0 | 36.1 / −3.7 | 40.2 / −8.0 |
| aryan:joseph holdout-sir0-none | 0.0 / +15.8 | 6.1 / +12.2 | 14.5 / +6.3 | 23.5 / +3.1 |
| joseph:aryan holdout-sir0-none | 4.2 / +13.7 | 6.6 / +10.6 | 19.4 / +4.3 | 21.2 / +0.6 |
| aryan:joseph same-sir0-cafe | 4.5 / +12.6 | 14.9 / +8.9 | 32.3 / −1.2 | 40.8 / −2.9 |
| joseph:aryan same-sir0-cafe | 2.6 / +12.7 | 9.9 / +8.7 | 37.2 / −0.0 | 36.4 / −3.5 |
| aryan:joseph holdout-sir0-snr0 | 11.6 / +8.1 | 17.4 / +5.5 | 52.2 / −4.5 | (in eval_ref.json) |
| joseph:aryan holdout-sir0-snr0 | 2.8 / +10.9 | 16.1 / +5.4 | 38.9 / −4.9 | (in eval_ref.json) |

Halving the teacher's window (≈ 0.75 s less delay) costs 3–10 points of wrong speaker and
about 3–4 dB. A student that matches teacher @0.5 s already moves the delay from ~1 s to ~30 ms.

**Ship rule for a student to go further.** On the four real-pair rows (holdout SIR 0,
café/none, both targets) it needs:
- fixed-slot wrong % ≤ teacher @0.5 s + 5 points, i.e. ≤ 17.1 / 16.7 / 11.1 / 11.6 %;
- SI-SDRi ≥ +6 dB.

## Mac speed (`rtf_mac.py`, laptop-a M4 Pro, torch on CPU/MPS, Oct 10 2026 23:32 EDT)
Run under `swarm-lock --exclusive`; load average 4.08 / 4.85 / 5.60 at the end.
- Streaming = `Streamer` fed 10 ms chunks (160 samples) over 10 s. Whole = one pass over the 10 s.
- The weights don't change speed, so these are random-init nets of the final 3-slot architecture.
- `cpu1` = one torch thread; `cpu` = default threads (10).
- Raw JSON: `$HEARING_RESULTS_DIR/T-424_rtf.json`.

| variant | device | stream RTF | ms / 10 ms chunk p50 / p95 / max | whole-pass RTF |
|---|---|---|---|---|
| tcn16 | cpu1 | 0.147 | 1.39 / 1.90 / 3.12 | 0.0077 |
| tcn16 | cpu | 0.252 | 1.92 / 4.64 / 20.46 | 0.0274 |
| tcn16 | mps | 0.694 | 6.81 / 8.29 / 10.48 | 0.0114 |
| tcn32 | cpu1 | 0.064 | 0.95 / 1.14 / 1.95 | 0.0035 |
| tcn32 | cpu | 0.072 | 1.09 / 1.27 / 1.59 | 0.0189 |
| tcn32 | mps | 0.619 | 8.98 / 11.21 / 16.70 | 0.0184 |
| tcn8s | cpu1 | 0.103 | 1.01 / 1.20 / 1.57 | 0.0089 |
| tcn8s | cpu | 0.249 | 2.44 / 2.86 / 3.17 | 0.0156 |
| tcn8s | mps | 0.753 | 8.05 / 8.85 / 9.52 | 0.0089 |

What this means:
- **Stream on one CPU thread.** It is 7–15x faster than real time with a p95 under 2 ms per
  chunk, and leaves the GPU to the other tiers.
- MPS is fine for whole passes but per-call bound when streaming: 6–9 ms per chunk, too close
  to the 10 ms budget.
- Default CPU threads add jitter on tiny ops (tcn16: 20 ms max).
- End-to-end delay is about window + one 10 ms chunk + ~2 ms compute: ~20–45 ms for the
  student, against ~1.5 s for the windowed teacher.

## Morning fetch (from the Mac; the ControlMaster may have expired, so log in again first)
```
ssh ap2772@ilab.cs.rutgers.edu 'squeue -u ap2772; cd ~/tmp/T-424; tail -n 3 run_*.log; cat eval_students.md eval_ref.md'
# metrics only (data and checkpoints stay on iLab unless a student wins):
scp 'ap2772@ilab.cs.rutgers.edu:tmp/T-424/eval_*.{json,md}' "$HEARING_RESULTS_DIR/"
scp 'ap2772@ilab.cs.rutgers.edu:tmp/T-424/run_*/log.jsonl' /tmp/   # or per run: run_tcn16/log.jsonl
# a winning student (10 MB) for the Mac RTF/listening check, into the gitignored models dir:
scp ap2772@ilab.cs.rutgers.edu:tmp/T-424/run_tcn16/best.pt "$HEARING_MODELS_DIR/T-424_tcn16.pt"
```
If the final eval job failed or ran on missing checkpoints, rerun it by hand on ilab1 with
`cd ~/tmp && bash T-424/code/T-424/submit.sh final <any finished job id>`. With a finished job
id the dependency is already satisfied.

## Integration plan (only if a student passes the ship rule)
1. **Package the code.** Move `student.py` (net + `Streamer`) into `hearing/tse/causal_student.py`.
   Add a `Stage` named `student`:
   - 16 kHz in, 10 ms chunks;
   - `latency_ms` = window, plus the chunk-alignment wait of at most one hop;
   - `HEARING_STUDENT_CKPT` selects the checkpoint, default unset = off;
   - lazy load through `hearing.models.load`;
   - a test that streaming output equals the whole-file output.
2. **Add a tier.** Use slot `fixed_slot(profile)` (aryan 0, joseph 1, as `hearing.tse.blind`).
   The output is target + (1 − s)·residual, where residual = mixture − target, through the
   existing mixer. Add the tier as `student` in the `tiers` ladder between `enrolled` and
   `gate`. It is team-only, like the fine-tune, because it was trained on two voices.
   Keep the T-407 learned gate optional on top.
3. **Measure live.** Run `hearing latency-test` and `demo_regress` with the student tier
   against the enrolled tier on the four real scenes. Record wrong-speaker time and listening
   notes. Delay should drop from ~1.5 s to ~30–60 ms end to end (DFN optional in front).
4. **Fallback.** The windowed teacher stays as Focus HQ. If the student swaps slots more than
   the teacher, use it only for the Natural fast path with the gate. For the demo story, wrong
   speaker at 30 ms beats right speaker at 1.5 s only if the swap rate is close to the
   teacher's.
