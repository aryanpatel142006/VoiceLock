# T-280: Pitch-conditioned frozen ECAPA metric

Offline research only. No server, Stage, profile, or routing contract changes.
`eval/pitch_conditioned_head.py` fuses a unit-normalized frozen 192-dimensional
ECAPA vector with log-F0 and log-spectral-centroid features, standardized using
training data only. The head is Linear(194,64), ReLU, Linear(64,32), L2-normalize.
Full-batch supervised cosine loss pulls same-speaker pairs together and penalizes
cross-speaker cosine above 0.1. AdamW runs 120 epochs at 0.001, seed 280.

## Reproduction

Create a JSON list with four or more records, for example:

```json
[
  {"split":"train", "speaker":"aryan", "path":"/absolute/aryan-profile.wav"},
  {"split":"train", "speaker":"joseph", "path":"/absolute/joseph-profile.wav"},
  {"split":"test", "speaker":"aryan", "path":"/absolute/aryan-fixture.wav"},
  {"split":"test", "speaker":"joseph", "path":"/absolute/joseph-fixture.wav"}
]
```

```sh
export HEARING_MODELS_DIR=/Users/joeroche/Developer/selective-hearing/models
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m eval.pitch_conditioned_head \
  /tmp/t280-manifest.json --crop-s 0.5 --epochs 120 --gains-db -12 0 \
  --output /Users/joeroche/Developer/selective-hearing/eval/results/T-280/pitch-head.json
.venv/bin/python -m pytest tests/test_pitch_conditioned_head.py -q
```

Uses the existing local-only `hearing.models.load` ECAPA adapter, selected device,
no downloads and no silent mock substitution. JSON includes resolved paths,
decoded/resampled audio hashes, host, device, timings, per-speaker errors and
margins. A `.pt` state dictionary beside it stores the head and feature statistics;
instantiate `PitchConditionedHead` then `load_state_dict(torch.load(path,
weights_only=True))`. No audio or weights are committed. CLI manifest paths should
be absolute; output belongs to the shared results directory.

WAVs must be finite mono in [-1,1]; non-16 kHz input is explicitly resampled.
Repeated paths or decoded audio hashes are rejected even under different names.
File-level splits happen before nonoverlapping complete crops; tails are dropped,
near-silence RMS <1e-5 is omitted. Each file needs at least two retained crops.
The split check detects identical copies, not partial copies or shared recording
sessions: users must supply genuinely independent source recordings.

## Acoustic features

40 ms frames / 20 ms hop at 16 kHz. F0 uses overlap-energy-normalized
FFT autocorrelation, local peak selection over 65–400 Hz, periodicity >=0.6,
and median voiced F0. Missing voiced F0 is encoded as zero. Spectral centroid is
the median frame magnitude-weighted frequency. Whole-crop peak normalization
makes both features gain invariant without using volume as identity. Digital
amplification that would clip is rejected. Defaults include +6 dB; explicitly
choose safe gains for a given recording.

Centroid is a spectral-envelope proxy, **not a formant estimator**. Autocorrelation
can make octave errors; unvoiced/noisy crops can provide misleading side features.
This is a two-feature R&D head, not a trained pitch model or causal streaming Stage.

## Measured real-audio checkpoint (2026-10-10)

MacBook-Air.local, CPU ECAPA and CPU head, one thread. Input root:
`/Users/joeroche/Developer/selective-hearing/eval/fixtures/generated/team/`.
Train: Aryan `aryan/profile-01.wav` (50 s), Joseph
`joseph/profile-01-20261009-1353.wav` (15.7 s).
Holdout: Aryan `aryan/fixture-01.wav` (58.692625 s), Joseph
`joseph/fixture-01-20261009-1353.wav` (48.15 s).
All four decoded-audio hashes differ; source-session independence is unverified.

Nearest normalized training-speaker centroid classification, identical test crops:

| Digital gain | Speaker | Test crops | Raw ECAPA errors | Fused head errors |
|---|---|---:|---:|---:|
| -12 dB | Aryan | 117 | 3/117 (2.56%) | 0/117 (0%) |
| -12 dB | Joseph | 96 | 12/96 (12.5%) | 6/96 (6.25%) |
| 0 dB | Aryan | 117 | 3/117 (2.56%) | 0/117 (0%) |
| 0 dB | Joseph | 96 | 12/96 (12.5%) | 6/96 (6.25%) |

Aggregate errors fall from 15/213 (7.04%) to 6/213 (2.82%) at each gain.
Gain conditions reuse the same recordings; they are not 426 independent examples.
Feature + ECAPA extraction p50 **23.05 ms**, p95 **24.10 ms**, median RTF **0.0461**
on 0.5 s crops, excluding model loading. Batched head + score compute
**0.00076–0.00081 ms/crop** (amortized, not single-crop latency).
Head training **64.52 ms**, loss 0.8711 → 0.002683. Audio acquisition still requires
500 ms; compute timing alone is not total streaming latency.

Raw artifacts: shared `eval/results/T-280/pitch-head.json` and `pitch-head.pt`.
The JSON records full hashes and paths. This result demonstrates digital gain
stability on clean held-out files. It does **not** establish room-volume/noise
robustness, unknown-speaker false accepts, separation quality, or live usefulness.
SI-SDRi, WER and listening results are not applicable to an embedding-only output.
No embedding-only learned-head ablation was run, so gain over raw ECAPA cannot be
attributed specifically to pitch/centroid rather than metric learning.
Next gate: session-disjoint same-mic recordings, unknown speakers, noisy separated
streams and an identical learned-head ablation before any runtime promotion.

## Model registry row for owning task

| Model | Role | Source / licence | Size / limits |
|---|---|---|---|
| T-280 pitch-conditioned head (194→64→32) | Offline Aryan/Joseph metric on frozen ECAPA | Own code, repository licence; team audio, consent provenance inherited/unverified here; existing ECAPA Apache-2.0 | 14,560 trainable parameters; no distributed checkpoint; closed-set clean-file evidence only |

`docs/MODELS.md` and `docs/BENCHMARKS.md` are outside this task's scope; their owner
can copy the row and measured checkpoint above. No dependencies were added.
