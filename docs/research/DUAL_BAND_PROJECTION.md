# T-283: dual-band speaker metric experiment

Status: working offline prototype, **not validated distance invariance**. No live
stages, thresholds, profiles, or selection defaults change. T-246's theory and
synthetic ablations motivate frozen-encoder metric learning and warn that lower
training loss can coexist with unacceptable target-absent false accepts. Its
reported 52% FAR is synthetic evidence, not team or room performance.

## Signal and embedding interface

`DualBandSplitter.process(audio)` accepts mono finite samples in [-1,1], explicitly
at 16 kHz, and returns two float32 arrays of the same length. The 129-tap Hamming
FIR low-pass has cutoff 2 kHz and 64 samples (4 ms) group delay. High band is
`delayed_raw - low`: approximately 2–8 kHz, with a finite transition at the split.
Both outputs sum to delayed raw. FIR state survives calls; reset at an independent
utterance. Empty chunks are allowed. Output ringing can exceed input bounds;
there is no clipping that would break reconstruction.

`extract_band_embeddings(audio, encoder)` starts fresh state, flushes 64 zero
samples and trims the shared delay. It calls an injected frozen encoder twice,
then validates and unit-normalizes the embeddings, returning `[2,D]`.
The encoder receives aligned crops with the original length. For actual ECAPA
research, load once using the existing `hearing.tse.enrolled._load_encoder()`
adapter and pass `lambda band: _unit_embedding(backend, band)`; caller is
responsible for availability and explicit resampling with
`hearing.audio.io.resample`. No encoder is loaded or downloaded by this module.
These private helpers can change; no new live consumer depends on them.

This is an offline utility rather than a registered Stage: the two-output band
interface and utterance embedding do not implement the single-output AudioChunk
Stage contract. Streaming equivalence is tested for the signal splitter. The
projection requires an utterance's cached embeddings and is not a 10 ms pipeline.

## Learned head and training

`DualBandProjection(192,32)` has two bias-free linear projections and a learned
`Linear(384,2)` softmax gate over normalized low/high embeddings. The weighted
sum is L2-normalized. It returns `(metric_vector, gate_weights)`. There are 13,058
trainable parameters; backbone weights remain frozen. A zero fused vector stays
zero and must be rejected, not interpreted as a valid speaker match.

`train_head` detaches cached embeddings, moves them to the head's device, and
optimizes the branches and gate with Adam. Full-batch pair loss pulls distinct
same-speaker crops toward cosine 1 and penalizes different-speaker cosine above
0.2. Each speaker needs at least two crops. The configurable epochs default to
100 and learning rate to 0.001. Quadratic pair storage makes this a small-cache
prototype; do not feed an unbounded corpus. The returned history includes the
loss before training and after the final update. The 0.2 training margin is not
an acceptance threshold.

Train on an explicitly prepared local NPZ cache with numeric arrays `low`, `high`
(`[N,192]`) and integer `labels` (`[N]`), with no pickled objects:

```sh
.venv/bin/python -m eval.dual_band_projection /tmp/train-bands.npz \
  --epochs 100 --checkpoint /tmp/dual-band.pt --seed 42
```

CLI device selection uses `hearing.device.pick()`. It writes a state dictionary
plus dimensions, sample rate, cutoff and seed; reconstruct the head with the
recorded dimensions and load its `state_dict`. Keep weights and caches outside
Git. JSON output explicitly labels the loss as `training_only`. No automatic
synthetic fallback is supplied. Data provenance and split manifests must be
retained alongside the cache by the evaluation owner.

## Measured checks (MacBook-Air.local, CPU, 2026-10-10)

14 targeted tests passed in 2.00 s including the explicitly slow-marked training
test. Signal tests verify 500 Hz/4 kHz band preference (>30 dB power ratio),
reconstruction, 160-sample chunk equivalence, reset, alignment and invalid data.
Gradient checks cover both branches and the gate; training checks checkpoint
state round-trip and loss reduction with seed 42.

| Check | Result | Conditions |
|---|---:|---|
| Head compute p50 / p95 | 0.0496 / 0.0532 ms | 192→32, batch 1, 20 warmups, 500 inference-mode calls |
| FIR compute | 0.351 ms | mean of 100 one-second clips, reset per clip |
| FIR compute RTF | 0.000351 | 16 kHz; excludes encoder and file I/O |
| FIR algorithmic delay | 4 ms | 64 samples, excludes embedding window |
| Synthetic training time | 0.517 s | 24 rows, 3 labels, 8→4, 80 epochs, LR 0.02 |
| Synthetic pair loss | 0.469985 → 0.000140 | informative low-band centers, random high-band vectors |

Synthetic embeddings are generated in vector space, not ECAPA outputs from
attenuated speech. These checks establish signal correctness and optimizer
operation. They do not measure generalization, ECAPA latency, WER, SI-SDRi,
wrong-speaker time, FAR, or room distance performance. No real listening test
was performed and no model weights are committed.

## Next experiment and promotion gate

Cache full-band and both band ECAPA embeddings on recording/session-disjoint
train/dev/test partitions. Include clean enrollment, near/far same-speaker
speech, separated stems, quiet target, absent target, same-gender hard negatives,
and an unknown third speaker. Never split overlapping crops across partitions.
Use dev only for hyperparameters and rejection calibration. Compare stock
ECAPA and current LDA, low-only, high-only, equal-weight dual branches and the
learned gate on identical held-out audio. Include real matched-microphone
near/far recordings; digital high-frequency attenuation alone does not reproduce
hall reverberation or competing speech. Normalized embeddings can erase simple
amplitude cues, so the gate may learn identity rather than distance reliability.

Reject promotion if held-out accuracy drops >2 percentage points from baseline,
FAR exceeds 2%, or quiet-target FRR fails to improve by at least 15% relative.
Report gate distributions, collapse/zero-vector frequency, speaker margins,
wrong-speaker time and end-to-end embedding overhead, plus level-matched
listening. These are proposed research gates, not claims of meeting them.

## Model registry note (docs/MODELS.md outside task scope)

| Model | Source / licence | Status |
|---|---|---|
| DualBandProjection (13,058 parameters) | Original task code; repository licence applies, no third-party weights | Offline research head, randomly initialized unless explicitly trained |
| Frozen ECAPA-TDNN | Existing project adapter/model; inherits the existing registry licence | Optional externally injected encoder; no newly introduced checkpoint |

Repository verification: `bash scripts/verify_fast.sh` passed (ruff, 1,889 Python
tests with 61 deselected in 138.18 s, and 67 Node tests). The slow synthetic
training check was also run explicitly in the targeted 14-test suite.
