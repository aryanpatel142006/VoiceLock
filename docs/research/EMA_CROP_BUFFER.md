# T-275: Causal short-crop EMA context

## Verdict

Keep as offline R&D. EMA reduces angular jitter but does not eliminate variance,
and its inertia can worsen wrong-speaker time. Do not promote the 1s half-life
prototype to the live matcher without held-out target-absent and stem-swap tests.
T-246 motivates improving frozen ECAPA geometry; this experiment adds no learned
head and does not inherit its synthetic adoption verdict.

## Buffer and replay contract

`EMACropBuffer.update(stream_id, embedding, elapsed_s=...)` consumes an existing
finite nonzero vector, normalizes it, and applies
`normalize(decay * previous + (1-decay) * current)`, with
`decay = 2 ** (-elapsed_s / half_life_s)`. Defaults: 1s half life, 2s gap expiry,
two streams. First crops and crops after a longer gap initialize immediately.
A cancelling antipodal update reacquires the current crop. Returned vectors are
copies; capacity and dimension changes fail explicitly. `reset()` clears all
context. No enrollment vector is mutated. Storage is O(streams * dimension).

Elapsed time is hop/time since last update, **not** crop length when windows
overlap. Callers must skip invalid/unvoiced evidence and pass the actual elapsed
time to the next accepted update. Supply stable associated stream keys; separator
stem indices alone are unsafe. Reset on target change, reassociation or projection
head version change. This is an embedding evaluator, not an audio Stage, and is
not registered in a live pipeline. It adds no lookahead but can add identity lag.

Replay NPZ keys: `embeddings [N,D]`, `profiles [K,D]`, integer `labels [N]`,
`elapsed_s [N]`, loaded without pickle. All embeddings/profiles must use the same
space. One replay follows one ordered stream, including identity transitions.
Nearest-profile closed-set picks are evaluation labels, not target-present proof.
Metrics include overall accuracy, duration-weighted wrong-speaker time, flips
within constant-label segments, mean adjacent cosine distance within those
segments, and first-correct switch delay (null if never correct). Delay starts
at the first completed new-speaker crop and excludes its acquisition duration;
first-correct does not prove a sustained lock. No warmup samples are discarded.

## Reproduction

Run with the worktree `.venv/bin/python`:

```sh
python -m eval.ema_crop_buffer                         # explicitly synthetic
python -m eval.ema_crop_buffer --replay /path/ordered.npz --half-life 1
HEARING_MODELS_DIR=/path/shared/models python -m eval.ema_crop_buffer \
  --audio /path/02_isolated_aryan.wav /path/03_isolated_joseph.wav
```

Audio replay requires local ECAPA weights, never downloads, and uses the existing
`hearing.models.load` adapter and device selection. Inputs must be mono 16kHz.
Two previously separated teammate recordings from main checkout's
`eval/fixtures/demo_listen` were used. Enrollment is the first 2.5s of each;
evaluation begins after it. A/B/A segments each retain up to 12 nonoverlapping
crops. Seed 275 is restarted per crop length. Independent Gaussian white noise
is RMS-scaled to 0dB SNR per crop; summed samples are clipped to [-1,1]. This is
speech plus artificial noise, not a fresh separator or real café benchmark.
The repeated A segment is not an independent holdout. Silence is not filtered.
Existing frozen `speechbrain/spkrec-ecapa-voxceleb` (Apache-2.0) is reused; no new
model, weights or dependencies were introduced.

## Measured results (2026-10-10)

macOS 26.6.2 arm64, NumPy CPU EMA, local ECAPA MPS; half life 1s.
Accuracy and wrong-speaker fractions include switches.

| Crop | Crops | Accuracy raw → EMA | Wrong time raw → EMA | Stable flips raw → EMA | Cosine step raw → EMA | Switch delay raw → EMA (s) | EMA p50/p95 ms |
|---|---:|---|---|---|---|---|---|
| 0.5s | 36 | 86.11% → 75.00% | 13.89% → 25.00% | 6 → 2 | .78176 → .05404 | [.5,0] → [3.5,1] | .00788/.00894 |
| 0.75s | 30 | 73.33% → 83.33% | 26.67% → 16.67% | 12 → 1 | .78075 → .11483 | [0,0] → [3.75,0] | .00915/.00965 |
| 1.0s | 21 | 95.24% → 95.24% | 4.76% → 4.76% | 1 → 1 | .71856 → .16916 | [0,0] → [1,0] | .00858/.00988 |

The separate 320-crop synthetic embedding-space stress test (192 dimensions,
prototype cosine .6, isotropic noise .45/sqrt(crop seconds), seed 275) increased
accuracy 75.31→89.38%, 80.00→95.00%, 82.81→96.25% at .5/.75/1s. Its longest
EMA switch delay was 6s. These simulated vectors are not measured ECAPA quality.

SI-SDRi, WER, audio RTF, leak and four-scene listening were not measured: the
buffer changes identity vectors, not audio. No microphone or speakers were used.
Next gate: held-out utterances/rooms, target silence, unknown speakers, actual
separator permutations and a half-life sweep including raw/no-memory baseline.
Lower jitter alone is insufficient evidence to adopt.

## Verification

Unit tests cover time weighting, independent stream state, defensive copies,
reset/gaps, invalid vectors/time/configuration, capacity, dimension changes,
antipodal cancellation, replay metrics/validation and deterministic offline
noisy-audio wiring. Real-weight measurement is a manual CLI run, outside fast
unit tests. Required full verification is recorded in the worker report.
