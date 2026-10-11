# T-282: confidence-weighted EMA identity memory

Offline R&D only. `eval/dynamic_ema_tracker.py` consumes embedding windows;
it does not load models, separate audio, implement an audio Stage, change saved
profiles, or alter the live demo path. No new model/dependency/licence is introduced.

## Formulation

Let `t` be the immutable normalized target enrollment, `n_j` optional negative
profiles, `e_i` normalized separator candidates, and `m` the EMA accumulator.
All vectors must share the same frozen encoder/projection revision. Initialize
`m = t`; compare candidates to `u = m / ||m||`. Choose the largest `u·e_i`.
The stem index is local to this window: separator permutations never become
identity IDs.

Define stream margin `d = u·e_best - max_other(u·e_other)`. For a single candidate,
use `d = margin_full` and still require the absolute enrollment/negative gates.
Freeze memory and return **no verified stem** if any gate fails:

- upstream laughter flag is true, or window speech probability < 0.60;
- `t·e_best < 0.55` (absolute identity floor);
- with negatives, `t·e_best - max_j(n_j·e_best) < 0.10`;
- `d < 0.05` (ambiguous competing stems).

Speech activity alone is not target presence. The immutable anchor prevents the
EMA from authorizing identity solely through its own accumulated drift. Unknown
speakers can still pass a cosine floor; this is not a perfect identity verifier.
If the best memory match fails the anchor gate, abstain rather than trying a
different candidate. This deliberately trades possible target dropouts for
avoiding an unverified selection.

For an accepted window:

```
c = clip((d - 0.05) / (0.30 - 0.05), 0, 1)
a_ref = 0.02 + (0.35 - 0.02) * c
a = 1 - (1 - a_ref) ** (hop_s / reference_hop_s)   # reference_hop_s = 1
m_next = (1 - a) * m + a * e_best
```

Alpha is the **new observation weight**, not the old-memory retention weight.
At minimum/full accepted confidence, reference-hop retention half-lives are
34.31/1.61 seconds of accepted evidence. Invalid/ambiguous windows contribute
zero updates. No accumulated pause time enters the first resumed weight.
`hop_s` denotes newly advanced audio duration, not overlapping window length.
Overlapping crops remain correlated; time scaling does not make them independent.
The accumulator is not re-normalized after updates; its public embedding is.
For identical evidence with fixed confidence this preserves equivalence between
one 1 s update and two 0.5 s updates. Changing evidence/confidence can change the
trajectory; this is not a general chunk-invariance promise for the encoder.

## API and consumer obligations

```python
from eval.dynamic_ema_tracker import DynamicEMATracker

tracker = DynamicEMATracker(target_embedding, negatives=[other_embedding])
result = tracker.update(candidate_embeddings, speech_probability=0.9,
                        laughter=False, hop_s=1.0)
# result.index is None on rejection; result.reason explains the gate.
# tracker.embedding is a defensive copy; tracker.reset() restores enrollment.
```

During silence/laughter `candidates=None` is allowed and avoids embedding work.
Freezing memory does **not** mean playing yesterday's stem: the caller must
mute/mark uncertain when `index is None`. At resumed speech it must use the new
window's index. Construct a new tracker on explicit target/profile/revision
changes; reset the old one when discarding session context. The prototype never
re-targets automatically or writes enrollment files. No current consumer changed.

Thresholds are configurable **uncalibrated research defaults**, not ECAPA or
projected-model calibration results. No laughter detector is implemented: the
flag and speech probability must come from an upstream system. Reliability of
those inputs, especially laughter within overlapping windows, is unmeasured.

## Dependency T-246

The checked-in `docs/research/theory/projection_head.md` is a theoretical and
synthetic ablation report, not proof of room performance. This tracker accepts
raw frozen ECAPA vectors or consistently projected vectors without importing
T-246's model/weights. Floors and margins must be recalibrated for a projection;
do not carry raw-ECAPA thresholds into a learned metric by assumption. No head
is trained or promoted by T-282.

## Deterministic synthetic ablation

Seed 282; 192 dimensions; 1,000 1 s hops; 800 speech and 200 non-speech hops.
Random stem permutations occur each hop. Three biased low-margin crops alternate
with one nearly clean crop. During pauses the candidates contain only a negative
and an antipodal vector. Speech/laughter flags are supplied by the generator.
Compare adaptive EMA against fixed `alpha_min=alpha_max=0.35` with **identical
gates**, and immutable-profile matching (`alpha_min=alpha_max=0`). These constructed vectors test mechanics and favor slow updates for
biased ambiguous evidence; they simulate neither actual ECAPA distributions
nor separation/audio quality. Thresholds were not fitted to recordings.

Measured on macOS 26.6.2 arm64, Python 3.14.7, NumPy CPU:

| Variant | Accepted speech hops / 800 | Wrong accepted speech hops | Accepted pause hops / 200 | Mean memory cosine error to enrollment | Update p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| Adaptive | 800 | 0 | 0 | 0.01989546 | 0.019125 / 0.020708 |
| Immutable profile | 800 | 0 | 0 | 0.00000000 | 0.018708 / 0.020292 |
| Fixed, same gates | 402 | 0 | 0 | 0.06738702 | 0.016500 / 0.020002 |

Immutable-profile matching performs equally well on selection in this constructed
sequence; this experiment does not establish a benefit over the current matcher.
Rejected speech hops are dropouts, not counted as correct selections. Memory
drift in the fixed baseline narrows subsequent margins and causes abstention.
All variants reacquire the first clean post-pause hop (0 additional hops);
this excludes encoder/window acquisition delay. Zero wrong accepts here does
not establish a real-room false-accept rate. Compute timing excludes encoding,
separation and audio I/O; SI-SDRi, WER, audio RTF and capture-to-ear latency are
not applicable to this embedding-only experiment.

## Verification and promotion gate

`tests/test_dynamic_ema_tracker.py` verifies confidence-dependent adaptation,
freeze across repeated silence/laughter, no stale-index playback authorization,
stem permutations, target-absent and negative-profile rejection, ambiguous ties,
pause-safe hop weighting, reset/defensive copies, single-stream floors, hop
equivalence for constant evidence/confidence, and input/config validation.

Run `.venv/bin/python -m pytest -q tests/test_dynamic_ema_tracker.py` and
`bash scripts/verify_fast.sh`.

Before integration, compare immutable-profile matching, fixed EMA with the same
gates, and adaptive EMA on identical cached separated windows. Include both
targets, target absence, an unknown speaker, laughter, long pauses and changed
stem order. Calibrate on development sessions only; measure false accepts,
wrong-speaker duration, rejected-target duration, reacquisition delay and CPU
overhead on session-disjoint same-mic holdouts. Require no wrong-person regression
and acceptable dropouts, plus consented real-room listening. Keep the live path
unchanged until its owning task explicitly implements and validates integration.

## Reproduce the synthetic measurement

Save the following as `/tmp/t282_bench.py`, then run
`PYTHONPATH="$PWD" .venv/bin/python /tmp/t282_bench.py` from this checkout.
Timing varies by host/load. No files, weights, microphone or network are used.

```python
import json
import platform
import time
import numpy as np
from eval.dynamic_ema_tracker import DynamicEMATracker

rng = np.random.default_rng(282)
dim = 192
target, negative = np.eye(dim)[:2]
trackers = {
    'adaptive': DynamicEMATracker(target, negatives=[negative]),
    'immutable_profile': DynamicEMATracker(target, negatives=[negative], alpha_min=0, alpha_max=0),
    'fixed_gated': DynamicEMATracker(target, negatives=[negative], alpha_min=.35, alpha_max=.35),
}
metrics = {k: {'accepted': 0, 'wrong': 0, 'pause_accepted': 0, 'error': [], 'times': []} for k in trackers}
for hop in range(1000):
    pause = hop % 20 >= 16
    # Low-margin biased target crops alternate with clean evidence. No acoustic simulation.
    if pause:
        streams = np.stack([negative, -target])
    elif hop % 4:
        streams = np.stack([.8*target+.6*negative, .7*target+np.sqrt(.51)*negative])
    else:
        x = target + rng.normal(0, .005, dim)
        streams = np.stack([x / np.linalg.norm(x), negative])
    perm = rng.permutation(2)
    streams = streams[perm]
    expected = int(np.flatnonzero(perm == 0)[0])
    for name, tracker in trackers.items():
        start = time.perf_counter_ns()
        result = tracker.update(streams, speech_probability=0 if pause else 1, laughter=hop % 20 == 19)
        metrics[name]['times'].append((time.perf_counter_ns()-start)/1e6)
        metrics[name]['accepted'] += result.index is not None
        metrics[name]['wrong'] += result.index is not None and not pause and result.index != expected
        metrics[name]['pause_accepted'] += pause and result.index is not None
        metrics[name]['error'].append(1-float(tracker.embedding @ target))
for name, rows in metrics.items():
    print(json.dumps({'name': name, 'hops': 1000, 'speech_hops': 800, 'pause_hops': 200,
        'accepted': rows['accepted'], 'wrong_speech_accepts': rows['wrong'],
        'pause_accepts': rows['pause_accepted'], 'mean_memory_cosine_error': float(np.mean(rows['error'])),
        'update_ms_p50': float(np.percentile(rows['times'],50)), 'update_ms_p95': float(np.percentile(rows['times'],95)),
        'device': 'NumPy CPU', 'machine': platform.platform(), 'python': platform.python_version()}))
```
