# Borrowed streaming separation research (T-301)

Inspected upstream GitHub code and licenses on 2026-10-10. All six repositories
were shallow-cloned into `/tmp/t301-upstream`; only the small attributed MIT
adaptation and license below are shipped. Revisions are pinned, not floating
claims about current branches. No weights or audio are committed.

The immediate borrow is Asteroid's overlap-based source permutation objective,
with a thin NumPy adapter in `hearing/tse/borrowed_streaming.py`. It runs on dummy
stems offline and does not change the live pipeline. ESPnet SkiM is the strongest
stateful model path inspected; MossFormer2 and SepFormer file/window interfaces
must not be called causal streaming models.

## Repository evaluation

| Repository / pinned revision | Code license inspected | Streaming classification | Exact entry points | Decision |
|---|---|---|---|---|
| [asteroid-team/asteroid](https://github.com/asteroid-team/asteroid/tree/c15708a04d3d28e9a1cd50553c456299dbc6d236) (`c15708a04d3d`) | MIT | Windowed continuous-source separation | `LambdaOverlapAdd.ola_forward`, `_reorder_sources` | Rank 1: vendor overlap channel ordering; full OLA buffers whole input. |
| [espnet/espnet](https://github.com/espnet/espnet/tree/fff206876ef5f9e51cf5e5c0ea78a42a36dca4a2) (`fff206876ef5`) | Apache-2.0 | Stateful frame-wise SkiM | `SkiMSeparator.forward_streaming`, `SkiM.forward_stream` | Rank 2: future causal model integration; add FIFOs around simulation frame/merge helpers. |
| [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio/tree/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61) (`6b3774dc79c4`) | Apache-2.0 root; subtrees differ | Windowed MossFormer2 SS / AV TSE | `decode_one_audio_mossformer2_ss_16k`, `decode_one_audio_AV_MossFormer2_TSE_16K` | Rank 3: direct model calls and aligned windows; no causal cache in inspected decoder. |
| [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain/tree/89ead74d163463d30c62329a09cfdb4c54f5abc1) (`89ead74d1634`) | Apache-2.0 | Whole-window SepFormer / ReSepFormer variant | `SepformerSeparation.separate_batch`; `lobes/models/resepformer.py` | Offline separator candidate only; inspected inference has no state argument or streaming method. |
| [alibabasglab/MossFormer2](https://github.com/alibabasglab/MossFormer2/tree/5727f739947605f884da8d7a1448c64f32d37b9d) (`5727f7399476`) | MIT | ModelScope 8 kHz whole-file sample repository | `README.md` pipeline example | No streaming code path to borrow; README delegates inference to ModelScope. |
| [tommy-fox/streaming-source-separation](https://github.com/tommy-fox/streaming-source-separation/tree/e6fa38d901e35ae6c4cd49b5138dea4defbf1430) (`e6fa38d901e3`) | No root license found: do not copy | Stateful Open-Unmix speech/noise demo | `OpenUnmix.forward(x,h,c)`, `unmix_separate_streamer` | Not two-talker TSE; 44.1 kHz file-to-speaker demo, large blocks, unsafe import-time device access. |

SpeechBrain inspection: `speechbrain/inference/separation.py` encodes the entire
`mix`, applies `masknet`, decodes each source and pads/trims to input length;
`separate_file` also peak-normalizes whole stems. ReSepFormer is an available
architecture variant, not proof that this inference interface carries live state.
The MossFormer2 root MIT license does not establish a downloaded ModelScope
checkpoint's license. Open-Unmix demo has explicit `(h,c)` propagation, but root
permission is missing; upstream Open-Unmix permission does not automatically
license this derivative. Its block length is `(4096 + 127*1024)/44100 = 3.042 s`
from source constants, not a measured latency. Its CLI condition
`if(mode == 'instrumental' or 'acapella')` is always true, and the script opens
PyAudio at import time; inspect only, do not execute it in our headless tests.

No GPL module is copied. Any future GPL or unlicensed candidate is documentation
only until permission is resolved. A WebRTC transport demo was not verified in
this bounded search; audio transport does not establish separator causality.
No separately verified “MossFormer2-streaming” repository was found; the concrete
ClearerVoice window path above is the evaluated alternative.

## Top-three borrow maps

- [Asteroid notes](../../eval/borrowed/T-301/asteroid_notes.md): exact vendored
  objective, changes, license, integration point and limitations.
- [ESPnet notes](../../eval/borrowed/T-301/espnet_notes.md): state transitions,
  encoder/decoder framing and a future true-streaming Stage plan.
- [ClearerVoice notes](../../eval/borrowed/T-301/clearvoice_notes.md): direct
  SS/AV calls, window alignment, RMS hazard and existing-code reuse.

## Delivered adaptation and usage

`third_party/borrowed/T-301/source_order.py` adapts Asteroid's `_reorder_sources`
from pinned `asteroid/dsp/overlap_add.py`. Full upstream MIT license is included.
The Torch/PIT batch machinery is replaced with two-source NumPy permutation
costs, preserving overlap selection and centered negative correlation. Ties
(including silent overlap) preserve current order. No new dependency is needed.

```python
import numpy as np
from hearing.tse.borrowed_streaming import SourceContinuityAdapter

ordering = SourceContinuityAdapter(window_size=32000, hop_size=16000)
first = ordering.process_window(np.zeros((2, 32000), dtype=np.float32))
second = ordering.process_window(np.zeros((2, 32000), dtype=np.float32))
ordering.reset()
```

Run from a source checkout: the adapter resolves the requested `third_party/`
location explicitly; it is not included in the current `hearing*` wheel package
selection. It operates on already-separated overlapping windows, not individual
160-sample capture frames, and is deliberately not registered as a waveform
Stage or candidate. Upstream `LambdaOverlapAdd` itself is whole-input processing.

A future opt-in experiment can call this immediately after separation and before
identity matching/OLA in `EnrolledCandidate.extract`. Existing `_follow_hop`
already correlates the previously selected stream; this utility instead aligns
both streams with a supplied fixed hop. Evaluate the additional behavior rather
than duplicating continuity logic in production blindly. Reset per session and
whenever the separator or window schedule changes. Keep speaker embeddings,
negative enrollment and target-absent guards authoritative: correlation is not
identity, and the first channel need not be the selected speaker.

## Verification and measurements

`.venv/bin/python -m pytest -q tests/test_borrowed_streaming.py` verifies a forced
channel swap across overlapping random stems with DC offsets, silence ties,
reset, copy ownership, geometry, two-channel shape and finite-value validation.
No network, weights, microphone or speaker access is involved.

Utility-only measurement on this arm64 Mac / CPU, Python 3.14.7, NumPy 2.5.3:
500 calls after 10 warmups, two float32 32000-sample windows, 16000-sample hop,
16 kHz: compute p50 **0.0916 ms**, p95 **0.0999 ms**, RTF **0.0000932**
(total compute / 500 one-second hops). Added lookahead is **0 samples** beyond
already available windows. This excludes separator inference, identity matching,
OLA, devices and transport. It is not end-to-end latency or separation quality.
SI-SDRi, WER, wrong-speaker time and real-scene listening are **not measured**;
no model or checkpoint is introduced or promoted by this task.

Continuation audit (2026-10-10): all six disposable clones still match the
revisions above, and the vendored MIT notice matches Asteroid's `LICENSE`.
The reported `docs/BENCHMARKS.md` conflict is absent in this checkout: both
T-284's demo-health results and T-301's utility row are retained, with no
unmerged index entries or conflict markers. The seven adapter tests pass.

## Next adoption gate

1. A/B the helper versus existing `_follow_hop` on unchanged held-out mixtures,
   including two similar male voices, target absence, simultaneous onset and
   source swaps. Save both stems and actual target decisions.
2. For SkiM, verify checkpoint license, sample rate and causal config; load lazily
   through the model registry, use device selection, implement bounded input and
   decoder FIFOs, and test 10 ms/batch equivalence with explicit startup/flush.
3. Measure full-path delay, RTF, SI-SDRi, wrong-speaker time and listening quality
   before any default change. Preserve the cafe-first live demo configuration.
