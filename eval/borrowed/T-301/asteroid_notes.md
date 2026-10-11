# asteroid: inspected borrow path

Inspected 2026-10-10. Clone: `/tmp/t301-upstream/asteroid` (disposable, not shipped).
Revision: `c15708a04d3d28e9a1cd50553c456299dbc6d236`. [Pinned source](https://github.com/asteroid-team/asteroid/tree/c15708a04d3d28e9a1cd50553c456299dbc6d236).

## Exact upstream functions

[`asteroid/dsp/overlap_add.py`](https://github.com/asteroid-team/asteroid/blob/c15708a04d3d28e9a1cd50553c456299dbc6d236/asteroid/dsp/overlap_add.py):
`LambdaOverlapAdd.ola_forward` unfolds complete audio, calls `nnet` per window,
uses `_reorder_sources` before synthesis weights, then folds all windows back.
`_reorder_sources` compares the current prefix against the previous suffix,
mean-centers both, computes negative cross-correlations and calls `PITReorder`.
This is continuous-source window ordering, **not** incremental capture or a causal model.
`DualPathProcessing` is feature segmentation, not an audio streaming server.

## What is actually borrowed

Vendor only `_reorder_sources` as `third_party/borrowed/T-301/source_order.py`:
retain overlapping-sample selection, mean subtraction and the objective;
replace Torch/PIT with NumPy and the two possible permutations. Remove batching,
add finite/shape validation, accumulate in float64 and keep identity on a tie.
The full MIT notice from upstream `LICENSE` is shipped beside it; the module
has copyright, revision and modification attribution. No Asteroid install needed.

## Plug into hearing/tse/

`SourceContinuityAdapter.process_window` takes unweighted `(2, window_samples)`
16 kHz float32 stems at a fixed positive overlapping hop. First window passes
through; subsequent windows align to a private copy of the previous ordered
window. `reset()` clears it. No model is supplied and no target is chosen.

For a future opt-in experiment, call after `separator.separate(mix, sr)` in
`EnrolledCandidate.extract`, before ECAPA matching and any per-stem OLA. Its
existing `_follow_hop` already follows the selected stream using inferred overlap;
this alternative aligns **both** streams with a known hop. Do not replace that
code or rely on source index for identity without A/B testing. Silence, similar
voices, negative polarity and low correlation can leave correspondence ambiguous.

Do not copy `LambdaOverlapAdd` wholesale: it buffers a complete waveform, needs
Torch/SciPy/PIT, and does not satisfy our 10 ms `Stage` contract. Existing
`SlidingWindowExtractor` handles target-waveform OLA; multi-source alignment must
happen before selecting one stem. This helper adds zero new lookahead beyond the
already available window, but cannot reduce the separator's window delay.

## Proof and remaining gate

`tests/test_borrowed_streaming.py` runs without weights/network/audio devices:
forced channel swap with DC offsets, silence, reset/copy ownership, invalid
geometry, invalid source count and NaNs. This is a continuity utility smoke test,
not SI-SDR, WER, speaker identification or a live separation benchmark.
