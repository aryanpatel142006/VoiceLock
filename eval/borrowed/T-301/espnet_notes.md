# espnet: inspected borrow path

Inspected 2026-10-10. Clone: `/tmp/t301-upstream/espnet` (disposable, not shipped).
Revision: `fff206876ef5f9e51cf5e5c0ea78a42a36dca4a2`. [Pinned source](https://github.com/espnet/espnet/tree/fff206876ef5f9e51cf5e5c0ea78a42a36dca4a2).

## Exact streaming path

[`SeparateSpeechStreaming`](https://github.com/espnet/espnet/blob/fff206876ef5f9e51cf5e5c0ea78a42a36dca4a2/espnet2/bin/enh_inference_streaming.py)
`__call__` invokes `encoder.forward_streaming`, carries `streaming_states` through
`separator.forward_streaming`, then calls `decoder.forward_streaming` per stem.
`reset` sets separator state to None. `frame` and `merge` are offline simulation
helpers: `ConvEncoder.streaming_frame` requires the whole audio array and
`ConvDecoder.streaming_merge` allocates the whole output. They are not live FIFOs.

[`SkiMSeparator.forward_streaming`](https://github.com/espnet/espnet/blob/fff206876ef5f9e51cf5e5c0ea78a42a36dca4a2/espnet2/enh/separator/skim_separator.py)
calls `SkiM.forward_stream`, reshapes one feature frame into per-speaker masks,
and applies masks to the encoder input. In
[`espnet2/enh/layers/skim.py`](https://github.com/espnet/espnet/blob/fff206876ef5f9e51cf5e5c0ea78a42a36dca4a2/espnet2/enh/layers/skim.py),
`SkiM.forward_stream` maintains `current_step`, `seg_state` and `mem_state`;
every `segment_size` steps it advances `MemLSTM.forward_one_step`, then each
`SegLSTM` processes the next feature frame. Use causal configuration
(unidirectional LSTMs / cLN); ordinary noncausal forward is not a streaming substitute.

## Exact borrow plan

Borrow/adapt `SkiM.forward_stream`, `MemLSTM.forward_one_step`, `SegLSTM` and
`SkiMSeparator.forward_streaming` together with the trained encoder/decoder's
`forward_streaming` methods. Do not lift only one LSTM and expect pretrained
weights to fit. Code is Apache-2.0 (`LICENSE` inspected); weights and any recipe
assets need separate model-card permission checks. Nothing from ESPnet is vendored.

Create an optional `hearing/tse/skim_streaming.py` Stage with a per-session input
FIFO: append 160 samples per call, wait for encoder `kernel_size`, advance
`stride` samples per feature frame, carry separator state and overlap-add decoder
frames into an output FIFO. Release exactly the requested 10 ms output including
explicit startup delay; reset all FIFOs and states on a new session. Use
`hearing.device.pick()` and lazy `hearing.models.load()` for a verified checkpoint.
Feed both stems into the existing enrolled identity matcher, never choose the
louder stem. This requires a streaming identity policy rather than pretending a
`TSECandidate.extract` whole-window call is state-equivalent.

## Risks / adoption gate

No checkpoint loaded, quality or speed measured here. Encoder kernel/stride and
sample rate must come from the actual checkpoint; 8 kHz CLI default is not our
16 kHz contract. Prove batch/10 ms equivalence after delay/flush alignment,
bounded state, target-absent rejection and RTF on the selected device. The true
stateful inference path makes this the strongest causal research candidate, but
it is a larger integration than the shipped continuity helper.
