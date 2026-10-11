# clearvoice: inspected borrow path

Inspected 2026-10-10. Clone: `/tmp/t301-upstream/clearvoice` (disposable, not shipped).
Revision: `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`. [Pinned source](https://github.com/modelscope/ClearerVoice-Studio/tree/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61).

## Exact windowed path

[`clearvoice/clearvoice/utils/decode.py`](https://github.com/modelscope/ClearerVoice-Studio/blob/6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61/clearvoice/clearvoice/utils/decode.py):
`decode_one_audio_mossformer2_ss_16k` computes window samples from `decode_window`,
sets stride to 75% of the window, pads the full input, calls `model(tmp_input)`
per window for long inputs, discards half the overlap on either edge, and finally
rescales each entire stem to the input RMS. Short input uses a whole-array model
call. No recurrent/cache state is returned. Source index is used directly across
windows: there is no permutation stitching in this function.

`decode_one_audio_AV_MossFormer2_TSE_16K` likewise uses audio/video windows, with
60% stride and aligned 25 fps video slices. It runs `model(tmp_audio,tmp_video)`
and handles the final full window separately. These are windowed offline
inference paths; the variable names do not demonstrate a causal MossFormer2.

## Exact borrow plan / hearing/tse integration

Borrow the direct model call and paired audio/video indexing from the two named
decode functions for an optional window experiment. Retain our own
`SlidingWindowExtractor` length conformance and normalized crossfade rather than
copying the full-input padding/output allocation or whole-stem RMS rescaling.
For SS, pass the raw two stems through the vendored source-order adapter at a
known hop **before** ECAPA selection, then mix with the existing residual policy.
Keep the AV 25 fps/16 kHz alignment in `windowed.frame_span`.

Source model definitions are under `models/mossformer2_ss/` (`mossformer2.py`,
`mossformer2_block.py`, `fsmn.py`); copying a block alone does not give a causal
checkpoint. Existing ClearVoice-backed candidates already cover model loading;
do not add duplicate default backends. Root code license is Apache-2.0, verified
in `LICENSE`; shipped toolkit subtrees contain other licenses and cannot inherit
that permission blindly. No ClearerVoice code/weights are copied in this task.

## Risks / evaluation gate

Changing window size needs listening/SI-SDR and target-absent checks on unchanged
mixtures. Upstream per-source RMS normalization can lift an unwanted stem and
uses future samples; omit it in a live experiment. The decoder is not evidence
for a repository named “MossFormer2-streaming”; no separate causal implementation
was verified in this search. Keep the existing cafe-first demo configuration.
