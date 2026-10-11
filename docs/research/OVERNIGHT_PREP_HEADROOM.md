# T-238: Quiet-target headroom and gain-sweep preparation

This is a **preparation artifact**, not a measured model or room result. The
replayable command reads clean target/interferer/noise stems, decodes them at
16 kHz, and records actual RMS, peak, silence, additive reconstruction, and
unprocessed clipping forecasts for SIR 0/-6/-10/-15 dB. The nominal target
level is -20 dBFS RMS. Gain is only a bounded, causal diagnostic candidate
(±12 dB, 10 ms frames, attack 3 frames, release 12 frames); it is not a
production AGC setting.

```sh
.venv/bin/python -m eval.prep_headroom \
  --fixtures "$HEARING_FIXTURES_DIR/tiny" \
  --out "$HEARING_RESULTS_DIR/overnight_prep/T-238" \
  --limit 24 --seed 0 --render
```

The result contains `config_sha256`, source manifest hashes, complete-word
intervals (the whole clean fixture utterance), licence evidence, and one atomic
checkpoint per case. Re-running with the same configuration skips verified
complete checkpoints; a failed case is recorded as `failed` and raises. The
default cap is 24 existing pairs and no more than the available count is
claimed. WAV derivatives are private and ignored under the results directory.

## Listening plan and interpretation

T-211/T-213/T-217 can consume `report.json` and shuffle the rendered
`waveforms/*_baseline_sum.wav` and target references at matched RMS. Record
quiet-target audibility, clipping/harshness, and wrong-speaker intrusion per
time interval. `float_overrange_samples` predicts values outside [-1, 1] before
conversion; `pcm_clip_samples` predicts samples that a PCM16 writer would clip.
These are predictions from clean stems, not measured output of a separator or
room. `input_stats` and `mix_additivity_error_rms` are measured after decoding
the existing PCM16 WAVs. No normalization is applied to the baseline sum.

Inputs are the existing LibriSpeech-derived fixture bank. Attribution and
licence evidence: `eval/fixtures/README.md#licences` (LibriSpeech, CC BY 4.0).
