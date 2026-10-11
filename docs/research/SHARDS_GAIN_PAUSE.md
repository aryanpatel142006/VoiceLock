# Gain-disparity and pause/absence shards

`eval.render_gain_pause` prepares offline, deterministic stress cases for the two team roles. It
renders each role as target and interferer, with SIR `{0, -6, -10, -15}` dB, schedules
`target_on`, `target_pause`, and `target_off`, and onset shifts `{0, 0.05, 0.25}` seconds.
Audio is mono float32 at 16 kHz and is deliberately not peak-normalised: `peak` and
`declared_overrange` make clipping stress explicit.

The renderer expects `aryan.wav` and `joseph.wav` in a source directory. Keep source recordings
and rendered WAVs outside git (the normal overnight location is
`$HEARING_RESULTS_DIR/gain_pause`):

```sh
.venv/bin/python -m eval.render_gain_pause \
  --sources "$HEARING_FIXTURES_DIR/team" \
  --out "$HEARING_RESULTS_DIR/gain_pause"
```

Each shard directory contains `mix.wav` and a metadata manifest with the recipe, silence
intervals, peak, and SHA-256 of the serialized waveform. The output root has a sorted aggregate
manifest. WAV serialization uses the repository's PCM16 writer, so values beyond `[-1, 1]`
remain visible in metadata even though the PCM16 file is clipped at write time.

This preparation layer does not alter enrollment profiles or production stages. It is intended to
be consumed by offline gain-disparity, target-silent, and target-absent evaluations; no quality,
latency, or speaker-privacy claim should be inferred from the synthetic shards alone.
