# Room/noise shard preparation

`eval/render_room_noise.py` prepares an offline challenge bank for replay and overnight
evaluation. Each example starts with one of the committed LibriSpeech-derived tiny speech
clips, applies a deterministic analytic room impulse response, and adds either the committed
six-talker babble or seeded white ambient noise. The additive bed is scaled by whole-clip RMS
to the requested SNR (`-5`, `0`, `5`, or `10` dB); the saved `target.wav`, `noise.wav`, and
`mix.wav` remain aligned at 16 kHz mono.

The default recipe writes 12 examples to three shards:

```console
.venv/bin/python -m eval.render_room_noise
```

Each `shard-XXXX/manifest.json` contains the example metadata and relative audio paths.
`checkpoint.json` is replaced atomically after each completed shard, and each audio file and
manifest is also written through a temporary file followed by `os.replace`. Re-running with
the same recipe resumes completed shards; changing `--count`, `--shards`, or `--seed` is
rejected until the old checkpoint is removed or a different `--out` is selected.

The room responses are intentionally analytic fixtures rather than measured venue responses:
`small_room`, `medium_room`, and `large_room` provide a direct path plus short exponentially
decaying early reflections. They are useful for deterministic model comparisons, not evidence
about the HackRU room. No recordings, weights, or network downloads are needed.
