# Overnight tempo/bandwidth render shard

This is an **offline experiment-preparation tool**, not a production profile and
not evidence that any augmentation improves recognition. It renders a bounded,
restartable shard from consented local recordings (for example
`$HEARING_FIXTURES_DIR/team/aryan/*.wav` and `.../team/joseph/*.wav`) without
uploading or modifying the originals.

## Exact command and consumer contract

```bash
.venv/bin/python -m eval.prep_render_tempo_band \
  --input "$HEARING_FIXTURES_DIR/team" \
  --out "$HEARING_FIXTURES_DIR/tempo-band" \
  --seed 260 --max-variants 600
```

`--input` may be repeated and may name a recording or a directory. The output
contains one directory per source stem, PCM16 WAV files at native 16 kHz, a
same-stem JSON sidecar, and `checkpoint.jsonl`. Re-running the exact command
skips a completed WAV when its source SHA-256 and variant definition still
match; deleting an output sidecar intentionally causes that item to render
again. The output root is intended to be an ignored generated-data location.

Consumers must read the sidecar before using a render:

* `baseline: true` is the untouched native-rate baseline.
* `variant.sample_rate` records a simulated input-rate excursion, resampled
  back to 16 kHz for a consistent output contract.
* `variant.bandlimit_hz`, `clip`, `gain_db`, and `channels` identify independent
  bandwidth, clipping, gain, and mono/stereo stress transforms. Stereo outputs
  must be downmixed explicitly by a consumer that requires pipeline mono.
* `tempo` is implemented by deterministic resampling. Values `0.9` and `1.1`
  therefore change **both tempo and pitch**; `tempo_pitch_change` is true and
  `identity_preserving` is false. Never use these files as identity-preserving
  training labels. A transcript, when present in a neighboring manifest, is
  copied as metadata only.

The deterministic plan has 432 possible transform combinations (including the
baseline); `--max-variants` selects at most 600 unique variants and always
keeps the baseline first. The seed only controls bounded plan selection; it
does not alter DSP samples for a selected variant. There are no downloads,
weights, GPU calls, or new dependencies.

## Observations and hypotheses

No team recordings were available in this worktree, so no audio was rendered
and no quality, WER, SI-SDR, or latency measurement is claimed. This is an
honest exclusion, not a fabricated overnight result. The hypothesis for a
future local run is that tempo/pitch stress and bandwidth/channel mismatch may
expose brittle speaker association; this must be checked against held-out
source groups, with originals and the final source-group holdout kept out of
training.
