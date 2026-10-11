# Overnight rendered-shard manifest audit

`eval.audit_manifests` is a metadata-first check for overnight rendered audio.
It recursively finds `manifest.json` files, resolves each manifest's `audio`
(or `audio_path`) relative to the manifest directory, reads the audio header,
and emits a JSON report. Audio is not decoded, so a large render bank can be
checked without loading it into memory.

## Expected manifest

Each rendered shard directory contains a manifest with these fields:

```json
{
  "schema_version": 1,
  "shard_id": "shard-001",
  "audio": "render.wav",
  "sample_rate": 16000,
  "channels": 1,
  "duration_s": 12.5
}
```

The rendered file must be mono, 16 kHz, 32-bit float audio (`FLOAT` in
libsndfile; normally a float32 WAV). `duration_s` is checked against the header
within 10 ms and against the configurable bounds (0.01 s to 3600 s by
default). A manifest may include `sha256` or `audio_sha256`; when present it
must match the measured SHA-256 digest. The report always includes a digest for
each readable audio file, making the `shards` section a reproducible hash
manifest without committing rendered audio.

## Run

```bash
.venv/bin/python -m eval.audit_manifests results/overnight/rendered \
  --output results/overnight/manifest-audit.json
```

The command prints the report and returns exit code 0 only when no issues are
found. Use `--min-duration` and `--max-duration` when a batch has a deliberate
duration contract different from the defaults. An empty or missing render
directory is an error (`no_manifests`), rather than a successful no-op.

Issue codes include `invalid_manifest`, `missing_field`, `missing_audio`,
`invalid_audio`, `audio_outside_root`, `sample_rate`, `channels`,
`sample_format`, `metadata_mismatch`, `duration`, `duration_mismatch`,
`invalid_hash`, `hash_mismatch`, and `duplicate_shard_id`.
