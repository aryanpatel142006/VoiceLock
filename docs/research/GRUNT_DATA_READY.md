# M13 overnight data-ready handoff

Generated 2026-10-09 from the already-completed public fixture bank. This is a
metadata-only handoff: `eval/overnight_data_ready.py` never downloads, mutates,
or synthesizes media. The ready JSONL embeds a SHA256, byte count, WAV frame
count, duration, sample rate, channel count, provenance/license, transcript
availability, and stem-addition error for every case.

## Artifact and rerun command

- Ready artifact: `$HEARING_RESULTS_DIR/overnight_grunt/ready/ready.jsonl`
- Observed path on this worker:
  `/Users/joeroche/Developer/selective-hearing/eval/results/overnight_grunt/ready/ready.jsonl`
- Artifact SHA256: `39022c0360383b74f2aba5d9f031f2986ac5ed84c74f9d27a3f35f733d56c050`
- Source root:
  `/Users/joeroche/Developer/selective-hearing/eval/fixtures/generated/public_overnight`
- Exact rerun:

```bash
.venv/bin/python -m eval.overnight_data_ready \
  --fixtures-root "$HEARING_FIXTURES_DIR/public_overnight" \
  --output "$HEARING_RESULTS_DIR/overnight_grunt/ready/ready.jsonl" \
  --limit 12
```

The default output is under `HEARING_RESULTS_DIR`; neither media nor this
handoff is committed to the repository.

## Coverage and disposition

| partition | candidates | READY | EXCLUDED | PENDING |
|---|---:|---:|---:|---:|
| `public_overnight` | 48 | 12 | 31 duplicate source partition | 5 deferred by `--limit 12` |
| completed `overnight_grunt` shards | 0 | 0 | 0 | not present |

The assembler stopped after the requested quota of 12; five additional
candidates were not evaluated in this bounded pass. The 12 READY cases contain
6 target-present and 6 target-absent controls,
7 distinct speaker IDs, 194.94 seconds of audio, and 49,908,480 bytes across
four PCM WAV stems per case. Every file is mono 16 kHz; each case is 23.315 s
and all 12 report `CC BY 4.0` LibriSpeech provenance. Every per-file SHA256
is in the JSONL. Stem addition passed at `<1e-3` maximum absolute error.
Target-absent controls have `sisdr_eligible: false`; they are not used as
SI-SDR references.

The validator also has offline coverage for malformed manifests, path
traversal, missing files, declared hash mismatches, invalid references,
invalid audio, inconsistent headers, missing licenses, and stems that do not
add up. No collected data is claimed for absent grunt shards; those remain
PENDING rather than being fabricated.

## Consumer mapping

- **T-211:** use target-present rows for separation/enrollment inputs; honor
  `sisdr_eligible` and never score target-absent rows as SI-SDR references.
- **T-212:** use `files.*.path`, `sha256`, `sample_rate`, `channels`, and
  `duration_s` for augmentation/fixture staging; source licenses are
  LibriSpeech CC BY 4.0.
- **T-213:** use `source_partition` to keep held-out original sources
  disjoint; use target-absent rows only for leak/negative checks.

No Muse or new overnight-grunt downloads were available in this shard, so the
additional acquisition budget was unused.
