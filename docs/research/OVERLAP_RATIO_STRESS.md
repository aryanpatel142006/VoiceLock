# Overlap-ratio stress shards

`eval/render_overlap_ratios.py` creates a deterministic, offline four-point
stress set at **0, -5, -10, and -15 dB target-to-interferer RMS ratio**. The
target source starts at 0.25 s and the competing speaker starts at 1.25 s;
both stop at their recorded timestamps, giving one exact 1.5 s overlap
interval (`[1.25, 2.75)`).

Run:

```bash
.venv/bin/python -m eval.render_overlap_ratios
```

The output is `eval/shards/overlap_ratios/`, which is intentionally ignored
because it contains voice recordings. Each shard has `target.wav`,
`interferer.wav`, `mix.wav`, and `manifest.json`; `index.json` records all
four manifests. Audio is 16 kHz mono PCM16, peak-limited to 0.9. Manifests
store SHA-256 hashes for source and rendered stems, relative stem references,
sample-aligned speaker intervals, pairwise overlap intervals, requested ratio,
and measured overlap ratio. Re-running with the same seed and sources is
byte-stable.

The interval helper is generic: `detect_overlaps({"speaker": [(start, end)]})`
returns every positive-duration pairwise intersection, sorted by time and
speaker names. Downstream separator evaluation should use
`overlap_intervals` for overlap-only scoring and the clean `target.wav` as
the reference.
