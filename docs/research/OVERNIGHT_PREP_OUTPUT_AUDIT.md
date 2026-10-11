# Overnight delivered-output audit preparation

`eval/prep_output_audit.py` is a CPU-only, offline audit for already rendered
WAV output. It does not claim naturalness and does not run PESQ, DNSMOS, or
any other learned scorer. It reports observations and the thresholds used:

* **Alignment:** cross-correlation lag (bounded by `--max-lag-ms`), overlap,
  length delta, and an unavailable flag for silent/degenerate signals.
* **Clicks/transitions:** unusually large output sample changes relative to
  the reference waveform's ordinary-change distribution. A boundary candidate
  is not proof of a click; inspect the reported threshold and waveform.
* **Clipping:** samples at full scale within `--clip-epsilon`.
* **Dropouts:** target-active 10 ms frames whose level is below the matched
  reference level by `--dropout-db`. A supplied `target_active` mask is
  preferred; otherwise the reference's upper three quartiles define activity.
  Output is RMS-matched to the reference before this comparison.

## Replayable command

Prepare a private manifest outside git (for example under
`$HEARING_FIXTURES_DIR/overnight_prep/T-241/`) with:

```json
{
  "source": {
    "speaker": "recording-or-synthetic-id",
    "utterance": "utterance-id",
    "interval": [12.0, 18.0],
    "sha256": "source-file-hash",
    "licence": "licence-evidence"
  },
  "cases": [
    {
      "id": "case-001",
      "reference": "reference.wav",
      "output": "delivered.wav",
      "target_active": "target-active-mask.wav"
    }
  ]
}
```

Run from the repository root:

```bash
.venv/bin/python -m eval.prep_output_audit \
  "$HEARING_FIXTURES_DIR/overnight_prep/T-241/manifest.json" \
  --results "$HEARING_RESULTS_DIR/overnight_prep/T-241" \
  --resume --max-cases 300 --max-seconds 3600 --max-bytes 1000000000
```

Each case is written atomically to a checkpoint named `<id>.json`; `--resume`
skips existing checkpoints. `report.json` records planned versus available
cases, resumed cases, input SHA-256 values, source provenance, configuration
and configuration hash. Missing or unavailable stages must be represented by a
case with an explicit provenance/status field; this tool does not fabricate
audio or measurements.

The exact report is consumed by T214/T217 as a recipe for delivered-output
alignment and click/dropout review. Their follow-up command should read
`$HEARING_RESULTS_DIR/overnight_prep/T-241/report.json`, preserve the
thresholds, and separately label any additional stage outputs as unavailable
when no output WAV exists.

