# Overnight runtime matrix preparation (T-242)

This is a **preparation-only** manifest for T-215. It does not load weights,
start a server, infer a device from a config string, or access a microphone.
The wrapper should run serially and record the required telemetry fields for
each window: `capture_ts`, `deadline_ts`, `emitted_ts`, `status`, `tier`, and
observed `device`. A missed deadline must use the declared fallback (`raw_mix`);
warmup and steady-state repetitions are separate so initialization cost is not
silently mixed into the steady-state result.

## Reproduce

From the repository root (hardware-free):

```bash
.venv/bin/python -m eval.prep_runtime_matrix \
  --repo . --repetitions 3 --case-cap 18 \
  --fixtures "$HEARING_FIXTURES_DIR/overnight_prep/T-242" \
  --output "$HEARING_RESULTS_DIR/overnight_prep/T-242/matrix.json"
```

The command reports the actual available tier-config files and SHA-256
revisions separately from the planned cases. It validates that requested
window/hop pairs occur in the current configs; it never broadens support to a
nearby pair. Event fixtures cover stale server revision, never-enrolled,
missing-tier, and late-window outcomes. Their `tier_evidence_required` field
means a tier claim must be backed by telemetry; a `config_device_claim` is
intentionally non-evidence.

## Matrix and accounting

The matrix requests 0.5/1/2-second windows with half- or quarter-window hops,
then retains only pairs explicitly present in `config/tiers*.yaml`. Each
candidate has warmup and steady-state repetitions with deterministic seeds,
case and byte caps, and a raw-mix fallback. `account_timeline()` computes
late and fallback counts from emitted versus deadline timestamps. GPU evidence
is counted only when an observed telemetry `device` is present, never from
pipeline/config text. T-215 can consume `matrix.json` and the fixture
directory directly; no private audio or transcript is generated.

Actual input availability on this checkout is the config-file list and hashes
in the dry-run report. No inference inputs are consumed by this preparation
task.
