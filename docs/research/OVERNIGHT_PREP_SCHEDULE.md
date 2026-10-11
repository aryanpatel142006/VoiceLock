# Overnight preparation schedule

`eval/prep_schedule.py` creates a metadata-only, deterministic case schedule for
the GPU owners. It does not render, augment, transcribe, open devices, or
download data. Pair metadata is read from the T-221 catalog and immutable
source-group partitions from T-229 when those manifests are available:

```bash
.venv/bin/python -m eval.prep_schedule \
  --pairs eval/manifests/overnight_grunt/pairs.json \
  --splits eval/manifests/overnight_grunt/splits.json \
  --output "$HEARING_RESULTS_DIR/overnight_prep/T-234/schedule.json"
```

Use `--existing <gpu-cases.json>` once per existing GPU result manifest. Every
existing `cases[]` row consumes the cumulative 300-case cap; rows marked
`status != "complete"` are eligible for regeneration. The output's
`checkpoint` records input-manifest hashes and deterministic resume behavior,
and `consumer.command` is the exact schedule-producing command for downstream
workers.

## Schedule contract

Each planned row has a stable `case_id`, pair and target role, SIR (0, -6, -10,
-15 dB), target state (`absent`, `pause`, `onset`), noise and room factors,
enrollment duration (1/2/3 s), and window/hop proposal. `smoke_cases` contains
the first 12 rows selected by a stable round-robin over role/SIR/target-state
cells. `coverage.primary_cells` and `missing_cells` make incomplete coverage
explicit. `accounting.available_input_cases` is the number of eligible
metadata combinations before the cap, separate from `planned_cases`.

The schedule requires both pair roles to have source IDs and SHA-256 evidence,
different speaker IDs, and a common `development` or `final` partition. If no
split manifest is supplied, a stable pair-ID hash selects a partition and the
output should be treated as a proposal only. Final source groups are never
reassigned or mixed with development/enrollment rows.

No overnight schedule was rendered in this task. Intervals for absent/pause/onset
are proposals for downstream owners; the scheduler does not invent timestamps,
transcripts, precision, or measurements.
