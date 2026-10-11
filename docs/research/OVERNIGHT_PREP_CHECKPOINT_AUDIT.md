# Overnight preparation checkpoint audit

`eval/prep_checkpoint_audit.py` is an offline, metadata-only integrity gate for
overnight experiment preparation. It never edits, deletes, or marks work
complete in an input directory. Results and checkpoints may be complete,
partial, malformed, stale, duplicated, or missing; all of those states remain
visible in the snapshot.

## Consumer command

```bash
.venv/bin/python -m eval.prep_checkpoint_audit \
  /path/to/manifest.json /path/to/results/*.json /path/to/checkpoints/*.json \
  --output "$HEARING_RESULTS_DIR/overnight_prep/T-240/audit.json"
```

The output is a single atomically replaced JSON snapshot. Do not point the
output at an input directory. The `resume.case_ids` and `resume.files` fields
are the exact bounded selection for the next run; consumers should process
those cases only after reloading and checking the same manifest. At most 500
input paths are inspected and at most 500 resume files/case IDs are emitted.
The script is append-free, deterministic (sorted paths and IDs), and safe to
rerun after an interrupted write.

## Integrity rules

The manifest's `cases` list defines planned coverage. Each case is keyed by
`case_id`; `available_cases` counts cases found in valid JSON inputs and is
reported separately from `planned_cases`. A case is `complete` only when its
result status is complete and it has no integrity issues. `partial` includes
running/in-progress checkpoints. Missing cases have no result/checkpoint.
Malformed JSON is reported under `invalid_files`; duplicate IDs, non-finite
metrics, and stale config/model/profile/source hashes make a case `invalid`.
Manifest hashes may be declared as `config_hash` or under `hashes`.

Metric groups are retained separately: `baseline`, `selected`, `oracle`,
`mixer`, `agc`, and `delivered`. `0.0` is a valid metric and remains distinct
from `null`. SI-SDR/SI-SDRi values are invalid for a manifest case with
`target_present: false` (`si_sdri_not_applicable`), rather than being scored
against silence. No metric is fabricated when a group or value is absent.

Inputs and snapshots must remain in the ignored overnight results/fixtures
locations. Public reports should contain hashes and case IDs, not private
transcripts, audio paths, or source content. The audit requires no models,
GPU, audio devices, network, or new dependencies.
