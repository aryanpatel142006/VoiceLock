# Overnight Muse ingest preparation

`eval/prep_muse_ingest.py` is an offline, provenance-preserving preparation step. It
does not load models or recompute measurements. It accepts prior JSON/JSONL/CSV
exports, normalizes quiet-target role, SIR, condition, and output stage, and
separates **missing** metrics from measured zeroes. Oracle, picked, and mixer
rows are summarized independently; telemetry missingness, stage-label conflicts,
and inconsistent task revisions are surfaced rather than silently merged.

## Replay

From the repository root:

```bash
.venv/bin/python -m eval.prep_muse_ingest \
  path/to/T-224.json path/to/T-225.json path/to/T-226.json path/to/T-227.json \
  --seed 235 --max-cases 160 \
  --checkpoint-dir "$HEARING_RESULTS_DIR/overnight_prep/T-235/checkpoint" \
  --out "$HEARING_RESULTS_DIR/overnight_prep/T-235/muse_ingest.json"
```

The command is deterministic for the same input bytes and seed. `--resume`
reuses a complete checkpoint only when its SHA-256 matches every input file.
Input count, bytes, and planned cases are capped (500, 1 GiB, and 300);
writes are atomic. The output records source filenames (never absolute paths),
the combined input hash, seed, available-record count, stratum counts, metric
missingness, losses, revision conflicts, and required provenance fields:
original speaker, utterance, interval, source hash, and licence evidence.
Private inputs and generated output belong under the ignored
`$HEARING_RESULTS_DIR/overnight_prep/T-235/` directory.

## Current evidence and planned coverage

No T-224--T-227 measurement export is present in this checkout, so the
currently available-record count is **0**; this is not a measurement of model
quality. The report contains five composable proposals (160 planned cases):

| Proposal | Cases | Factors | Causal question |
|---|---:|---|---|
| role-x-sir | 48 | quiet role × SIR | Does a quiet target increase picked-selection loss? |
| condition | 32 | condition × stage | Does occlusion widen oracle-to-picked loss? |
| stage | 32 | oracle/picked/mixer | Does mixer reintroduction dominate played leakage? |
| telemetry | 24 | telemetry present/missing | Are deadline failures hidden by missing telemetry? |
| revision | 24 | revision/config | Do inconsistent revisions explain metric shifts? |

The proposals stay below the global 300-case cap and preserve final held-out
groups; the overnight executor must provide its own input hashes and per-case
checkpoints. No fabricated transcript, precision, licence check, or coverage
claim is produced here.

Consumers should use the generated JSON as the manifest, not this narrative:
T-211 consumes strata and provenance for augmentation planning; T-213 consumes
oracle/picked/mixer loss and telemetry gaps; T-217 consumes the bounded
proposal list. Exact consumer command is the replay command above with the
consumer's output directory substituted.
