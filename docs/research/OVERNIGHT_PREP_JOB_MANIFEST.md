# Overnight preparation job manifest (v2)

`eval.prep_job_manifest` validates recipes for Sol without executing scripts,
loading models, probing devices, syncing hosts, or submitting jobs. This is
experiment preparation, not a queue or GPU runner.

## Consumer command and incremental inputs

Run from the intended source checkout. Keep unavailable plans in the arguments:
missing, malformed, or incompatible plans get explicit `plans[].status=pending`.
Available plans are processed immediately. Re-run the same command when another
owner exports a plan; nothing is executed or retried.

```bash
.venv/bin/python -m eval.prep_job_manifest \
  --plan "$HEARING_RESULTS_DIR/overnight_prep/T-210/job-plan.json" \
  --plan "$HEARING_RESULTS_DIR/overnight_prep/T-211/job-plan.json" \
  --plan "$HEARING_RESULTS_DIR/overnight_prep/T-212/job-plan.json" \
  --plan "$HEARING_RESULTS_DIR/overnight_prep/T-213/job-plan.json" \
  --plan "$HEARING_RESULTS_DIR/overnight_prep/T-214/job-plan.json" \
  --plan "$HEARING_RESULTS_DIR/overnight_prep/T-215/job-plan.json" \
  --plan "$HEARING_RESULTS_DIR/overnight_prep/T-217/job-plan.json" \
  --repo-root "$PWD" --dry-run \
  --out "$HEARING_RESULTS_DIR/overnight_prep/T-236/job-manifest.json"
```

Those filenames are a proposed export convention, not a claim that the owners
have supplied plans. Existing T210/212/213/214/215 research and scripts and
Muse T224/225/226 catalogs were inspected read-only. Their native input manifests
are not automatically treated as job recipes. T227's export and normalized
owner job plans are unavailable here. No downloads or discovery were repeated.
Use the owners' actual export paths when available; never invent source clips,
transcripts, CUDA capability, or measured results to make a plan ready.

## Input schema

A plan is an object with a `source_revision` (7–40 hexadecimal Git commit ID)
and a `jobs` list. At most 500 input entries are hashed per preparation batch. All relative paths resolve from `--repo-root`. Each job has:

| Field | Requirement |
|---|---|
| `id` | Unique identifier using letters, digits, underscore or hyphen |
| `script`, `script_sha256` | Existing script with matching SHA256 and exact Git blob at source revision |
| `inputs` | List of `{path, sha256}`; bare paths remain pending until a hash is supplied |
| `config_path`, `config_hash` | Required existing immutable configuration |
| `model_path/hash`, `profile_path/hash` | Optional for jobs that use neither; supplied pairs must match |
| `seed` | Explicit integer |
| `args` | Literal string argument list; never evaluated here |
| `cases`, `timeout_s` | Positive integer case count and finite positive per-case timeout |
| `max_output_bytes` | Explicit positive per-job artifact byte cap, at most 1 GiB |
| `outputs` | Nonempty list of distinct artifact paths |
| `requires_cuda` | Explicit boolean |
| `capability` | Required name for CUDA jobs; must be explicitly confirmed |
| `metrics` | Names from the supported dependency table below |
| `metric_dependencies` | Map required dependency names to verified `inputs[].path` |
| `thresholds` | Known requested metrics mapped to finite numeric acceptance thresholds |

Media inputs (`wav`, `flac`, `mp3`, `m4a`, `mp4`) additionally carry `provenance`:
`speaker_id`, `utterance_id`, `interval_s: [start, end]`, `source_sha256`,
`licence`, `licence_evidence`, `source_group`, `partition` (`development` or
`final`). These refer to original sources, not newly generated identities.
Preserve the original speaker/utterance/interval and original hash when rendering
derivatives. Licence evidence is supplied from existing catalogs; this CLI
checks presence and structure, not legal authenticity or audio contents.

Supported metrics: `si_sdri` requires `target_reference` and `mix_reference`;
`leak_db` requires `target_reference` and `interferer_reference`; `stoi` requires
`target_reference`; `wer` requires a hashed `transcript` input. Unknown metrics
stay pending until their verified dependency contract is added. The builder
never reads transcript text into the output.

## Readiness and caps

`--git-revision` is an additional assertion, never an override for actual Git
HEAD. A job is pending unless its intended commit resolves locally, equals the
checkout's HEAD, and contains exactly the working script blob. Uncommitted
scripts cannot claim a committed source revision. Plans and recipes carry SHA256
fingerprints; seeds, configuration, inputs, script, arguments, and source identity
are included in the recipe hash.

CUDA requires explicit `--capability cuda` plus the job's capability, for example
`--capability a100`. These flags are the executor's assertion about the intended
machine; this preparation CLI never probes or allocates a GPU. Omit them when
capabilities are unknown. All GPU jobs use the shared serial lock
`/tmp/selective-hearing-overnight-gpu.lock`; alternative locks are rejected.

Duplicate IDs and normalized output-path collisions make **every participant**
pending, including the first occurrence. Total planned work across all supplied
plans/shards must fit **300 cases / 28,800 seconds**. Any aggregate overrun makes
**all jobs** pending. Invalid counts, NaN/Infinity, hash mismatches, missing
inputs, unknown capabilities, and unverified metric dependencies also make jobs
pending and enter the error list. Exit **0** means all supplied recipes passed;
exit **2** means there are pending jobs/plans or validation errors.

`timeout_total_s = cases * timeout_s` is the hard per-job wall limit, including
setup and teardown. The executor must also enforce each job's `max_output_bytes`
cap before writing additional outputs. Sol's executor must enforce both limits, hold the serial
lock for the entire job, and preserve a suite-wide ledger across shards. The
first 12 suite cases are smoke cases; proceed only after their review gate.
No implicit repeats or retries. Preserve final heldout source groups and never
use them for tuning. These are executor requirements, not actions performed
by this builder.

Counts distinguish available plans from pending plans, and existing input entries
from planned inputs/cases. Input counts count entries across recipes, not unique
files or independent examples. A file that exists but mismatches its declared
hash is counted as existing while its job remains pending. Missing plans carry
no invented case or input counts. Every recipe's evidence is `planned`;
`intended_evidence` records the requested downstream category (`oracle` or
`deployable`) without claiming any measurement.

## Resume/checkpoint contract

`--resume` labels recipes for resumption and does not execute or skip work.
The recipe fingerprint is identical in dry-run and resume modes. Write per-case
checkpoints to `checkpoints/{job_id}/{case_id}.json` using a temporary file,
fsync, and atomic rename on the same filesystem. The checkpoint contains:

```json
{"status":"complete", "recipe_sha256":"<recipe fingerprint>",
 "case_id":"<case id>", "outputs":[{"path":"<artifact>","sha256":"<hash>"}]}
```

Sol can call `verified_checkpoint(path, recipe_sha256, case_id, root)` from
`eval.prep_job_manifest`; it returns true only for a complete checkpoint with
matching identities and a nonempty output list whose bytes all rehash correctly.
Missing/corrupt outputs, failed markers and different cases/recipes never skip.
The executor must check that the output list names that case's expected artifacts.
Write `status=failed` on failure and stop the job without retries. Resume retains
the full original case/time reservation; it never resets the global budget.

## Offline verification and artifact policy

Tests use tiny local committed script fixtures that raise if executed, immutable
byte fixtures, and incremental T210/T211-shaped recipe exports. They check the
first plan remains ready while another is missing, then both become ready when
the second arrives, with stable fingerprints and cumulative budget accounting.
They also cover 8-hour boundary/overrun, duplicates, collisions, missing inputs,
hash/source mismatches, unknown CUDA/metrics, malformed fields, and checkpoint
corruption. These are fixture integration results, not owner-plan readiness or
GPU measurements. Actual GPU executions by T236: **zero**.

Private plans and derivatives belong under ignored
`$HEARING_FIXTURES_DIR/overnight_prep/T-236/` and
`$HEARING_RESULTS_DIR/overnight_prep/T-236/`. Keep manifests containing private
paths there; never commit media, private transcripts, model weights, or private
path-bearing reports. The public document contains only schema and commands.
