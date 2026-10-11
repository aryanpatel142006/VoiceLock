# Bounded overnight CPU preparation executor

`eval.prep_cpu_executor` runs only a reviewed JSON allowlist. It is intended
to prepare deterministic, model-free inputs for the overnight GPU consumers;
it does not select cases, discover scripts, fetch data, or run a model.

## Plan format

```json
{
  "output_root": "$HEARING_RESULTS_DIR/overnight_prep/T-245",
  "total_timeout_s": 28800,
  "jobs": [
    {
      "case_id": "fixture-001",
      "argv": ["/absolute/path/to/python", "-m", "eval.some_existing_tool",
               "--case", "fixture-001"],
      "output_dir": "fixture-001",
      "reviewed": true,
      "available": true,
      "input_hashes": {"manifest": "<sha256>"},
      "config_hash": "<sha256>"
    }
  ]
}
```

`output_root` and each job's `output_dir` are resolved before execution; the
latter must remain below the former. The executable must exist and be
runnable. `argv` is passed to `subprocess.Popen(..., shell=False)` exactly as
an argument vector. Shells, network/file-transfer tools, package managers,
Git, and Slurm tools are rejected. Each case must have a unique ID and both
`reviewed` and `available` set to true. Plans are capped at 300 cases,
30 minutes per command, and 8 hours total.

## Replay, resume, and stopping

Dry run (does not create the output directory):

```bash
.venv/bin/python -m eval.prep_cpu_executor PLAN.json --dry-run \
  --report /tmp/prep-plan.json
```

Replay with a deterministic stop file:

```bash
.venv/bin/python -m eval.prep_cpu_executor PLAN.json \
  --stop-file "$HEARING_RESULTS_DIR/overnight_prep/T-245/STOP" \
  --report "$HEARING_RESULTS_DIR/overnight_prep/T-245/report.json"
```

The executor checks free space before starting and before every case, stopping
without starting another case below 5 GiB. A SIGTERM or stop file prevents
the next case from starting but allows the current subprocess to finish and
write its checkpoint. Every case gets `stdout.log`, `stderr.log`, and an
atomic `checkpoint.json`. A successful checkpoint is skipped only when its
command, executable-content hash, input hashes, and config hash still match;
failed or changed cases are eligible for one explicit replay. There are no
automatic retries or filler jobs.

The report distinguishes `planned_cases`, `executed_cases`, and
`skipped_cases`; `available_cases` is the number of reviewed/available jobs
present in the plan. The report and checkpoints contain the exact argv and
hashes needed by consumers to reproduce preparation. Consumer jobs should
read the report's case paths and verify checkpoint status before using output.
