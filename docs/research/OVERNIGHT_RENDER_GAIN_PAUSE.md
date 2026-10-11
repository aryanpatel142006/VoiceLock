# Overnight render: gain disparity and pause/absence

## Consumer instructions

This preparation shard is offline and does not change production profiles. From the
repository root, run:

```sh
.venv/bin/python -m eval.prep_render_gain_pause \
  --sources eval/fixtures/generated \
  --out "$HEARING_RESULTS_DIR/gain_pause" --limit 12
```

The first 12 cases are a complete, inspectable prefix. Increase `--limit` up to 600
for the long batch. `checkpoint.json` is atomically updated after every case; rerunning
the same command resumes completed case IDs. The batch stops at 90 minutes or 2 GiB.
For a worktree-safe overnight run, copy this script to a versioned ignored artifact
under the shared results directory and invoke the main checkout's `.venv`; record the
copy path, PID, command, and stop/resume command in the run log.

Each case directory contains `mix.wav` and `manifest.json`. `source_crops` records
absolute immutable source paths, source IDs, speaker IDs, and roles. `enrolment_group`
and `final_group` remain separate. `sir_db` is one of 0, -6, -10, or -15 dB;
`target_on`, `target_off`, and `target_pause` schedules and onset shifts are explicit.
A third stem is emitted only when three distinct speaker IDs exist. Truncated or
paused cases have an empty transcript rather than an inaccurate full-clip transcript.

## Rendering contract

Audio is loaded as mono float32 at 16 kHz. Source amplitudes are untouched. The
interferer is scaled to the requested SIR and the mix is a float32 sum with no implicit
normalisation. `declared_overrange` and `peak` identify cases exceeding [-1, 1];
WAV writing clips only as a serialization limitation. This is preparation data, not
production adoption.

## Observations and hypotheses

No source/profile recordings were available in this worktree, so no overnight render
measurements are claimed. The hypotheses to test are that stronger negative SIR
increases gain-disparity stress, and that explicit target pauses/absence separate
“target silent” from “target missing”. Results must be reported with the source set,
case count, host, seed, and actual measured metrics; never infer them from this plan.
