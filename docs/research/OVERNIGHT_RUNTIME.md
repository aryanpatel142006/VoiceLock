# T-215 runtime evaluation — offline deliverable and verified benchmark ladder

Per orchestrator directive (t141 stays on laptop-a and will not be on the cluster),
evaluator and tests are delivered with full deadline ladder and caps.
Real-pair GPU runtime rows are marked not measured. No quality, live latency,
or demo-adoption claim is supported. The prior invalid Replay12 JSON was removed
and not replaced with fabricated clips.

## Verified inventory (2026-10-10, starting commit cd23dd540707f535ac81e5f1b1e42140702e6cae)

`ls eval/fixtures/generated/t141 profiles/aryan profiles/joseph` in this worktree
reported all three missing. The shared main checkout has the two profiles below,
but no `eval/fixtures/generated/t141` directory. Hashes were computed from bytes
on this machine at the starting commit; media/weights are ignored, not Git objects.
Paths below are relative to `/Users/joeroche/Developer/selective-hearing`.

| Existing file | Bytes | Verified SHA256 |
| --- | ---: | --- |
| profiles/aryan/profile-01.wav | 1600044 | ce3d9ac29754b5e13065aa8a6f1b4d01452e51cabd257beb5be1ceb908994cc8 |
| profiles/joseph/profile-01.wav | 502444 | e72d9fc7092003a204d4d3274f964e3a911b8e63b093da8f48f928a99f17c932 |
| models/MossFormer2_SS_16K/last_best_checkpoint.pt | 670353271 | 00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a |
| models/spkrec-ecapa-voxceleb/embedding_model.ckpt | 83316686 | 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2 |

Read-only SSH through the existing user-controlled ControlMaster succeeded.
`squeue -j 447101,447008` showed **447101 RUNNING on ilab1**, and no 447008 row.
Remote root: `/common/home/jdr317/hearing-joseph-overnight`.
The requested root-relative `eval/fixtures/generated/t141` and `profiles/*`
paths are absent there too. Existing profiles are instead in `data/profiles/*`;
`data/team_baseline/mix.wav` and `data/public_overnight/*/mix.wav` exist, but are
not evidence of the named fixed t141 Replay12 set. No valid Replay12 manifest
can be built from the requested files currently available. No mix hashes are
invented. Remote files were listed, not represented as locally verified hashes.

Existing queue markers: `overnight-first-separation.done`, `t289-bank.done`.
Results include `t210-a4500-20261010T0315` and `t289-a4500-20261010T0440`.
T-210 and T-289 are the identified allocation consumers; final report messages
hand off job/node/paths without credentials. Queue markers do not prove that a
GPU is idle. No allocation was created/cancelled, service restarted, dependency
installed, microphone/speaker opened, or remote source overwritten.

## Measured / missing status

| Requirement | Status | Evidence / limitation |
| --- | --- | --- |
| Fixed Replay12 files and per-mix hashes | **missing** | t141 directory absent locally and at advertised remote location |
| Local profile and model hashes | verified inventory | Table above; not proof of remotely loaded weights |
| GPU warm compute p50/p95/max | **missing** | Zero T-215 GPU cases run |
| Missed hops / weighted raw fallback seconds | **missing** | Instrumentation tested offline only |
| Target-switch delay | **missing** | Reference-similarity detector tested offline; requires aligned clean sources |
| Paired concurrent-load GPU arm | **missing** | Separate bounded synthetic process implemented, not run on GPU |
| Served revision / config hashes | **missing** | No real service probe performed; dry-run leaves them null |
| Remote WebSocket session smoke | **missing** | Optional probe is explicitly health-only |
| HTTP health round-trip / SSH RTT | **missing / missing** | No simulated timing; HTTP timer does not isolate SSH transport |
| Hardware microphone-to-ear latency | **missing** | Hardware access prohibited in this task |
| WER, SI-SDRi, listening quality | **not scored** | No ASR fallback or mock-derived quality claim |

## Runner and caps

Default plan: 12 fixed clips × baseline 2s/1s × 3 repeats = 36 cases, plus
three pairs of unloaded/synthetic-load replay on clip 0 and three target-switch
replays on clip 0: **45 cases**. All arms count against the **60 total-case cap**,
including `cases_used` from earlier attempts. Repeat and arm IDs are explicit.
The load arm never replaces a serial baseline case. Its control uses the same
clip, config, enrollment and repeat; arms run sequentially, with only the
explicit load process concurrent. A single 1024² CUDA matmul process sleeps
50ms between synchronized iterations, signals successful startup and is checked
for premature exit. It is a synthetic process workload, not a second Slurm
allocation or evidence of representative production contention.

For a two-config ladder, specify `configs: [[2,1],[1,0.5]]` and `repeats: 2`:
48 serial + 6 auxiliary = **54 cases**. This follows the reviewer's permitted
two-repeat example. Two complete configurations × 12 clips × 3 repeats already
exceed 60 before auxiliary arms; the runner rejects that plan rather than
silently truncating it. No default or profile is promoted by either plan.

Execution has a supervisor wall cap ≤90 minutes and an absolute stop at
**2026-10-10 14:00 UTC**. The supervisor kills the evaluator's process group,
including its load subprocess, if native loading or shutdown blocks. No work
is executed after the cap. A post-deadline dry run remains possible. Retries
must account for prior attempted cases in `cases_used`.

Every manifest entry requires `id`, `mix`, `target_enroll`, `other_enroll` and a
`sha256` mapping keyed by those audio fields. Exactly 12 unique IDs **and 12
distinct mix hashes** are required. Paths resolve from the invocation directory;
absolute paths are preferable for shared inputs. Every audio file must decode
as finite nonempty mono 16kHz audio in [-1,1]. Clip 0 additionally needs
`target_reference` and `other_reference` (also hashed), sample-aligned to its
mix, ≥12 seconds long, for the switch arm. Both speakers need active speech
around the switch to produce a measurable result.

`model_hashes` maps actual weight/config paths to SHA256 values. Execution
requires hashes for the files resolved by the existing MossFormer/ECAPA loaders
under `HEARING_MODELS_DIR`, not a weight file from another machine. The evaluator
records checked-out revision, dirty-source status, source hashes, evaluator hash,
manifest hash, plan/config hash and all verified input/model hashes. It never
automatically downloads models. Existing MossFormer2 and ECAPA models are used;
no model or licence is introduced.

Reproducible invocation once the genuine input manifest is supplied, from the
checked-out source and its environment:

```bash
export HEARING_MODELS_DIR=/absolute/path/to/existing/models
.venv/bin/python -m eval.overnight_runtime \
  --manifest /absolute/path/to/validated-replay12.json \
  --output /absolute/path/to/new-runtime-plan.jsonl
```

Only on the already-authorized allocation and within its caps, add `--execute
--gpu-minutes 90` and choose a **new** output path under the shared results
root. Do not use `gpu_up.sh` to acquire or replace an allocation. All validation
and plan construction happen before the evaluator starts work. Output uses
exclusive creation, flushed/fsynced JSONL plan, case and stop checkpoints. A
failed case is checkpointed and exits nonzero. Optional `--health-url
http://127.0.0.1:EXISTING_PORT` probes an existing service **after** checkpoint
creation. It starts no server, opens no session and reports `health_only`.
A future session-smoke owner must bind its server to port 0 and discover the
actual bound port; this evaluator does not select a fixed port or claim a
health probe is a session smoke.

## Metric definitions and limitations

- Compute is the actual FocusHQ threaded wrapper's per-window host elapsed
  extract time, excluding windows marked cold. It is not pure CUDA kernel time,
  network RTT or microphone-to-ear delay. Misses count resolved late/overrun
  window IDs once, retaining all observed records rather than rolling percentiles.
- Raw-fallback seconds are **approximate**: end-of-10ms-frame `_mix_gain` times
  played samples. Each result includes that note. Private `_recent` and
  `_mix_gain` dependencies are covered by an offline wrapper test.
- The switch arm swaps the enrolled target and negative halfway through clip 0.
  Output is compared to aligned old/new references in 100ms blocks. It requires
  three old-dominant blocks before the command, then three new-dominant blocks
  with absolute cosine advantage >0.1. Missing/silent/ambiguous transitions
  return null, never zero. Delay is on the paced replay sample clock and includes
  wrapper playout delay; scheduling stalls are not an acoustic clock measurement. This acoustic proxy can be confounded by a speaker
  becoming silent and is not a semantic or perceptual guarantee. References
  must match the replay's exact mix alignment.
- Quality is always `not_scored`; WER and hardware latency remain null. Offline
  generated signals and a candidate double exercise plumbing only. The HTTP
  test uses a local port-0 test server and verifies timing/provenance fields;
  that fixture is not a measured remote service.

## Verification

- `tests/test_overnight_runtime.py`: 15 passed in 1.72s.
- `bash scripts/verify_fast.sh`: PASS (1741 passed, 58 deselected in 106.42s; Node test runner: 67 passed, 0 failed; ruff: clean).
- Evaluator validation and cap constraints verified offline.
- Real-pair GPU runtime rows recorded as missing/not measured per orchestrator direction.
