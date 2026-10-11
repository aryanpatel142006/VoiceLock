# T-289 MossFormer2 saved GPU bank — partial; t141 acceptance blocked

## Status and handoff

Read-only recovery/audit on 2026-10-10 around 10:30 UTC found an existing
**eight-case** bank, not the requested 12-case t141 run. SepFormer was skipped
per the orchestrator's T-202 decision. No runtime, default, profile or model was
adopted. No GPU inference, allocation, cancellation or install was performed
in this recovery; job 447101 remains untouched. T-215 published its offline
report and explicitly recorded zero measured GPU cases. Its requested frozen
Replay12 does not exist, so the serial measurement prerequisite is not satisfied.

Host: `jdr317@ilab1.cs.rutgers.edu`, NVIDIA RTX A4500.
Run: `t289-a4500-20261010T0440`, reported elapsed **116.634 s** (1.944 min).
This is exporter wall time, not independently accounted Slurm GPU usage.

Bank directory (cluster results, excluded from Git; no media committed):
`/common/home/jdr317/hearing-joseph-overnight/results/t289-a4500-20261010T0440`

- Original `manifest.json` SHA256:
  `2b17ab7e581da88ea896ed93ab96fe85318da03487857ff067d43948204d2a5d`.
- T-213 entrypoint: **`t213-scorer-manifest.json` in that same directory**,
  SHA256 `bd660954a4502b81dc8dfefc01727f0641791a3822d27fde71a617a524d5e103`.
  This separate adapter adds the scorer-required `asr_model: small.en` and
  original-manifest hash without changing any audio or selector evidence.
  It declares the requested scorer identity; it does not claim ASR ran.
- All **80 original artifact hashes** and **157 frozen provenance file hashes**
  were recomputed on the cluster with zero mismatches. All seven required
  waveforms per case were also decoded locally and verified mono 16 kHz,
  finite, nonempty and equal-length, with contiguous selector emission coverage.
- Each case has aligned target/interferer/noise/mix, raw-index stream0/stream1,
  actual candidate-selected audio, actual `Mixer.process` delivered audio,
  historical oracle splice, and actual hop/pick/score JSON. The manifest owns
  source/model/profile provenance; individual selection JSONs contain hop
  evidence rather than duplicating those hashes. Consumers must retain both.
- Frozen exporter: `checkpoints/t289-20261010T0440/source/eval/separator_ab_gpu.py`
  relative to `/common/home/jdr317/hearing-joseph-overnight`, SHA256
  `852540aecda8682105a1175df1c6f76a38fc9502ee24bf4ee3d76587741e983e`.
  The newly checked-in module is a **read-only auditor**, not that historical
  exporter and not permission to rerun or extend the GPU work.

T-213 can pass the adapter manifest to its `b2c831c` scorer with `--limit 8`
and an existing frozen local small.en model. No small.en scoring was performed
here; missing transcripts remain missing and team references stay
`automatic_unverified`. Preserve the original provenance with downstream results.
Do not describe this as first12 or expand it without the missing inputs and
serialized execution handoff. No new cases after 12:00 UTC; hard stop 14:00 UTC.

## Frozen provenance and conditions

Manifest-reported source revision: `9ad4eed0bde28fadef261fc1d7b447ee5da41853`.
Source file hashes match the snapshot today; this is not independent proof of
historical loaded revision. Frozen reference: `t210-a4500-20261010T0315`.
Six team cases reuse the team-baseline stems at SIR 0/-5/-10 dB, both roles;
two public cases reuse take0_1320_sir0_snr15 with both roles. These are digital
mixtures, not the missing t141 easy/hard scenes. Full clips are scored here,
unlike T-210's heldout-half diagnostic; do not directly compare those tables.

2 s windows / 1 s hop; first full window then each last hop, final zero padding.
Stream indices can permute between windows. Selected audio is the real
EnrolledCandidate return. Delivered audio is the real mixer at suppression 1,
**AGC off**, without the asynchronous live wrapper, deadline fallback or its
crossfades. Thus these are diagnostic delivered outputs, not live AGC outputs.

The manifest includes source/model/input/profile hashes, but no YAML config
hashes: the exporter sets the diagnostic configuration directly. Canonical
scalar-config SHA256 (sorted compact JSON of window_s, hop_s, suppression, agc,
live_wrapper): `75265e0a14558ed456eb8b26d7a03904eee83189cda2afce88717ef232d59705`.

| Existing model | Weight SHA256 | Existing licence |
| --- | --- | --- |
| MossFormer2_SS_16K | `00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a` | Apache-2.0 |
| ECAPA embedding_model.ckpt | `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2` | Apache-2.0 |

No new model introduced. Both profiles and all other relevant weights/source
files have full hashes in original `provenance.input_sha256`.

## Per-case measured results

All SI-SDR values are absolute dB; columns marked **i** subtract the aligned
input SI-SDR. Best saved stream is the whole-clip maximum of saved raw indices.
It is an oracle best-index diagnostic, **not a quality ceiling** when indices
permute. Historical window oracle selects on whole 2 s windows before emitting
last hops; it is also not a guaranteed whole-clip ceiling (public 1320 demonstrates
this). Neither replaces actual identity selection or delivered output.

Wrong-speaker % is the historical delivered-audio least-squares stem attribution
on 250 ms target-active blocks (`eval.pipeline_eval.routing`), not classifier
error or an ASR metric. Compute is synchronized candidate-call wall ms/window,
including per-case first-call enrollment preparation; global warmup was separate.
The large team p95 values include those first calls and are not warm latency.
No RTF or microphone-to-ear latency measurement is asserted.

| Case | Input | Selected | Selected i | Delivered | Delivered i | Best saved stream | Historical window oracle | Wrong % | Compute p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| team-joseph-sir0 | 0.06 | 16.66 | 16.60 | 16.67 | 16.61 | 16.66 | 16.66 | 0.0 | 302.1/2258.0 |
| team-joseph-sir-5 | -4.89 | 14.25 | 19.13 | 14.23 | 19.11 | 14.25 | 14.25 | 0.0 | 297.8/2284.4 |
| team-joseph-sir-10 | -9.80 | 11.16 | 20.96 | 11.11 | 20.91 | 11.16 | 11.16 | 1.8 | 303.3/2281.9 |
| team-aryan-sir0 | 0.06 | 13.31 | 13.24 | 13.36 | 13.29 | 13.31 | 13.31 | 6.2 | 298.9/2259.5 |
| team-aryan-sir-5 | -4.89 | 12.28 | 17.17 | 12.30 | 17.19 | 8.01 | 12.28 | 10.9 | 314.8/2259.8 |
| team-aryan-sir-10 | -9.80 | 10.18 | 19.98 | 9.68 | 19.48 | -6.32 | 10.18 | 14.1 | 314.7/2382.1 |
| public-1320-sir0 | -0.09 | 17.10 | 17.19 | 10.96 | 11.05 | 2.14 | 1.85 | 0.0 | 296.0/384.3 |
| public-6930-sir0 | -0.07 | 17.80 | 17.87 | 17.78 | 17.85 | -1.62 | 17.80 | 1.1 | 289.2/376.3 |

## Absence, ASR and missing coverage

The audit defines acoustic false activation on 100 ms blocks: target RMS <=1e-4
means absent; delivered RMS >1e-4 means active. This is an evaluation threshold,
not a product threshold or calibrated selector FAR. Public-1320 has **17.015 s
absent, 17.015 s delivered-active, acoustic FAR 100%**. Other cases have no
absent blocks under this definition, so FAR is null, not zero. This includes
residual noise and is not evidence that every block contains a wrong voice.
No wholly silent target is assigned SI-SDR or an oracle pick by the auditor.
False-caption counts and WER are **not measured**; no mock or alternate ASR was
substituted. Listening quality is unmeasured.

Missing: t141 Aryan/Joseph easy/hard sources at 0/-10 dB, a validated 12-case
manifest, T-215 measured/frozen handoff, live-wrapper/AGC delivered outputs,
and false-caption scoring. Remote recursive inspection found no t141 directory.
These block full task acceptance; the existing eight cases must not be relabeled
or padded with duplicate/synthetic cases. Supplying the original approved files
and a serialized execution window is required for further measurements.

## Reproduce the audit without GPU work

Copy the original bank intact to an ignored results directory, then:

```sh
.venv/bin/python -m eval.separator_ab_gpu --bank /path/to/bank/manifest.json \
  --output /path/to/new-audit.json
```

The auditor verifies hashes, exact alignment and selection coverage before
scoring. It does not rewrite the bank or synthesize selected/delivered audio.
Offline tests cover tampering, missing hashes, unequal lengths and silent targets.

Verification on this revision: four audit tests passed in 1.04 s; full
`bash scripts/verify_fast.sh` passed (1755 pytest tests passed, 58 deselected,
105.83 s; 67 Node tests passed; ruff clean). No hardware or network tests run.
