# T-210 frozen baseline — partial; acceptance blocked

This is the current evidence record, not a completed suite or demo adoption.
Historical authorized A4500 Slurm step 447101.1 ran 2026-10-10 03:12–03:14 UTC,
exit 0:0, 2 min 4 s. Both setup and measured-baseline caps were 45 minutes.
This revision performs read-only retrieval/offline checks, zero additional GPU
minutes, no allocation requests/cancellation, installs, credentials, or audio hardware.
The original reuse-only A100 job 447008 was cancelled externally. Later feedback
authorized the historical A4500 run. No current allocation authorization is inferred.
The recorded no-new-work deadline was Oct 10 12:00 UTC, hard stop 14:00 UTC;
this revision does not extend either cap/deadline.

## Frozen hashes and reproducible commands

Remote root: `~/hearing-joseph-overnight` (jdr317). Frozen source archive:
`checkpoints/t210-20261010T0250/source`, REVISION 34e567386606885f7a453eaae898cb2483242993.
`results/t210-freeze-20261010T0250.json` inventories 876 collection-time hashes.
`results/t210-a4500-20261010T0315/hashes.json` inventories source, profiles,
models and team/public inputs. Media and weights remain remote and uncommitted.

## Identity and controls

Local starting HEAD: `34e567386606885f7a453eaae898cb2483242993`.
Remote has no original `REVISION` marker. Post-run comparison of all 775 tracked
regular source files against that commit found no byte differences; the
`.agents/skills` symlink is absent from the file inventory. This is collection-time
verification, **not proof of historical loaded identity**. Collector intentionally
leaves `source_revision_verified=false`. No shared remote repository was overwritten.

Config: 16 kHz, 2 s window/1 s hop, both roles, heldout second-half scoring,
SIR 0/-5/-10 dB, separation threshold 4 dB, drop threshold 3 dB, AGC off,
noise off. Scene is a digital combination of existing disjoint team holdouts,
first 16 seconds; enrollment is the first sorted profile WAV per person.
No room-acoustics, easy/hard coverage or public-data generalization claim.
Active profile/config byte hashes are in the checkpoint; no defaults changed.

Loaded model files present at collection:

| Model | SHA256 | Licence recorded by existing project |
| --- | --- | --- |
| MossFormer2_SS_16K last_best_checkpoint.pt | 00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a | Apache-2.0 (existing docs/MODELS.md) |
| ECAPA embedding_model.ckpt | 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2 | Apache-2.0 (existing docs/MODELS.md) |


The historical measurement script is now reviewable as `measure()` in
`eval/overnight_baseline.py`. Original script SHA256:
`652b70fb35398cc4f6720a2fc24ad52b16024648fbb3869b748a4d6e763462e0`.
Versioned module SHA256: `dbd7129b98ba7e7b8e7bd988d1cf839b8e7debc7e5de597662752996a02e1aec`.
The repo version parameterizes root/output, binds loop profiles explicitly and
uses min rather than sorted-first; measurement conditions remain the same.
It deliberately reproduces the incomplete historical public crop, not overlap.

| A4500 artifact under results/t210-a4500-20261010T0315 | SHA256 |
| --- | --- |
| readiness.json | 9b6548c142cf625671ced55900af39bb98f9ec09a1cdf9d7404b7eb997c88adb |
| team.json | 9fa7a8cc2529852f50c61d0f58cf97ba8260d2b707debef5dac3a0ef2aa5cc2a |
| public.json | 521edd21b1dea5c58fe54338f0d54f4d12c2f3bd215b90ddb0e557dbded986a4 |
| asr.json | 21a1680fc505162a066efd54a749a38424c729da2111b3c1e687eded1732d645 |
| hashes.json | 2a2242f98b971ddbbea5b2a2ec946d7f2a1b5fac1e3d6f3782b7e00b4732bf1e |

Read-only collector (uses real ASR artifact shape, rejects mock evidence,
records output hashes and unverified-reference scores):

```bash
.venv/bin/python -m eval.overnight_baseline --root /path/to/overnight \
  --source-revision 34e567386606885f7a453eaae898cb2483242993 \
  --results-dir /path/to/overnight/results/t210-a4500-20261010T0315 \
  --out /path/to/new-freeze.json
```

Historical replay recipe, **only with renewed authorization within new caps**:
copy this versioned module to the remote host as `/path/to/overnight_baseline.py`;
use the existing frozen source via PYTHONPATH and a fresh output directory.
Do not use a mutable remote script or request another allocation.

```bash
srun --jobid=AUTHORIZED_EXISTING_JOB --overlap bash -c '
cd ~/hearing-joseph-overnight/checkpoints/t210-20261010T0250/source
export PYTHONPATH="$PWD" HEARING_DEVICE=cuda OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HEARING_MODELS_DIR="$HOME/hearing-joseph-overnight/models"
timeout 2700 ~/hearing-joseph-overnight/.venv/bin/python \
  /path/to/overnight_baseline.py --measure \
  --root "$HOME/hearing-joseph-overnight" \
  --source-revision 34e567386606885f7a453eaae898cb2483242993 \
  --out "$HOME/hearing-joseph-overnight/results/UNIQUE_REPLAY"
'
```

## A4500 measured results

### Readiness and steady-state evidence

CUDA true; **NVIDIA RTX A4500**, allocation log 19,190 MiB, torch 2.8.0+cu128.
Real MossFormer2/ECAPA loading: **10.0899 s**; first public 2-second window
warmup: **7.0250 s** (includes profile preparation and inference; CUDA synchronized).
Warmup was excluded from the table; subsequent condition preparation can still
be included in first-window timing. Existing diagnostic compute measures wall
call time, not independently CUDA-synchronized kernel timing or live latency.

| Target | SIR dB | Actual/oracle SI-SDR dB | Mean compute ms | Max compute ms |
| --- | ---: | ---: | ---: | ---: |
| Joseph A | 0 | 16.517 | 227.339 | 305.536 |
| Joseph A | -5 | 14.277 | 278.312 | 314.247 |
| Joseph A | -10 | 11.612 | 313.079 | 383.020 |
| Aryan B | 0 | 13.262 | 265.775 | 327.879 |
| Aryan B | -5 | 12.257 | 330.120 | 628.229 |
| Aryan B | -10 | 9.087 | 275.289 | 344.624 |

Eight scored heldout windows per team condition; no diagnostic deadline misses.
Aryan -10 has one separation and one mix-back failure, six OK. Oracle agreement
is not measured wrong-speaker duration. Knobs unchanged: 2/1 seconds, thresholds
4/3 dB, AGC off, no team noise, both roles, first sorted existing profiles.

Small public baseline: existing LibriSpeech CC BY 4.0
`take0_1320_sir0_snr15`, profile 1320/6930, both roles, ratio 0,
noise enabled, 2/1 seconds, heldout half. **Role A has zero active target windows**:
11 absent-target windows, leak -34.161 dB vs mix, mean/max 263.155/334.496 ms.
Role B has 11 target-only windows, actual/oracle SI-SDR 23.654 dB,
mean/max 282.364/350.453 ms. This is a small target-absent/clean-tail check,
**not a public overlapping-speech quality result**. Padding and utterance-length
imbalance mean the second half cannot supply that claim.

### Actual ASR clean/mix comparison and remaining reference blocker

Existing model row: faster-whisper **base.en**, MIT code/weights,
`Systran/faster-whisper-base.en`, snapshot
`3d3d5dee26484f91867d81cb899cfcf72b96be6c`, model.bin SHA256
`2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef`.
CUDA float16, English, beam size 5, whole clip, no adapter fallback.
Load **3.4733 s**, first 2-second transcription warmup **0.5318 s**.
23.315-second public target: **0.2087 s**, RTF **0.00895**;
mix: **0.4792 s**, RTF **0.02055**. Loading excludes prior download time.
The clean transcript matches the supplied reference apart from punctuation/case (this match is only a consistency check, not independent verification);
mix substitutes “sending” for “something” and also transcribes interferer speech.
Complete hypotheses and supplied reference are in `asr.json`.

No team ASR quality, room generalization,
wrong-speaker duration, live-wrapper latency, or demo adoption is claimed.
The flagged team references remain unverified; independent audit is unavailable
and is not the blocking condition. No microphone/speaker was opened. Missing
acceptance measurements are listed below.


## WER against current references

No independent audit was available; reference audit is not a demo blocker per
orchestrator correction. References are not corrected from ASR agreement.
Word-level Levenshtein uses lowercase Unicode word tokens, strips punctuation,
retains internal ASCII apostrophes, counts insertions; WER can exceed 100%.
Empty references raise ValueError. No number below is verified-reference WER.

| Reference provenance | Public input | Errors / words | WER |
| --- | --- | ---: | ---: |
| reference unverified (auto transcript) | clean target | 0 / 22 | 0.00% |
| reference unverified (auto transcript) | raw mix | 50 / 22 | 227.27% |

The raw-mix hypothesis includes interferer words, compared to a target-only
reference. This is whole-clip ASR, not speaker attribution or team ASR.

## Gaps and blocked acceptance

- Team clean/mix ASR hypothesis set: not measured.
- Public overlapping-speech scored crop: not measured; existing heldout tail is
  target-absent for A and target-only for B.
- SepFormer comparison, wrong-speaker duration, live-wrapper deadline ladder,
  contrast and floor_ab: not measured; no claims inferred from oracle agreement.
- Independently audited/corrected references: unavailable, explicitly unverified.
- Four real-scene listening notes, team ASR RTF/live latency: not measured.

Further measurements need a current authorized executor/allocation and renewed
stop window. Status is blocked for missing acceptance measurements, not for
reference audit alone. No quality from mocks and no automatic demo adoption.

Superseded: A100-era timing tables, pending-confirmation blocker logs and three
older verify blocks were removed from this current-state document. Original
A100 artifacts remain historical remote evidence, not mixed into A4500 timings.

## Verification of this revision

`bash scripts/verify_fast.sh` exited 0: Ruff passed; 1473 pytest passed,
56 deselected in 81.00 s; Node 53 passed, 0 failed; `verify_fast: PASS`.
Targeted offline tests: 6 passed in 1.63 s. Regression observed red before fix
(ASR status missing), then green. `git diff --check` exited 0.
Historical replay logic was retrieved/reviewed, not rerun on GPU this turn.
