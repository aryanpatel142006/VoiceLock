# T-319: frozen speaker-head loss experiment

This prepares an offline comparison for Joseph's existing **447101** allocation on
**ilab1 / RTX A4500**. It does not launch a job, download models, access audio devices,
or change SAME_VOICE in the live backend. The calibrated probability is an offline
candidate replacement for the fixed cosine `.7` rule; deployment requires a separate
consumer change and real holdout validation.

## Reproduce on CPU

From a checkout with the shared `.venv` (existing `ml` extra supplies torch and
scikit-learn; no new dependencies):

```sh
.venv/bin/python -m pytest -q gpu_jobs/T-319/test_job.py
.venv/bin/python -m ruff check gpu_jobs/T-319
bash gpu_jobs/T-319/reproduce.sh synthetic /tmp/T-319-cpu
bash scripts/verify_fast.sh
```

Synthetic mode explicitly generates a deterministic confusable 192-d pair, seed
309, with noise varying by duration (0.5/1/2 seconds). **This is a stand-in, not the
T-309 dataset**: this checkout's T-309 documents describe an AGC evaluation, and no
named T-309 synthetic artifact or T-291/T-310 embedding cache was available.
Supply the authentic artifact via `DATA` in real mode after adapting its metadata.
No real-mode failure falls back to synthetic. Synthetic proves execution and
bookkeeping only, never male/male voice performance. The default 100 CPU steps
are a smoke run, not the 2000-step GPU experiment.

## Input contract and upstream handoff

Flat NPZ, `allow_pickle=False`, one row per frozen embedding:

| Key | Shape | Meaning / aliases |
| --- | --- | --- |
| `x` | N x D float | Nonzero finite embeddings; alias `embeddings` |
| `speaker` | N string or int | Speaker identity; aliases `labels`, `y` |
| `duration` | N float | Positive crop seconds; aliases `duration_s`, `crop_duration_s` |
| `source_id` | N string | Original recording ID; alias `recording_id` |
| `split` | N string | Exactly `train`, `enroll`, `calibrate`, `test` |

Extra arrays such as `aug` are allowed. T-310 augmentations **must inherit the
original recording ID and split**, not a new ID per augmented clip. Never randomly
split overlapping crops or label an augmentation as a separate source. A recording
ID may occur in only one split. Enrollment must cover every calibration/test
speaker; training requires >=2 speakers with >=4 distinct crops each. The reader
rejects missing metadata instead of guessing. It does not flatten padded 4-D crop
caches. Upstream T-291/T-310 schema compatibility is provisional until their actual
files are supplied: adapt explicitly to this contract, preserving source provenance.
IDs are trusted metadata; the script cannot detect dishonest IDs or aliases for
the same recording. If each teammate has only one recording, an independent
recording-disjoint team evaluation cannot be claimed.

## Methods and artifacts

All three methods train the same D -> 64 -> 32 ReLU MLP, normalize input/output,
use identical seeds, speaker-balanced batches without crop replacement, Adam
(lr=.001), and the same number of updates. No backbone is trained.

* AP: clean-room angular prototypical cross entropy. First crop is the query;
  remaining crops form a normalized centroid. Learnable positive cosine scale,
  capped at 100. No external source was copied; T-307 also already has AP code.
* SubCenter ArcFace: k=3 trainable centers per speaker, max cosine center, target
  angular margin .3 rad, scale 16, with monotonic correction near pi.
* CE: stock linear classifier plus cross entropy on normalized head outputs.

Loss layers are discarded for inference. `{ap,subcenter,ce}.pt` hold head and
loss-layer state dictionaries, dimensions, class map and CLI settings. Load a
`Head(input_dim)`, its `head` state dict, set eval, and feed frozen embeddings.
They are final checkpoints, not resumable optimizer snapshots. `.pt`/`.npz` outputs
stay outside git. `metrics.json` records seed, host/device, steps, data SHA256,
wall time, mean/p05 margin and flip rate. Enrollment centroids come solely from
enrollment rows. Margin is own-profile cosine minus strongest other-profile
cosine; flip rate counts nonpositive margins (ties conservatively count as flips),
not temporal label-switch frequency. No EER is computed.

`*_pairs.npz` contains all query/profile pairs on calibration/test splits with
(score, duration, **input embedding norm before this script's normalization**).
`calibrate.py` standardizes these features on calibration rows only and fits
regularized logistic regression. JSON stores mean/scale/coefficients/intercept;
`predict(path, features)` returns P(same speaker). Offline acceptance uses P>=.5.
Metrics compare it to cosine>=.7 and include held-out Brier/log loss. If upstream
ECAPA exports unit vectors, norm carries no quality information. All-profile
pair sampling sets the prior (1 / number of profiles); deployment with a different
prior, unknown speakers, or target-absent windows needs separate validation.

## ilab1 execution

Sync this directory and prepared NPZ into `~/hearing-remote` with its existing
`.venv`. From the login node, after confirming allocation ownership and remaining
time externally:

```sh
cd ~/hearing-remote
srun --jobid=447101 --overlap --nodelist=ilab1 bash gpu_jobs/T-319/run_ilab.sh
```

The launcher refuses any other allocation/node and verifies an A4500 CUDA device.
`job_spec.yaml` is the precise step specification. **sbatch cannot attach to an
existing job ID**; do not submit a new allocation or use jobs 435775/447008 or
ap2772's resources. No remote operation was performed during preparation.
Default data: `~/hearing-remote/gpu_jobs/T-319/data/team_slices.npz`.
Set `DATA` to the adapted T-310 cache if desired, `OUT` for a unique result directory,
and `STEPS` for bounded timing. Output defaults to `eval/results/T-319/seed{0,1,2}`.
Expected GPU runtime 5–20 minutes is an **unmeasured estimate**; operator must stop
at two hours and first time 100 steps to confirm remaining budget. No CPU/memory
SLURM flags are added. The reuse launcher deliberately has no fallback reservation.

## What would justify the bet

On a frozen recording-disjoint team test manifest, both targets and duration bins,
over seeds 0/1/2: a candidate must improve mean pair margin by >=.05 **and** reduce
flip rate by >=.02 absolute versus CE, without degrading either target's flip rate.
Inspect per-target/duration results using the same manifest; the aggregate JSON
alone does not establish that gate. Bootstrap by recording for confidence bounds,
not overlapping crops. Calibrated acceptance must improve Brier/log loss and reduce
held-out threshold error versus .7, with target-absent/unknown-speaker false accepts
checked before deployment. Synthetic or LibriSpeech stand-ins cannot pass the team
gate; absent independent recordings means no adoption verdict.

Measured CPU smoke (MacBook Air, seed 0, 100 updates, 96 test queries):

| Method | Mean margin | Flip rate | Wall seconds (train + eval) |
| --- | ---: | ---: | ---: |
| raw | .07082 | 5.21% | .00042 |
| AP | .63713 | 6.25% | 1.28486 |
| SubCenter | .88065 | 3.13% | .04322 |
| CE | 1.71858 | 4.17% | .02724 |

The AP time includes first-use optimizer initialization, so these wall times are
not GPU throughput comparisons. AP increased mean margin yet worsened flips;
the smoke result does not establish a winning head. Calibration Brier scores
raw/AP/SubCenter/CE: .17945/.05649/.03574/.03401; threshold errors with calibration:
30.73%/6.77%/4.17%/4.17% (192 test pairs).

Model register proposal for the owning docs task: `T-319 D->64->32 frozen-ECAPA
MLP + AP/SubCenter/CE | clean-room project code; no repository-wide redistribution license found
(no new third-party model/weights) | offline, 32-d output; backbone license unchanged`.
No external model or GPL implementation was introduced.

Verification: task tests 5 passed (2.15 s); task ruff passed; synthetic and
three-seed real-mode command paths completed on CPU (real mode used the synthetic
NPZ only). Repo verify_fast passed: 1762 Python tests, 58 deselected, 128.40 s,
and 67 Node tests. Git staging was refused by the sandbox at
`.git/worktrees/T-319/index.lock`; the harness must commit these files.
