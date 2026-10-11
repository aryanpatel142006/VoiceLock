# Angular prototypical loss head vs cross-entropy head (T-307)

Steal #1 from muse-c's SOTA list: train the compact 32-d speaker head with the **angular prototypical (AP)
loss** (Chung et al., Interspeech 2020), which optimises the cosine geometry the enrolled tier uses at
inference, instead of a classification proxy. Prediction from the synthesis: +2 to 5 points on hard
male/male pairs. (The synthesis file `~/workspace/swarm-sota/SWARM_SYNTHESIS.md` was not on this host;
the method comes from the task text and the paper.)

**Verdict: not shown on real data.** On synthetic speakers AP beats both CE and raw ECAPA. On real ECAPA
features, with heads trained on the only 30 speakers available locally, **raw ECAPA (no head) has the
lowest flip rate**. AP beats the CE head on nearly every real metric and widens the team margin, but it
also makes more team crops flip. The GPU run (251 to 1172 training speakers) is the real test of the idea.

## Files
- `eval/heads/T-307/ap_head.py`: `MLPHead` (192→64 Linear, BatchNorm, ReLU, 64→32 Linear, L2 norm),
  `AngularPrototypicalLoss`, `train_ap_head`, `train_ce_head` (same MLP plus a linear softmax classifier
  that is thrown away after training), `evaluate_prototypes`, a synthetic speaker generator, and ECAPA
  feature extraction for LibriSpeech / the T-306 set. CLI: `python eval/heads/T-307/ap_head.py synthetic|real`.
- `tests/test_ap_head.py`: 8 unit tests, about 2 s in total, no weights needed.
- `experiments/gpu/T-307/train_ap_head.py` + `run_ilab.sh`: the iLab A100 run (below).

## Method
- **AP loss.** Each episode takes N speakers × M = 4 crops. Crop 0 is the query and the mean of crops 1-3 in the
  32-d space is the prototype. Logits are `w·cos(query_i, proto_j) + b`, with learnable w (starts at 10, clamped > 0)
  and b (starts at -5). The loss is cross entropy on the true prototype, i.e. `-log softmax` over the cosines.
  N = 32 locally. Adam, lr 1e-3, wd 1e-4, 1500 steps.
- **CE baseline.** The same MLP, a linear classifier on the raw 32-d output and softmax CE. Batch of 128 crops,
  same optimiser and number of steps. Evaluation uses the normalised 32-d output.
- **Raw.** Unit ECAPA 192-d with no head: what the live enrolled tier uses today.
- **Metrics (open-set).** Each speaker's prototype is the mean of its enrollment crops. Every query crop is scored
  by cosine against all prototypes.
  - `margin` = cos(own prototype) − max cos(other prototypes).
  - **`flip rate`** = share of queries with margin < 0. This simulates the enrolled stage picking the wrong
    speaker's stream for that crop.
  - `hard_*` = the same metrics against each speaker's hardest partner only: the same-sex speaker whose raw
    prototype is closest. For the team set this is the other team member.
  - All numbers are means over 3 seeds.

## Results

### Synthetic speakers (`ap_head.py synthetic`, 3 seeds, 1500 steps, CPU, 2.7 s per seed)
Setup: 192-d centres in confusable pairs (centre cosine 0.8), a shared rank-8 nuisance subspace
(channel/codec stand-in, scale 0.35) and isotropic noise 0.05. Training uses 200 speakers × 20 crops. Evaluation
uses 40 unseen speakers: 5 enrollment crops and 15 query crops each, 600 queries.

| head | margin | flip rate | hard-pair margin | hard-pair flip |
|---|---|---|---|---|
| raw | 0.112 | 19.7% | 0.114 | 19.6% |
| **AP** | **0.139** | **10.1%** | 0.156 | **8.7%** |
| CE | 0.127 | 13.9% | **0.161** | 10.7% |

AP vs CE: −3.8 points of flip and −2.0 points of hard-pair flip. That is inside the predicted +2 to 5 points, but
only on data built to contain a removable nuisance subspace.

### Real ECAPA features (`ap_head.py real`, 3 seeds, 1500 steps, CPU, wall 25.8 s incl. cached features)
- Features: ECAPA (`spkrec-ecapa-voxceleb`, MPS) on 2.0 s crops.
- LibriSpeech test-clean: 40 speakers (20 M / 20 F), up to 40 utterances × 2 crops = 2781 crops. Each seed holds
  out 5 M + 5 F speakers (10 enrollment crops each, 597 queries; male/male hardest pairs give 321 queries) and
  trains on the other 30 speakers (2084 crops).
- T-306 team set: aryan vs joseph, both adult male, 1784 crops at a 1 s hop. Enrollment = the clean crops of
  `profile-*`. Queries = every `fixture-*` crop in all 9 augmentation groups (1193), or the clean ones only (117).
- Host: laptop-a (Aryans-MacBook-Pro, M4 Pro), torch 2.14.1. Raw JSON is in `$HEARING_RESULTS_DIR/T-307/`.

| set | head | margin | flip rate | hard margin | hard flip |
|---|---|---|---|---|---|
| LibriSpeech held-out (10 spk) | raw | **0.402** | **0.17%** | 0.431 | **0.06%** |
| | AP | 0.267 | 10.6% | **0.455** | 3.4% |
| | CE | 0.213 | 14.7% | 0.423 | 6.2% |
| LibriSpeech male/male hardest pair | raw | | | **0.413** | **0.0%** |
| | AP | | | 0.378 | 4.8% |
| | CE | | | 0.384 | 6.0% |
| team, clean fixture queries | raw | 0.533 | 2.6% | | |
| | AP | **0.623** | 2.6% | | |
| | CE | 0.425 | 5.4% | | |
| team, all augmentations | raw | 0.337 | **10.7%** | | |
| | AP | **0.369** | 17.7% | | |
| | CE | 0.274 | 19.5% | | |

Training-length check, same setup:
- 300 steps: AP team-aug flip 19.7%, LibriSpeech flip 11.2%.
- 4000 steps: AP team-aug flip 18.0%, team-clean flip **1.4%** (raw 2.6%), LibriSpeech flip 11.1%.

The gap to raw is not an under- or over-training effect. It is the 30-speaker training set: a 32-d bottleneck
fitted on 30 voices throws away directions that separate unseen voices.

Team flip rate by augmentation group (seed mean; queries per group in brackets):

| group | raw flip | AP flip | CE flip | raw margin | AP margin |
|---|---|---|---|---|---|
| clean (117) | 2.6% | 2.6% | 5.4% | 0.533 | 0.623 |
| opus16 / opus32 (117 each) | 3.4% / 1.7% | 4.6% / 2.8% | 6.3% / 5.7% | 0.48 / 0.51 | 0.55 / 0.60 |
| reverb 0.2 / 0.5 / 0.8 s (117 each) | 1.7% / 0.9% / 0.9% | 2.8% / 2.8% / 3.4% | 5.7% / 6.0% / 5.7% | ~0.49 | ~0.57 |
| pitch ±2 st (234) | 9.0% | 27.9% | 32.8% | 0.172 | 0.143 |
| pitch ±4 st (232) | 34.1% | 49.0% | 43.2% | 0.039 | −0.009 |
| combo (25) | 60.0% | 41.3% | 60.0% | −0.001 | 0.002 |

Pitch-shifted queries cause most of the team flips for every head. A ±4 semitone shift really changes the
apparent voice, so it is a questionable "same speaker" test. On codec and reverb crops the AP head keeps a
wider margin (+0.06 to 0.09), but a few more crops cross over.

## Reading
1. On real data AP beats CE on 9 of the 10 comparable flip/margin rows. "Optimise the inference geometry,
   not a proxy" holds against the CE head.
2. Neither head trained on 30 speakers beats raw ECAPA, which was trained on about 7k VoxCeleb speakers. The
   predicted +2 to 5 points on male/male pairs is **not reproduced**: on LibriSpeech male/male hardest pairs the
   AP head loses 4.8 points against raw.
3. Wider average margins with more flips (team, augmented) mean the AP head pulls most crops towards the
   prototype but leaves a heavier tail. The flip rate is the number that matters for the enrolled stage.
4. Do not wire this head into `hearing/tse/enrolled.py` on this evidence.

## GPU run for ilab (prepared, not run: no iLab access from this worker)
`experiments/gpu/T-307/run_ilab.sh` runs on the iLab login node from the `scripts/gpu_up.sh` checkout:
`ssh <netid>@ilab.cs.rutgers.edu 'cd hearing-remote && bash experiments/gpu/T-307/run_ilab.sh'`.
1. Downloads LibriSpeech test-clean and train-clean-100 (6.3 GB) to `/freespace/local/$USER/librispeech` if they
   are missing.
2. Fetches the ECAPA weights if they are missing.
3. Runs `train_ap_head.py` in `srun --gres=gpu:a100:1 --time=01:00:00`, or as an `--overlap` step inside a running
   job with `REUSE_JOB=<id>`.

What the run does:
- ECAPA features for 251 training speakers × 60 utterances × 2 crops (about 30k crops).
- 20k steps of 128-speaker episodes per head, 3 seeds.
- Scores on all 40 test-clean speakers, none of them seen in training. Tests the male/male hardest pairs and
  the team set, if the T-306 wavs were rsynced (`rsync -a <main>/eval/augmented/T-306/ <host>:hearing-remote/eval/augmented/T-306/`).
- `SPLITS=train-clean-100,train-clean-360` raises training to 1172 speakers.
- Features are cached in `$OUT` (default `/freespace/local/$USER/t307-runs`), so a re-run only trains.
- Estimated time is about 30 min, mostly the download and feature extraction (not measured on iLab).

Smoke-tested on laptop-a:
- `--dry-run` (synthetic, CPU).
- The real path on MPS with `--train-splits test-clean --utts 20 --steps 200`: 14 s wall, all four eval sets produced.

**Promotion bar** for the GPU result: AP hard-pair flip on LibriSpeech male/male below raw ECAPA's (0.0% here,
321 queries), and the team clean flip no worse than raw (2.6%).

## Limits
- Only 30 training speakers locally. Team set: 2 speakers, about 200 s of source audio.
- Pitch groups are a questionable positive class.
- The flip rate is per 2 s crop with no temporal smoothing. The live tier averages windows, so real flips
  would be rarer for every head.
