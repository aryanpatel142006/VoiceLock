# Evidence: projection-head margins on real team audio (T-305)

Claim checked (`docs/research/theory/projection_head_theory.md`, T-294): *"0.5 s-window speaker margin 8× (0.15 → 1.19) with 97.5% held-out accuracy"*.
Every number below comes from the real Aryan/Joseph recordings. Nothing synthetic.
Script: `eval/evidence/T-305/reproduce_margins.py` (11 s end to end). Raw numbers: `eval/evidence/T-305/margins_raw.json`.
Host: laptop-a (Aryans-MacBook-Pro, macOS arm64, Python 3.12.8, torch 2.14.1), ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) on MPS, head on CPU. Run 2026-10-10.

## TL;DR

1. **The "0.15 → 1.19" headline was never measured on team audio.** It traces to `docs/research/T291_DATACARD.md` (T-309 branch). There, the embeddings and head outputs are **synthetic and calibrated to produce 0.136 and 1.190 by construction**, on LibriSpeech stand-ins (speakers 1580 F and 61 M), not Aryan/Joseph. `models/team_projection_head.pt` did not exist on any disk of this host either. Treat those two numbers as placeholders, not evidence.
2. **On real team audio the head's margin gain is real and larger.** At 0.5 s the stock ECAPA margin is **0.121 [0.107, 0.136]**; the head's is **1.594 [1.517, 1.664]** (seed 0). That is **13.2× [11.9, 14.6]**, and 1.42–1.59 across 5 training seeds.
3. **The gain is not specific to 0.5 s, but the ratio is.** The head's margin is roughly flat (1.59 / 1.73 / 1.84 at 0.5 / 1.0 / 2.0 s). Stock ECAPA's margin grows with window length (0.12 / 0.21 / 0.35), so the ratio falls: **13.2× / 8.2× / 5.3×**. An "8× gain" figure corresponds to 1.0 s windows on this data, not 0.5 s.
4. **Accuracy barely moves.** Nearest-prototype accuracy at 0.5 s: stock **97.2% (206/212) [94.0, 98.7]** vs head **99.1% (210/212) [96.6, 99.7]**; the CIs overlap. At 1.0 s stock is 99.2% vs head 98.3%, and at 2.0 s both are 100%. The margin widens a lot, but the decision boundary was already almost right. "97.5% accuracy" is not evidence for the head: stock ECAPA scores the same within session.
5. **⚠ THE SESSION CHECK CANNOT BE RUN: there is no cross-session data.** Every labelled clip of a speaker is a cut of **one voice memo** (Aryan: Mac Voice Memos, Oct 9, profile = 0–50 s, fixture = 50–108.7 s; Joseph: one `joseph-original.m4a`, cuts at 8.3–24.0 / 29.85–78.0 / 85.9–113.7 s). Session == speaker. **Every number in this file, and in T-273, is within-session. The cross-session margin is unmeasured (N = 0).** It is the biggest open risk to the head claim. The only proxy available (non-speech windows, below) shows no channel leak, but it is weak (N = 29).

## Method

- **Training** (T-273 pilot recipe, reusing its `slice_audio`, `TeamProjectionHead` and `train_projection_head`): 0.5 s voiced crops at 0.15 s step from each `profile-01.wav` (first 120: Aryan 120, Joseph 100 windows), contrastive loss (negative target cos −0.5, weight 2.5), 120 epochs, AdamW lr 1e-3, CPU. Seeds 0–4. Seed 0 is saved to `$HEARING_MODELS_DIR/team_projection_head.pt` (gitignored, not committed).
- **Test:** the disjoint fixture cuts (Aryan `fixture-01.wav`; Joseph `fixture-01.wav` + `fixture-02-namecall.wav`). Windows are non-overlapping (hop = window) so they are closer to independent for the CI, and pass the same voiced gate (peak ≥ 0.02, RMS ≥ 0.005).
- **Margin** = mean intra-speaker cosine − mean cross-speaker cosine, averaged over both speakers, **self-pairs excluded**. The pilot's `compute_metric_stats` includes the diagonal (cos = 1), which inflates intra-speaker similarity on small N. That convention is also in the JSON (`margin_pilot`): 0.129 stock / 1.595 head at 0.5 s.
- **CI:** 95% percentile bootstrap, 2000 resamples of windows within each speaker; pairs of the same source window are dropped (a resampled duplicate would add cos = 1). Accuracy CIs are Wilson 95%. The ratio CI uses paired resampling of the same windows in both spaces.
- **Accuracy:** nearest prototype; the prototype is the mean of the speaker's profile training embeddings, in each space.

## 1–2. Margin per window length (Aryan vs Joseph, held-out fixtures)

| window | N windows (A + J) | stock intra / cross | **stock margin [95% CI]** | head intra / cross | **head margin [95% CI]** (seed 0) | head margin, 5 seeds | **ratio head/stock [95% CI]** |
|---|---|---|---|---|---|---|---|
| 0.5 s | 86 + 126 = 212 | 0.177 / 0.056 | **0.121 [0.107, 0.136]** | 0.846 / −0.748 | **1.594 [1.517, 1.664]** | 1.420 – 1.594 (mean 1.505) | **13.2× [11.9, 14.6]** |
| 1.0 s | 50 + 69 = 119 | 0.278 / 0.068 | **0.210 [0.180, 0.240]** | 0.906 / −0.822 | **1.728 [1.635, 1.803]** | 1.619 – 1.728 (mean 1.670) | **8.2× [7.4, 9.4]** |
| 2.0 s | 27 + 36 = 63 | 0.447 / 0.098 | **0.349 [0.303, 0.396]** | 0.957 / −0.883 | **1.840 [1.754, 1.889]** | 1.739 – 1.840 (mean 1.791) | **5.3× [4.7, 6.0]** |

Accuracy (nearest prototype, Wilson 95% CI):

| window | stock | head seed 0 | head, 5 seeds |
|---|---|---|---|
| 0.5 s | 206/212 = 97.2% [94.0, 98.7] (A 86/86, J 120/126) | 210/212 = 99.1% [96.6, 99.7] (A 86/86, J 124/126) | 97.2 – 99.1% |
| 1.0 s | 118/119 = 99.2% [95.4, 99.9] | 117/119 = 98.3% [94.1, 99.5] | 98.3 – 100% |
| 2.0 s | 63/63 = 100% [94.3, 100] | 63/63 = 100% [94.3, 100] | 98.4 – 100% |

Reading: the head pushes the two speakers to opposite hemispheres (cross cosine −0.75 to −0.88), which is what the contrastive loss asks for. Within one recording session, stock ECAPA prototypes already separate them almost perfectly. All of stock's errors at 0.5 s are Joseph windows (6/126).

## Per fixture take (time distance from the training cut, same recording)

| window | take | source span in memo (s) | gap after profile (s) | N | stock margin / acc | head margin / acc |
|---|---|---|---|---|---|---|
| 0.5 s | aryan/fixture-01 | 50.0 – 108.7 | 0 | 86 | 0.121 / 86/86 | 1.594 / 86/86 |
| 0.5 s | joseph/fixture-01 | 29.85 – 78.0 | 5.9 | 90 | 0.127 / 86/90 | 1.626 / 89/90 |
| 0.5 s | joseph/fixture-02-namecall | 85.9 – 113.7 | 61.9 | 36 | 0.119 / 34/36 | 1.516 / 35/36 |
| 1.0 s | aryan/fixture-01 | 50.0 – 108.7 | 0 | 50 | 0.210 / 50/50 | 1.728 / 50/50 |
| 1.0 s | joseph/fixture-01 | 29.85 – 78.0 | 5.9 | 47 | 0.235 / 47/47 | 1.815 / 47/47 |
| 1.0 s | joseph/fixture-02-namecall | 85.9 – 113.7 | 61.9 | 22 | 0.173 / 21/22 | 1.553 / 20/22 |
| 2.0 s | aryan/fixture-01 | 50.0 – 108.7 | 0 | 27 | 0.349 / 27/27 | 1.840 / 27/27 |
| 2.0 s | joseph/fixture-01 | 29.85 – 78.0 | 5.9 | 24 | 0.390 / 24/24 | 1.891 / 24/24 |
| 2.0 s | joseph/fixture-02-namecall | 85.9 – 113.7 | 61.9 | 12 | 0.293 / 12/12 | 1.743 / 12/12 |

(Per-take margin = that take against the other speaker's full test set.) Joseph's later take (fixture-02, ~1 min after the profile, name-call content) has the lowest margin in both spaces at every window length, and at 1.0 s the head makes 2 errors there vs stock's 1. That is mild drift within one memo, not collapse. Its N (12–36) is too small to call.

## 3. Session-artifact check

**Within-session vs cross-session margin:**

| split | 0.5 s margin stock → head | N |
|---|---|---|
| within-session (train and test cuts of the same voice memo) | 0.121 → 1.594 | 212 windows |
| **cross-session** (train on one recording session, test on another) | **NOT MEASURABLE: no second labelled session exists for either speaker** | **0** |

**⚠ FLAG: the T-273 "0 errors / 97.5%" and the margins above are all within-session.** Each speaker was recorded once, on a different device/app (Aryan: Mac Voice Memos at speech RMS ≈ 0.015; Joseph: his own voice memo at RMS ≈ 0.065, peak-normalised, `TEAM_AUDIO_AUDIT.md`). So a head trained to separate "Aryan" from "Joseph" is equally a head trained to separate *recording A* from *recording B*. Its behaviour on a new session (the demo room, the judge's phone mic, the RØDE) is unknown. The Joseph pack's own `processing-report.json` says the same: "One uploaded recording, split into non-overlapping time intervals; not independently recorded sessions."

**Proxy that real audio does allow: does the recording alone carry the label?** Non-speech windows (every 10 ms frame ≤ 2× the clip's 10th-percentile frame RMS: room tone, mic/codec noise, no speech) from every clip were scored with the speech-trained prototypes:

| window | N (A + J) | stock: label from noise | head: label from noise | noise-window margin stock / head |
|---|---|---|---|---|
| 0.5 s | 18 + 11 = 29 | 11/29 = 37.9% [22.7, 56.0] | 12/29 = 41.4% [25.5, 59.3] | 0.013 / −0.007 |
| 1.0 s | 3 + 3 = 6 | 4/6 = 66.7% [30.0, 90.3] | 4/6 = 66.7% [30.0, 90.3] | 0.004 / −0.015 |

The noise windows are at chance and their margin is ~0 in both spaces, so the head does not pull the room/mic signature of each recording toward that speaker's side. This **rules out the crudest artefact** (the head reading the noise floor). It **does not** rule out a channel colouring the *speech* itself (mic frequency response, codec, AGC, reverberation), which only a second recording session can test. The noise-only nearest-centroid classifier (train on the profile clips' silences, test on the fixtures') could not run: Joseph's 15.7 s profile cut has no window quiet enough (0 at 0.5 s). N is small (29 / 6). Treat this as weak evidence.

**What would settle it** (needs a human: recordings): each speaker records ≥ 60 s again in a *different* session (other room, or the phone/RØDE used at the demo). Drop the WAVs into `$HEARING_FIXTURES_DIR/team/<name>/` and add a second session entry to `RECORDINGS` in `reproduce_margins.py`. Then train on session 1 and test on session 2 (and the reverse). Success bar: cross-session 0.5 s head margin within the within-session CI's lower bound (≥ 1.5) and accuracy ≥ stock's.

## Discrepancies with earlier docs

| earlier claim | source | on real team audio (this run) |
|---|---|---|
| 0.5 s stock margin 0.136–0.15 | T-309 datacard, **synthetic** | 0.121 [0.107, 0.136] (pilot convention 0.129) |
| 0.5 s head margin/gap 1.19 | T-309 datacard, **synthetic, calibrated** | 1.594 [1.517, 1.664] (seeds 1.42–1.59) |
| 8× gain at 0.5 s | T-294 theory doc | 13.2× at 0.5 s; 8.2× at 1.0 s; 5.3× at 2.0 s |
| 97.5% held-out accuracy (head) | T-294 / LDA doc | head 97.2–99.1% (5 seeds) vs **stock 97.2%**, 0.5 s |
| baseline 17/97 errors, 25 flips at 1.0 s | T-273 pilot doc | stock 1/119 errors at 1.0 s with profile prototypes. The pilot uses a 10-window prototype and a different test file set. Also, on this host its `resolve_team_audio_paths()` finds none of `profiles/*/profile-01.wav` or `fixture-01-20261009-1353.wav` and silently falls back to synthetic data, so its numbers cannot be reproduced here as written. |

## Reproduce

```bash
export HEARING_FIXTURES_DIR=<main>/eval/fixtures/generated HEARING_MODELS_DIR=<main>/models
.venv/bin/python eval/evidence/T-305/reproduce_margins.py            # 5 seeds, writes margins_raw.json
.venv/bin/python eval/evidence/T-305/reproduce_margins.py --seeds 1 --out /tmp/m.json
```
Inputs: `$HEARING_FIXTURES_DIR/team/{aryan/profile-01,aryan/fixture-01,joseph/profile-01,joseph/fixture-01,joseph/fixture-02-namecall}.wav` (gitignored, never committed). Output is deterministic up to MPS float noise in ECAPA; head seeds are fixed.
