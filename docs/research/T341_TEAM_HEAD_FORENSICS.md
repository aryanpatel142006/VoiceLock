# T-341: forensic audit of Gemini's TeamProjectionHead "97.5% / margin 1.14"

Harness: `eval/team_head_forensics.py` (extract + score, CPU). Raw numbers: `eval/evidence/T-341/forensics.json`;
input hashes: `eval/evidence/T-341/cache.manifest.json`. Complementary to T-336 (logistic head on 5 selector
features); this task does not retrain T-336's model, it dissects T-273's.

## Phase 0: artifacts (correcting the earlier "unrecoverable" note)

| Artifact | Found | Notes |
|---|---|---|
| Method code | **yes, on main**: `eval/projection_head_pilot.py` + `docs/research/PROJECTION_HEAD_PILOT.md` (T-273, commit `7410956a`, authored on Joseph's machine, `/Users/joeroche/Developer/...` paths) | `TeamProjectionHead`, `ContrastiveProjectionLoss`, `ArcFaceLoss`, `slice_audio`, eval loop. This *is* "Gemini's method"; `eval/team_voice_discriminator.py` exists nowhere on laptop-a |
| `team_projection_head.pt` (61,425 B) | **yes**, laptop-a `models/team_projection_head.pt`, sha256 `7ca5539fafb5d89cd8aa365b574020f22285544ab67cb9cf03cbbb3863fe9661`, mtime 2026-10-10 01:19 | State dict matches `TeamProjectionHead` exactly (`net.0` Linear 192→64, `net.1` LayerNorm 64, `net.3` Linear 64→32). T-305's `reproduce_margins.py` writes its seed-0 retrain to exactly this path, so this file is most likely **T-305's retrain of the T-273 recipe, not Gemini's own binary** (which lived on Joseph's Mac) |
| Q-836 / Q-838 text | not on this host | the wording quoted in the task feedback ("97.5% test accuracy" vs "initial in-sample pilot") is UNVERIFIED here |

## Phase 1: what the method is and what the numbers measure

- **Architecture**: Linear(192,64) → LayerNorm(64) → GELU → Linear(64,32) → L2 norm; 14,560 params.
- **Objective**: contrastive (positive pairs → cos 1, negatives pushed below cos −0.5, weight 2.5) or ArcFace
  (m 0.5, s 16); full-batch AdamW lr 1e-3, 120 epochs, no validation split, no early stopping.
- **Training data**: 0.5 s crops, 0.15 s step (70 % overlap), of `profiles/<name>/profile-01.wav`
  (first 120 crops; Aryan 120, Joseph 90 here).
- **Scoring**: 2-way nearest prototype; prototype = mean of the **first 10 training crops** (~1.85 s of
  audio). No threshold, no "neither" class, no other streams: false accepts cannot be measured.
- **Test**: 1.0 s non-overlapping crops of `team/<name>/fixture-01.wav`: 50 + 47 = **97 windows**,
  two classes. That is the denominator behind "0/97 errors"; Wilson 95 % CI for 97/97 is [0.962, 1.0].
- **"97.5 %"**: not printed by the pilot code (which reports 100 % at 1.0 s and 96.6 % at 0.5 s).
  The nearest measured numbers are T-305's 0.5 s nearest-prototype accuracy, head 97.2-99.1 % vs stock
  ECAPA 97.2 % (206/212) within session. Treat "97.5 %" as a rounded within-session 2-way accuracy.
- **"margin 1.14"**: T-291 (`eval/embeddings/T-291/README.md`) reports the 0.5 s head margin
  (intra − cross cosine) median **1.147**, computed over profile-01 **plus** the fixtures pooled: the
  head's own training audio is inside that statistic.

## Phase 2: leakage audit

| Vector | Finding |
|---|---|
| Same session / same mic | **Present, decisive.** `team/aryan/meta.json`: "profile-01 = 0-50 s, holdout-01 = 50-108.7 s" of **one** Mac Voice Memo. Joseph: "one uploaded recording, split into non-overlapping time intervals" (`joseph_pack/processing-report.json`, quoted by T-305). The T-273 doc's "completely separate recordings made at different times" is false. Speaker identity and recording/channel identity are perfectly confounded (Aryan RMS ≈ 0.015, Joseph ≈ 0.065 peak-normalised, `TEAM_AUDIO_AUDIT.md`), so a 2-class head can separate *recording A from B*. |
| Shared utterances / overlapping crops train↔test | none in time for the pilot split (0-50 s vs 50-108.7 s); **present** for the T-291 "margin 1.14" statistic (training file pooled into the eval set) |
| Train fragments derived from eval recordings | **Present for the t141 scenes**: their target/other stems are cut from the same `fixture-01` takes (manifest `fixture_takes`), and enrollment from the same session's `profile-01`. So the t141 test below is still same-session; it is disjoint in *utterances*, adds babble, room mixing and the separator. A true new-session test is impossible with the audio on disk (one session per speaker). |
| Threshold calibrated on eval | n/a in the pilot (argmax, no threshold); here thresholds come from `demo-team-same` (dev) only |
| Closed-set evaluation | **Present.** The pilot only asks "Aryan or Joseph?" on clean single-speaker audio. Stock ECAPA already gets 97.2 % at 0.5 s on the same question (T-305). |
| Silent synthetic fallback | `resolve_team_audio_paths()` falls back to **synthetic** embeddings (`generate_synthetic_team_dataset`, cos 0.60 by construction) when the team files are not at the hard-coded roots; `data_source` is only a field in the JSON. On laptop-a the worktree cannot find them (it needs `/Users/joeroche/...` or a sibling `selective-hearing/`). |

## Phase 3-4: rebuild and identical-trial evaluation (laptop-a, CPU, 2026-10-10)

Trials: t141 `demo-team-easy` and `demo-team-hard` for target = aryan and target = joseph (4 scenes,
30.2 s each). The mix is separated per 1.0 s window (0.5 s hop, 59 windows per scene, T-336's
windowing) by the live `BlindSeparator`; the target stream is the one correlating best with the target
stem; a window is target-present when the target stem is above −45 dBFS (**212 windows**). Target-absent
= the same scene without the target stem (other teammate + babble, 236 windows); unknown = babble only
(236 windows). Threshold per scorer: the 95th percentile of target-absent scores on `demo-team-same`
(dev). Every scorer sees the identical 2 streams per window. Heads are retrained here on the T-273 split
(5 seeds, T-273 code unchanged) because Gemini's own binary is not on this host. "Pilot" = T-273's own
2-way test (97 windows of the same-session fixtures).

| Scorer | Pilot acc (n=97) | t141 top-1 (n=212) [95 % CI] | TA @ dev thr | wrong accept | FA target-absent (n=236) | FA unknown (n=236) | quiet target top-1 | d′ margin |
|---|---|---|---|---|---|---|---|---|
| **A raw ECAPA, cos to full enrollment** | (n/a: single-profile) | **191 (90.1 %) [85.3, 93.4]** | 60.9 % | 0.5 % | 4.7 % | 8.5 % | 29/38 | **1.40** |
| A2 raw ECAPA, cos to T-273 prototype (10 crops) | 85.6 % | 136 (64.2 %) | 26.4 % | 0.9 % | 4.7 % | 8.5 % | 22/38 | 0.50 |
| A3 raw ECAPA closed set cos(t)−cos(o) | **100 %** | 179 (84.4 %) | 53.8 % | 0.5 % | 4.2 % | 7.6 % | 27/38 | 1.06 |
| B on-disk `team_projection_head.pt`, prototype | **100 %** | 103 (48.6 %) | 28.8 % | 3.8 % | 7.2 % | 8.5 % | 15/38 | −0.13 |
| B on-disk `team_projection_head.pt`, enrollment | n/a | 182 (85.9 %) | 59.4 % | 2.8 % | 5.9 % | 8.5 % | 28/38 | 0.90 |
| B′ T-273 contrastive head, 5 seeds, prototype (Gemini's scoring) | **100 %** | 181 (85.4 %) [80.0, 89.5] | 62.3 % | 2.4 % | 7.6 % | 8.5 % | 28/38 | 1.00 |
| B′ T-273 contrastive head, 5 seeds, enrollment | n/a | 181 (85.4 %) | 61.8 % | 2.4 % | 7.2 % | 8.5 % | 28/38 | 1.01 |
| B′ T-273 contrastive head, seed 0 | 100 % | 181 (85.4 %) | 52.8 % | 0.5 % | 5.5 % | 8.5 % | 28/38 | 0.99 |
| B′ T-273 ArcFace head, 5 seeds, enrollment | n/a | 180 (84.9 %) | 57.1 % | 0.5 % | 5.5 % | 8.5 % | 28/38 | 1.06 |
| D1 ablation: linear 192→32 (no LayerNorm/GELU), contrastive | 100 % | 182 (85.8 %) | 64.6 % | 2.8 % | 4.7 % | **0.0 %** | 28/38 | 1.05 |
| D3 ablation: train-mean-centred raw ECAPA | n/a | 189 (89.2 %) | 59.4 % | 0.5 % | 5.1 % | 8.5 % | 31/38 | 1.34 |
| D4 production `fit_lda` on the same train crops | 100 % | 184 (86.8 %) | 61.8 % | 2.4 % | 6.4 % | 4.2 % | 29/38 | 1.04 |

Paired (McNemar, exact, same 212 windows) vs A: T-273 contrastive head 17 windows only raw gets right,
7 only the head gets right (p = 0.064); on-disk .pt 18 vs 9 (p = 0.12); closed-set A3 20 vs 8 (p = 0.036).
Head CPU cost: 57 µs per window for 2 streams (laptop-a CPU, under load ~6, not exclusive), negligible; the
ECAPA+separator cost per window is unchanged. Uncertainty: 4 scenes from 2 recordings, so windows are
not independent trials; the CIs above are optimistic.

T-336 for reference (its own harness, not re-run here): head accept-right / accept-wrong per scene
(aryan easy 41/1, hard 38/0; joseph easy 34/1, hard 29/6) on 46/46/44/37 target windows, target-absent
accepted 0.5 s (aryan) and 4.0-4.5 s (joseph) of 29.5 s. Its features (LDA score, cosines, level) differ
from this head's, so only the direction is comparable: T-336 trades target acceptance for far fewer
wrong accepts; the T-273 head does neither.

## Verdict: FAILED VALIDATION (nothing to salvage into R28)

1. **What Gemini built**: T-273's `TeamProjectionHead` (192→64→LN→GELU→32, L2), contrastive
   (neg cos ≤ −0.5) or ArcFace, trained full-batch for 120 epochs on 0.5 s crops of one profile take per
   person, scored by 2-way nearest prototype (mean of 10 training crops).
2. **Why it reported 97.5 % / 1.14**: a closed-set Aryan-or-Joseph question on clean audio from the
   same recording session and microphone as training (the "held-out" fixture is the second half of the
   same voice memo), compared against a crippled baseline (raw ECAPA vs a 10-crop prototype, 85.6 %);
   raw ECAPA with a closed-set score already gets 97/97 on that split. The 1.14 margin is a median over a
   set that includes the head's own training audio.
3. **What survives disjoint eval**: on separated t141 windows the head ranks the right stream 85.4 %
   of the time vs 90.1 % for raw ECAPA cosine to the enrollment; false accepts are equal or worse. The
   gain does not survive; the direction reverses (p = 0.064, paired).
4. **Components**: the non-linearity (LN+GELU) adds nothing over a linear map (D1 = 85.8 %); ArcFace vs
   contrastive makes no difference; the scoring prototype is the most damaging piece (raw ECAPA drops
   90.1 → 64.2 % with it, the on-disk head to 48.6 %); any 2-class closed-set transform (A3, LDA,
   heads) turns "closer to Aryan than Joseph" into the decision and loses ~4-6 points on separated
   streams, where both streams often carry some of both voices.
5. **Beats calibrated raw ECAPA on real separated speech?** No, on top-1, TA, wrong accepts, quiet-target
   and target-absent FA. The only cell it wins is D1's unknown-babble FA (0/236 vs 20/236), a single
   ablation run with no replication; not enough to adopt.
6. **Worth integrating into R28**: none of the head. Keep raw ECAPA cosine to the full enrollment
   (as the live `enrolled` stage does); do not use short-crop prototypes. Train-mean centring (D3) is
   within noise of raw. T-336's logistic gate stays the only live candidate from this line.
