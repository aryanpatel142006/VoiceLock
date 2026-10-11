# T-250 Active enrollment and speaker-profile maintenance (theory R&D)

Status: research design + toy synthetic-embedding prototype. **Nothing here changes a profile, default or runtime.** All numbers are from synthetic unit vectors (dim 64), not real ECAPA embeddings or audio; they show mechanism behaviour in a toy model, not product accuracy.

## Replay
```
.venv/bin/python experiments/theory/T-250/enroll_sim.py --seed 0 --out $HEARING_RESULTS_DIR/overnight_prep/T-250/seed0.json
.venv/bin/python -m pytest tests/test_theory_enrollment_active.py
```
Caps: 2000 cases, 60 s (`--max-seconds`), exit 2 with `status: stopped_cap` if hit. Output is written atomically; a rerun with the same seed and script hash that is already `complete` is skipped. Output records seed, script sha and config sha. Run: seeds 0–2, 600 cases each, ~1 s CPU, 0 inputs, <20 KB derived (ignored results dir). Config hash `config_sha` is in each JSON.

## Design
1. **Only explicit, confirmed, clean examples update a profile.** No self-training on uncertain output: `ProfileStore.add_version` raises unless `confirmed=True`. The UI path is "play the clip, user taps Yes, this is them". Auto-found clips (NameCaller, camera) are *candidates* shown for confirmation, never written silently.
2. **Piece-level selection.** Segment a candidate recording into pieces (VAD segments of 1.5–4 s); score each by single-speaker evidence (overlap detector, SNR, cosine to the current centroid). Ask the user to confirm the few pieces that maximise coverage (farthest-point after the medoid, `uncertainty_pick`) rather than the most typical ones; the user can discard individual pieces.
3. **Aggregation.** Mean of unit embeddings; with >=3 clips use leave-one-out rejection (drop the lowest 30%); with <3 clips outliers cannot be judged, so ask for another confirmation instead of guessing.
4. **Drift / cross-mic / domain.** Keep per-condition sub-centroids (mic label, room tag) beside the pooled centroid; when scores for confirmed clips in a new condition fall below the enrolment-time 5th percentile, prompt for one confirmed clip in that condition rather than moving the profile.
5. **Identity update / revert.** Versions are immutable (`v1`, `v2`, ...), `current` is a pointer; update = new version; revert = move pointer back; both O(1). This matches the existing `<name>/current -> v<N>/` layout described in CONTRACTS (T-172); no change proposed there.
6. **Fast cached swap.** Hold pre-normalised embeddings for every enrolled profile in memory so switching target is a lookup (`swap`), not a recompute; reload only on version change.
7. **Negative profile.** A confirmed clip of a known competing talker forms a negative centroid; score = cos(x,p) - 0.5·max(0, cos(x,n)). Useful only when the interferer is a known, confusable voice.
8. **Closed vs open set.** Closed-set argmax always names someone (an unknown talker is 100% falsely accepted in the toy). Use open-set: accept only above a threshold calibrated from the user's own confirmed enrolment clips (5th percentile of genuine scores). Cost: a lower true-accept.
9. **Privacy.** Embeddings and clips stay in the local profile dir, are never sent to an external service, and have an explicit per-profile delete. Candidates not confirmed are discarded in memory.

## Toy results (mean over seeds 0–2; seed 0 shown first, all in the JSON)
Metric is AUC (target vs impostor scores) unless stated.
| Experiment | Result |
|---|---|
| E1 clip count, 25% wrong-speaker outliers (seed 0, earlier config identical) | mean: n=1 0.83, n=2 0.95, n=3 0.96, n=5 0.99, n=8 1.00; robust: 0.83, 0.91, 0.96, 1.00, 1.00 |
| E2 cross-mic (offset 1.5) | enrol A/test A 0.958–0.960; A/test B 0.951–0.955; +1 confirmed B clip 0.957–0.961 |
| E3 drift after 10 steps | static 0.60–0.63; confirmed EMA 0.75–0.77; self-training on unconfirmed 0.74–0.76 |
| E4 open set (seed 0) | closed-set false accept of unknown speakers 1.00; open-set 0.00 at true-accept 0.89 |
| E5 negative profile, similar interferer (seed 0) | 0.81 -> 0.93 |
| E6 3 confirmed clips from pool of 12 (high noise) | random 0.87–0.89; spread-based 0.90 |

## What this does and does not show
- More clips help in the toy, but with n<=2 robust rejection is slightly *worse* than the mean (it cannot tell outliers; E1) — a reason to require a minimum of 3 before rejecting. This is a property of the toy noise model, not proved accuracy.
- Cross-mic effect is small in this model (a few AUC points); real mic/codec mismatch is not additive-Gaussian, so the real size is **unmeasured**.
- Confirmed-only updates ≈ self-training in this toy (differences 0.01–0.02 are within seed spread), so the toy gives **no evidence that self-training is worse**; the rule to avoid it rests on the safety argument (a wrong capture silently changes identity and compounds), not on these numbers. Confirmed updates beat static by ~0.14 AUC under the toy drift.
- E6 spread selection beats random by 0.01–0.03 AUC; small and unvalidated.
- More examples is not proved accuracy: it was only shown on synthetic data.

## Testable next steps (not done; need real embeddings and a human)
1. On existing ECAPA enrolment clips, repeat E1/E2/E4 with a leave-speaker-out split; keep final heldout source groups untouched, tune thresholds on the dev groups only.
2. Real cross-mic: same speaker on two mics, measure score shift distribution (needs recordings).
3. User study of confirm-pieces UX (time to confirm, errors).
