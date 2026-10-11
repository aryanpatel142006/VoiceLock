# Profile learning curve: 1/2/4/8 confirmed examples, frozen ECAPA (T-201)

Optional, offline, research only. Tool: `eval/profile_learning_curve.py`; tests: `tests/test_profile_learning_curve.py`.
It operationalizes the progression in `docs/VOICE_EMBEDDINGS.md`. No live pipeline, registry, server, config or
profile is changed, no profile is promoted, the cafe-first core demo is untouched.

```
.venv/bin/python -m eval.profile_learning_curve --dry-run   # synthetic voices + fake embedder: wiring only, labelled FAKE
.venv/bin/python -m eval.profile_learning_curve              # existing team recordings + real ECAPA (about 7 s)
```
Missing recordings or ECAPA weights exit 2 with `BLOCKED: ...`; the dry run is never evidence. Results JSON goes to
`$HEARING_RESULTS_DIR/profile_learning_curve/<stamp>.json`; proposed profile versions go to
`$HEARING_RESULTS_DIR/profile_learning_curve/proposed/<name>/exp-n{1,2,4,8}/` (gitignored: `eval/results/*`, `*.wav`).

## Design (fixed before looking at results)
- **Recordings (existing, Oct 9 Voice Memos, 16 kHz):** Aryan pool = `team/aryan/profile-01.wav` (0-50 s, sha256 prefix `2ef0c878bdc9356e`), holdout = `team/aryan/fixture-01.wav` (50-108.7 s, different sentences, `8a8703eceb88b55e`). Joseph pool = `team/joseph/profile-01.wav` (15.7 s, `e72d9fc7092003a2`), holdout = `team/joseph/fixture-01.wav` (48.15 s, `a66a49d4a1e37a5f`). No new real-room recording was captured.
- **Split:** `check_disjoint` fails the run if any holdout block occurs in the pool. Holdout windows (hop 1 s) are split in time: first half calibrates one accept threshold per profile (95th percentile of dev negatives = 5% dev false-accept target), second half (one window gap) is scored. The two speakers are each other's fixed negative (profile and holdout).
- **Examples:** 1.5 s clean pieces (voiced share >= 0.6 above -45 dBFS, clipping <= 0.1%), nested and time-spread (n=1 first clean piece, n=2 adds the last, then the gaps). 20 clean pieces for Aryan and 9 for Joseph; fewer than 8 blocks the target.
- **Methods:** `first` (one prototype), `mean` (normalised mean of unit ECAPA embeddings: this is what `hearing.tse.enrolled` already does for a saved profile, so it is **not new**), `multi` (max cosine over prototypes). **Reference row `reference_full`** = the existing whole saved profile (2.5 s pieces, mean). Encoder frozen (`spkrec-ecapa-voxceleb`), no separator.
- **Conditions:** target present clean, target present quiet (-12 dB + noise floor -45 dBFS), wrong speaker clean and quiet, target absent (noise only, -35 dBFS). Crop 1 s and 2 s (crop-duration check).
- **Rule (`promotion_verdict`):** mean TPR (clean+quiet) up >= 0.05 absolute AND wrong-speaker FAR, target-absent FAR, wrong-speaker seconds not worse AND closed-set accuracy (target vs negative profile margin) not worse. Best possible output is `embedding_gain_only`; it can never say "promote" because word/dropout counts, intelligibility and delay were **not measured** (no separator or ASR run).

## Measured (2026-10-09, laptop-a, MPS/CPU, seed 201, one run)
Test windows per condition: Aryan 10 (1 s) / 9 (2 s) present, 40/42 wrong total; Joseph 20/18 wrong total. One window = 0.05-0.11 of a rate. "wrong s" = wrong-speaker windows accepted x 1 s hop.

| target | crop | profile | TPR clean | TPR quiet | FAR wrong | FAR absent | wrong s | closed-set acc | verdict vs first@1 |
|---|---|---|---|---|---|---|---|---|---|
| aryan | 1 s | first@1 | 1.00 | 0.40 | 0.00 | 0.00 | 0/40 | 0.97 | baseline |
| aryan | 1 s | mean@2 / @4 / @8 | 1.00 | 0.40 / 0.60 / 0.50 | 0.00 | 0.00 | 0 | 1.00 / 0.98 / 0.98 | no_win / gain / gain |
| aryan | 1 s | multi@2 / @4 / @8 | 1.00 | 0.50 / 0.70 / 0.60 | 0 / 0 / 0.03 | 0.00 | 0 / 0 / 1 | 1.00 | gain / gain / no_win |
| aryan | 1 s | reference_full | 1.00 | 0.40 | 0.00 | 0.00 | 0/40 | 1.00 | no_win |
| aryan | 2 s | first@1 | 1.00 | 0.89 | 0.00 | 0.00 | 0/42 | 1.00 | baseline |
| aryan | 2 s | mean@2/4/8 | 1.00 | 0.89 | 0.00 | 0.00 | 0 | 1.00 | no_win (no gain) |
| aryan | 2 s | multi@2/@4/@8 | 1.00 | 1.00 / 1.00 / 0.89 | 0.00 | 0.00 | 0 | 1.00 | gain / gain / no_win |
| aryan | 2 s | reference_full | 1.00 | 1.00 | 0.00 | 0.00 | 0/42 | 1.00 | gain |
| joseph | 1 s | first@1 | 1.00 | 1.00 | 0.10 | 0.05 | 2/20 | 1.00 | baseline |
| joseph | 1 s | mean@2/4/8, multi@2/4/8 | 1.00 | 0.90-1.00 | 0.00-0.05 | 0.00 | 0-1 | 0.98-1.00 | no_win (no TPR headroom) |
| joseph | 2 s | first@1 | 1.00 | 1.00 | 0.11 | 0.00 | 2/18 | 1.00 | baseline |
| joseph | 2 s | mean@2/4/8, multi@2/4 | 1.00 | 1.00 | 0.00 | 0.00 | 0 | 1.00 | no_win (no TPR headroom; FAR 0.11 -> 0) |

Order sensitivity (20 random k-subsets, mean TPR clean+quiet, mean +- sd): Aryan 1 s mean@2/4/8 0.78/0.79/0.79 (first@1 0.70); Aryan 2 s mean 0.97-0.98 (first@1 0.945); Joseph 1 s mean 0.96-0.97 (first@1 1.00); Joseph 2 s 1.00. Full per-row numbers are in the results JSON.

## Result: no win to adopt
- Recognition is **saturated** for clean speech (TPR 1.00 for both targets at both crops) and for Joseph's quiet target. The only headroom is Aryan's quiet target, +0.1 to +0.3 TPR at 1 s and +0.11 at 2 s for `multi`, i.e. one to three windows out of 9-10, with `multi@8` (and Joseph's `multi@8`) giving some of it back. The gain is not monotone in n (mean@2 0.40, @4 0.60, @8 0.50) so it is not a learning curve; it is within the noise of a 9-10 window test and the random-subset spread (sd 0.03-0.10).
- The one gain-looking pattern that is consistent: **aggregation (mean or multi) lowers Joseph's wrong-speaker FAR from 0.10 to 0.00-0.05** at the same dev-calibrated threshold, but Joseph's TPR cannot rise (already 1.00), so the rule says `no_win`.
- The existing full profile (`reference_full`, already a mean) is as good as or better than every 1-8-example variant at 2 s (Aryan TPR quiet 1.00): the whole-profile averaging that exists today captures what these variants do. This is not a new capability; "learning" = more examples averaged, which Fable's profile loader already does.
- **Verdict: no promotion, no default change.** At best `embedding_gain_only` on one target/condition, unverified for word retention, dropouts, intelligibility or delay.

## Why embeddings have not reliably worked (diagnosis; T-200 measurements cited)
| mechanism | status | evidence |
|---|---|---|
| Embedding cannot tell the two men apart (clean speech) | **ruled out** | target-vs-other profile cos 0.06; clean TPR 1.00, closed-set 0.97-1.00, AUC 1.00 here and in T-200's clean stems |
| Negative profile dropped by cosine > 0.7 (`SAME_VOICE`) | **ruled out for Aryan/Joseph** | profile cos 0.06 (tool prints `negative_rejected_at_same_voice=false`); saved negatives also bypass the gender guard; T-201 does not test same_bucket or tier state (live wrapper only) |
| ECAPA crop duration | **demonstrated** | 1 s crop: Aryan quiet TPR 0.40 vs 0.89 at 2 s; AUC 0.87 vs 0.96; T-200: margin 0.42 at 0.5 s to 0.77 at 4 s. Intervention: use >= 2 s embedding crops for selection (cheapest measured gain, bigger than any enrollment count) |
| Quiet target / noise | **demonstrated** | only quiet condition loses TPR; clean never does |
| Contamination by the other voice | demonstrated in T-200 (margin 0.69 -> 0.26 at 50% overlap); **not re-tested** here (no mixed windows) |
| Profile-vs-demo-mic mismatch | **unknown, likely confounded**: Aryan (Mac Voice Memos) and Joseph (his own device) were recorded on different mic paths, and pool/holdout are one session each, so wrong-speaker rejection and TPR are optimistic. Needs both men through the demo mic (T-172 re-enroll) |
| Target loss before selection (separation), mix-back/AGC, deadline | **not testable here** (no separator run); T-200: not the cause on `t141/aryan/demo-team-same` at 0/-10 dB |
| Missing-camera strict pause decisions | **unknown**, not exercised offline |
| T-069 male/male ceiling gap (.930 ceiling vs .742 accuracy) | **not reproduced**; here the selector is near-perfect on real clean speech, so that gap points to synthetic/overlapped windows or separation, not to the profile |

Measured intervention that is most likely to matter: >= 2 s crops and a fixed quiet-target test with the real separator (T-200 harness) plus manual word retention. Enrollment count alone did not move closed-set accuracy beyond 0.97-1.00.

## Not measured (explicit)
Target word/dropout counts, intelligibility and delay (no separator/ASR in this tool: set to unmeasured, not to zero); real mixed two-speaker scenes; swapped positions and equal/unequal levels beyond the -12 dB quiet proxy; a different room or mic; any session independent of the enrollment session; statistical significance (n = 9-21 windows per cell).

## Consent, version and rollback
- Use only recordings whose speaker consented to be used for this evaluation (Aryan and Joseph each supplied their own memo; nothing was sent elsewhere). Derived embeddings are as sensitive as the audio: delete both on request.
- Every proposed version is a separate directory `exp-n{N}/` with `manifest.json` (source pool sha256, example start samples, run stamp, `promoted: false`, encoder name). They are never written to `HEARING_PROFILES_DIR`, `profiles/<name>/current` or the live profile directories.
- Rollback: nothing to roll back, the active profile is untouched. If a version is ever adopted later: keep the original as the anchor version, switch only via the `current` pointer (T-172 versioning), and revert by restoring the pointer. Update only from user-confirmed clean examples, never from the system's own uncertain output.
