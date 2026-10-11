# T-249 · Speaker-selection confidence and reject option (theory + synthetic prototype)

Status: offline modelling only. All score traces are **synthetic** with **assumed** class distributions
(`CLASSES` in the script); nothing here is a measurement of the real ECAPA/LDA scores, and no production
threshold is proposed. Real-score replay is the follow-up (below).

## Replay
```
.venv/bin/python experiments/theory/T-249/selection_calibration.py --out "$HEARING_RESULTS_DIR/overnight_prep/T-249" [--seed 249] [--max-cases 400] [--force]
.venv/bin/python -m pytest tests/test_theory_selection_calibration.py -q
```
Config hash `0a51b731deb78198` (seed 249, 24 source groups x 5 scenarios, 120 frames at 100 ms, 400-case cap, 1 trace per group x scenario).
Writes one `result.json` atomically; an existing `status: complete` result with the same config hash is reused (resume),
a failure writes `status: failed` and returns 1 (no retry). Runtime < 2 s, a few hundred KB, CPU, numpy only.
Actual available inputs: 0 real recordings used; 120 synthetic traces (60 dev / 60 holdout, 12 groups each, 7200 frames per split).

## Model
- Score per frame = cosine to the target profile. Classes: target, target masked by a louder competitor, other, similar other (hard negative, stands in for the saved-negative clip), silence.
- Scenarios: target_present, target_absent, similar_voice_absent, pause_hijack (target, 1 s silence, similar competitor, target), louder_competitor.
- Calibrators fit on **dev groups only**: raw cosine (identity), temperature (`sigmoid((s-0.4)/T)`), Platt, isotonic (PAV). Holdout groups are disjoint (asserted) and used only for reporting.
- Policy (accept / hold / abstain): hard dwell+hold hysteresis (accept 0.7, release 0.4, dwell 5, hold 20 frames; defaults, not tuned) vs a two-state online filter with bounded LLR (+-2) and a switch cost (HMM dwell cost). Abstain = pass nothing (Raw/Open decision stays with the existing UI); hold = keep last decision through low-confidence frames.

## Results (holdout, config 0a51b731deb78198)
| calibrator | ECE | Brier | HMM: target-absent false accept | HMM: target retention | HMM: pause-hijack wrong-speaker |
|---|---|---|---|---|---|
| raw cosine | 0.108 | 0.178 | 0.005 | 0.595 | 0.000 |
| temperature | 0.028 | 0.158 | 0.269 | 0.897 | 0.294 |
| Platt | 0.025 | 0.158 | 0.227 | 0.879 | 0.256 |
| isotonic | 0.014 | 0.157 | 0.206 | 0.859 | 0.189 |

Observations (synthetic, so qualitative only):
1. Any calibrator cuts ECE ~4-8x vs raw cosine and improves Brier; isotonic was best by dev ECE, Platt/temperature are smoother (2 and 1 parameters) and better choices when real dev data is small.
2. Calibration alone does not fix the policy: with the assumed similar-voice negatives (cosine 0.38 vs target 0.60) a probability-space accept threshold lets through ~41-45% of similar-voice-absent frames. Reject cannot beat the overlap of the score distributions; that needs a better score (e.g. LDA-projected or negative-clip-contrasted margin), not a better calibrator.
3. The hard dwell policy is safe on target-absent (false accept 0.001-0.026) but loses ~41% of target frames (retention 0.58) and still passes 100% of pause-hijack competitor frames once it is open, because the hold window bridges the pause. The HMM filter retains ~0.86-0.90 but pays false accepts. Retention vs false-accept is a real trade-off; the dev dwell sweep (1/3/5/10 frames) shows retention 0.99/0.79/0.63/0.30 against target-absent false accept 0.55/0.16/0.04/0.00.
4. Pause hijack is the dominant failure: low-confidence silence frames should *extend* the hold only for a bounded time and a competitor frame with a *positive negative-margin* (closer to the saved negative than to the target) should break the hold immediately. Prototype does not yet implement a negative-margin term, which is the next experiment.

## Measurable gates (proposed; evaluate on holdout, never tune there)
| gate | threshold | verdict this run (best-by-dev-ECE = isotonic) |
|---|---|---|
| holdout ECE | <= 0.05 | pass |
| Brier vs raw cosine | no worse | pass |
| target-absent false accept (frame level) | <= 0.02 | **fail** (0.206) |
| target-word retention | >= 0.85 | pass (0.859) |
| pause-hijack wrong-speaker share | <= 0.15 | **fail** (0.189) |
| holdout-dev ECE gap | <= 0.03 | pass |
Failing two of six gates on synthetic data is the honest outcome: the prototype does not clear the bar, so no policy is recommended for runtime.

## Counterexamples (emitted by the script)
- similar-voice absent: pass rate 0.41, cosine cannot separate close voices.
- pause hijack: hold window bridges the silence into the competitor.
- louder competitor: target kept 0.66 < 0.85, masked cosine falls below release (an abstain looks like a rejection; the UI should show hold, not "target lost").
- group bias: per-group score shift makes holdout ECE exceed dev ECE; calibrate per session or with a group-level offset.

## Comparison with current cosine/LDA and saved negatives
Current runtime uses fixed cosine thresholds (0.45 retarget guard, 0.55 grow, 0.4/0.25 not-heard; see CONTRACTS) which behave like the raw-cosine row: well-behaved as ranking, poorly calibrated as probability. Not run on real LDA scores: no scored traces were available to this task. Saved negative semantics are modelled only as the `similar_other` class.

## Follow-up (consumer command once real scored traces exist)
Replace `make_dataset` with a loader of per-frame (score, speaker-group id, target-speaking label) CSVs from `$HEARING_RESULTS_DIR`, keep the group split (exact source-group holdout, final heldout groups untouched until the last run), and run the same CLI. Add the negative-margin term and per-group offset first.
