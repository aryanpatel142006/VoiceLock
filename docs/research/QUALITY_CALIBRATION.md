# Quality-aware calibration vs the fixed `SAME_VOICE = 0.7` (T-308)

Steal #2 from the SOTA synthesis: quality-aware score calibration (Thienpondt et al., ICASSP 2021).
A logistic regression maps (cosine score, window duration, raw embedding norm) to P(same speaker),
so the score needed for 95 % precision moves with window quality instead of being one number.

Code: `eval/calibration/T-308/quality_cal.py` (numpy IRLS, no new dependency). Tests:
`tests/test_quality_calibration.py` (synthetic, 0.7 s).

```
.venv/bin/python eval/calibration/T-308/quality_cal.py          # synthetic table
.venv/bin/python eval/calibration/T-308/quality_cal.py --team   # + real teammate pairs (ECAPA, ~5 s on CPU/MPS)
```
Results JSON: `$HEARING_RESULTS_DIR/T-308_quality_cal.json`.

## Model
`P(same) = sigmoid(w · [1, s, log d, log n, s·log d, s·log n])`, where `s` is the cosine to the
enrolled profile, `d` the window length in seconds, and `n` the raw (pre-normalisation) ECAPA norm.
The interaction terms let the score's slope change with quality, not only its offset.
`predict_threshold(d, n, precision=0.95)` solves `P(same) = c` for `s`.
- **Default (set-level):** `c` is the posterior cut at which the accepted *training* pairs reach 95 % precision.
  This is one cut on the calibrated output, as in the paper.
- **`per_pair=True`:** `c = 0.95` for each pair. This is stricter and gives lower recall.
- **Base rate:** `prior=` shifts the logit when the deployment base rate differs from training.

## Where 0.7 is used today
| Site | Compares | Constant |
|---|---|---|
| `hearing/tse/enrolled.py:122` `SAME_VOICE` | target profile vs negative profile (two long enrollments) | 0.7: the negative is dropped as "same person" above it |
| `hearing/server/namecall.py:69` `SAME_VOICE` | name utterance (~1-3 s) vs target clip | 0.45 |
| `eval/profile_learning_curve.py`, `eval/isolation_diagnostic.py` | copies of the 0.7 above | 0.7 |

The 0.7 in `enrolled.py` is a profile-vs-profile check on long, clean clips. The quality problem below
applies to **window-level** same-speaker decisions: Name Call's 0.45/0.55, the enrolled window's stream choice,
and any future "is this window the target?" gate.

## Results (laptop-a, Oct 10 2026)
P / R = precision / recall of "same speaker" decisions. Adaptive = per-window threshold from
`predict_threshold(d, n)` at 95 % precision. Oracle = the best single raw-cosine threshold for 95 %
precision chosen *on that bin's own labels* (an upper bound a fixed constant cannot reach across bins).

### Synthetic (40 toy speakers, impostor centroids at cosine 0.2-0.8, 10 000 pairs per bin, fit on 20 000 mixed-length pairs)
| window | same p50 | diff p50 | fixed 0.7 P / R | adaptive thr p50 | adaptive P / R | oracle thr | oracle P / R |
|---|---|---|---|---|---|---|---|
| 0.5 s | 0.48 | 0.24 | 1.000 / **0.015** | 0.49 | 0.958 / **0.473** | 0.49 | 0.950 / 0.488 |
| 1 s | 0.61 | 0.31 | 1.000 / 0.214 | 0.62 | 0.991 / 0.498 | 0.54 | 0.950 / 0.697 |
| 2 s | 0.74 | 0.37 | 0.996 / 0.618 | 0.70 | 0.996 / 0.677 | 0.59 | 0.950 / 0.859 |

### Real teammate pairs (ECAPA, Aryan vs Joseph)
Setup:
- **Enrollment:** each speaker's `profile-01.wav` (50 s / 15.7 s).
- **Windows:** non-overlapping crops of their `fixture-*.wav` takes (134 s total), each scored against both profiles.
- **Thresholds:** adaptive thresholds are out-of-fold (5-fold by window).

| window | pairs | same p50 | diff p50 | fixed 0.7 P / R | adaptive thr p50 | adaptive P / R | oracle thr | oracle P / R |
|---|---|---|---|---|---|---|---|---|
| 0.5 s | 536 | 0.28 | 0.01 | (none accepted) / **0.000** | 0.16 | 0.921 / **0.698** | 0.19 | 0.952 / 0.660 |
| 1 s | 266 | 0.45 | 0.04 | 1.000 / 0.015 | 0.26 | 0.982 / 0.835 | 0.23 | 0.957 / 0.835 |
| 2 s | 132 | 0.65 | 0.07 | 1.000 / 0.288 | 0.33 | 1.000 / 0.924 | 0.21 | 0.953 / 0.924 |

**Thresholds from the full team fit** (at each crop's median norm): 0.17 at 0.5 s, 0.26 at 1 s and 0.33 at 2 s.

**Raw ECAPA norm p50:** 372 at 0.5 s, 343 at 1 s and 314 at 2 s. On real audio the norm *falls* with window
length; the synthetic generator assumes the opposite. The regression learns the sign from the data, so this
does not change the method, but the synthetic weights must not be reused on ECAPA scores.

**Profile-vs-profile cosine:** Aryan vs Joseph `profile-01` is **0.066**, far below 0.7.

## Where fixed 0.7 fails
1. **Short windows:** same-speaker cosine is low because ECAPA on 0.5 s is noisy. Same p50 is 0.28 on real
   audio and 0.48 synthetic, so 0.7 accepts **0 %** of true 0.5 s windows (real) and 1.5 % (synthetic).
   Even at 2 s it keeps only 29 % (real). Used as a window gate, 0.7 means "never the target".
2. **No single constant fits every length:** the 95 %-precision point moves from 0.16 (0.5 s) to 0.33 (2 s) on
   real pairs and from 0.49 to 0.70 synthetic. A constant tuned for 2 s windows lets in impostors at 0.5 s
   only if it is set low, and one tuned for 0.5 s throws away long-window recall. In the pooled synthetic
   test, the best single cut for 95 % overall loses short-window recall against the adaptive one.
3. **Precision of 0.7 is never the problem in our data:** wherever it accepts anything, it is
   ≥ 99.6 % precise. It fails on **recall**: it is roughly 2-4× too high for window decisions on ECAPA scores.

The adaptive threshold gets close to the per-bin oracle without knowing the bin's labels. It reaches
0.473 vs 0.488 recall at 0.5 s (synthetic) and 0.698 vs 0.660 on real pairs. Its real 0.5 s precision is
**0.921**, under the 95 % target: there are only 536 pairs from two speakers, and the 5-fold fits see about 750 pairs each.

## Recommendation: hybrid
- **Keep `enrolled.SAME_VOICE = 0.7`** for its actual job. It compares two long profiles, where the team pair
  sits at 0.066, so it never fires wrongly for two different teammates and only catches a polluted negative.
  Quality calibration has nothing to adapt there, because both clips are long.
- **Use the calibrator for window-level same-speaker decisions** (Name Call's 0.45 re-target/0.55 grow,
  any per-window target gate). Pass the window's duration and raw ECAPA norm (`backend.embed()`
  before `_unit_embedding`), then threshold at `predict_threshold(d, n)`.
  This needs a refit on more speakers than two first, because the current real fit is two male voices and
  134 s of takes. The synthetic fit's weights are not usable on ECAPA scores (scale and norm direction differ).
- **Do not lower a global constant instead.** A single cut is right for at most one window length.

Not done here (follow-ups): wiring into `namecall.py`/`enrolled.py`, which needs the raw norm plumbed through
`_unit_embedding`; and a refit on a multi-speaker set (GRID/LibriSpeech fixtures), which is what the
production weights need.
