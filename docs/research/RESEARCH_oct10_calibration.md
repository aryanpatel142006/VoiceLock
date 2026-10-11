# Calibrating the enrolled gate on 1 s crops of separated speech (research, Oct 10)

Scope: the confidence and gating decision in `hearing/tse/enrolled.py` (`extract`). No repo was edited. Sources are linked inline. "Repo" means `~/CODE/selective-hearing` at `origin/main`.

## What the repo shows (the problem)
- A window is gated when the picked stream's closed-set LDA score is below `contrast` (0). Confidence is the margin divided by `CONF_MARGIN_LDA` = 0.58, clipped to [0, 1]. Repo: `hearing/tse/enrolled.py` L128-131, L443-492.
- The LDA (`fit_lda`) is fit on the **clean profile-piece** embeddings. The live scores come from **1.0 s crops of MossFormer2 output**. `CONF_MARGIN*` was fit on GRID s1/s4, "not on the teammates' takes yet". Repo: `enrolled.py` L126-129, L179-198, L321-330.
- Gated windows cause 34 of the 42 wrong-speaker windows. On the joseph-as-target rows, 30 of 31 wrong windows are gated, and in 22 of those 30 **Joseph was talking but his stream scored on Aryan's side**. Repo: `docs/BENCHMARKS.md` "Wrong-speaker windows by pick margin" (T-323).
- `HEARING_GATED_FLOOR_DB=20` lowers wrong-speaker time from 14.8 to 1.4 % (easy) and from 19.7 to 2.8 % (hard). The cost: target-active time pushed down by at least 10 dB rises from 0.8 to 3.8 s (easy) and from 2.0 to 6.2 s (hard). Repo: `docs/BENCHMARKS.md` "Gated-window floor live A/B" (T-326). Fewer *false* gates would cut that cost directly.
- **Reading:** this is a **score shift** (bias) between the clean-piece LDA and separated 1 s crops. The ranking between the streams mostly works: in 26 of the 42 wrong windows the oracle stream was picked and was clean on its own (`mix_back`), and only 13 were wrong picks. What fails is the absolute threshold.

## Background numbers on short-duration verification
- ECAPA-TDNN on VoxCeleb1 with the test cut to 0.5 / 1 / 2 s / full: EER 26.83 / 12.82 / 5.39 / 2.05 % ([arXiv 2506.14226, Table 1](https://arxiv.org/html/2506.14226v1)).
- A second ECAPA re-implementation, full / 5 / 2 / 1.5 / 1 s: EER 1.03 / 1.05 / 1.76 / 2.12 / 3.04 %, with no AS-norm ([arXiv 2509.19721, Table 2](https://arxiv.org/html/2509.19721)).
- Both agree that going from 1 s to 2 s roughly halves the EER.
- The ECAPA paper itself scores with cosine plus adaptive s-norm, using a cohort of 1000 for VoxCeleb and 50 for the cross-dataset test ([arXiv 2005.07143](https://arxiv.org/abs/2005.07143)).
- PLDA needs many training speakers. With two known speakers, the shrunk 2-class LDA is already the right closed-set back end. I found no primary source that compares PLDA and cosine at 1 s for ECAPA, so I make no claim on that.

## 1. Score normalization and calibration on short segments
**1a. Duration- and domain-matched cohort (AS-norm).**
- Plain AS-norm made 0-5 s i-vector trials *worse*: EER 8.74 to 10.51 %, Cllr 0.95 to 1.18 ([Nautsch et al., Odyssey 2014, Tables 3, 4 and 6](https://download.hrz.tu-darmstadt.de/pub/FB20/Dekanat/Publikationen/CASED/odyssey.pdf)).
- Selecting the cohort from the *same duration class* fixed it: EER 5.63 %, minDCF 0.932 to 0.646, Cllr 0.35 (same paper).
- Adaptive s-norm with the top 200 cohort: minDCF 0.9538 to 0.6771 with a domain-matched cohort, but only to 0.8922 with a mismatched one ([Matějka et al., Interspeech 2017, Tables 1 and 2](https://fit.vut.cz/research/group/speech/public/publi/2017/matejka_interspeech2017_IS170803.pdf)).
- **Lesson:** normalization helps only if the cohort looks like the test crops. Here that means 1 s separated crops.
- Knob: `HEARING_ENROLLED_ASNORM=<cohort .npy>`. Each stream's raw score is replaced by `(s - mu_top) / sd_top` over the top-N cohort scores.
- Data: cohort crops from non-team speakers, separated the same way. The repo has GRID s1 and s4 profiles; more cohort speakers would need new audio.
- Effort: about 3 h. Risk: medium. With a closed set, the LDA already contrasts against the one impostor who matters, so the expected gain is the smallest of the ideas here.

**1b. Logistic (Platt-style) recalibration plus quality measures.**
- Linear calibration (offset + scale, fit by logistic regression) is the standard back end for verification scores ([van Leeuwen & Brümmer, arXiv 1304.1199](https://arxiv.org/abs/1304.1199)).
- Adding a quality term (duration or other side information) to the calibration lowered EER in short conditions ([arXiv 1901.10345, Tables 5 and 7](https://ar5iv.labs.arxiv.org/html/1901.10345)): 2s-2s from 32.95 to 31.75 %, 5s-2s from 30.37 to 27.23 % (SRE08), and on SRE10, 5s-2s from 31.44 to 27.44 %.
- Log-duration quality functions are the classic version ([Mandasari et al., via SRI](https://www.sri.com/publication/speech-natural-language-pubs/quality-measure-functions-for-calibration-of-speaker-recognition-systems-in-various-duration-conditions/); [Hasan et al., ICASSP 2013](https://personal.utdallas.edu/~jxh052100/Publications/CP-ICASSP13-HasanSaeidiHansenLeeuwen-DurationSID-0007663.pdf)).
- Our duration is fixed at 1 s, so the useful quality signal would be stream level or stream SNR.
- Knob: `HEARING_LDA_CALIB=<json>` holding `a, c` (and an optional `q` weight on stream dBFS), so the score becomes `a*(w@e - b) + c + q*level_db`. The gate stays at `< contrast`, but now on calibrated log-odds.
- Effort: about 2 h on the data from idea 2. Risk: low. It is an affine change, and with the knob off, behaviour is unchanged.

## 2. Separated (artifacted) speech: train/test mismatch
- Target speaker extraction followed by verification on overlapped speech: EER 22.67 to 10.40 %. Adding *extracted* speech to the verification training pool brought it to **7.77 %** ([arXiv 1902.02546, Table 1](https://ar5iv.labs.arxiv.org/html/1902.02546)). Clean-test EER also moved slightly, from 3.33 to 3.07 %.
- An enhancement front end tuned for audio can *hurt* verification: 17.68 to 18.32 % at 0 dB with an L2-trained DAE. Training the front end for speaker identity helps instead: 16.56 % ([VoiceID loss, arXiv 1904.03601, Table 2](https://ar5iv.labs.arxiv.org/html/1904.03601)).
- So enrolling on, or fitting the back end on, the *same kind of separated output* that is scored live is the documented fix for a mismatch like ours.
- **Idea 2 (main lever): refit the LDA, or just its offset and scale, on 1 s separated crops.**
  1. Mix the two profiles' audio at the live levels, with 30-70 % overlap and no-overlap stretches. Include the café bed if available.
  2. Run MossFormer2 at the live 1.0 s window and 0.5 s hop.
  3. Embed each stream and label it by its oracle stem, as `eval/wrong_speaker_audit.py` already does.
  4. Fit `fit_lda` on these crop embeddings, or keep the clean `w` and fit only the bias and scale (1b).
  5. In the same pass, refit `CONF_MARGIN_LDA` on team data (the code comment already asks for this).
- Knob: `HEARING_LDA_FIT=separated:<npz>`.
- Data: only the ~60 s per person of profile audio. The t141 test scenes are cut from disjoint takes, so the evaluation stays honest.
- Effort: about 3-4 h, including an offline `wrong_speaker_audit` run (accuracy only, so no `swarm-lock` needed).
- Risk: overfitting with ~120 s of source audio. Also, the LDA is antisymmetric: moving the bias toward Joseph makes the Aryan-as-target gate stricter. Aryan-as-target has about 0 % gated windows now (T-326), so there is headroom, but re-run all 4 rows.

## 3. Closed-set 2-speaker decision: joint logistic vs one threshold
- Today the gate tests one stream's score against 0. With two streams and two known voices, the real question has three answers: H_A (target is stream A), H_B (target is stream B), or H_0 (target absent).
- A model trained jointly on these classes beat combining a VAD with a thresholded verification score. Personal VAD mAP went from 0.900 to 0.946 (ET) and 0.969 (SET), and in noisy multi-condition training from 0.801 to 0.928 ([Ding et al., arXiv 1908.04284, Table 1](https://ar5iv.labs.arxiv.org/html/1908.04284)).
- Multiclass logistic calibration is the standard tool for this ([Brümmer's publication list, Odyssey 2006](https://sites.google.com/site/nikobrummer/)).
- **Idea 3:** a 3-class logistic over five features: `s_A`, `s_B`, `s_A - s_B`, the stream energy ratio and the mix level. The gate fires when P(H_0) is greater than P(H_choice). Confidence is P(H_choice).
- This turns a window where both streams look "Aryan-ish" but A clearly beats B into a pick instead of a gate. That is the T-323 pattern: 26 of the 42 wrong windows are `mix_back` (the oracle stream was picked, then gated or mixed back).
- Knob: `HEARING_GATE_MODEL=<json weights>`.
- Data: the separated crops from idea 2, plus H_0 windows (the other voice alone, room noise only).
- Effort: about 3 h, sharing data with idea 2. Risk: a small training set. Keep the five features with L2, and check H_0 recall so that true gates (8 of 30 in T-323) still fire.

## 4. Temporal smoothing across overlapping windows
- The separator's stream order is arbitrary in each window, so smooth the **target-present log-odds**, not the stream index. The existing `_follow` correlation already maps streams between windows.
- VBx puts a speaker-turn prior `P_loop` on the x-vector sequence (x-vectors every 0.25 s from 1.5 s windows). VBx against AHC on CALLHOME: DER 8.10 to 4.42 % (forgiving setting), and SER 7.53 to 4.10 % (fair setting) ([Landini et al., arXiv 2012.14952](https://ar5iv.labs.arxiv.org/html/2012.14952)). The paper has no ablation that isolates the HMM's share of that gain.
- Online diarization averages overlapping 5 s windows taken every 500 ms. DIHARD III DER is 27.6 % at 1 s latency and 25.0 % at 5 s ([Coria et al., arXiv 2109.06483, Table 1](https://ar5iv.labs.arxiv.org/html/2109.06483)). More look-ahead helps, but live we cannot afford it.
- HMM smoothing gave small DER gains, and over-smoothed turns with many speakers ([arXiv 1904.11130](https://arxiv.org/pdf/1904.11130)).
- **Idea 4a, causal 2-state forward filter:** `HEARING_GATE_SMOOTH=hmm:0.9` filters P(target present) with a stay probability of 0.9 per hop. It adds no latency (no look-ahead).
- **Idea 4b, longer embedding context:** `HEARING_SCORE_CONTEXT_S=2.0` embeds the aligned previous stream plus the current one. Going from 1 s to 2 s of speech halves ECAPA EER (12.82 to 5.39 %; 3.04 to 1.76 %, sources above). It costs about one extra ECAPA pass per window and adds no output latency.
- Data: none new. First read the run lengths of gated windows in `wrong_speaker_audit_20261010-015949.json`. If the 22 false gates come in runs of 2 or more windows, a 0.9 stay probability cannot flip them, and only idea 4b helps.
- Effort: 2 h. Risk: a lag at real turns, i.e. 0.5-1 s of the target's onset muted or the other voice's onset leaking. Check true dropouts and the T-326 "pushed down >= 10 dB" column.

## Ranked top 3
1. **Idea 2 + 1b: refit or recalibrate the LDA on 1 s separated crops from the profile audio, and refit `CONF_MARGIN_LDA` on team data.** It goes straight at the 22 false gates (a bias shift), it is source-backed (extracted-speech training 10.40 to 7.77 %; a duration-matched cohort is what fixes short-segment normalization), it needs no new recordings, and it takes about 3-4 h.
2. **Idea 4b: 2 s scoring context from the aligned previous stream.** It has the largest source-backed effect (1 s to 2 s halves ECAPA EER), uses no look-ahead and is one env knob. Risk: lag at speaker turns.
3. **Idea 3: a 3-class logistic gate (H_A / H_B / H_0) on the two scores plus energy.** It uses the permutation structure that a single threshold throws away, and it reuses idea 2's data. Smoothing (4a) and AS-norm (1a) come after these.

All knobs default off. Measure offline first with `eval/wrong_speaker_audit` (accuracy only). Then run the T-284 command live on all 4 rows against the T-180 bar, and add the T-326 "target pushed down >= 10 dB" check.
