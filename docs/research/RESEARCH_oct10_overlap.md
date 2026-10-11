# Using the delay and window structure better on gated / overlap windows (research, Oct 10)
No repo edited. [B] = `git show origin/main:docs/BENCHMARKS.md` (section named); [S] = `origin/main:hearing/tse/stage.py`, [E] = `hearing/tse/enrolled.py`.
[P] = my replay of the T-323 per-window file `eval/results/wrong_speaker_audit_20261010-015949.json` (4 scenes x 59 windows; fields `scores`, `margin`,
`gated`, `follow`, `oracle`, `target_active`). [P] is an in-sample, accuracy-only proxy: it counts windows, not wrong-speaker seconds. Nothing here was measured live.

## 0. Facts that bound every idea
- **There is no free lookahead.** Delay = window + hop = 1.5 s threaded [S, docstring and `self.delay = self.window + self.hop`]. Output of input
  sample s plays at s + 1.5 s; window k = [kh, kh+1 s) finishes by kh + 1.5 s, so when region [kh, kh+0.5) starts playing only windows <= k exist,
  and both windows covering it are already crossfaded [S, `_finish` overlap-adds each window's chosen stream on arrival]. Every extra window of
  lookahead costs +0.5 s (2.0 s total for one, 2.5 s for two). T-324's "one-hop lookahead" was the same min-over-covering-windows trick [B, T-324].
- **The gate is the problem, not the pick.** 34 of 42 wrong windows are gated mix-back; 22 of 30 joseph-row gated wrong windows had the target
  talking [B, T-323]. Floor 20 fixes wrong-speaker but adds 2.8-4.2 s of target >= 10 dB down per row [B, T-326, T-342].
- **False gates come in runs.** [P] joseph rows: 17 + 22 false gates (gated, target active) vs 4 + 4 true gates. Runs are 1-3 windows, with one run
  of 8 (hard), interleaved with accepted windows, e.g. `.GG..GGG.GGG..GG..GGG.` (easy). The relative pick is right in most of them: the oracle
  stream is the argmax in 13/17 and 17/22 false gates. The absolute "target present" score is biased low; the ranking between streams is not.
- **Stream identity across windows is shaky.** Following `follow` [E `_follow_hop`, waveform correlation on the 0.5 s overlap] the oracle stream
  changes track label 7-12 times in 58 hops per scene [P]. Some of that is real turn-taking, but it means "decide once per speaker turn" is not free.

## 1. Bidirectional smoothing of the pick (forward-backward / Viterbi with a turn prior)
- Method: a 2-state HMM (target present / absent) over hops with a stay probability, decoded with a fixed lag L. VBx uses the same prior,
  p(s|s') = (1-P_loop)pi_s + delta(s=s')P_loop, on x-vectors every 0.25 s from 1.5 s windows ([Landini et al., arXiv 2012.14952 §2.2, §3.2](https://ar5iv.labs.arxiv.org/html/2012.14952));
  the paper gives no ablation of P_loop alone. Fixed-lag smoothing is the online form of forward-backward and trades a fixed delay for lower error
  ([UVic CSC421 slides](https://www.cs.uvic.ca/~gtzan/csc421/Spring2005/slides/27.pdf); [Anderson, "From Wiener to HMMs"](https://researchportalplus.anu.edu.au/en/publications/from-wiener-to-hidden-markov-models-2/)).
- **Proxy result [P]:** evidence = scale x max LDA score, stay 0.8/0.9, lag 0/1/2. Joseph false gates 17 -> 13-16 and 22 -> 17-20; true gates
  4 -> 3-5. Lag 1-2 is no better than lag 0. The bias is sustained over 1-1.5 s runs, so a turn prior cannot outvote it (as the calibration note
  predicted, RESEARCH_oct10_calibration.md §4). A "sandwich" rule (un-gate when the accepted windows on both sides picked the same track) fixes
  2/3/4/5 of 17 and 4/4/4/4 of 22 at lag 1/2/3/4 hops, with 0-2 wrongly un-gated true gates [P].
- Effect: wrong-speaker ~flat, target muting down ~10-20 % of false gates; compute negligible; +0.5 s latency per hop of lag. Test: replay `eval.wrong_speaker_audit` rows through the smoother (done above as [P]); then T-284 command live. Build: 3-4 h (the stage must hold
  both streams per window and commit L hops late). Knob: yes (`HEARING_PICK_LAG=0` = today). **Verdict: low value for the cost.**

## 2. Stitching streams across windows by permutation continuity
- CSS systems split audio into overlapping chunks (2.4 s, 0.8 s hop) and pick the permutation with the smallest mask distance on the 1.6 s overlap,
  then average the overlap ([Raj et al., arXiv 2011.02014 §3.1](https://ar5iv.labs.arxiv.org/html/2011.02014)); RSAN does the same on past+future frames
  ([arXiv 2110.14838 §2.2](https://ar5iv.labs.arxiv.org/html/2110.14838)). Online EEND keeps a speaker-tracing buffer and picks the permutation with the
  largest correlation to buffered outputs: CALLHOME DER 38.29 % without vs 12.84 % with the buffer at 1 s chunks ([Xue et al., arXiv 2006.02616](https://ar5iv.labs.arxiv.org/html/2006.02616)); Streaming Sortformer solves it with an arrival-order speaker cache ([arXiv 2507.18446](https://arxiv.org/abs/2507.18446)).
- The repo already stitches the chosen stream by correlation on a 0.5 s overlap [E `_follow`]; CSS overlaps are 1.6 s, ours is 0.5 s. [P] shows
  frequent track flips. Turning the per-window pick into a per-track pick: causal EMA of each track's LDA score (a = 0.5) matched the oracle 55/53/50/48
  of 59 vs the per-window argmax 53/53/49/48 [P]. **No real gain; skip as a standalone** (cost ~0, 2-3 h, knob-able); keep it as plumbing for #3.

## 3. Longer matching context: 2-4 s running embedding per tracked stream
- ECAPA EER on VoxCeleb1 with the test cut to 1 / 2 s: 12.82 / 5.39 % ([arXiv 2506.14226 Table 1](https://arxiv.org/html/2506.14226v1)); another
  re-implementation 3.04 / 1.76 % ([arXiv 2509.19721 Table 2](https://arxiv.org/html/2509.19721)). That is the bias source: one 1 s separated crop.
- diart only updates a speaker centroid when that speaker's summed activity in the chunk exceeds rho_update, and weights embedding frames
  away from overlap (AMI DER 30.5 -> 27.5 % with overlap-aware weights) ([Coria et al., arXiv 2109.06483 §2.2](https://ar5iv.labs.arxiv.org/html/2109.06483)).
- Transfer: embed the stitched track audio of the last 2 s (previous chosen stream's second half + current stream), only when the stitch
  correlation is high; else fall back to 1 s. Expected: fewer false gates where the target talks across >= 2 hops (most runs [P]); risk is lag at
  turns (0.5-1 s of the other's onset carried). Compute: one extra ECAPA pass per stream per hop on 2 s (estimate, unmeasured; hop p95 is ~255 ms of a
  500 ms budget [B, T-284]). Test: a `--context-s 2` arm in `eval.wrong_speaker_audit`, then T-284 live + T-326 check. 2-3 h. Knob: yes (= calibration note 4b + stitch guard).

## 4. Soft decisions instead of a hard gate
- Noise suppressors with speech-presence uncertainty use a weighted geometric mean of gains, G = G_H1^p x G_min^(1-p), i.e. in dB a linear
  interpolation by p ([Cohen & Berdugo 2001, Signal Processing](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/sp_Nov2001.pdf)).
  Same shape here: play the argmax stream at gain_dB = -(1-p) x D_stream and the residual at -20 dB, with p from the T-336 head or a margin map.
- What listeners prefer: silent gaps in speech hurt; filling them with noise raised intelligibility 9 % ([Baskent 2012, JARO](https://pmc.ncbi.nlm.nih.gov/articles/PMC3441953))
  and 30-33 % for 40 ms gaps in another study ([CiNii record, interrupted time-reversed speech](https://cir.nii.ac.jp/crid/1050861482657471744), via search summary, not read).
  Hearing-aid users trade noise removal against speech distortion and split into "NR lovers" and "NR haters"; strong NR cut word scores 60 -> 52 %
  ([Neher & Wagener 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017568)). Attenuation distortion of the target had the least effect on intelligibility;
  amplification > 6.02 dB the most ([Loizou & Kim 2011, IEEE TASLP](https://pmc.ncbi.nlm.nih.gov/articles/PMC3169296)) (that is per T-F bin, not 3 s whole-band dropouts). Reading: a target 6-10 dB down is tolerable, a 3 s target
  dropout is not, so prefer "slightly leaky" over "drops out". No study tests wrong-talker leak vs target dropout directly (TSE "target confusion"
  work uses SI-SDRi plus a small listening check: [Zhao et al., arXiv 2204.01355](https://ar5iv.labs.arxiv.org/html/2204.01355)).
- Data for it [P]: in false gates the argmax is the oracle 30/39; the margin is small there (median 0.20-0.22) and large in true gates (median 0.57-0.61).
  With the existing hold threshold (margin < 0.4) as "unsure, likely overlap": 32/39 false gates vs 3/8 true gates flagged (n = 8, in-sample).
- Zero-code first step: `HEARING_GATED_FLOOR_DB=20 HEARING_GATED_PLAY=stream` (K2+K5). It was only run on the aryan rows, where gating is rare
  [B, T-180]; never on the joseph rows where it matters. Expect target-down time to fall back toward unset and wrong-speaker to rise by the
  ~8 true-gate + ~9 wrong-argmax windows, so likely between unset and floor 20. Note K5 alone gave no help because confidence 0 still sets the residual floor to 6 dB [B, T-323].
- Soft knob = a tunable point between those ends. Caveat: `session_score` counts other-dominant only when ducked < 10 dB, and
  T-326 counts target >= 10 dB down, so D near 10 dB can game both metrics; choose D by listening. Compute 0. Build 1.5-2 h. Knob: yes.

## 5. What online diarization does that transfers
- diart: 5 s buffer every 500 ms, per-chunk local speakers matched to global centroids by Hungarian assignment with a cannot-link constraint (two
  local speakers never map to one centroid), new speaker if distance > delta_new; outputs averaged over all buffer positions; DIHARD III DER
  27.6 % at 1 s latency vs 25.0 % at 5 s, confusion is what rises at low latency [Coria et al. §2.2-2.3, Table 1].
- Transfers: (a) **cannot-link**: with two known talkers and two speech-active streams, one stream must be joseph; assign by relative score and
  use the absolute gate only when one stream is active. This is exactly the [P] pattern (ranking right, absolute score biased). Needs per-stream
  activity (Silero per stream, RESEARCH_oct10_pvad.md #1). (b) **update profiles only on confident solo speech** (rho_update). (c) More latency
  buys less confusion, but only slowly (2.6 DER points for 4 s), so do not raise the 1.5 s delay for it.

## Ranked top 3
1. **Cannot-link soft gate (#4 + #5a).** In a window the extractor would gate: if both streams are speech-active (or margin < hold as the fallback
   cue), play the argmax stream at a p-dependent gain and duck the residual 20 dB; otherwise keep floor 20. Goes at the 30/39 false gates with the
   right argmax. 0.5 h for the zero-code K2+K5 joseph-row A/B, then ~3 h for the knob.
2. **2 s stitched-track embedding (#3).** Attacks the score bias at its source (1 s -> 2 s halves EER). ~3 h. No added latency.
3. **Fixed-lag sandwich smoother (#1)**, only if #1-#2 leave runs of 1-2 false gates. +0.5 s delay per hop of lag; [P] gain is small.

## Algorithm sketch for #1 (knob `HEARING_GATED_SOFT=1`, default off; all else unchanged)
```
per window k (inside EnrolledCandidate.extract, after scores/margin/gated are computed):
  if not gated: return as today
  act = [stream_active(y) for y in streams]        # Silero p>0.5 on >=20% of 30 ms frames, or rms > noise_floor+6 dB
  overlap = all(act) or (not any_vad and margin < hold)   # fallback cue when no VAD is wired
  if not overlap: return zeros, confidence 0       # floor 20 as today (HEARING_GATED_FLOOR_DB=20)
  j = argmax(scores)                               # cannot-link: one of two active streams is the target
  p = head_p[j] if team head else sigmoid(a*(scores[j]-scores[1-j]) + b)   # fit a,b on t141 tuning scenes (T-340 split)
  g_db = -(1 - p) * D_STREAM                       # OM-LSA style; D_STREAM default 12, sweep 6/9/12
  self.soft_gated = True; self.confidence = 0      # mixer: residual at the gated floor (20 dB)
  return streams[j] * 10**(g_db/20)                # overlap-add + 30 ms ramp smooth the step
offline: eval.wrong_speaker_audit --gated-soft D (add the arm), joseph+aryan, easy+hard; report wrong windows,
         other_dominant_s and the T-326 target>=10 dB-down seconds; keep only if target-down falls >= 1.5 s/row and
         wrong-speaker stays within +1 pp of floor 20.  live: the T-284 command x 4 rows in one swarm-lock batch; then a listening pass.
```
