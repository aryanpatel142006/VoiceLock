# Who is talking right now? Target-activity signals for the gated-window floor (Oct 10)

Problem, from the repo [R1]. Most wrong-speaker time is mix-back from gated windows: 34 of 42 wrong windows, 11.4 of 14.4 s (T-323).
In 22 of the 30 gated joseph-as-target wrong windows the target *was* talking. In the other 8 he was silent and the 6 dB floor let aryan
through. `HEARING_GATED_FLOOR_DB=20` cut joseph rows 14.8 -> 1.4 % and 19.7 -> 2.8 % wrong-speaker, but it added 2.8-4.2 s of target-active
time at >= 10 dB down (T-326) [R1]. The missing input is a per-window "target active?" probability p. Integration point for every option below:
`confidence_floor_db()` in `hearing/fusion/mixer.py`, where `HEARING_GATED_FLOOR_DB` (K2) is read [R2]. Behind a new env knob (default off),
a gated window gets floor = 20 dB if p < 0.3, 6 dB if p > 0.7, and an interpolation in between.
Caveat for all identity-based options: the LDA already confuses joseph's stream for aryan [R1]. A detector conditioned on the same ECAPA
embedding (`spkrec-ecapa-voxceleb`, the model in `hearing/tse/enrolled.py` [R2]) may share that blind spot.

## 1. Personal VAD / target-speaker VAD
- **Google Personal VAD (2019).** A 2-layer LSTM (64 cells) on 40-d log-mel input at a 10 ms step, conditioned on a 256-d d-vector. It outputs
  per frame {non-speech, target, non-target}. ET variant: 0.13 M params, ~130 KB at 8 bit. tss AP is 0.932-0.955 clean and 0.878-0.916 in
  multistyle noise (cafe noise included) [P1]. The test data are concatenated LibriSpeech turns, not overlapped speech [P1]. No code or weights
  are mentioned [P1].
- **Personal VAD 2.0 (Interspeech 2022).** A causal Conformer (4 layers, dim 64, no right context) with FiLM speaker modulation: 1.0 MB at
  8 bit, 9.58 M FLOPs. It is evaluated only by ASR WER (no frame metrics), and no release is mentioned [P2].
- **Comparative PVAD study (UT Austin and Apple, 2024).** Frame EER for PVAD is 10.2-12.2 % for the end-to-end variants vs 26.2 % for a
  score-combination baseline. Median detection latency is 185-240 ms vs 432 ms. Models are 0.08-0.40 M params [P3]. No code statement [P3].
- **ASE-PVAD (ICASSP 2026).** It adapts the enrollment embedding online and targets short enrollment. Code is at an anonymous 4open link [P4].
  No numbers on the abstract page, and the code licence is not stated [P4].
- **FireRedChat pVAD (Sept 2025): the only downloadable, permissively licensed PVAD found.** `pvad.onnx` is 3.94 MB, Apache-2.0, and ships
  with SpeechBrain `spkrec-ecapa-voxceleb` (also Apache-2.0) [P5]. The architecture is a causal conv + GRU over mel features. The ECAPA
  embedding is concatenated to the conv features, and the model outputs a per-step speaking probability. It was trained on about 2,000 h of
  Mandarin/English with interferers or MUSAN noise at 0-30 dB [P7]. Inference code: 160-sample (10 ms at 16 kHz) steps, CPU ONNX, a
  carried mel buffer (1x80x15) and GRU state (2x1x256), and a 192-d L2-normalised ECAPA speaker input [P6]. No frame-level accuracy is published.
  The paper gives only a system-level false barge-in of 10.2 % vs 33.4 % (LiveKit) with a second speaker at 20 dB SNR [P7].
  **Fit:** the repo already uses this exact ECAPA, so saved profiles plug in directly [R2]. The LiveKit wrapper is not needed: the ONNX call is 4 tensors [P6].
- **Online TS-VAD (Wang et al., 2022).** A block-wise ResNet front-end that updates target embeddings online, for meeting diarization
  (AliMeeting) [P8]. The authors note the ResNet front-end is not time-efficient for online use [P8]. No weights found. Research only.
- **Effect.** It aims straight at the 22 "target talking but gated" windows: keep 6 dB there (fewer target mutes) and go to 20 dB in the 8
  target-silent windows [R1]. In overlap, a mix-level PVAD should say "target active", so the other voice still leaks at 6 dB. Running it
  per separated stream (2 streams) instead also tells *which* stream holds the target, a second opinion to the LDA (projection, not measured).
- **Cost per 0.5 s hop.** 50 ONNX steps on the mix, or 2x100 per 1 s window per stream from a reset state. Not measured. For scale,
  Silero (2 MB) takes 189 us per 31 ms chunk with 1-thread ONNX [S2]. Time to try offline: **3-4 h**.

## 2. Overlapped-speech detection (OSD)
- **pyannote segmentation-3.0.** MIT, gated on HF (accept the terms) [O1]. Input is 10 s of 16 kHz mono. Output is 7 powerset classes
  (silence, speakers 1-3, pairs), so at most 2 speakers per frame [O1]. Architecture: SincNet + 4 BiLSTM + FC [O2]. 589 frames per 10 s =
  17 ms per frame. An ungated MIT ONNX export exists [O3]. The BiLSTM is not causal inside the window, which suits the existing 1.0 s
  window and 1.5 s delay. Speaker slots are local and unordered, so they need an ECAPA or stream match to name the target. Cost: diart
  reports 12 ms CPU / 8 ms GPU for the older `pyannote/segmentation` [O4]. Seg-3.0 on MPS has not been measured.
- **diart (streaming).** MIT. Segmentation + embedding + incremental clustering. Latency is adjustable from 500 ms to 5 s, and the buffer updates
  every 500 ms [O4]. It accepts ECAPA/WeSpeaker embeddings, but enrolment of known speakers is not documented [O4].
- **NVIDIA Streaming Sortformer 4spk-v2.** CC-BY-4.0, 117 M params, 80 ms frames. A 0.32 s latency config has RTF 0.18 on an RTX 6000 Ada,
  and the 1.04 s config has CALLHOME 2-spk DER 6.57 % [O5]. It needs NeMo; no CPU or Apple numbers are given [O5]. Slots are by arrival
  order, not enrolment. Heavy next to MossFormer2 on the same M4.
- **Effect.** OSD alone says "two voices now" (where T-141 put all wrong-speaker time [R1]), not who. Rule: a gated window with overlap gets
  the hard floor, and a gated window with a solo voice keeps 6 dB. That cuts other-voice leak in overlap and mutes the target there. Solo
  speech, where the target is otherwise muted by mistake, is spared. Time to try offline: **3-5 h** (seg-3.0 ONNX).

## 3. Separated-stream energy + cheap VAD (identity-free)
- **Silero VAD.** MIT, about 2 MB JIT, 8/16 kHz, under 1 ms per 30+ ms chunk on one CPU thread [S1]. v5 ONNX: 189 us per 31.25 ms chunk at
  16 kHz, 1 thread [S2]. Per hop, 2 streams x 1.0 s = 64 chunks, about 12 ms (derived from [S2]). Silero claims robustness to background
  noise, with no numbers given [S1].
- **WebRTC VAD (py-webrtcvad).** MIT (GitHub licence file). 10/20/30 ms frames at 8-48 kHz, 16-bit mono. Aggressiveness 0-3. Speech vs
  non-speech only [S3]. Cost is negligible. It is weaker than Silero in café babble (no source found; judgement).
- **Signal.** Per 10-30 ms frame, VAD(stream A) and VAD(stream B), plus each stream's energy relative to the window mix. "Both active" means
  overlap (an OSD substitute). "One active" means solo, and in solo the pick is right [R1], so a gated solo window can trust the louder
  stream. MossFormer2 leakage of about -21 dB on the real pair [R1] should stay under the VAD threshold (to verify).
- **Effect.** Like option 2: fewer target mutes in solo gated windows, a hard floor only in overlap. Zero new models.
  Time to try offline: **2-3 h** (one column in `eval/wrong_speaker_audit.py`, which already replays the gated windows with stems [R1]).

## 4. How hearing aids and earbuds decide to duck
- **Hearing-aid noise reduction** detects speech by the modulation index per band. If time constants are too fast, a brief /d/ or /t/
  burst is cut "as if it were a noise". If too slow, noise is caught late. The gain cut is spectrally weighted, least in the mid bands
  where the speech cues are [H1].
- **Signia OVP 2.0** detects only the *wearer's* voice, from spatial cues learned at fitting ("not the sound quality" of the voice). It
  switches "almost instantly" to an own-voice path with different compression [H2].
- **AirPods Conversation Awareness** ducks media when *you* start speaking, emphasises voices in front, stays down "as long as your
  conversation continues", then restores [H3].
- **Lesson.** Products key ducking on a cheap, reliable cue (modulation, own-voice geometry, beam direction), not on telling two similar
  voices apart. They also hold the decision through the conversation instead of flipping per window. Spectral-weighted ducking (keep the
  mids) would soften a wrong mute. The repo's own asymmetric attack/release knob (T-324) raised wrong-speaker on 2/4 offline rows and was
  not adopted [R1], so hold time alone is not the fix.

## Ranked top 3
1. **Per-stream Silero activity, with overlap gating the hard floor (option 3).** About 12 ms per hop, MIT, no identity risk, 2-3 h offline.
   Targets the gated solo windows (fewer target mutes) and keeps 20 dB only in overlap.
2. **FireRedChat pVAD on the mix and per stream (option 1).** The only open PVAD (Apache-2.0, 3.9 MB, 10 ms steps), on the same ECAPA as
   the repo. It answers "is joseph talking" directly, but the accuracy on two similar male voices is unknown and the ECAPA blind spot may carry over. 3-4 h offline.
3. **pyannote seg-3.0 OSD on the 1 s window (option 2).** MIT, 17 ms frames, about 12 ms CPU class [O4]. A better-trained overlap
   detector than #1's heuristic; use it if Silero's per-stream activity is noisy on separator output. 3-5 h offline.
Skip for Sunday: Sortformer (NeMo, 117 M params, no Mac numbers [O5]), TS-VAD and Google PVAD (no weights [P1][P2][P8]).
Offline bar for each: rerun the T-323 audit and T-326 target-attenuation check, and beat floor 20 on "target >= 10 dB down" without giving
back more than 1 pp of wrong-speaker on the joseph rows [R1].

## Sources
- [R1] `git show origin/main:docs/BENCHMARKS.md`, sections T-323, T-326 (`HEARING_GATED_FLOOR_DB`) and T-324, repo ~/CODE/selective-hearing
- [R2] origin/main `hearing/fusion/mixer.py` (GATED_ENV, `confidence_floor_db`), `hearing/tse/enrolled.py` (MODEL = spkrec-ecapa-voxceleb)
- [P1] https://arxiv.org/abs/1908.04284 · [P2] https://arxiv.org/abs/2204.03793 · [P3] https://arxiv.org/html/2406.09443v1 · [P4] https://arxiv.org/abs/2601.12769
- [P5] https://huggingface.co/FireRedTeam/FireRedChat-pvad (files, NOTICE) · [P6] https://github.com/fireredchat-submodules/livekit-plugins-fireredchat-pvad (`vad.py`)
- [P7] https://arxiv.org/abs/2509.06502 (FireRedChat, streaming pVAD section) · [P8] https://arxiv.org/abs/2207.05920
- [O1] https://huggingface.co/pyannote/segmentation-3.0 · [O2] https://arxiv.org/abs/2310.13025 · [O3] https://huggingface.co/soniqo/Pyannote-Segmentation-ONNX
- [O4] https://github.com/juanmc2005/diart · [O5] https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
- [S1] https://github.com/snakers4/silero-vad · [S2] https://github.com/snakers4/silero-vad/wiki/Performance-Metrics · [S3] https://github.com/wiseman/py-webrtcvad
- [H1] https://hearingreview.com/inside-hearing/events/speech-or-noise-using-signal-detection-and-noise-reduction
- [H2] https://www.signia-pro.com/en/signia-library/2022-05-signia-ax-own-voice-processing-20/ · [H3] https://support.apple.com/guide/airpods/dev6d977ff21/web
