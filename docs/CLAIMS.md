# CLAIMS: what the slides may say (judging Sun Oct 11, 2026)

Rules: claim only what was measured, with its condition. Nothing prerecorded shown as live. No canned captions.
Sources: app repo main at 3b233bd for sections 1-3 as first written; section 1b and the Oct 10 lines checked against main 5960c8e (line numbers are `git show origin/main:` lines).
B = docs/BENCHMARKS.md, D = docs/DEMO.md, R = docs/REHEARSAL.md, PLC = docs/research/PROFILE_LEARNING_CURVE.md. "Notes" = orchestrator notes, Oct 9. Written Oct 9, 2026; offline results added Oct 10 ~04:00 UTC (previous version: CLAIMS-prev.md).

## 1. SAFE TO CLAIM (number | source | slide wording)
1. Real voices, solo speech right | Notes (T-141, 12 runs, Mac) | "With our two real voices, the chosen speaker is followed correctly when they talk alone."
2. Wrong-speaker time 6-25%, only during overlap | Notes (T-141) | "When both of us talk at once, the wrong voice is heard 6-25% of the time. That is the hard case."
3. Best case, Joseph as target: 8.6-9.1% wrong, other voice 21 dB down, SI-SDRi +4.6 dB | Notes (T-141) | "Best run: other voice 21 dB quieter, 4.6 dB cleaner (SI-SDRi), 2 male voices, Mac."
4. Name Call: 2 of 4 clean takes, 0 of 4 with noise or overlap, 0 false triggers | Notes (Joseph's take) | "Saying the name worked 2 of 4 times in quiet, 0 of 4 with noise or overlap, and never fired by mistake."
5. Name Call false triggers: 0 in 5.06 min of speech without the name | B:1141 (T-120, GRID + 4 Kokoro voices, not our voices) | Only if the condition is said: "0 false triggers in 5 min of other speech."
6. Output lag: about 1.5 s on the Mac (Balanced preset, by design), about 1 s on the A100 (950 ms measured) | B:162 and B:429 (T-065, window+hop 1500 ms), B:856 (T-110 runs 211552/703/815, replay clip), Notes | "The isolated voice plays about 1.5 s behind the room (about 1 s on a GPU)." Say "delay", never "real-time".
7. Engine-only: 13 ms algorithm + 0.7 ms p95 compute per 10 ms chunk (RTF 0.06), noise-only denoiser, M4 Pro | B:160 (bench --full) | "The noise filter itself adds 13 ms. This excludes Bluetooth and the Focus delay." Tag it "engine-side".
8. Stand-in clip (GRID male/female, replay, target locked from the manifest, Mac): separation +8.3 to +8.5 dB SI-SDRi, other talker 18.1-18.8 dB down, locked to Isolated in 5.5-5.9 s | B:1325 (T-135 runs regress-20261008-203649 and -204144), B:161 | "On a test clip with two different voices, separation was +8.3 dB with the other voice 18 dB down." Never say this is our voices.
9. Captions: word error 71-83% (3 runs 83/71/82) on a GRID stand-in clip with a competing talker, no caption prompt, Mac | docs/decisions/T-139.md (batch B), B:1129 | "Our live captions are rough: 71-83% word error on a hard test clip." Disclosure only. Captions are whatever Whisper heard.
10. Each session is recorded (in.wav, out.wav, events.jsonl) | CLAUDE.md, D | "Every run is logged, so numbers come from recordings, not memory." Use only if true on stage.

## 1b. OFFLINE RESULTS (measured, but not live; background or Q&A only, always with the word "offline")
None of these changed the demo. None is what the judge hears. Section 1 is unchanged: nothing here was measured live.
1. Our two saved profiles are far apart: ECAPA cos(aryan, joseph) = 0.018 (the "same voice" trap is at 0.7) | B:1696 (T-178), R:73; current profiles = close-mic Voice Memos (aryan 50.0 s, joseph 13.95 s), laptop-a | "Offline, the speaker model scores our two saved voices as clearly different (similarity 0.018 of 1)." Do not turn this into "it never confuses us": overlap still does (section 1, line 2).
2. Contrast-margin sweep (T-178), offline, laptop-a M4 Pro MPS, scenes t141/aryan demo-team-easy/-hard, run `contrast_sweep_20261009-200458`: at contrast 0.3 wrong-speaker windows fell 11.3% -> 3.8% (easy) and 11.3% -> 5.7% (hard), but 13.2% / 11.3% of the target's windows were gated (live, those would play the ducked mix). Session-level check never ran (enrolled tier overran). **Not adopted; default stays 0.** | B:1690-1724, docs/decisions/T-178.md | Only as "we tried a stricter voice match offline; it cut wrong picks but dropped too much of the target, so we did not ship it."
3. Mixer-floor knobs K2/K3/K5 (T-180), offline, laptop-a MPS, same two scenes, run `floor_ab_20261009-212547`: K2 leak 0.4 / 2.0 dB better but +0.5 s dropouts; K3 wrong-speaker up (5.2 -> 6.2%, 6.2 -> 7.3%); K5 no measurable change. **No win, none adopted**; config/demo-team.yaml unchanged. | B:1743-1800, docs/decisions/T-180.md | Not a slide claim. Q&A only: "tested, no gain, not shipped."
4. More enrollment examples (T-201), offline, laptop-a, frozen ECAPA, 1/2/4/8 clean 1.5 s examples, one run, 9-21 windows per cell: no reliable gain; the existing full saved profile matches or beats every variant at 2 s crops. Aryan and Joseph were recorded on different devices, so it is optimistic. **Verdict: no promotion.** | PLC:23-47, docs/decisions/T-201.md | Say "saved voice profile". Never "learns over time" or "gets better the more you use it".
5. Learned projection head on ECAPA (T-246), offline, **synthetic embeddings** (experiments/theory/T-246/synthetic_data.py, 300 cases, seed 42), not real audio: the non-linear MLP head false-accepts 52% of target-absent windows (cosine baseline and closed-set LDA: 0%). **Not adopted** (the demo keeps closed-set LDA). | **pending merge**: branch task/T-246 (1dd5616), docs/research/theory/projection_head.md section 8, docs/decisions/T-246.md | Not a slide claim.
6. GPU separation baseline, run `t210-a4500-20261010T0315` (Joseph's iLab job 447101, NVIDIA RTX A4500), **offline separation, not the live demo**: MossFormer2 + ECAPA pick on a digital mix of the team holdouts (first 16 s, 8 scored 2 s windows per condition, AGC off, no noise). Absolute **SI-SDR** (not SI-SDRi): 16.5 dB (Joseph target) / 13.3 dB (Aryan target) at 0 dB SIR; 11.6 dB / 9.1 dB at -10 dB SIR (Aryan -10 dB: 1 separation failure and 1 mix-back failure of 8 windows). Pick agreed with the oracle stream in every window. Compute 227-330 ms mean per 2 s window. | **pending merge**: branch task/T-210 (98c04c2) docs/research/OVERNIGHT_BASELINE.md "Readiness and steady-state evidence"; branch task/T-262 (ecd2009) docs/research/gpu_model_review.md section 2.2. Neither is on main. | Only with the label: "Offline, on a digital mix of our two voices, the separator on a GPU kept the selected voice clean (SI-SDR 9-17 dB)." Never as an improvement number, never as live, and it is not the Mac path the demo runs on.

## 2. NOT YET MEASURED (do not put on slides until filled)
- A100 with saved voice profiles (keys 1/2 on the GPU): T-129 (A100 tuning + ILAB_PROFILES). Until then the A100 learns voices live (target alone 3-5 s).
- Whether the GPU improves quality at all: T-129. Only the delay gain (about 1 s vs 1.5 s) is measured.
- Mouth-to-ear round trip through MacBook mic -> AirPods Pro 3: scripts/bt_latency.sh (T-144), rehearsal step 2. Bar median <= 300 ms, p95 <= 350 ms is a target, not a result. B:166-168 leave blanks for it.
- Start -> Isolated with a live person: rehearsal sheet step 5 (T-147/T-157; bars 6 s GPU, 8 s Mac). The 5.5-5.9 s above is a replay clip.
- Share of time still Isolated while Aryan talks over Joseph with cafe noise: rehearsal step 7 (bar 90% of ticks).
- Space A/B flips and Name Call counts on the real devices: rehearsal steps 8 and 9 (bars >= 3 and >= 2).
- Caption WER on the team's own voices (and Quiet-mode captions from one person): rehearsal step 10 with captions.txt; T-149/T-150 (Focus/Quiet) have no numbers yet.
- Real-voice eval after any new fix: T-141 numbers are from the Mac only, 12 runs.
- Worklet underruns (the sheet always prints NOT MEASURED, R:56). Real-room DNSMOS (B:135: 0 clips). AirPods ANC effect: never measured.
- Contrast 0.3 (T-178) and the K2/K3/K5 floor knobs (T-180) at session level: both batches failed because the enrolled tier overran on the Mac (B:1726-1735, B:1755-1762). Offline only (section 1b).
- Voice profiles recorded on the demo mic (T-172 re-enroll, Sat mic gate) and a same-mic held-out test (`team/<name>/holdout-0N.wav`): not recorded yet. T-201's caveat (different mic paths) stays until then.
- Live GPU quality with the saved profiles on the A4500 or A100: the t210 numbers (1b.6) are offline separation on a digital mix.
- Effect on people with sensory overload: no user study. The pitch is motivation, not a measured benefit.

## 3. DO NOT CLAIM (with the measured reason)
- "Works on any two voices" / "separates similar voices": T-141 is NO-GO. Blind separation fails on two overlapping male voices (Q-399); even a perfect stream pick leaves about 3.8% wrong-speaker time. Pitch (F0) cannot help, ranges overlap (Notes).
- "+13.8 dB", "+11.8 dB", "+13.7 dB", "WER 1.41 to 0.78" (B:143-147): they need lip video, clean enrollment or an oracle speaking signal on GRID/LibriSpeech. The demo has no camera and real voices score +4.6 dB.
- Association accuracy 0.945 and similar (D:194): GRID, oracle lock, camera era. Same-gender was 0.742 (M/M). Our pair is M/M.
- Name Call "0.0 s wrong, -36 dB" (B:1308, T-132): GRID stand-in pair, not real voices. Real takes: 2/4 and 0/4.
- "The GPU is better": the A100 replay on the easy stand-in clip scored worse than the Mac: SI-SDRi -3.8 dB, leak -4.7 dB, about 0.97 s delay (Notes). Older A100 runs (B:856, B:876-890) swing from +0.1 to +5.8 dB. No quality claim either way.
- "Low latency", "real-time", "under 100 ms": the lag is 1.5 s (Mac) or 1 s (A100). The 13 ms and the 64 ms passthrough (T-013, "not the demo path", B:163) are not what the judge hears. Bluetooth adds an unmeasured 150-250 ms (D:194, an estimate).
- "Accurate captions" or "captions from only that person": WER 71-83%, and the ASR was fed the mixture, so the other talker's words became captions (T-139 table, 6.2-15 s). No caption prompt or vocabulary hints may be used (Q-375).
- Anything from REPLAY / "Replay clip" as live. It shows a REPLAY badge. Saved profiles (keys 1/2) are cuts of Voice Memos: say "saved voice profile", not "learned just now".
- "Learns your voice over time" / "gets better with use": T-201 found no reliable gain from more enrollment examples (PLC:43-47). Say "saved voice profile".
- Camera, lip tracking or face lock: cut from the demo (Aryan + Joseph directive, Oct 9).
- Synthetic mode / re-voiced speech (1.70 s p50 first word, B:1129): Kokoro, stand-in clip, ElevenLabs never measured; not in the Focus/Quiet pitch.
- That the judges' voices are isolated: only Aryan's and Joseph's profiles are tuned.
- "+16.5 dB" or any t210 number as SI-SDRi, as live, or as the demo: it is absolute SI-SDR, offline, on a digital mix, on Joseph's RTX A4500, and pending merge (1b.6).
- "Tuned the voice match" / "smarter speaker model": contrast (T-178), floor knobs (T-180) and the projection head (T-246, synthetic, 52% false accepts on target-absent windows) were all not adopted.

## How to update this sheet
After T-129 posts, each rehearsal (T-147/T-157 writes results/rehearsal_<stamp>.md), and any T-141 re-run: move a line from section 2 to section 1 with the run id, or add a row to section 3 if it came out worse.
