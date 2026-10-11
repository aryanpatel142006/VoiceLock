# Muse research digest (Oct 10, 2026; 43 posts Q-1032..Q-1085)

## 1. What the Muse agents did
muse-b (Joseph's) set a "challenger tournament" policy (Q-1033/1034), ran six investigation packages P1-P6 (REAL-TSE checkpoints, wrong-speaker rescue, target-presence state machine, DSP post-processor, ASHA scheduler, recording-manifest schema) and posted six paper leads (Q-1064/1065/1082-1085). muse-c (Ahmad's) reviewed the T-307 head training (found a CE-head train/eval normalization mismatch and seed-noise reporting), built `analyze_results.py` (mean+/-SD, McNemar), and ran the Q-1035 "deep research" mission: 15 ranked ideas whose headline is "attack the loud interferer" and "near-field proximity gating". Roughly half the 43 posts are duplicates (the T-307 review was posted 5x; the Q-1035 summary 6x under T-256/T-358/T-365/T-370/T-367). Nothing was measured on our pair; the only measured item (P4) was on a simulated separator. Board state: Q-1086 (open) says Joseph's GPU freeze stands, so laptop-b GPU work (T-374) is gated on Joseph.

## 2. Distinct findings

### Audio engine
| # | Idea | Posts | Evidence | Covered by | Verdict / reason |
|---|------|-------|----------|-----------|------------------|
| A1 | P2 detect->verify->recover: impostor flag, switch to the other separated stream, hold-last fallback, hysteresis | Q-1037, Q-1066 | design only; cited Zhao >1 dB, DCF-Net 0.4% are lab numbers | T-370 (variants A-E on the T-323 cache) | TEST NOW (already running); tell T-370 to verify "mixture consistency by construction" on our cache first (residual energy of mix-s1-s2 per window) before trusting variant C |
| A2 | P3 presence state machine: Schmitt-trigger hysteresis, explicit UNKNOWN, trust-gated prototype EMA | Q-1042, Q-1066 | design; USEF-TP paper; repo already has margin/floor/pause embryos | T-370 hook (per Q-1066 answer); laptop-a pvad note (Silero per-stream) | TEST NOW only the hysteresis+UNKNOWN wrapper inside T-370; trained PVAD head LATER |
| A3 | Adaptive enrollment from confident crops (self-enrollment F3, EvoTSE reliability retrieval, ASE-PVAD keyframe fusion, AFW fast weights) | Q-1047, Q-1064, Q-1065, Q-1082 | papers (AFW +3 dB under mismatch, EvoTSE OOD gains); none on our data; offline R&D exists in `eval/ema_crop_buffer.py` (T-275) and `eval/dynamic_ema_tracker.py` (T-282), never benchmarked | T-371 (2 s stitched embedding) is the static cousin; T-375 retrains the pick | TEST NOW the inference-only variant (new task N1); AFW/EvoTSE training LATER. Risk: drifting onto the wrong male voice during overlap, which is our exact failure, so solo-only updates with a kill rule |
| A4 | Interferer-first cancellation / "kill the bully" / estimate-and-subtract the loud talker | Q-1047-1053, Q-1074/1075, Q-1085 relay | physics argument, no data | none | SKIP. Single-channel: the blind 2-stream separator already yields mix-minus-target as stream 2 (muse-b says so in Q-1037). Multi-channel: we have one mono input (MacBook mic or iPhone over USB). And our measured failure is equal-level overlap, not a quiet target |
| A5 | Spatial: near-field proximity gating, differential mic-pair probe, direction x identity fusion, body-shadow calibration, WPE+LCMV+Wiener front end | Q-1050, Q-1053, Q-1077, Q-1085 relay | physics; muse-c itself notes a 3-6 dB ceiling with 2 mics | none | SKIP for Sunday: needs >=2 synchronized mics; the web SDK sends one Float32 PCM stream (`web/sdk/hearing-client.js`) |
| A6 | P1 REAL-TSE BSRNN checkpoints (tfmap_context_causal) | Q-1061, Q-1067 | paper DEV DNSMOS 1.5-1.66, TER 0.65; weights unlicensed, 250-300 MB, no streaming code | RESEARCH_oct10_models.md section 3; T-373 (PS4, Apache-2.0 BSRNN cousin) | SKIP; T-373 covers the BSRNN family with a licensed, meeting-tuned checkpoint |
| A7 | Frame-level / contextual conditioning (USEF-TP, TF-map graft into R28) | Q-1061, Q-1083 | papers; needs training | models note (USEF-TSE: CC BY-NC, offline-only) | LATER (post-hackathon) |
| A8 | P4 DSP post-processor: mask-conditioned residual restoration, dry/wet blend, band floors | Q-1044, Q-1067 | measured on a SIMULATED separator: SI-SDR 7.05 vs 9.29 (worse), 0.01 RTF; naive restoration -3.8 dB | none; Q-1067 answer: research-only | SKIP for Sunday: the only objective number is a regression; MossFormer2 masks are not exported; no blind listen possible before judging |
| A9 | Generative / flow-matching refiner; causal Mamba (SepMamba) backbone; glimpse reconstruction | Q-1049, Q-1052 | papers | models note section 2 (too slow for a 0.5 s hop), latency note C | SKIP / LATER; glimpse idea is the soft gate in spirit (T-370) |
| A10 | Input-conditional cue choice (Ashihara: verifier EER != TSE cue quality) | Q-1084 | paper | R24 ablation (laptop-b line) | SKIP for Sunday: our only stratum is same-gender; closed-set 2-speaker pick |
| A11 | Codec / waveform augmentation of enrollment and test (ReDimNet2+) | Q-1085 (muse-b) | paper | none | SKIP: no codec in our path (raw PCM over WebSocket; iPhone over USB is PCM) |
| A12 | Report SD across seeds + McNemar on paired per-window outcomes; CE head pre/post-norm bug | Q-1032/1036/1074/1075/1078/1079 | measured on a CPU dry run: a 2.3 pt "win" had 4.0 pt SD | nothing in T-370/T-371/T-375 acceptance | TEST NOW as an acceptance rule (new task N2 + `swarm tell` T-375) |
| A13 | Shared quiet-target eval slice, 300+ mixtures, -12..0 dB, 3 seeds (F1) | Q-1047 | methodology | T-377 (public male-male set) | fold into T-377: strata at -6/0/+6 dB (our failure is equal level), 3 seeds; tell T-377 |
| A14 | P5 ASHA scheduler; T-331 "unverifiable" | Q-1043 | design; muse-b's clone lacks 9b8c3f9f (it IS on origin/main) | T-331 done | SKIP for Sunday (infra) |
| A15 | P6 recording manifest schema + validator; R48 distill B/C matrix and the crop time-alignment join bug | Q-1038, Q-1060 | validator smoke-tested; join misalignment measured (mean 0.285 s, max 7.75 s) | Q-1038 answered (research-only) | LATER; approve the fail-loud join fix as a research-line change, no GPU before Sunday |
| A16 | Smart-squelch per-bin ducker; "inverted psychoacoustic masking" | Q-1077 | speculative | T-370 ducks the residual | SKIP |

### Frontend / UX
| # | Idea | Posts | Evidence | Covered by | Verdict / reason |
|---|------|-------|----------|-----------|------------------|
| U1 | Show an explicit UNKNOWN / "not sure who is talking" state instead of silently muting | Q-1042 | design | T-370 will emit gate/confidence; judge page has no such pill | TEST NOW (new task N3): honest, cheap, and it explains the mute to judges |
| U2 | "New voice - Follow?" tap-to-follow pill, hold-to-learn, never auto-enroll | Q-1042 | design | T-365/T-367 app shell (no) | LATER: demo uses two saved profiles; adds a live failure surface |
| U3 | Phone-runtime presence state machine; IMU recordings (F5) | Q-1042, Q-1047 | none | none | LATER / SKIP |
| U4 | Arena blind-listening batches as the promotion gate | Q-1044, Q-1043 | T-255 prototype, T-333 blocked | T-255 | SKIP for Sunday; the three teammates listen informally at rehearsal |
| U5 | Lip-reading / camera cues | Q-1049 | none | decided: no camera | SKIP |

## 3. New tasks worth filing before Sunday
- **N1. Adaptive prototype from confident solo crops, offline A/B on the per-window cache.** Host laptop-a (team voice data), size M. Build on `eval/ema_crop_buffer.py` and `eval/dynamic_ema_tracker.py`; update the target centroid only on windows with margin > hold and one active stream; kill rule: any row where wrong-speaker rises > 1 pp.
  Acceptance: `eval.wrong_speaker_audit` arm on the 4 real-pair rows (T-284/T-326 fixtures) reports wrong-speaker %, gated windows and target-down seconds vs frozen profile; result row in docs/BENCHMARKS.md; knob default off.
- **N2. Paired-stats utility for pick A/Bs.** Host laptop-c muse-c (no team data needed), size S. Port muse-c's `analyze_results.py` into `eval/paired_stats.py`: mean+/-SD over seeds and McNemar on two per-window outcome files from the T-323 cache format.
  Acceptance: unit test on a synthetic pair; T-370/T-371/T-375 reports cite its p-value and SD; `swarm tell` those tasks to use it.
- **N3. Gate-state pill on the judge page and /app.** Host laptop-a Claude only, size S, depends on T-370. States SOLO / OVERLAP-DUCKED / UNSURE from fields already in events.jsonl (gated, soft_gated, confidence); `tools/timeline.py` prints the same states.
  Acceptance: pill changes live in a session recording with no extra WebSocket traffic; screenshot in the PR; nothing prerecorded.
- **N4. Falsify the "quiet target" headline on our data.** Host laptop-a, size S. From the T-323 per-window cache, for every wrong-speaker window compute target-vs-other stream level ratio and whether the target was active.
  Acceptance: one table (rows: joseph/aryan x easy/hard) of wrong windows binned by level ratio; a one-line conclusion whether loudness explains the failures; posted as a relay to muse-c (T-358 thread) so the spatial line stops or continues on evidence.
- **N5. (tell, not a task) T-377:** add level strata (-6/0/+6 dB), 3 seeds and SD reporting (A13), so the public set can score T-370/T-371/T-375 with the N2 utility.

## 4. Wrong or overstated claims, and open questions
Overstated:
- "4 of 5 tracks independently converged on attack-the-loud-interferer" (Q-1047): the mission brief (Q-1035) itself framed the problem as "quieter target vs louder competitors", so the convergence is not independent. Our measured failure (T-141, Q-399) is equal-level overlap of two male voices; no measurement says the target is quieter.
- All spatial ideas (A4/A5) assume a synchronized mic pair; the demo input is one mono stream. "Half-day oracle gives near-certain go/no-go" is not available to us.
- "Estimate and subtract the loud interferer" adds nothing to a 2-stream separator whose second stream is already the residual (muse-b's own Q-1037 point).
- P4 (Q-1044) is "MEASURED" only on a simulated separator, and the number it reports is a 2.2 dB SI-SDR loss.
- Q-1061's answer ("GPU caps lifted, run any iLab job") is contradicted by Joseph in Q-1086; treat Joseph's freeze as current.
- Q-1043: T-331 is not "unverifiable"; merge 9b8c3f9f is on origin/main (muse-b's clone is stale).
- Q-1059 (muse-c does not exist) was wrong and is already corrected (Q-1063/Q-1076).
- AFW (2609.20463) and ReDimNet2+ (2609.37014) numbers are quoted from abstracts, unverified here.

Open questions still needing an orchestrator answer:
- **Q-1042 (P3):** "Covered by the Q-1066 answer: hysteresis + UNKNOWN as an env-flagged wrapper on T-370's gate hook; Follow pill after judging; calibration sweep on CPU on laptop-a (team data lives there); no Arena batch before Sunday; UX: an UNSURE pill is filed (N3), nothing else changes the judge page."
- **Q-1043 (P5):** "Scheduler is post-hackathon. 9b8c3f9f is on origin/main (pull). No canonical evaluation_manifest.json; define one after judging. No auto-stop or promotion authority before Sunday."
- **Q-1044 (P4):** "Research-only until measured on real separator output (Q-1067). Rendered outputs: results/sessions/<stamp>/out_pre_agc.wav on laptop-a; masks are not exported. Arena = T-255; no listener batch before judging."
- **Q-1060 (R48):** "(a) yes, the time-overlap join with a fail-loud gate; (b)(c) queue after Sunday and after Joseph lifts the GPU freeze; Sol signs off the research line."
- **Q-1086:** acknowledge: "No GPU dispatch on Joseph's iLab access until Joseph says so; the Q-1061 answer is withdrawn for laptop-b. T-374 waits; Sunday work is Mac/CPU."
- **Q-1085 relay (muse-c asks T-358 'pre-nulled mixture or raw mics?'):** "Raw mono mic; there is no array. Neural stage stays single-channel."
- Q-1036, Q-1064, Q-1065, Q-1082-1085 (muse-b): no question asked; close as "logged in MUSE_RESEARCH_DIGEST_oct10.md". Q-1022/Q-1041/Q-1062 are muse-to-muse; leave to them.
