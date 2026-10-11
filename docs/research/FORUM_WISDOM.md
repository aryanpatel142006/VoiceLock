# Forum wisdom for muse-c: separation and speaker identity

T-304 · researched 2026-10-10 · 30 actionable tips. This is a research backlog,
not a change to the demo configuration or evidence that these techniques work here.

Our constraints: one mono 16 kHz stream, 10 ms pipeline hops, saved-person
selection, café-noise reduction first, delayed Focus HQ, and an honest Raw/Open
return. Diarization labels *who spoke when*; it does not itself produce an isolated
waveform or identify a saved person.

**Relevance:** 5 = directly useful to the current demo; 4 = useful bounded
experiment; 3 = useful diagnostics or later captions; 2 = needs different capture
hardware; 1 = distant roadmap. Relevance is not evidence strength.

**Evidence:** Report = first-person success/failure, not independently reproduced;
Advice = a concrete community recommendation; Maintainer = project-maintainer
statement. Each row separates the source observation from **our proposed action**.
Forum numbers retain their original conditions and must not become our defaults.

## Separation

| ID | Source observation and actionable use | Relevance | Source |
|---|---|---:|---|
| S1 | **Advice:** In the January 2022 thread, a reply corrects the ICA suggestion: conventional ICA generally needs as many observations as sources. **Action:** Reject “run ICA on the phone mono channel” as the two-voice rescue plan; retain the learned separator plus profile matching. | 5 | [r/MachineLearning: Speech Separation Algorithm](https://www.reddit.com/r/MachineLearning/comments/s71bli) |
| S2 | **Advice:** Laurent Duval explains that PCA can yield voice sum/difference components instead of individual voices. **Action:** Do not replace the separator with PCA; reserve whitening for an explicitly tested downstream method. | 4 | [DSP StackExchange: PCA and the cocktail party problem, 2021 answer](https://dsp.stackexchange.com/questions/72296/can-principal-component-analysis-pca-solve-the-cocktail-party-problem) |
| S3 | **Advice:** The DSP reply describes Wiener filtering *after* estimating source power spectra, applied to the mixture STFT. **Action:** For a replay-only experiment, derive a soft mask from estimated target/interferer spectra and compare it against the unfiltered selected stem; a Wiener filter alone supplies no identity evidence. | 4 | [r/DSP: DSP project, May 2021 reply](https://www.reddit.com/r/DSP/comments/nk52nm) |
| S4 | **Advice:** The one-microphone answer uses framewise voice/noise spectrum estimates for Wiener soft masking. **Action:** Evaluate spectral attenuation on café ambience separately from competing speech; do not score a speech-band filter as two-person separation. | 4 | [DSP StackExchange: human speech noise filter](https://dsp.stackexchange.com/questions/2993/human-speech-noise-filter) |
| S5 | **Advice/report:** A recording practitioner warns that even shotgun microphones may not reject the other speaker sufficiently and that ASR may still recognize the quieter voice. **Action:** Test non-target transcript intrusions after acoustic attenuation; a quieter interferer is not automatically absent. | 5 | [r/audio: Affordable Speaker Separation, December 2021 reply](https://www.reddit.com/r/audio/comments/rksfcw) |
| S6 | **Advice:** Keegs suggests delay-and-sum as an initial array beamformer and notes frequency-dependent beam width. **Action:** If a future capture path exposes independent synchronized microphones, compare a directional baseline before neural separation; do not infer array access from the phone having multiple physical mics. | 2 | [DSP StackExchange: PCA thread, December 2020 comments](https://dsp.stackexchange.com/questions/72296/can-principal-component-analysis-pca-solve-the-cocktail-party-problem) |

## Embeddings and short-utterance verification

| ID | Source observation and actionable use | Relevance | Source |
|---|---|---:|---|
| E1 | **Report:** ECAPA outputs changed markedly between native 16 kHz recording and 44.1 kHz downsampling; the reply suggests checking codec/container for lossy capture. **Action:** Log original sample rate and codec for enrollment/query, decode to PCM, and compare resamplers using one identical source recording. The thread does not establish resampling as the root cause. | 5 | [HF ECAPA discussion #15, October–December 2024](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/discussions/15) |
| E2 | **Advice, unvalidated:** One ECAPA reply proposes **0.7–0.75 on a [0,1] scale**, without defining the transformation or validation set. **Action:** Log whether scores are cosine similarity, distance, or transformed scores; sweep our held-out positive/negative pairs instead of copying this number. | 5 | [HF ECAPA discussion #7, April 2025 reply](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/discussions/7) |
| E3 | **Report:** X-vector agglomerative clustering looked good on training samples but failed to generalize because `distance_threshold` was sensitive. **Action:** Select the speaker threshold on different utterances/room takes, then freeze it before evaluating target-silent false accepts. | 5 | [HF Forums: unknown speaker count, Ollie, April 2022](https://discuss.huggingface.co/t/diarization-with-unknown-number-of-speakers/17281) |
| E4 | **Maintainer:** hbredin says SpeechBrain/NeMo wrappers are being deprecated because they are not native to pyannote `Inference`. **Action:** Keep ECAPA behind our own adapter; do not assume `Inference.crop([segments])` works with a SpeechBrain embedding wrapper. | 4 | [pyannote discussion #1956, November 2025](https://github.com/pyannote/pyannote-audio/discussions/1956) |
| E5 | **Advice:** The same discussion batches manually cropped segments as `(N, 1, samples)` in one forward pass. **Action:** Batch offline profile candidates of similar length rather than repeatedly calling inference; keep live windows independent of this offline batching optimization. | 4 | [pyannote discussion #1956, July 2026 reply](https://github.com/pyannote/pyannote-audio/discussions/1956) |
| E6 | **Advice, unvalidated:** That reply recommends duration buckets so short segments are not dominated by padding. **Action:** Bucket short-utterance experiments by voiced duration and compare padded-batch embeddings with unpadded individual inference before relying on score equivalence. Zero-padding is not extra speech evidence. | 5 | [pyannote discussion #1956, July 2026 reply](https://github.com/pyannote/pyannote-audio/discussions/1956) |

No inspected forum establishes a validated ECAPA minimum duration for our room,
microphone, and two speakers. Test short interjections against longer voiced
windows; retain the project's clean enrollment requirement until measured evidence
supports a change. E6 is a batching recommendation, not a solution to insufficient
phonetic content.

## Streaming and latency

| ID | Source observation and actionable use | Relevance | Source |
|---|---|---:|---|
| L1 | **Report:** A Whisper Inference Endpoint accepted whole files rather than an audio stream. **Action:** Check incremental audio input/output behavior at the adapter boundary; submitting short files repeatedly is windowed processing, not proof of stateful streaming. | 4 | [HF Forums: realtime speech-to-text solution, 2023–2024](https://discuss.huggingface.co/t/realtime-speech-to-text-solution/58648) |
| L2 | **Report:** A distilBERT endpoint logged **50 ms inference** but the user's notebook measured **500 ms round trip**. **Action:** Time capture, queue, transport, inference, and playback separately for remote Focus HQ. This is a general serving warning, not an audio benchmark. | 5 | [HF Forums: 50 ms inference, 500 ms latency, February 2024](https://discuss.huggingface.co/t/50-ms-inference-500-ms-latency/75042) |
| L3 | **Report:** A live pipeline user reported better results using Speechmatics `max_delay` to aim below **2 s**, after Whisper interruption/code-switch problems. **Action:** In a future captions adapter, expose endpointing/delay independently from compute settings and A/B interruptions; this delay is unsuitable as a Natural listening target. Exact `max_delay` value and hardware were not supplied. | 3 | [r/speechtech: realtime diarization pipeline, July 2025 reply](https://www.reddit.com/r/speechtech/comments/1kl5ndq/has_anyone_worked_on_a_realtime_speech/) |
| L4 | **Advice, builder-affiliated:** A reply recommends independent transcription with speaker/emotion results joined by audio timestamps. **Action:** Keep optional captions off the audible processing critical path; preserve capture timestamps when joining delayed annotations. | 4 | [r/speechtech: realtime pipeline, September 2026 reply](https://www.reddit.com/r/speechtech/comments/1kl5ndq/has_anyone_worked_on_a_realtime_speech/) |
| L5 | **Advice:** A latency-comparison commenter asks whether numbers represent average/p95, audio-to-word/render, or draft/final. **Action:** Report separate p50/p95 for first partial, final caption, and audible output; name the endpoints rather than one unlabeled “latency.” | 5 | [r/speechtech: comparing STT latencies, March 2026](https://www.reddit.com/r/speechtech/comments/1rubxjg/tool_for_comparing_latencies_across_different_stt/) |
| L6 | **Advice with concrete configuration:** HF users show pyannote `min_duration_off=0.5817029604921046` and segmentation `threshold=0.4442333667381752`; a reply explains `min_duration_off` fills shorter non-speech gaps. **Action:** Inspect instantiated parameters before diagnosing delayed speech release; sweep gap filling separately from detection threshold. These are that user's pipeline values, not our defaults. | 4 | [HF Forums: segmentation parameters, March–September 2023](https://discuss.huggingface.co/t/what-are-min-duration-off-and-threshold-means-segmentation/35006) |

## Phone microphone and capture quality

| ID | Source observation and actionable use | Relevance | Source |
|---|---|---:|---|
| P1 | **Report/advice:** Phone compression flattened a pianist's dynamics; a reply proposes a soft-to-loud single-key test. **Action:** Record a fixed speech-level ramp plus silence on each actual phone/browser path and inspect whether gain lifts pauses or compresses peaks before tuning our AGC. | 5 | [r/audioengineering: disproportional phone noise levels, December 2023](https://www.reddit.com/r/audioengineering/comments/18i3vz4) |
| P2 | **Report:** A practitioner could not bypass voice/noise processing on some Xiaomi devices even with third-party apps selecting raw input. **Action:** Verify capture processing acoustically on the exact device; a requested “raw” setting is not evidence of an unprocessed recording. | 5 | [r/audioengineering: cellphone mics and untreated rooms, April 2020](https://www.reddit.com/r/audioengineering/comments/fusne8) |
| P3 | **Advice:** Put the phone close to the mouth but angle it to avoid blowing directly into the mic. **Action:** Use a fixed off-axis enrollment position and repeat a plosive-rich phrase; reject clipped/wind-burst clips before profile building. | 5 | [r/audioengineering: Is it possible?, April 2023 reply](https://www.reddit.com/r/audioengineering/comments/12j3yvo) |
| P4 | **Advice:** Compare the phone's onboard microphone with its earbud microphone because their tone differs. **Action:** Record the same enrollment script through both candidates and verify the selected input label; do not silently reuse a profile after switching microphones. | 5 | [r/audioengineering: Is it possible?, April 2023 reply](https://www.reddit.com/r/audioengineering/comments/12j3yvo) |
| P5 | **Report/advice:** Cellphone recordings had room reverberation, comb filtering, speaker bleed, and overload; a reply recommends a quiet room/blanket for isolated recording. **Action:** Make consented clean reference stems in a damped space, then retain separate venue takes; do not present clean-room enrollment performance as café performance. | 4 | [r/audioengineering: cellphone recordings, April 2020](https://www.reddit.com/r/audioengineering/comments/fusne8) |
| P6 | **Advice/report:** An outdoor phone-mic user recommends a wind shield and emphasizes proximity even with a cardioid attachment. **Action:** If the venue has fan drafts, compare a shielded mic against the unshielded take before changing denoising strength; do not expect software to repair wind overload. | 4 | [r/audioengineering: Android recording at the beach, May 2023](https://www.reddit.com/r/audioengineering/comments/13ee75f) |

## Demo and evaluation gotchas

| ID | Source observation and actionable use | Relevance | Source |
|---|---|---:|---|
| D1 | **Practitioner statement:** A Rev speech-group commenter separates WER, DER, and word-level diarization error. **Action:** Score intelligible target words and wrong-person attribution separately; ASR correctness cannot certify isolation or saved-person identity. | 5 | [r/speechtech: benchmark diarization separately, June 2026](https://www.reddit.com/r/speechtech/comments/1u5g9hm/diarization_should_be_benchmarked_separately_from/) |
| D2 | **Advice:** The same thread lists late joining, similar voices, overlap, and short interjections as separate tests. **Action:** Add these as distinct replay scenes; report each failure category instead of averaging away the difficult cases. | 5 | [r/speechtech: benchmark diarization separately, June 2026](https://www.reddit.com/r/speechtech/comments/1u5g9hm/diarization_should_be_benchmarked_separately_from/) |
| D3 | **Report:** A court-reporting user says diarization drifts/loses speakers despite vendor claims. **Action:** Rehearse a sustained fixed-profile conversation and inspect late speaker swaps; a clean opening sentence does not certify lock persistence. | 5 | [r/speechtech: October 7, 2026 follow-up](https://www.reddit.com/r/speechtech/comments/1u5g9hm/diarization_should_be_benchmarked_separately_from/) |
| D4 | **Report:** A developer observed role swaps in quiet two-person Italian/French dialogues even allowing **1.5–2 s** latency. **Action:** Include actual demo speakers/languages in held-out testing; increasing the buffer cannot be assumed to solve identity swaps. | 5 | [r/software: bad realtime diarization experience, October 2025](https://www.reddit.com/r/software/comments/1nx08g9) |
| D5 | **Report, author benchmark:** A CPU diarization author explicitly used preloaded models on an **Apple M2 Max**, reporting **RTF 0.12** versus **0.86** for their pyannote comparison on VoxConverse. **Action:** Separate cold-start and warm-run timing and name hardware; faster-than-real-time batch throughput alone says nothing about first audible output latency. Not independently reproduced here. | 4 | [r/speechtech: CPU-only diarization library, March 2026](https://www.reddit.com/r/speechtech/comments/1rit76u/i_built_a_cpuonly_speaker_diarization_library_it/) |
| D6 | **Advice:** Separate per-speaker audio streams avoid having to infer who spoke from a mixture. **Action:** Where feasible, capture separate consented clean scripts as reference stems for digital mixtures; retain mono real-room evaluation so this easier capture does not inflate the live demo claim. | 4 | [r/speechtech: realtime pipeline, April 2026 reply](https://www.reddit.com/r/speechtech/comments/1kl5ndq/has_anyone_worked_on_a_realtime_speech/) |

## Bounded next experiments for muse-c

These are our synthesis of the rows above, not additional sourced success claims:

1. **Capture audit (E1, P1–P4):** same script, exact device/codec/processing labels,
   level ramp and pauses; retain the original PCM and fixed microphone positions.
2. **Identity audit (E2–E3, E6, D1–D4):** duration-bucket positive/hard-negative
   pairs, target-only/interferer-only/overlap/late-join scenes, threshold freeze on
   held-out utterances; report false accepts and reacquisition delay together.
3. **Timing audit (L2, L5, D5):** cold and warm runs; capture-to-playback p50/p95,
   window accumulation and transport shown separately from model compute.
4. **Replay post-filter (S3–S4):** compare the same selected stem with/without soft
   masking, level matched, while checking lost consonants and wrong-speaker leak.

No model, dependency, runtime threshold, or interface is introduced by this task.
No local latency, SI-SDRi, WER, RTF, or listening experiment was performed.

## Search coverage and evidence limits

Searched r/MachineLearning, r/signalprocessing, r/DSP, r/speechtech,
r/audioengineering, DSP StackExchange, and Hugging Face Forums/Hub discussions.
Used the five requested query phrases and variants such as “ECAPA duration,”
“speech separation microphone,” and “diarization unknown speakers.”
r/signalprocessing returned no sufficiently concrete, attributable result in this
pass; it contributes no invented tip. Added one relevant pyannote GitHub discussion
and r/audio/r/software practitioner threads to fill specific gaps.

Reddit evidence above came from search-tool indexed post/comment text; direct page
opens on sampled r/DSP URLs failed with cache misses. HF, StackExchange, and the
GitHub discussion were also inspected through opened pages where available. URLs
identify threads; dates and reply descriptions identify the mined statements.
Indexed excerpts may omit replies, and a link is not a guarantee the thread will
remain accessible. No private communities or recordings were accessed.

Low-vote experiences are hypotheses worth testing, not consensus. The Speechmatics
reports do not supply controlled comparisons, hardware, or an exact delay setting;
the Oruk reply is builder-affiliated. The ECAPA score recommendation has no
validated operating point. Numerical precision in a copied configuration is not
evidence that it transfers. In particular, this search found no battle-tested
forum recipe establishing stable single-mic streaming target extraction under
our target-absent/overlap conditions; do not replace the current measured demo
path on forum enthusiasm alone.
