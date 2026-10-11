# T-435: combination and mechanism audit

Research brief from the checked-out [ranked report](ranked-approaches.md). All 63 entries were reviewed. This is a recommendation and experiment plan, not a trained system or a phone benchmark. No runtime defaults, checkpoints or interfaces change here. T-432, T-434 and T-408 outputs were unavailable and are not prerequisites for this brief.

“New” means a mechanism newly actionable relative to the inspected implementation, rather than a claim about personal memory or world-first novelty. Evidence anchors: `hearing/tse/enrolled.py` already caches pooled ECAPA embeddings, contrasts known negative profiles, tracks streams and gates output; `hearing/tse/blind.py` separates first; `hearing/tse/windowed.py` explicitly delays offline extraction by a complete window. `experiments/gpu/T-387/train_sep.py:slot_losses/objective` already averages per-slot dB losses, floors active-reference error and handles silent slots. It does not condition that separator on enrollment. Thus equal source weighting, negative-profile comparison, cached enrollment and silent-slot suppression are existing foundations, not discoveries. “New candidate” below means no equivalent found in those inspected paths, not proof of absence throughout the repo.

## Chosen combination

Keep **2 + 5 + 4 + 1** as the research default: small causal neural waveform extractor; cached enrollment-frame features attended to by current/past mixture frames; enrollment noise/reverb augmentation with controlled target-quieter SIR strata; independent output identity verifier with veto and bounded recovery. The full enrollment can be read at initialization without violating mixture causality. Bound its frame bank and cache projections so enrollment length does not create unbounded speech-time cost. A pooled ECAPA vector remains valid for independent verification, but is not the sole separator cue. Add training mechanisms one at a time, rather than enabling the report's whole loss stack at once.

Two qualifications to the proposed stack:

- **Approach 3 is a conditional fallback.** [VoiceFilter-Lite](https://arxiv.org/abs/2009.04323) supplies streaming single-channel ASR separation, asymmetric training and INT8 precedent. An ASR feature masker does not by itself provide the reconstructed 16 kHz waveform Focus needs. Keep its lightweight topology as a challenger only after a causal reconstruction path passes listening and identity tests; it is not an available drop-in stage.
- **Approach 1 needs a timing contract.** A trailing 1–2 s verifier can veto updates and future playback, but cannot revoke wrong audio already heard. Buffer-and-verify adds that evidence interval to delay; asynchronous verification leaves exposure before the next decision. Start with frozen enrollment and no adaptation, measure exposure/reacquisition, and reject any claim of zero wrong-speaker delivery. Retry uses a separate saved state or second instance, one attempt maximum, and cannot run silently on the critical frame path. A failed retry remains uncertain/muted; Raw/Open remains an explicit user choice.

Approach 8 stays optional. Test both ideal interferer removal (upper bound) and imperfect causal masking before training a second engine. An oracle win justifies testing an estimator; it does not establish deployable recovery. No GPU jobs are launched: this worker is Ahmad/Linux, not muse-b, and the GPU-job sentence is conditional and unverified. No scope/resource change is inferred from that relay.

## Top three newly actionable findings

### 1. Nontarget trials prevent identity laundering (#13)

A positive-only enhancer can make another voice resemble the enrollment cue. Train an enrolled extractor on mixtures where the enrolled person is absent, with an explicit suppression objective; do not merely train a selector to reject afterward. [TASE-SVNet](https://arxiv.org/abs/2103.08781) supports nontarget sampling for robust speaker verification; transfer to natural waveform TSE is our hypothesis, not an established 0 dB phone result. This **strengthens training (#4) and the bouncer (#1)**; it replaces neither.

**Cheapest decisive test:** first score the frozen model on enrollment-swapped and target-absent mixtures. Then compare equal-budget positive-only and positive-plus-absent fine-tunes from the same checkpoint, with the same active loss and a separate energy loss for silent references. Use held-out speakers/sessions, not another crop from the training memo. Reject if fewer absent false accepts come primarily from muting the target: track active-target deletion, wrong-speaker delivered duration and target-only quality jointly. A fixed-slot separator cannot test enrollment laundering directly; it only supplies a silent-slot baseline. Implement the enrollment-conditioned arm before claiming this mechanism was tested.

### 2. Clean the guidance, preserve the raw reconstruction path (#6)

Denoise mixture features used to compute enrollment interaction, while the extractor retains the original mixture. This **strengthens frame conditioning (#5)** without imposing denoiser waveform distortion on every output. [D-LGTSE sections 2.2–2.4](https://arxiv.org/html/2508.19583) describe this interaction plus denoised-input training augmentation; their final recipe jointly fine-tunes the denoiser. Our frozen-denoiser pilot is a cheaper ablation, not a reproduction of that recipe.

**Cheapest decisive test:** on the same small causal frame-conditioned model, compare raw guidance/raw reconstruction, denoised guidance/raw reconstruction, and denoised waveform cascade. Freeze the denoiser and equalize updates/data. Test café noise and speech-only overlap separately at 0 and −5 dB; use −15 dB as a quiet-target destruction probe. Reject if the denoised-guidance arm loses identity or target words, even when noise quality improves. Attention entropy alone is not a success criterion. Time the additional denoiser on the actual phone only after this accuracy gate passes.

### 3. Train on local confusion rather than whole-utterance averages (#16/#45)

Short wrong-speaker bursts can hide inside a good utterance score. [X-SepFormer sections 3–4](https://arxiv.org/html/2303.05023) defines local improvement statistics and confusion-sensitive training; its 250 ms chunks overlap by 125 ms during training and do not overlap in the reported inference evaluation. This **strengthens training (#4)** at zero additional inference work.

**Cheapest decisive test:** first compute local improvement on existing saved outputs; if utterance success coexists with frequent local failures, compare baseline and a chunk-statistic-weighted fine-tune using identical seeds, LR, steps and data. Preserve a differentiable reconstruction objective: literal hard histogram counts alone are not a usable autograd loss; audit the authors' implementation before reproducing the weighted variant. Require improvement in both local failure and independently scored wrong-speaker duration, without higher deletion. Report paper-compatible nonoverlap and an explicitly labelled sliding diagnostic separately. This is cheaper and less risky than replacing the separator architecture.

## Common falsification protocol

Use a fixed manifest with held-out speaker/session pairs, both target orders, separate enrollment utterances and SIR bins 0, −5, +5 dB; add −15 dB as exploratory. Include target-only, target-absent, wrong-enrollment, same-sex similar voices, contaminated enrollment, onset/offset and changing-level cases. Equal loudness at the intended SIR must be documented with the mixing convention; log realized per-active-region SIR after reverb, rather than treating nominal gain as truth. With real mixtures lacking stems, do not invent oracle SIR or SI-SDR.

Each paired ablation fixes training budget, seed and validation thresholds. Recheck promising results across multiple seeds. Bootstrap by independent speaker/session pair, not by correlated windows. A pilot decides whether to continue; a claimed general benefit requires a held-out paired confidence interval favoring the change and no material regression in target deletion or absent false acceptance. Practical tolerances must be set before the run, not chosen after seeing results.

Report median SI-SDR improvement and fraction of nonpositive-improvement utterances (an extraction-failure proxy), local failure rate, true wrong-speaker duration from stems/independent adjudication, target-absent output activity, deletion/TSOS, intelligibility and listening. Nonpositive SI-SDR improvement does not uniquely identify a wrong speaker; silence/distortion also fail. Exclude silent-reference windows from SI-SDR, score them with leakage/activity, and expose deleted-target duration separately so activity filtering cannot hide muting.

For causal streaming acceptance, test suffix perturbation (future mixture changes must not alter earlier emitted samples), reset/target switch, chunk-size equivalence and bounded state. Run paced 160-sample hops on the specified phone CPU with the intended thread count and build; report p50/p99, 10 ms deadline misses, end-to-end algorithmic delay including STFT and bouncer evidence, plus RTF/memory. A 10 ms hop is not a 10 ms latency promise. Wall-time runs require `swarm-lock --exclusive`, contemporaneous `uptime` and a same-run baseline; accuracy-only heavy training/evals use the shared lock. Linux desktop timings cannot satisfy phone acceptance.

## Coverage of all 63 ranked entries

**E** = existing foundation in inspected paths; **N** = newly actionable candidate or discipline; **P** = newly considered but parked/excluded. Layer references: engine 2/3, conditioning 5, training 4, bouncer 1. Except for the five primary sources linked above, these are mechanisms transcribed from the ranked report, not independently verified paper claims. No survey performance numbers or unverified publication status are promoted here. Each test is proposed, not executed.

| Rank | Status and mechanism | Effect on combination | Cheapest test / stop rule |
|---|---|---|---|
| 1 | E/N: identity comparison exists; residual check and bounded subtract-retry are additional | Strengthens bouncer | Saved-output swap probes: check whether residual correlation adds discrimination beyond identity; reject if pure wrong voice passes; test state-safe single retry |
| 2 | N: direct causal enrollment-conditioned extraction | Replaces blind separation as research engine | Same-size blind+select versus conditioned model; future-suffix test and held-out 0 dB identity |
| 3 | N: asymmetric filterbank masker | Conditional engine fallback | Demonstrate causal waveform reconstruction and identity before phone timing |
| 4 | E/N: level strata exist; enrollment corruption/consistency are additions | Strengthens training | Noise/RIR cue ablation with clean and mismatched enrollment; reject clean regression |
| 5 | N: cache frames, attend from mixture to enrollment | Replaces pooled-only conditioning | Fixed bank-size frame versus pooled ablation, enrollment swaps, 0/−5 dB |
| 6 | N: denoised guidance with original mixture path | Strengthens conditioning | Top-three test 2 |
| 7 | E/N: accuracy/timing metrics exist; stratified confusion and paced deadlines add discipline | Strengthens all acceptance gates | Re-score saved outputs locally; phone paced audit before deployment claim |
| 8 | N: explicit undesired mask/residue extraction | Optional engine/recovery substitution | Oracle residue then imperfect masker versus direct; stop if even oracle adds no benefit |
| 9 | N: identity-admitted embedding refinement | Strengthens conditioning/bouncer | Frozen versus gated versus ungated update replay; reject drift/false admissions |
| 10 | E/N: negative profile comparison exists; Siamese conditioning is additional | Strengthens conditioning | Contaminated positive cue with/without matched negative cue; test negative containing target |
| 11 | N: extract then dereverb, explicit early-reflection target | Strengthens engine target discipline | Fixed RIR stems, order and target-definition ablation; check identity and delay |
| 12 | N: three-way personal VAD soft prior | Strengthens engine compute and bouncer | Offline posterior/oracle activity comparison; count deleted onsets before using compute skip |
| 13 | N: disentangle identity/prosody; absent-enrollment suppression | Strengthens conditioning/training | Top-three test 1; separately vary prosody without changing identity |
| 14 | P: identity-filtered real-data pseudo-labels | Strengthens training after demo | Audit teacher admissions before retraining; stop if coverage removes quiet-target strata |
| 15 | N: re-encoded cache plus enrollment-seeded bank | Optional engine memory replacement | Long replay, frozen/gated/blind bank; contamination burst and rollback |
| 16 | E/N: source dB averaging/floor exist; consistency and local losses add supervision | Strengthens training | Verify existing loss algebra first; one addition per equal-budget ablation |
| 17 | N: enrollment-seeded attractor with rollback | Strengthens conditioning memory | Seed target versus dominant-stream seed on onset inversion; reject persistent wrong admission |
| 18 | E/N: soft attenuation exists; clean-target bypass needs validation | Strengthens output guardrails | Target-only and absent-target battery; count deletion and leak together |
| 19 | E: blind separation then identity selection | Engine baseline | Frozen checkpoint routing versus oracle best stream; separates routing from separation failure |
| 20 | P: detect unreliable segment, mask, enrollment-guided repair | Optional recovery replacement | Oracle mark bad segments first; reject identity invention or added delay |
| 21 | N: coarse deletion then limited refinement | Optional engine decomposition | One-stage versus two-stage at same runtime budget; inspect identity after each handoff |
| 22 | E/N: PCM/cached enrollment exist; asymmetric encoder and mixed quantization add deployment choices | Strengthens conditioning/bouncer economics | Quantize interiors versus I/O separately; recalibrate identity on phone enrollment |
| 23 | N: curriculum admission separately within SIR strata | Strengthens training | Static coverage versus scheduled coverage at equal steps; confirm quiet strata actually admitted |
| 24 | N: augmented-enrollment averaging/keyframe fusion | Strengthens conditioning | Enrollment-only augmentation first; reject mixture-mined frames that drift to interferer |
| 25 | N: augmentation-adversarial identity invariance | Strengthens conditioning/bouncer | Independent-session probe and speaker probe; stop if invariance destroys speaker separation |
| 26 | P: meta-trained delta-rule fast weights | Replaces heuristic identity updater | Closed-loop mismatch/contamination replay; no claim from open-loop embedding scores |
| 27 | N: fast audio engine, slower bounded identity refresh | Strengthens runtime cadence | Disable/enable refresh on long replay; burst exposure and stale identity timing |
| 28 | N: cached prompt bank derived from enrollment onset prompt | Alternative conditioning | Offline prompt is an upper bound only; causal cached-bank challenger must pass suffix test |
| 29 | P: speaker-derived learnable prompt slots | Alternative conditioning | Train adapter with correct/wrong/no prompts; stop if cue does not change speaker selection |
| 30 | P: text/style/dialogue as secondary cues | Optional conditioning support | Same-style different-speaker counterexample; kill any identity override |
| 31 | P: foreground activity as degraded gate | Activity support only | Loud interferer/quiet target inversion; kill prominence-based identity selection |
| 32 | N: fixed-state S4D; selective Mamba challenger | Alternative engine recurrence | Same conditioning and budget; state growth and paced phone benchmark |
| 33 | N: skip-memory recurrence, predictive memory training | Alternative engine skeleton | Long stream versus reset segments; auxiliary prediction on/off |
| 34 | N: causal dilated encoder plus shallow decoder | Alternative engine | Matched-capacity conv-only comparison and actual phone layout/attention cost |
| 35 | P: enrollment anchors top-down multiscale modulation | Alternative conditioning/engine | Running-pool causal prototype versus global pool; stop future leakage |
| 36 | N: pure-conv multiscale blocks with group compression | Alternative engine | Causal 16 kHz conditioned prototype at matched quality; phone timing |
| 37 | N: bounded/separable attention to avoid layout overhead | Strengthens engine/conditioning economy | Compare no attention, bounded attention and separable arm on same runtime/build |
| 38 | P: close-talk projected to phone for training pseudo-targets | Strengthens real-data training | Synchronized pilot and cross-talk audit; extra training mic does not change inference device |
| 39 | N: dynamic loudness/RIR strata and natural effort splits | Strengthens training/evaluation | Fixed versus changing gains; log local SIR and separate natural from gain-scaled voices |
| 40 | P: staged teacher→causal student→QAT/KD | Replaces heavy engine with student | One student pilot versus supervised student; wrong-teacher admission audit |
| 41 | P: distill conditioning interaction with identity admission | Strengthens conditioning/training | Correct/wrong teacher cue counterfactual; reject imitation of unavailable oracle input |
| 42 | P: training-only noncausal interferer branch | Strengthens engine learning without runtime branch | Remove branch at evaluation; verify target output never reads its features |
| 43 | E/N: hard PIT exists; soft assignment/recursion are alternatives | Changes training assignment | Soft versus hard only for unlabelled slots; fixed target output must retain identity supervision |
| 44 | N: asymmetric deletion coefficient by regime | Strengthens training | Small coefficient grid, per-bin deletion/leak; no single loud-target coefficient copied blindly |
| 45 | N: local confusion-dependent weighting | Strengthens training | Top-three test 3 |
| 46 | N: reference-free separation confidence | Strengthens bouncer diagnostics | Include pristine wrong-speaker output; kill use as identity substitute |
| 47 | E/N: stem replays exist; real-phone adjudication extends them | Strengthens acceptance protocol | Blinded real-room clip audit; compare simulated ranking with human identity judgment |
| 48 | N: norm/cache/lookahead audit and frame budgeting | Strengthens engine causality | Future-suffix/chunk equivalence tests then paced deadline test |
| 49 | P: voice-converted target-adjacent interferers | Strengthens training hard negatives | Real-only held-out hard pairs; reject gains confined to conversion artifacts |
| 50 | P: delta/acceleration multi-resolution continuity loss | Strengthens training deletion guard | One loss addition; compare word deletion and wrong identity, not SI-SDR alone |
| 51 | E/N: silent-slot objective exists; activity-aware supervision is new | Strengthens training | Oracle activity versus predicted activity weighting; retain separate absent suppression |
| 52 | P: noisy-target noise discount/orthogonality | Changes training target objective | Known clean/noisy target pair; apply only if training references are actually noisy |
| 53 | P: inverse-energy quiet-channel weighting | Changes training aggregation | Compare current equal dB averaging first; silence-floor/gradient audit before train |
| 54 | P: KD plus ASR auxiliary to reduce deletion | Strengthens training | Equal-budget KD-only versus KD+ASR; monitor insertions and identity |
| 55 | E/N: cached rich enrollment exists; attentive capture filtering extends it | Strengthens conditioning/bouncer | Reject contaminated capture; preserve independent-session identity, park TTS repair |
| 56 | P: distill small enrollment/runtime encoder | Replaces expensive verifier after calibration | Float anchor versus small encoder on swapped/contaminated cues; no compression headline claim |
| 57 | P: frozen causal codec latent masker | Alternative engine, post-demo | Codec reconstruction identity ceiling first; reject extra delay/voice drift |
| 58 | P: one/two-step generative correction | Optional output refinement, post-demo | Feed deliberately wrong extraction; never admit a prettier wrong voice |
| 59 | P: one-step flow target extraction | Engine replacement, post-demo | Capacity/runtime feasibility screen before any training |
| 60 | P: virtual/spatial supervision, mono deployment | Optional training supervision | Verify inference graph needs exactly one mic; compare matched mono-trained baseline |
| 61 | P: dual-mic beamforming/demixing | Kills these inference layers under one-mic constraint | Input-dependency audit; exclude any second-channel runtime requirement |
| 62 | P: visual/lip cues | Kills visual inference layers under audio-only constraint | Dependency audit; only audio-derived confidence/repair transfers may compete |
| 63 | P: foreground prominence used as selector | Kills identity substitution | Equal-loudness and quieter-target counterexamples |

## Handoff and evidence limits

Run the saved-output diagnostic before scheduling architecture work. Then pursue nontarget conditioning, local confusion loss and denoised-guidance ablations in that order when a causal enrollment-conditioned baseline exists. The top-three mechanisms do not license claiming that the current blind/fixed-slot checkpoint implements the combination.

For T-408: its floor sweep is a playback tradeoff, not an identity fix. Report wrong-speaker exposure, target muting and target-absent leakage together; a quieter wrong stream is still wrong. This brief does not modify its checkpoint, defaults or sweep.

The ranked report labels entry 7's 250/125 ms grid without distinguishing training from reported inference; the primary X-SepFormer text does distinguish them. Candidate B's pooled-only FiLM sketch also differs from the requested frame-level combination. Use those as comparator arms, not as evidence that the specified combination has already shipped. Some report sources are dated after this worker's environment date; unverified later entries remain survey hypotheses until primary-source checks.

Latency, SI-SDRi, WER, RTF and phone device results: **not measured (research-only task; no model executed)**. No models or dependencies introduced; consequently no model licence row or benchmark row is added. Future weight/code adoption requires its own verified code and weight licences in `docs/MODELS.md`.
