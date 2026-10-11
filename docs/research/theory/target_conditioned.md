# T-251 · Target-conditioned extraction vs blind two-stream separation + ECAPA pick

Theory/prep only. Nothing trained, downloaded or run on audio; no GPU; no runtime, config or shared doc touched. Numbers marked **computed** come from `experiments/theory/T-251/prep.py` (random weights, CPU); numbers marked **assumed** are planning ranges, not measurements. Primary-source facts about WeSep/REAL-TSE are reused from `docs/research/CONDITIONED_TSE_FEASIBILITY.md` (T-203) and not re-derived. Other literature named here (SpeakerBeam, VoiceFilter, SpEx+, TD-SpeakerBeam, X-TF-GridNet-class models) is cited from memory of the papers, **not re-opened in this task**: check before quoting.

## 1. Baseline and why it can fail

Today (`hearing/tse/enrolled.py`): MossFormer2_SS_16K (`blind.py`) emits exactly **two** streams per window; the ECAPA (speechbrain `spkrec-ecapa-voxceleb`) cosine to the enrollment picks one, with margin/pause/contrast guards. Structural limits:

1. **Pick errors are all-or-nothing.** A wrong pick plays the other person for a whole window (the 6–25 % wrong-speaker time in PLAN.md).
2. **Three speakers.** A 2-output separator must put the third voice into one or both streams. The pick then chooses a stream that is target + leak, and the ECAPA cosine of a mixed stream is lowered and noisier. Conditioning cannot fix a wrong separation; a conditioned extractor outputs "target only" for any N, so it is N-agnostic by construction.
3. **Unseen speakers.** ECAPA is a fixed verification embedding (trained on VoxCeleb); it is strong for unseen voices but degrades for short, noisy enrollment and similar same-sex voices. A *trained* conditioned extractor adds a second generalisation gap: it saw N training speakers, and an unseen target's embedding can land off-manifold (common failure in the speaker-conditioned literature: extractor leaks the interferer or suppresses the target when the cue is weak). The pick has no such gap; it only loses the ability to repair a bad separation.
4. **Delay.** Separation windows (`SlidingWindowExtractor`-style) are windowed, not frame-causal. A frame-causal conditioned model avoids the window but must be trained causal and typically loses quality versus its offline twin (T-203: REAL-TSE reports causal and non-causal baselines separately; their numbers were not reproduced).

## 2. Interfaces: what permits, what needs extension

| Interface | File | Permits today | Needs for conditioned extraction |
|---|---|---|---|
| `TSECandidate.extract(mix, sr, *, lips, lips_fps, p_speaking, enroll)` | `hearing/tse/base.py` | `enroll` is a raw waveform; a conditioned model embeds it inside `load()/extract()`. Registering a new candidate needs no core change (`register_candidate`, `needs={"enroll"}`). | Nothing for a windowed offline candidate. |
| `Stage` streaming protocol | `docs/CONTRACTS.md` | 10 ms hops, `latency_ms`, `reset()`. | A frame-causal model must carry state (conv caches) across `process()` calls; windowed candidates already go through `windowed.py`. |
| Enrollment/profile cache | `hearing/tse/enroll.py`, `enrolled.py` (`CACHE_ENTRIES`, profile version) | Caches ECAPA embeddings per clip/profile. | A **different embedding space** (e.g. WeSpeaker, or learned) is not interchangeable with stored profiles: needs a profile-version bump and re-enrolment. Projection-only conditioning on the *existing* ECAPA vector reuses profiles. |
| Negative enrollment (`other`) | `enrolled.py` | Used as contrast. | A conditioned model could take the negative as a second cue (SpeakerBeam-style "interferer" conditioning): extension of the candidate kwargs only. |
| Blind separator | `blind.py` `BlindSeparator.separate` | Returns 2 streams. | Adapter variant needs the streams (already exposed) — no change. |

Verdict: a **windowed offline candidate** and the **StreamGate adapter** need zero core edits; a **frame-causal stateful model** and **non-ECAPA embedding** need a stage wrapper and a profile-version bump.

## 3. Mechanisms compared (prototype: `experiments/theory/T-251/prep.py`)

One tiny causal dilated-conv mask estimator (6 blocks, 64 ch, 2 ms encoder window) with the cue entering four ways. **Computed** parameter counts (fp32 weights, seed 251):

| Mode | Params | fp32 MB | How the 192-d enrollment embedding enters | Notes |
|---|---:|---:|---|---|
| none (blind control) | 60 551 | 0.23 | — | what the ECAPA pick selects among, per stream |
| project | 72 903 | 0.28 | `Linear` bias added once at encoder output | cheapest; reuses ECAPA profiles; a one-shot additive cue that deeper blocks may forget |
| film | 208 775 | 0.80 | per-block γ,β from the embedding | standard, cheap; cue re-injected at every depth |
| xattn | 234 503 | 0.90 | each block attends over K enrollment frame features | can use a *sequence* (more speaker detail, robust to short/odd clips); needs enrollment frames, not a vector; most data-hungry |

Also sketched: **StreamGate** (163 params for 2 streams, 228 for 3 — computed): a tiny MLP over per-stream cosines (+ negative cosines + previous gate) producing stream logits and a "target absent" logit. It generalises today's argmax+margin+pause rules into a learned rule and is the lightest "adapter"; it cannot repair separation quality.

All four prototype variants pass a numerical causality check (change only samples ≥ cut; outputs before cut−64 unchanged, max leak 0.0, **computed**). Design rule recorded: no time-global normalisation (GroupNorm/global LN would leak the future); use cumulative LN or none. Receptive field **128 ms**, algorithmic latency **2 ms** (one encoder window) — these are properties of the sketch, not of a trained model.

Recommendation by evidence quality: (1) keep the ECAPA pick as default; (2) the first thing worth building is E0 (oracle-pick ceiling, no training); (3) if the pick is the bottleneck, StreamGate; (4) only if separation itself is the bottleneck (E1) consider a conditioned extractor, FiLM first, projection-only as the control, cross-attention only if E2 shows short/odd enrollment is the failure.

## 4. Causal vs offline

Causal extraction removes window latency but forfeits future context to resolve who is talking at onsets; offline/long-window models resolve identity better. The repo's windowed path already has look-ahead equal to the window length, so the fair comparison is *same latency budget*: causal model at 2 ms + context vs window model at window length. E4 measures this; claims about the size of the gap are **not** made here.

## 5. Decision experiments (exact; emitted in `manifest.json`)

| ID | Question | Method | Stop/decision rule | GPU? |
|---|---|---|---|---|
| E0 | Is the pick the bottleneck? | Oracle stream choice (higher SI-SDR to clean target) vs shipped ECAPA pick on existing 2-speaker fixtures | gap < 1 dB SI-SDRi or < 3 pts wrong-speaker ⇒ stop, nothing to build | no |
| E1 | Third-speaker leak | Target + 2 interferers at 0/−3/−6 dB; best-stream SI-SDR, interferer share in picked stream | best-stream drop > 4 dB vs 2-speaker ⇒ conditioned extractor justified | yes |
| E2 | Unseen speaker / short enrollment | Enrollment 3/6/10 s, similar same-sex pairs, enrollment in a different room; held-out speakers | adopt only if ≥ 2 pts wrong-speaker better with CI excluding 0 | yes |
| E3 | Projection vs FiLM vs xattn | Train the four prototype variants on identical simulated mixes/seeds | cheapest within 0.5 dB of best; xattn only if ≥ 1 dB better on unseen speakers | yes |
| E4 | Causal vs look-ahead | Causal vs 20/40/100 ms look-ahead, same model | accept look-ahead only inside the ARCHITECTURE latency budget | yes |

Final heldout source groups are not touched by any of these; tuning uses development splits only. GPU execution belongs to the GPU owner, within the global 300-case cap with a 12-case smoke first (this task schedules none).

## 6. Assumed data/VRAM/train ranges (assumptions, not measurements)

| | StreamGate | small conditioned TCN (this sketch scaled) | BSRNN/MossFormer-class conditioned |
|---|---|---|---|
| Params | < 0.01 M | 0.5–3 M | 10–60 M |
| Training mixes | 5–20 h (reuse separator outputs) | 200–500 h simulated 2–3-speaker | 500–2000 h (REAL-TSE baselines: Libri2Mix-100 clean, per T-203) |
| Distinct speakers | 100+ | 1 000+ | 1 000–5 000 |
| Train VRAM | < 2 GB (CPU ok) | 8–16 GB | 24–48 GB |
| GPU-days | < 0.1 | 1–4 | 4–14 |
| Inference extra VRAM | ~0 | < 0.5 GB | 1–3 GB |

Real-room data is the larger risk than any of these: simulated-clean training did not transfer to our laptop mic in the related evidence cited in T-203, and nothing here changes that.

## 7. Impact, migration, rollback

- **Impact if adopted:** a conditioned path mainly helps E1 (three speakers). It cannot help if E0 says the separator is the ceiling or the pick is already near the oracle.
- **Migration:** (a) register a new `TSECandidate` (e.g. `conditioned_film`) beside `enrolled`; (b) benchmark with the existing bench harness; (c) only after a win, add a tier/preset entry; (d) a non-ECAPA cue bumps the profile version and requires re-enrolment.
- **Rollback:** the default stays `enrolled`; the candidate is opt-in by name. Rolling back = unregister/ignore; profiles stay valid when the cue is the existing ECAPA vector. No schema, wire or contract change is proposed.

## 8. Replay

```
.venv/bin/python experiments/theory/T-251/prep.py sketch   --out $HEARING_RESULTS_DIR/overnight_prep/T-251
.venv/bin/python experiments/theory/T-251/prep.py manifest --out $HEARING_RESULTS_DIR/overnight_prep/T-251
.venv/bin/python -m pytest tests/test_theory_target_conditioned.py -q
```
Seed 251 (default, `--seed`). `manifest.json` records the config SHA-256, planned counts (smoke 12 / cap 300) and **available** input counts separately (a capped `*.wav` file count under `$HEARING_FIXTURES_DIR`, never read; cap 500). Both commands write atomically and are deterministic, so a rerun overwrites with identical content; no private audio is read and no result is committed.

## 9. Limits

Not measured: any quality, latency or VRAM number of a trained model. Literature other than T-203's cited facts was not re-read. The prototype's receptive field and parameter counts describe the sketch only.
