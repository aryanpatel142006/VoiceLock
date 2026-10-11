# Focus HQ bake-off v2 (T-056)

Replaces the T-053 recovery (which copied T-024's 16-clip, mixed-layout bench rows). Every number below comes from a fresh `eval.bakeoff` JSON on the **GRID two-face** set (`av_full --layout two_face`, 80 clips of 3 s, babble at −5/0/5/10 dB SNR), run on the iLab GPUs and on laptop-a.

## Decision

1. **Tier 1: `enrolled`** (MossFormer2_SS_16K separation + ECAPA-TDNN match against the target's voice), **1 s window / 0.5 s hop on the Mac** (local, MPS), **0.5 s window / 0.25 s hop on the GPU backend**. It plays once the session has 3 s of ASD-gated enrollment (`hearing/tse/enroll.py`: the target's track speaking with p ≥ 0.7, every other track < 0.5, voiced). Enrolled is the only candidate that keeps its quality at short windows: **+11.46 / +10.84 / +10.08 dB** SI-SDRi at 2 / 1 / 0.5 s on the GPU (−1.4 dB from 2 s to 0.5 s). Its compute is 109 ms per window at 0.5 s and 123 ms at 1 s on an RTX A4500, so a 0.25 s hop keeps up (RTF 0.42) and the live delay is 0.75 s plus RTT, close to PLAN's 0.7 s stretch goal. On the Mac (M4 Pro, MPS) the same model costs 236 / 323 ms (p50 / p95) per 1 s window, inside a 0.5 s hop (RTF 0.43), but 150 / 234 ms per 0.5 s window, only 16 ms inside a 0.25 s hop under load: hence 1 s / 0.5 s locally (1.5 s delay). The Mac reproduces the GPU's SI-SDRi to ±0.3 dB on every row.
2. **Tier 1b: `av_mossformer2`** (ClearerVoice AV_MossFormer2_TSE_16K) at **2 s window / 1 s hop**, only for the **cold start**, before enrollment exists. +11.20 dB at 2 s, but **+3.91 dB at 1 s and −0.37 dB at 0.5 s**: it needs 2 s of context and cannot follow enrolled to short windows.
3. **Tier 2: gate** (`dfn_ll>gate>mixer`), then **Tier 3: audio-only** (`dfn_ll`), unchanged.
4. **Rejected:** `dolphin` (negative at every window, worse the longer the window: −4.11 / −7.57 / −10.88 dB at 0.5 / 1 / 2 s; non-commercial checkpoint). `blind_assoc` is not shipped: +8.69 dB at 1 s needs an oracle speaking detector here, and it collapses to +2.12 dB at 2 s for reasons not investigated.

`config/tiers.yaml` implements this ladder (enrolled → focus_hq → gate → audio-only), with the GPU-backend specs under `remote:`; `tests/test_tiers.py` pins it.

**Delay, honestly.** A threaded `tse` stage delays its output by window + hop: enrolled 1.5 s on the Mac, 0.75 s on the GPU; AV-MossFormer2 at 2 s 3.0 s. `hearing/fusion/tiers.py` aligns every tier to the slowest one so a switch never jumps in time. With the cold-start AV tier in the ladder, **Focus HQ therefore plays 3.0 s behind even while the enrolled tier plays**, which breaks PLAN's ≤ 2 s target. Two ways out: `tiers(only="enrolled,gate,audio-only")` (1.5 s on the Mac, no cold-start tier), or a ladder change in `tiers.py` that drops the AV tier from the alignment once enrollment is ready (a one-time re-anchoring crossfade). The second is a `tiers.py` change, outside this task (see "Follow-ups").

## Results

### GPU (iLab, CUDA)

Built from `$HEARING_RESULTS_DIR/ilab/bakeoff-gpu-{A,C,D}.json/*.json` (run B only has the `identity` / `oracle` controls; its `enrolled` / `blind_assoc` rows failed to load and were re-run as D). Ranked by candidate, then window. `Live delay` = window + hop, what the threaded `tse` stage adds. Compute is per window, including the GPU lock wait. `by SNR` is the mean SI-SDRi per input SNR (20 clips each).

| Candidate | Window / hop s | Host, device | n | SI-SDRi dB | by SNR -5/0/5/10 dB | Compute p50 / p95 ms per window | RTF | Live delay ms (window + hop) | GPU MB | WER | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| enrolled | 2 / 1 | ilab1, cuda (RTX A4500) | 80 | **+11.46** | 5.9/12.6/13.8/13.7 | 191 / 218 | 0.130 | 3000 | 554 | mock | 20261005-153514 |
| enrolled | 1 / 0.5 | ilab1, cuda (RTX A4500) | 80 | **+10.84** | 7.4/11.8/11.9/12.3 | 123 / 175 | 0.216 | 1500 | 431 | mock | 20261005-153514 |
| enrolled | 0.5 / 0.25 | ilab1, cuda (RTX A4500) | 80 | **+10.08** | 7.8/9.9/10.9/11.8 | 109 / 154 | 0.418 | 750 | 370 | mock | 20261005-153514 |
| av_mossformer2 | 2 / 1 | rlab4, cuda (GPU not logged) | 80 | **+11.20** | 10.8/12.0/11.0/11.0 | 261 / 263 | 0.174 | 3000 | 525 | mock | 20261005-153227 |
| av_mossformer2 | 1 / 0.5 | rlab4, cuda (GPU not logged) | 80 | **+3.91** | 1.4/2.7/5.3/6.3 | 144 / 149 | 0.242 | 1500 | 399 | mock | 20261005-153227 |
| av_mossformer2 | 0.5 / 0.25 | rlab4, cuda (GPU not logged) | 80 | **-0.37** | -0.4/-0.8/-0.3/0.0 | 101 / 116 | 0.383 | 750 | 335 | mock | 20261005-153227 |
| blind_assoc | 2 / 1 | ilab1, cuda (RTX A4500) | 80 | **+2.12** | 1.9/-1.4/4.1/3.8 | 170 / 199 | 0.116 | 3000 | 554 | mock | 20261005-153514 |
| blind_assoc | 1 / 0.5 | ilab1, cuda (RTX A4500) | 80 | **+8.69** | 4.6/9.7/10.3/10.2 | 100 / 144 | 0.175 | 1500 | 431 | mock | 20261005-153514 |
| blind_assoc | 0.5 / 0.25 | ilab1, cuda (RTX A4500) | 80 | **+7.79** | 5.8/7.9/8.4/9.1 | 86 / 130 | 0.330 | 750 | 370 | mock | 20261005-153514 |
| dolphin | 2 / 1 | ilab1, cuda (RTX A4500) | 80 | **-10.88** | -7.4/-10.2/-13.1/-12.8 | 115 / 146 | 0.079 | 3000 | 3077 | mock | 20261005-152817 |
| dolphin | 1 / 0.5 | ilab1, cuda (RTX A4500) | 80 | **-7.57** | -4.2/-8.0/-9.3/-8.7 | 73 / 116 | 0.127 | 1500 | 908 | mock | 20261005-152817 |
| dolphin | 0.5 / 0.25 | ilab1, cuda (RTX A4500) | 80 | **-4.11** | -2.9/-5.3/-5.1/-3.1 | 69 / 104 | 0.269 | 750 | 308 | mock | 20261005-152817 |

Controls (run B on rlab3, RTX A4000, and laptop-a run 20261005-150453): `identity` 0.00 dB and `oracle` +101.34 dB at every window, as expected; the mixture's SI-SDR is −1.34 dB.

### Mac (laptop-a, MPS)

Built from `$HEARING_RESULTS_DIR/bakeoff-mac.json/bakeoff_20261005-161323.json` (`Aryans-MacBook-Pro.local`, Apple M4 Pro 24 GB, torch 2.14.1 on `mps`, run 20261005-161323, 64 min wall). Same set, same `--enroll self`, but **Whisper (`small.en`) WER, measured**: the mixture's WER is **0.827**. Compute ran with other swarm workers on the machine (load average 9-12), so treat the Mac compute as an upper bound.

| Candidate | Window / hop s | n | SI-SDRi dB | by SNR -5/0/5/10 dB | Compute p50 / p95 ms per window | RTF | Live delay ms (window + hop) | MPS MB | WER (mix 0.827) | WER rel. change |
|---|---|---|---|---|---|---|---|---|---|---|
| enrolled | 2 / 1 | 80 | **+11.46** | 5.9/12.6/13.8/13.7 | 431 / 1069 | 0.384 | 3000 | 3307 | 0.571 | −31% |
| enrolled | 1 / 0.5 | 80 | **+10.84** | 7.4/11.8/11.9/12.3 | 236 / 323 | 0.425 | 1500 | 3307 | 0.679 | −18% |
| enrolled | 0.5 / 0.25 | 80 | **+10.07** | 7.8/9.9/10.8/11.8 | 150 / 234 | 0.613 | 750 | 3307 | 0.715 | −14% |
| av_mossformer2 | 2 / 1 | 80 | **+10.90** | 10.8/12.1/10.4/10.3 | 405 / 498 | 0.280 | 3000 | 262 | **0.485** | **−41%** |
| av_mossformer2 | 1 / 0.5 | 80 | **+4.13** | 1.7/2.8/5.0/6.9 | 216 / 270 | 0.376 | 1500 | 262 | 0.679 | −18% |
| av_mossformer2 | 0.5 / 0.25 | 80 | **-0.39** | -0.4/-0.8/-0.3/0.0 | 127 / 181 | 0.505 | 750 | 262 | 0.760 | −8% |
| blind_assoc | 2 / 1 | 80 | **+2.12** | 1.9/-1.4/4.1/3.8 | 473 / 1206 | 0.411 | 3000 | 3307 | 0.710 | −14% |
| blind_assoc | 1 / 0.5 | 80 | **+8.69** | 4.6/9.7/10.3/10.2 | 206 / 303 | 0.375 | 1500 | 3307 | 0.850 | +3% |
| blind_assoc | 0.5 / 0.25 | 80 | **+7.79** | 5.8/7.9/8.4/9.1 | 125 / 205 | 0.519 | 750 | 3307 | 0.775 | −6% |
| dolphin | 2 / 1 | 80 | **-10.97** | -7.4/-10.2/-13.1/-13.2 | 418 / 1010 | 0.319 | 3000 | 3001 | 0.815 | −1% |
| dolphin | 1 / 0.5 | 80 | **-7.68** | -4.3/-8.0/-9.7/-8.7 | 148 / 278 | 0.281 | 1500 | 982 | 1.048 | +27% |
| dolphin | 0.5 / 0.25 | 80 | **-4.11** | -3.0/-5.4/-5.5/-2.6 | 75 / 127 | 0.313 | 750 | 478 | 1.002 | +21% |

Controls on the Mac: `identity` 0.00 dB (WER 0.83-0.87), `oracle` +101.34 dB (WER 0.115, the transcriber's own floor on GRID's command grammar).

**Mac vs GPU.** SI-SDRi agrees to ±0.3 dB on every row (same weights, same fixtures; AV-MossFormer2 2 s is +10.90 on MPS vs +11.20 on CUDA), so the quality ranking does not depend on the host. Compute does: enrolled costs **236 / 323 ms** (p50 / p95) per 1 s window on the Mac vs 123 / 175 ms on the GPU, and **150 / 234 ms** per 0.5 s window vs 109 / 154 ms. On the Mac a 0.25 s hop leaves 16 ms of p95 headroom under load (RTF 0.61), a 0.5 s hop leaves 177 ms (RTF 0.43): hence 1 s / 0.5 s locally and 0.5 s / 0.25 s only on the GPU.

**WER does not follow SI-SDRi.** On Whisper, enrolled's WER gain shrinks with the window (−31% at 2 s, −18% at 1 s, −14% at 0.5 s) while its SI-SDRi barely moves, and only AV-MossFormer2 at 2 s meets PLAN's −40%. Likely causes (not yet separated): the window seams of short windows (the 50% overlap-add crossfade is short at 0.5 s), and GRID's 3 s clips leaving a 1 s window little context. SI-SDRi stays the decision metric (it is what the slider mixes); the WER gap is a follow-up below and a slide caveat.

## Caveats every number carries

- **WER in the GPU JSON is not a measurement.** All four GPU runs used the `mock` transcriber, which echoes the reference, so every WER there is 0.000. Those values are not quoted anywhere in this decision; the table shows `mock`. The PLAN WER check (≥ 40% relative reduction) is measured only on the Mac run, which used Whisper `small.en`.
- **Enrollment was an oracle.** Every run used `--enroll self` (the default): the first 3 s of the clean target, which on 3 s GRID clips is the whole clean target of the same utterance. Live enrollment is 3 s of *other* speech by the same person, picked by ASD in a noisy room. The enrolled numbers above are therefore an upper bound on the speaker match; MossFormer2's separation is the same either way.
- **`blind_assoc` used an oracle speaking detector** (clean-target frame energy), not Light-ASD.
- **Three GPU hosts.** Run A (Dolphin) and D (enrolled, blind_assoc) ran on `ilab1.cs.rutgers.edu`, whose log reports an **NVIDIA RTX A4500** (Ampere, not an "RTX 4500 Ada"); run C (AV-MossFormer2) ran on `rlab4`, which did not log its GPU; run B (controls) on `rlab3`, an RTX A4000. Compare compute across candidates with that in mind.
- **Small speaker pool.** The two-face set has two target speakers (GRID s1, s4), five utterances each, mixed with each other's speech and babble. It is a fair A/B between candidates, not proof of performance on venue voices.
- **Not comparable with T-053's numbers.** T-053/T-024 measured 16 clips of mixed layouts (AV-MossFormer2 2 s +13.75 dB); this run uses all 80 two-face clips (+11.20 dB).

## PLAN quality check

PLAN asks Focus HQ for ≥ 8 dB SI-SDRi over the mixture and ≥ 40% relative WER reduction, at ≤ 2 s behind (stretch ≤ 0.7 s).

| Candidate @ window | SI-SDRi ≥ 8 dB (GPU / Mac) | WER −40% (Mac, Whisper) | Own delay ≤ 2 s |
|---|---|---|---|
| enrolled @ 0.5 s (GPU backend) | pass (+10.08 / +10.07) | **fail** (−14%) | pass (0.75 s + RTT) |
| enrolled @ 1 s (Mac) | pass (+10.84 / +10.84) | **fail** (−18%) | pass (1.5 s) |
| av_mossformer2 @ 2 s (cold start) | pass (+11.20 / +10.90) | pass (−41%) | fail (3.0 s) |
| av_mossformer2 @ 1 s | fail (+3.91 / +4.13) | fail (−18%) | pass (1.5 s) |
| blind_assoc @ 1 s | pass (+8.69 / +8.69, oracle detector) | fail (+3%) | pass (1.5 s) |
| dolphin, any window | fail | fail | - |

No candidate passes all three. Enrolled at short windows wins on SI-SDRi and delay, which is what Natural mode's slider mixes; it does not deliver the WER bar, so Text/Synthetic should not feed their ASR from the short-window enrolled output until the WER follow-up below is done (the bench's Text rows run `synth` on the mix or the `dfn_ll` stream).

## Licences

All shipped models are Apache-2.0 (rows already in [MODELS.md](MODELS.md)): AV_MossFormer2_TSE_16K, MossFormer2_SS_16K, ECAPA-TDNN (speechbrain spkrec-ecapa-voxceleb). Dolphin's checkpoint is non-commercial; it is rejected on quality anyway. No new model is introduced by this decision.

## Follow-ups

- **Ladder alignment** (`hearing/fusion/tiers.py`, owner of T-019's ladder): drop the cold-start AV tier from the delay alignment once enrolled is healthy, or the whole Focus HQ mode stays 3.0 s behind.
- **Real enrollment**: re-run enrolled with `--enroll other` (another utterance of the same speaker) and, at the rehearsal, with ASD-gated enrollment from the live camera.
- **WER at short windows**: enrolled's Whisper WER falls off with the window (−31% / −18% / −14% at 2 / 1 / 0.5 s) though its SI-SDRi does not. Check the overlap-add seams (`hearing/tse/windowed.py`) and re-score on clips longer than GRID's 3 s before quoting WER on slides.
- **Real WER on the GPU**: re-run with `--transcriber whisper` on the GPU box (the Mac run is the only WER we have).
- **`tse(candidate="auto")`** still prefers Dolphin after AV-MossFormer2 (`hearing/tse/stage.py` `AUTO_ORDER`); the ladder now names its candidates explicitly, but `auto` should drop Dolphin.
- **`blind_assoc` at 2 s** (+2.12 dB vs +8.69 at 1 s): unexplained, worth a look only if a no-enrollment, no-video tier is ever wanted.
