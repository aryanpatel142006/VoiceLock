# GPU Model Review: Separation Baseline and Next Measurable Decision

## 1. Review Summary and Core Findings

This review evaluates actual completed GPU measurements recorded from the overnight remote run (jdr317 job `447101` on ilab1 / NVIDIA RTX A4500, checkpoint `t210-a4500-20261010T0315` and `team-baseline.json`).

### Key Conclusions:
1. **Model Evaluated**: The actual executed model in `t210-a4500-20261010T0315` and `team-baseline.json` is `MossFormer2_SS_16K` with `spkrec-ecapa-voxceleb` speaker embedding conditioning (`EnrolledCandidate` / `BlindSeparator`). **SepFormer WHAMR16k comparison was NOT executed** in this run; weights and code for SepFormer (`speechbrain/sepformer-whamr16k`) were not present in the completed evaluation scripts.
2. **Measured Bottleneck**: Separation compute per 2-second window (1-second hop) averages **202–330 ms** on NVIDIA RTX A4500 (max spike **628 ms** on Aryan B -5dB). While well below real-time for an offline 1000 ms hop (RTF ~0.25–0.33), this establishes that a live streaming deadline ladder with small buffers (e.g., 0.5 s window, 0.25 s hop) is impossible without either model distillation, architecture changes, or larger latency buffers. Warmup cost is high: initial model load was **10.09 s** and first-window warmup was **7.03 s**.
3. **Target Quality vs Interference**: On the team pair (Joseph & Aryan), MossFormer2 provides strong separation under 0 dB and -5 dB SIR (actual choice SI-SDR: Joseph A 16.52 dB @ 0 dB, 14.28 dB @ -5 dB; Aryan B 13.26 dB @ 0 dB, 12.26 dB @ -5 dB). However, under hard -10 dB SIR, quality drops significantly (Joseph A: 11.61 dB; Aryan B: 9.09 dB with 1 separation failure and 1 mix-back failure out of 8 windows).
4. **Public Baseline Anomaly & Skipped Metrics**: The public evaluation on `take0_1320_sir0_snr15` (LibriSpeech) evaluated only the second half (`split="heldout"`). Because the target utterance ends early (active speech ~6.0 s out of 23.3 s total) while the interferer speaks for 21.6 s, the heldout split for Role A had **0 active target windows** (11 target-absent windows, -34.16 dB leak), and Role B had **11 target-only windows** (23.65 dB SI-SDR). Consequently, true public overlapping-speech separation quality and wrong-speaker duration under realistic overlap were **not measured**.
5. **ASR Reference Verification Gap**: Faster-Whisper `base.en` ran at RTF 0.009 (target) and 0.021 (mix) on public audio. However, team audio ASR was not scored, and transcript references remain unverified against ground truth.

---

## 2. Quantitative Evidence and Audit

### 2.1 Hardware and Runtime Readiness
- **Device**: NVIDIA RTX A4500 (19,190 MiB VRAM), CUDA True, PyTorch `2.8.0+cu128`.
- **Model Load Time**: 10.0899 s (`MossFormer2_SS_16K` + `ECAPA-TDNN`).
- **Warmup Latency (first 2s window)**: 7.0250 s (CUDA synchronized).
- **ASR Model Load (`base.en`)**: 3.4733 s; first warmup transcription: 0.5318 s.

### 2.2 Team Baseline Separation (`data/team_baseline`)
- **Configuration**: Window 2.0 s, hop 1.0 s, split `heldout` (8 windows), thresholds `sep_min_db=4`, `drop_db=3`, AGC off, noise off.

| Condition | Target Role | SIR (dB) | Oracle SI-SDR (dB) | Actual Choice SI-SDR (dB) | Ceiling Gap (dB) | Mean Compute (ms) | Max Compute (ms) | Failures (Sep / Mix-back / Deadline) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Role A (Joseph) | Joseph | 0 | 16.517 | 16.517 | 0.00 | 227.34 | 305.54 | 0 / 0 / 0 |
| Role A (Joseph) | Joseph | -5 | 14.277 | 14.277 | 0.00 | 278.31 | 314.25 | 0 / 0 / 0 |
| Role A (Joseph) | Joseph | -10 | 11.612 | 11.612 | 0.00 | 313.08 | 383.02 | 0 / 0 / 0 |
| Role B (Aryan) | Aryan | 0 | 13.262 | 13.262 | 0.00 | 265.78 | 327.88 | 0 / 0 / 0 |
| Role B (Aryan) | Aryan | -5 | 12.257 | 12.257 | 0.00 | 330.12 | 628.23 | 0 / 0 / 0 |
| Role B (Aryan) | Aryan | -10 | 9.087 | 9.087 | 0.00 | 275.29 | 344.62 | 1 / 1 / 0 |

*Observations*:
- Selector agreed with oracle on 100% of tested windows (oracle gap = 0.0 dB).
- At -10 dB SIR, Aryan's target track suffered 1 separation failure (< 4 dB SI-SDR) and 1 mix-back failure (< 3 dB improvement over mix), dropping SI-SDR to 9.09 dB.

### 2.3 Public Baseline (`data/public_overnight/take0_1320_sir0_snr15`)
- Target duration: 23.315 s, active speech: ~6.03 s. Interferer active speech: ~21.61 s.
- **Role A**: 11 target-absent windows. Leak vs mix: **-34.16 dB**. Mean compute: 263.15 ms.
- **Role B**: 11 target-only windows. Actual SI-SDR: **23.74 dB**. Mean compute: 282.36 ms.
- **Defect**: The heldout split completely avoided actual simultaneous speech overlap. Public overlapping SI-SDR and wrong-speaker leak remain unmeasured.

### 2.4 ASR Quality & Reference Audit
- Model: `faster-whisper` `base.en` (MIT license, snapshot `3d3d5dee...`, `model.bin` SHA256 `2a166925...`).
- Public target transcription: 0.2087 s (RTF 0.00895). Text matches reference except for punctuation.
- Public mix transcription: 0.4792 s (RTF 0.02055). ASR outputs both target and interferer segments ("There was sending in his air... With which, she was treated by two or three...").
- Reference status remains `auto-unverified`. Team speech was not transcribed or evaluated for WER.

---

## 3. Discrepancies and Omissions against Target Caps

1. **SepFormer vs MossFormer2 Comparison (Item 1 of caps)**:
   - Cap allowed <= 90 GPU min to compare MossFormer2 and SepFormer WHAMR16k.
   - **Finding**: SepFormer was omitted completely from the executed script (`t210-measure-20261010T0315.py`). Only MossFormer2 was evaluated.
2. **Live-Wrapper Deadline Ladder (Item 2 of caps)**:
   - Cap allowed <= 90 GPU min for 1.0/0.5s and 2.0/1.0s deadline ladder.
   - **Finding**: Only 2.0s window / 1.0s hop offline chunking was measured. No streaming live-wrapper (`LiveEngine` or deadline ladder) was executed.
3. **Contrast + floor_ab (Item 3 of caps)**:
   - Not measured on GPU.
4. **Wrong-Speaker Duration**:
   - Selector-oracle agreement was 1.0 on team audio, but wrong-speaker leak duration across continuous streams was not calculated.

---

## 4. Source Revision and Environmental Drift

- Snapshot code SHA: REVISION `34e567386606885f7a453eaae898cb2483242993` was used under `t210-20261010T0250/source`.
- `main` branch has progressed (currently at `159aaed` via `T-228` and `T-220`).
- Model weights were sourced dynamically from Hugging Face Hub (`faster-whisper-base.en`, `MossFormer2_SS_16K`, `spkrec-ecapa-voxceleb`) without pinned local file verification in the repo, although snapshots exist in `~/hearing-joseph-overnight/models`.
- Unauthenticated HF requests warning observed in execution log.

---

## 5. Next Measurable Decision and Falsifying Experiment

### Decision: Reject Immediate Production Adoption; Require 1 Focused Bounded Measurement
MossFormer2 cannot be promoted to the live demo or default settings based on these results:
1. Average compute time of 270 ms per 2-second window with 628 ms peaks indicates that live streaming at low latency requires careful buffer sizing and deadline handling that has not been benchmarked.
2. The comparison with SepFormer WHAMR16k has not yet occurred.
3. Overlapping public speech and verified team transcript WER are missing.

### Proposed Next Bounded Experiment (Delegated to local Sol/Claude-a queue):
**Bounded Test**: Run exactly **one focused A/B script** on the active allocation (`447101` or successor):
- **Inputs**: The 16 s team heldout audio (`data/team_baseline`) and a verified overlapping crop of LibriSpeech.
- **Conditions**: 
  1. MossFormer2 vs SepFormer WHAMR16k (if weights download <= 5 min) on identical 2.0s/1.0s and 1.0s/0.5s windows.
  2. Compute latency p50/p95, memory peak, SI-SDRi, and wrong-speaker time (% of audio where interferer is louder than target by > 3 dB).
- **Hard Cap**: <= 30 GPU minutes, max 30 evaluation cases, no model sessions left running.
- **Success Gate**: Separation must achieve > 12 dB SI-SDR on -5 dB SIR, with p95 compute < 250 ms and wrong-speaker time < 10%.
