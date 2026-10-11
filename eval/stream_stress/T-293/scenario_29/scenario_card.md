# Scenario Card: scenario_29 (asymmetric_pauses_150_200ms)

## Overview
- **Category:** `micro_pause_200ms`
- **Description:** Intra-turn 150ms pause followed by inter-turn 200ms pause
- **Seed:** `29329`
- **Duration:** 4.800 s (76,800 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.08708
- **SHA-256:** `a228c9cbb83853ceadd6bd8a805c95b5e50a7d9bf32e669fe45dd8a709668ee2`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.200 | 1.200 | +0.0 | Spk 1 segment 1 |
| 2 | `speaker_1` | 1.350 | 2.550 | 1.200 | +0.0 | Spk 1 segment 2 (150ms pause) |
| 3 | `speaker_2` | 2.750 | 4.250 | 1.500 | +0.0 | Spk 2 handoff (200ms pause) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.200 | 1.350 | 0.150 | 150 ms |
| 2.550 | 2.750 | 0.200 | 200 ms |
| 4.250 | 4.800 | 0.550 | 550 ms |

## EMA Tracking Evaluation Relevance
- **T-275 (Short-crop EMA Context Buffer):**
  - **Inertia & Lag:** Evaluates exponential moving average adaptation speed across short turn boundaries (<300ms) and pauses.
  - **Wrong-Speaker Time:** Assesses wrong-speaker accumulation when switching between profiles.
  - **Smoothing Stability:** Verifies suppression of angular jitter during stationary turns without excessive sluggishness.
- **T-282 (Confidence-Weighted EMA):**
  - **Switch-Stick Vulnerability:** Detects whether adaptive gain $\alpha_t = \alpha_{\text{base}} \cdot \sigma(\beta \cdot \text{margin})$ collapses during turn transitions and stalls adaptation.
  - **Coasting Across Pauses:** Verifies whether 200ms silence pauses cause confidence degradation or allow clean state coasting.
  - **Gain Asymmetry Balance:** Tests whether 6dB gain asymmetry creates asymmetric tracking locks or pulls embeddings towards the louder talker.

## Verification Checkpoints
- [x] Deterministic rendering verified with fixed seed `29329`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
