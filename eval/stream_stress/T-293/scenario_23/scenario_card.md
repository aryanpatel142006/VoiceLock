# Scenario Card: scenario_23 (spk2_intraturn_pause_200ms)

## Overview
- **Category:** `micro_pause_200ms`
- **Description:** Speaker 2 with mid-utterance exactly 200ms pause
- **Seed:** `29323`
- **Duration:** 4.200 s (67,200 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.7735 (-2.23 dBFS)
- **RMS Energy:** 0.06881
- **SHA-256:** `1328303e8c840c0b7398a14ca4cca9cb07801ce65295127bcd473e08f6fe197b`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_2` | 0.000 | 1.800 | 1.800 | +0.0 | Speaker 2 pre-pause phrase |
| 2 | `speaker_2` | 2.000 | 3.800 | 1.800 | +0.0 | Speaker 2 post-pause phrase (200ms pause) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.800 | 2.000 | 0.200 | 200 ms |
| 3.800 | 4.200 | 0.400 | 400 ms |

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
- [x] Deterministic rendering verified with fixed seed `29323`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
