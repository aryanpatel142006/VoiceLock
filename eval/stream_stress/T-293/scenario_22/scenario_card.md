# Scenario Card: scenario_22 (intraturn_double_pause_200ms)

## Overview
- **Category:** `micro_pause_200ms`
- **Description:** Speaker 1 with two 200ms hesitation pauses across a continuous 4.0s utterance
- **Seed:** `29322`
- **Duration:** 4.400 s (70,400 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.08953
- **SHA-256:** `97c61cd1034973a8f8d4e4d6e9a1dfeb76a776befc801273932c47e880e3bbb1`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.200 | 1.200 | +0.0 | Segment 1 |
| 2 | `speaker_1` | 1.400 | 2.600 | 1.200 | +0.0 | Segment 2 (after 200ms pause) |
| 3 | `speaker_1` | 2.800 | 4.000 | 1.200 | +0.0 | Segment 3 (after 200ms pause) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.200 | 1.400 | 0.200 | 200 ms |
| 2.600 | 2.800 | 0.200 | 200 ms |
| 4.000 | 4.400 | 0.400 | 400 ms |

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
- [x] Deterministic rendering verified with fixed seed `29322`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
