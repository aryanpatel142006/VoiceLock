# Scenario Card: scenario_26 (pingpong_with_200ms_pauses)

## Overview
- **Category:** `micro_pause_200ms`
- **Description:** Three-turn dialogue where every transition is exactly 200ms pause
- **Seed:** `29326`
- **Duration:** 5.000 s (80,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07929
- **SHA-256:** `e7b969c03b311f590dde028f449ac71be4f7b69ced7f0231ae3787cddef2161a`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.400 | 1.400 | +0.0 | Turn 1 (Spk 1) |
| 2 | `speaker_2` | 1.600 | 3.000 | 1.400 | +0.0 | Turn 2 (Spk 2, 200ms pause) |
| 3 | `speaker_1` | 3.200 | 4.600 | 1.400 | +0.0 | Turn 3 (Spk 1, 200ms pause) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.400 | 1.600 | 0.200 | 200 ms |
| 3.000 | 3.200 | 0.200 | 200 ms |
| 4.600 | 5.000 | 0.400 | 400 ms |

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
- [x] Deterministic rendering verified with fixed seed `29326`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
