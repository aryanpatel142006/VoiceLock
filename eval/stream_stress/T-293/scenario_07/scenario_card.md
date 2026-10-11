# Scenario Card: scenario_07 (rapid_pingpong_200ms)

## Overview
- **Category:** `rapid_turn_taking`
- **Description:** Four-turn rapid ping-pong exchange with 200ms gaps
- **Seed:** `29307`
- **Duration:** 6.000 s (96,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07839
- **SHA-256:** `1ec9fda07c1bfb57cf93049d1d1aa89bf513fc35e15b94ae1ecceda5228a6764`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.200 | 1.200 | +0.0 | Speaker 1 turn 1 |
| 2 | `speaker_2` | 1.400 | 2.600 | 1.200 | +0.0 | Speaker 2 turn 1 (200ms gap) |
| 3 | `speaker_1` | 2.800 | 4.000 | 1.200 | +0.0 | Speaker 1 turn 2 (200ms gap) |
| 4 | `speaker_2` | 4.200 | 5.400 | 1.200 | +0.0 | Speaker 2 turn 2 (200ms gap) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.200 | 1.400 | 0.200 | 200 ms |
| 2.600 | 2.800 | 0.200 | 200 ms |
| 4.000 | 4.200 | 0.200 | 200 ms |
| 5.400 | 6.000 | 0.600 | 600 ms |

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
- [x] Deterministic rendering verified with fixed seed `29307`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
