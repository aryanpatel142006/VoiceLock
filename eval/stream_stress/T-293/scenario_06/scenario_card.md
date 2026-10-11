# Scenario Card: scenario_06 (rapid_pingpong_100ms)

## Overview
- **Category:** `rapid_turn_taking`
- **Description:** Four-turn rapid ping-pong exchange with 100ms gaps
- **Seed:** `29306`
- **Duration:** 5.500 s (88,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07839
- **SHA-256:** `e719c74870d60486ab2ca1018a1e5697b86dd8b921546b5cb98e5fab5e8bd1a5`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.200 | 1.200 | +0.0 | Speaker 1 turn 1 |
| 2 | `speaker_2` | 1.300 | 2.500 | 1.200 | +0.0 | Speaker 2 turn 1 (100ms gap) |
| 3 | `speaker_1` | 2.600 | 3.800 | 1.200 | +0.0 | Speaker 1 turn 2 (100ms gap) |
| 4 | `speaker_2` | 3.900 | 5.100 | 1.200 | +0.0 | Speaker 2 turn 2 (100ms gap) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.200 | 1.300 | 0.100 | 100 ms |
| 2.500 | 2.600 | 0.100 | 100 ms |
| 3.800 | 3.900 | 0.100 | 100 ms |
| 5.100 | 5.500 | 0.400 | 400 ms |

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
- [x] Deterministic rendering verified with fixed seed `29306`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
