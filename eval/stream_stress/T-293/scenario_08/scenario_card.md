# Scenario Card: scenario_08 (accelerating_turn_gaps)

## Overview
- **Category:** `rapid_turn_taking`
- **Description:** Accelerating conversational cadence with diminishing gaps: 280ms -> 180ms -> 80ms
- **Seed:** `29308`
- **Duration:** 5.500 s (88,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07495
- **SHA-256:** `471043b2b675ae1ecd9ccee5184b7b3943d2442d11917169c79e4d26ebbcc481`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.100 | 1.100 | +0.0 | Speaker 1 turn 1 |
| 2 | `speaker_2` | 1.380 | 2.480 | 1.100 | +0.0 | Speaker 2 turn 1 (280ms gap) |
| 3 | `speaker_1` | 2.660 | 3.760 | 1.100 | +0.0 | Speaker 1 turn 2 (180ms gap) |
| 4 | `speaker_2` | 3.840 | 4.940 | 1.100 | +0.0 | Speaker 2 turn 2 (80ms gap) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.100 | 1.380 | 0.280 | 280 ms |
| 2.480 | 2.660 | 0.180 | 180 ms |
| 3.760 | 3.840 | 0.080 | 80 ms |
| 4.940 | 5.500 | 0.560 | 560 ms |

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
- [x] Deterministic rendering verified with fixed seed `29308`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
