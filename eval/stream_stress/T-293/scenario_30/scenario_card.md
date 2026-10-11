# Scenario Card: scenario_30 (triple_rhythmic_pause_200ms)

## Overview
- **Category:** `micro_pause_200ms`
- **Description:** Three periodic 200ms hesitation pauses across a continuous 5.0s sentence
- **Seed:** `29330`
- **Duration:** 5.200 s (83,200 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.08590
- **SHA-256:** `166b67ca18b69e88dfddc9187e64a76770e121ea2ae66bf6dceaea744e8c5055`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.000 | 1.000 | +0.0 | Chunk 1 |
| 2 | `speaker_1` | 1.200 | 2.200 | 1.000 | +0.0 | Chunk 2 (200ms pause) |
| 3 | `speaker_1` | 2.400 | 3.400 | 1.000 | +0.0 | Chunk 3 (200ms pause) |
| 4 | `speaker_1` | 3.600 | 4.600 | 1.000 | +0.0 | Chunk 4 (200ms pause) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.000 | 1.200 | 0.200 | 200 ms |
| 2.200 | 2.400 | 0.200 | 200 ms |
| 3.400 | 3.600 | 0.200 | 200 ms |
| 4.600 | 5.200 | 0.600 | 600 ms |

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
- [x] Deterministic rendering verified with fixed seed `29330`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
