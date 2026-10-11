# Scenario Card: scenario_09 (asymmetric_gaps)

## Overview
- **Category:** `rapid_turn_taking`
- **Description:** Asymmetric turn gaps: rapid 80ms reply followed by 240ms handoff
- **Seed:** `29309`
- **Duration:** 5.200 s (83,200 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07599
- **SHA-256:** `31cd179c5007191e6e013732aef3c04cca8c05d4da63a68f17eba9f02085596e`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.400 | 1.400 | +0.0 | Speaker 1 turn 1 |
| 2 | `speaker_2` | 1.480 | 2.880 | 1.400 | +0.0 | Speaker 2 quick retort (80ms gap) |
| 3 | `speaker_1` | 3.120 | 4.520 | 1.400 | +0.0 | Speaker 1 resumed handoff (240ms gap) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.400 | 1.480 | 0.080 | 80 ms |
| 2.880 | 3.120 | 0.240 | 240 ms |
| 4.520 | 5.200 | 0.680 | 680 ms |

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
- [x] Deterministic rendering verified with fixed seed `29309`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
