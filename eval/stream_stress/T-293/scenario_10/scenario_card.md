# Scenario Card: scenario_10 (rapid_staccato_75ms)

## Overview
- **Category:** `rapid_turn_taking`
- **Description:** Five short staccato bursts (0.75s) separated by 75ms conversational gaps
- **Seed:** `29310`
- **Duration:** 4.600 s (73,600 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.08870
- **SHA-256:** `229e3ffa1b9980cfe37b1e3cb6ef2106dd211d311d4667a7fc2d65c09cf88184`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 0.750 | 0.750 | +0.0 | Turn 1 (Spk 1) |
| 2 | `speaker_2` | 0.825 | 1.575 | 0.750 | +0.0 | Turn 2 (Spk 2, 75ms gap) |
| 3 | `speaker_1` | 1.650 | 2.400 | 0.750 | +0.0 | Turn 3 (Spk 1, 75ms gap) |
| 4 | `speaker_2` | 2.475 | 3.225 | 0.750 | +0.0 | Turn 4 (Spk 2, 75ms gap) |
| 5 | `speaker_1` | 3.300 | 4.050 | 0.750 | +0.0 | Turn 5 (Spk 1, 75ms gap) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 0.750 | 0.825 | 0.075 | 75 ms |
| 1.575 | 1.650 | 0.075 | 75 ms |
| 2.400 | 2.475 | 0.075 | 75 ms |
| 3.225 | 3.300 | 0.075 | 75 ms |
| 4.050 | 4.600 | 0.550 | 550 ms |

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
- [x] Deterministic rendering verified with fixed seed `29310`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
