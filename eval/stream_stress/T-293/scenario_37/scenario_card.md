# Scenario Card: scenario_37 (gain_asymmetry_int_burst_6db)

## Overview
- **Category:** `gain_asymmetry_6db`
- **Description:** Speaker 1 continuous at 0dB; Speaker 2 injects a +6dB burst intrusion [1.5, 2.7s]
- **Seed:** `29337`
- **Duration:** 4.200 s (67,200 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.09252
- **SHA-256:** `c60c497e2227d871410b5e3f66dec9875cd1b91111ef992d50655a708bbf405f`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 4.000 | 4.000 | +0.0 | Spk 1 continuous sentence at 0dB |
| 2 | `speaker_2` | 1.500 | 2.700 | 1.200 | +6.0 | Spk 2 loud intrusion burst (+6dB) |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 1.500 | 2.700 | 1.200 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 4.000 | 4.200 | 0.200 | 200 ms |

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
- [x] Deterministic rendering verified with fixed seed `29337`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
