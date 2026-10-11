# Scenario Card: scenario_15 (simultaneous_burst_short_500ms)

## Overview
- **Category:** `simultaneous_onset`
- **Description:** Simultaneous onset at t=0.0s with brief 500ms overlap collision; Spk 1 resolves solo
- **Seed:** `29315`
- **Duration:** 3.500 s (56,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.06972
- **SHA-256:** `ed7190070084974602ebb97288f3a21d40451dc2077fcd3440ea5d3120b86f56`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 3.000 | 3.000 | +0.0 | Speaker 1 sustained speech |
| 2 | `speaker_2` | 0.000 | 0.500 | 0.500 | +0.0 | Speaker 2 brief 500ms collision |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 0.000 | 0.500 | 0.500 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 3.000 | 3.500 | 0.500 | 500 ms |

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
- [x] Deterministic rendering verified with fixed seed `29315`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
