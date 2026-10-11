# Scenario Card: scenario_12 (simultaneous_onset_spk2_continues)

## Overview
- **Category:** `simultaneous_onset`
- **Description:** Simultaneous onset at t=0.0s for 2.0s, Spk 1 drops out, Spk 2 continues solo to 4.0s
- **Seed:** `29312`
- **Duration:** 4.500 s (72,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07828
- **SHA-256:** `dea019ef4064f50de15b8a9ddf1ea06764b00ae7aa3e86c3bad09ebbbc6adecf`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 2.000 | 2.000 | +0.0 | Speaker 1 simultaneous onset t=0, drops at 2.0s |
| 2 | `speaker_2` | 0.000 | 4.000 | 4.000 | +0.0 | Speaker 2 active throughout (onset t=0) |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 0.000 | 2.000 | 2.000 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 4.000 | 4.500 | 0.500 | 500 ms |

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
- [x] Deterministic rendering verified with fixed seed `29312`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
