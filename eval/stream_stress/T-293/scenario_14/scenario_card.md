# Scenario Card: scenario_14 (simultaneous_onset_post_silence)

## Overview
- **Category:** `simultaneous_onset`
- **Description:** 1.0s lead silence followed by simultaneous onset at t=1.0s for 2.5s
- **Seed:** `29314`
- **Duration:** 4.000 s (64,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.08523
- **SHA-256:** `1432ea0700f9b640060fe45c8f091c3cd16d9c18359150d548bb78147f3875e8`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 1.000 | 3.500 | 2.500 | +0.0 | Speaker 1 onset after 1s silence |
| 2 | `speaker_2` | 1.000 | 3.500 | 2.500 | +0.0 | Speaker 2 onset after 1s silence |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 1.000 | 3.500 | 2.500 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 0.000 | 1.000 | 1.000 | 1000 ms |
| 3.500 | 4.000 | 0.500 | 500 ms |

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
- [x] Deterministic rendering verified with fixed seed `29314`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
