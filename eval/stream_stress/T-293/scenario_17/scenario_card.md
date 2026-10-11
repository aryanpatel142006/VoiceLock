# Scenario Card: scenario_17 (midstream_simultaneous_onset)

## Overview
- **Category:** `simultaneous_onset`
- **Description:** Speaker 1 speaks solo [0.0, 2.0s], pause [2.0, 2.4s], then both onset simultaneously at t=2.4s
- **Seed:** `29317`
- **Duration:** 5.000 s (80,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.08216
- **SHA-256:** `779588269c1a9a86b67a194b71de011562d1ffb787e0bdfa5d7df478b036951f`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 2.000 | 2.000 | +0.0 | Speaker 1 initial solo segment |
| 2 | `speaker_1` | 2.400 | 4.400 | 2.000 | +0.0 | Speaker 1 midstream simultaneous onset |
| 3 | `speaker_2` | 2.400 | 4.400 | 2.000 | +0.0 | Speaker 2 midstream simultaneous onset |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 2.400 | 4.400 | 2.000 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 2.000 | 2.400 | 0.400 | 400 ms |
| 4.400 | 5.000 | 0.600 | 600 ms |

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
- [x] Deterministic rendering verified with fixed seed `29317`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
