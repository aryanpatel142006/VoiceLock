# Scenario Card: scenario_18 (simultaneous_onset_handoff)

## Overview
- **Category:** `simultaneous_onset`
- **Description:** Simultaneous onset [0.0, 1.5s], then Spk 1 solo [1.5, 2.8s], then Spk 2 handoff [3.0, 4.5s]
- **Seed:** `29318`
- **Duration:** 5.000 s (80,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07398
- **SHA-256:** `f9bfc1f8c71dac85ac04e8274babedfe853849e3ef27b1b4819a5689bad70f86`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 2.800 | 2.800 | +0.0 | Speaker 1 onset and continued solo |
| 2 | `speaker_2` | 0.000 | 1.500 | 1.500 | +0.0 | Speaker 2 simultaneous onset |
| 3 | `speaker_2` | 3.000 | 4.500 | 1.500 | +0.0 | Speaker 2 sequential handoff |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 0.000 | 1.500 | 1.500 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 2.800 | 3.000 | 0.200 | 200 ms |
| 4.500 | 5.000 | 0.500 | 500 ms |

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
- [x] Deterministic rendering verified with fixed seed `29318`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
