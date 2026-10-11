# Scenario Card: scenario_39 (compound_simultaneous_asymmetry_pause)

## Overview
- **Category:** `compound_stress`
- **Description:** Simultaneous onset at 6dB asymmetry, resolving into 200ms pause, followed by rapid 100ms handoff
- **Seed:** `29339`
- **Duration:** 5.500 s (88,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07138
- **SHA-256:** `6e6f676b9c5b420bec1b748f5e480a4129208caa08e4ed1668f1b39eab7266a4`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 2.000 | 2.000 | +0.0 | Spk 1 onset (0dB) |
| 2 | `speaker_2` | 0.000 | 1.000 | 1.000 | +6.0 | Spk 2 simultaneous onset (+6dB) |
| 3 | `speaker_1` | 2.200 | 3.500 | 1.300 | +0.0 | Spk 1 post-pause continuation (0dB) |
| 4 | `speaker_2` | 3.600 | 5.000 | 1.400 | +6.0 | Spk 2 rapid reply (+6dB, 100ms gap) |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 0.000 | 1.000 | 1.000 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 2.000 | 2.200 | 0.200 | 200 ms |
| 3.500 | 3.600 | 0.100 | 100 ms |
| 5.000 | 5.500 | 0.500 | 500 ms |

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
- [x] Deterministic rendering verified with fixed seed `29339`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
