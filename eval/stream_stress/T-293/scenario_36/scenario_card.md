# Scenario Card: scenario_36 (gain_step_change_6db)

## Overview
- **Category:** `gain_asymmetry_6db`
- **Description:** Speaker 1 starts at 0dB, pauses 200ms, then resumes at +6dB (intra-speaker 6dB step)
- **Seed:** `29336`
- **Duration:** 4.200 s (67,200 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.07332
- **SHA-256:** `940270fbca37afdb61ce677121c2b13f5c845c75077e91a0ad1aa467e346d95f`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.800 | 1.800 | +0.0 | Spk 1 initial phrase at 0dB |
| 2 | `speaker_1` | 2.000 | 3.800 | 1.800 | +6.0 | Spk 1 post-pause phrase stepped up to +6dB |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.800 | 2.000 | 0.200 | 200 ms |
| 3.800 | 4.200 | 0.400 | 400 ms |

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
- [x] Deterministic rendering verified with fixed seed `29336`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
