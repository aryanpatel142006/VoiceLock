# Scenario Card: scenario_31 (gain_asymmetry_loud_interferer_6db)

## Overview
- **Category:** `gain_asymmetry_6db`
- **Description:** Turn-taking with +6dB louder interferer (Spk 1 0dB, Spk 2 +6dB, 150ms gap)
- **Seed:** `29331`
- **Duration:** 4.500 s (72,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.06835
- **SHA-256:** `4f9ea63e5c6a3dfe0ee701bdecddf42a494c9e14c533a1898e6ff973da6fe0a3`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 2.000 | 2.000 | +0.0 | Speaker 1 at baseline 0dB |
| 2 | `speaker_2` | 2.150 | 4.150 | 2.000 | +6.0 | Speaker 2 loud interferer (+6dB, 150ms gap) |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 2.000 | 2.150 | 0.150 | 150 ms |
| 4.150 | 4.500 | 0.350 | 350 ms |

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
- [x] Deterministic rendering verified with fixed seed `29331`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
