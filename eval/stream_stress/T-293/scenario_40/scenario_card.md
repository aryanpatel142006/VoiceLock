# Scenario Card: scenario_40 (compound_full_cocktail_stress)

## Overview
- **Category:** `compound_stress`
- **Description:** Full cocktail stress: rapid staccato (<150ms gaps), alternating 6dB asymmetry, two 200ms pauses, and collision
- **Seed:** `29340`
- **Duration:** 6.500 s (104,000 samples @ 16000 Hz)
- **Noise Condition:** `none` (Clean / No additive noise)
- **Peak Amplitude:** 0.9000 (-0.92 dBFS)
- **RMS Energy:** 0.06597
- **SHA-256:** `eb55183a87abb946d45a81acd00c3ab377f174036edfa2ffed59cecc99e08e0f`

## Ground-Truth Timeline
| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `speaker_1` | 0.000 | 1.200 | 1.200 | +0.0 | Spk 1 initial (0dB) |
| 2 | `speaker_2` | 1.320 | 2.520 | 1.200 | +6.0 | Spk 2 loud reply (+6dB, 120ms gap) |
| 3 | `speaker_1` | 2.720 | 4.220 | 1.500 | +0.0 | Spk 1 resume after 200ms pause (0dB) |
| 4 | `speaker_2` | 3.500 | 4.500 | 1.000 | +6.0 | Spk 2 collision intrusion (+6dB) |
| 5 | `speaker_1` | 4.700 | 6.000 | 1.300 | +0.0 | Spk 1 final wrap-up after 200ms pause (0dB) |

### Overlap Intervals
| Start (s) | End (s) | Duration (s) | Overlapping Speakers |
|:---:|:---:|:---:|:---|
| 3.500 | 4.220 | 0.720 | `speaker_1, speaker_2` |

### Pauses & Silent Intervals
| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |
|:---:|:---:|:---:|:---:|
| 1.200 | 1.320 | 0.120 | 120 ms |
| 2.520 | 2.720 | 0.200 | 200 ms |
| 4.500 | 4.700 | 0.200 | 200 ms |
| 6.000 | 6.500 | 0.500 | 500 ms |

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
- [x] Deterministic rendering verified with fixed seed `29340`.
- [x] Audio rendered strictly at 16000 Hz mono float32 without clipping.
- [x] Ground-truth timestamps match audio onset/offset sample boundaries.
