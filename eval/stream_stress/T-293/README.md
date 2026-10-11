# Streaming Stress-Test Corpus for EMA Tracking Variants (T-293)

Corpus generated for muse-c to evaluate causal streaming speaker tracking variants, specifically:
- **T-275**: Short-crop EMA context buffer for streaming ECAPA stability
- **T-282**: Confidence-weighted EMA ($\alpha_t = \alpha_{\text{base}} \cdot \sigma(\beta \cdot \text{margin}_t)$)

## Corpus Summary
- **Total Scenarios:** 40 scripted scenarios
- **Sample Rate:** 16,000 Hz mono float32 / PCM16
- **Determinism:** Fixed seeds (`29301` to `29340`), bit-exact reproducible
- **Output Structure:**
  - `manifest.json`: Master index with scenario specifications, seeds, parameters, and audio hashes
  - `spot_check.json`: Spot-check validation on representative scenarios confirming sample-aligned label accuracy
  - `scenario_01/` ... `scenario_40/`:
    - `mix.wav`: Rendered 16 kHz audio mixture
    - `turn_labels.json`: Ground-truth speaker-turn timestamps, sample ranges, active/overlap/silence intervals
    - `scenario_card.md`: Detailed markdown scenario documentation with timeline and EMA evaluation notes

## Stress Conditions Covered

| Category | Count | Key Test Conditions | Primary EMA Tracking Failure Modes Tested |
|:---|:---:|:---|:---|
| **Rapid Turn-Taking** | 10 | Gaps < 300 ms (50 ms, 100 ms, 150 ms, 200 ms, 250 ms, 4-turn ping-pong, staccato 75 ms, accelerating) | Context inertia, transition lag, switch-delay, switch-stick when margin drops during speaker handover |
| **Simultaneous Onset** | 10 | Both speakers start at identical instant ($t=0$ or post-silence/midstream), 500 ms collision bursts, full double-talk overlap, babble noise | Indeterminate initial embedding, lack of prior prior, margin collapse preventing gain adaptation |
| **200 ms Pauses** | 10 | Intra-utterance hesitation pauses (single, double, rhythmic triple), inter-turn 200 ms pauses, pauses under 12 dB babble | Embedding decay during silence, noise pull, coasting state preservation vs unwarranted confidence drop |
| **Gain Asymmetry (6 dB)** | 8 | Interferer +6 dB louder, interferer -6 dB softer, simultaneous onset with 6 dB disparity, intra-speaker +6 dB step | Asymmetric tracking bias, loud talker embedding dominance, low-margin stall on quiet talker |
| **Compound Stress** | 2 | Simultaneous onset + 6 dB asymmetry + 200 ms pause + rapid <150 ms turn handoff | Multi-modal stress combining all failure modes in realistic challenging dialogue |

## Spot-Check Verification
Three representative scenarios (`scenario_01`, `scenario_11`, `scenario_21`) were spot-checked for label accuracy:
1. `scenario_01` (Rapid Turn-Taking, 50 ms gap): Verified active speech energy during Turn 1 (RMS 0.099) and Turn 2 (RMS 0.073), with exact zero energy (RMS 0.0) during the 50 ms gap [2.000, 2.050 s].
2. `scenario_11` (Simultaneous Onset): Verified simultaneous onset at $t=0.000$ s with double-talk energy (RMS 0.092 and 0.106) and exact silence in tail [4.000, 4.500 s].
3. `scenario_21` (200 ms Pause): Verified speech energy before pause (RMS 0.096), clean zero energy during the 200 ms pause [1.800, 2.000 s] (RMS 0.0), and resumed speech (RMS 0.076).

All spot-checks passed and results are recorded in `spot_check.json`.
