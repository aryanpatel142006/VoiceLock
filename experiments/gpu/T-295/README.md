# T-295 GPU sweep: joint alpha / metric / head tracking

Validates the composition-theory predictions in
`docs/research/theory/streaming_tracking_theory.md` on real phone-mic team
audio (ilab1, RTX A4500).

## What it tests

| # | Prediction | How the sweep tests it |
|---|-----------|------------------------|
| P1 | α*(0.5s) < α*(1.0s) < α*(2.0s) | α grid × 3 windows, cosine |
| P2 | α*_phone < α*_directional | SIR ladder: mix interferer at 0/3/6/12 dB |
| P3 | phone/dir α* gap widens with w | SIR ladder at 0.5 s and 2.0 s |
| P4 | α*_head > α*_maha > α*_cosine | metric × α grid |

## Round 2: gated tracker + switch-cue comparison (P8)

`sweep_gated.py` / `config_gated.json` test the Round-2 gated design
(theory §5b): fixed-α* base path + dwell-forced boost (`α_boost=0.7`,
`τ_dwell=3`) on detected switches. The simulator says the gated tracker
beats fixed α* on flip rate **iff** the switch cue reaches
(precision ≥ 0.6, recall ≥ 0.5) jointly — margin-only cues cannot at
0.5 s phone. This sweep compares:

- `fixed` (α=0.3 baseline) vs `naive` (T-282 confidence-weighted, β=4)
- `gated-margin` (θ_m ∈ {0.12, 0.20}, persist 2 hops)
- `gated-f0` (F0 discontinuity > 4 semitones, persist 2 hops;
  autocorrelation F0, numpy only — abstains on unvoiced hops)

Turn-taking streams are synthetic concatenations of **real**
single-speaker clips with known switch points (the switches are
constructed; the audio and ECAPA embeddings are real). Metrics: cosine,
head (if `models/team_projection_head.pt` loads). 10 configs, ~20 min —
fits the 30-min cap on job 447101.

```bash
python3 sweep_gated.py --config config_gated.json --dry-run  # check grid
python3 sweep_gated.py --config config_gated.json
# Results: results/t295/t295_gated_<ts>.json  (flip_rate,
#          mean_switch_latency_hops, detection P/R per cue)
```

Success criterion for P8: a gated cue with measured (P ≥ 0.6, R ≥ 0.5)
and flip_rate < fixed-α* flip_rate. If neither cue clears it, the
simulator's verdict stands: ship fixed α* = 0.30 and spend the
complexity budget elsewhere.

## Grid

- Main: α ∈ {0.1,…,0.9} × metric ∈ {cosine, mahalanobis, head} ×
  window ∈ {0.5, 1.0, 2.0} s = **81 configs**
- SIR ladder: SIR ∈ {0,3,6,12} dB × α ∈ {0.3,0.5,0.7} × window ∈ {0.5,2.0} s
  = **24 configs** (cosine only; interferer mixed into clean segments)
- ~30 s/config → **~45 min** total on the A4500.

## Run (on ilab1, inside the job 447101 allocation)

```bash
# 1. Point config.json at the real clips (edit the "clips" list;
#    filenames containing "enroll" become the speaker profiles).
# 2. Sanity check:
python3 sweep_tracking.py --config config.json --dry-run
# 3. Run:
python3 sweep_tracking.py --config config.json
# Results: results/t295/t295_sweep_<ts>.json
```

Notes:
- ECAPA comes from `hearing.tse.enrolled`; the script exits clearly if
  weights are absent.
- `head` metric activates only if `models/team_projection_head.pt` loads;
  `mahalanobis` only if `maha_stats_path` points at a within-speaker
  covariance (T-274 output). Otherwise those configs are skipped, not faked.
- Success criterion: the measured α*(w) curve reproduces the predicted
  ordering (P1–P4); exact α* values will differ (sim is calibrated, not exact).
- Do **not** run on the A100 reservation (job 447008 is superseded).
