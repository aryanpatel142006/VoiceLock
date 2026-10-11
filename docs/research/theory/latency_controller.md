# T-252 — Latency/quality adaptive controller (theoretical, offline)

Status: **theoretical**. Everything below is a simulation on synthetic traces with assumed constants. No measured latency, no runtime edits, no audio, no GPU. Predictive claims need replay against real telemetry before use.

## Delay model: three separate things
Mouth-to-ear (m2e) delay = **algorithmic** (capture/output buffers + the window that must fill, `window + 0.02 s`) + **scheduled slack** that must absorb **hardware compute** (`rtf × window × load`) and **network** (RTT to a remote tier, plus loss). The slack is fixed per config, so delay is predictable within a config; a deadline miss happens when compute+network > slack. Shrinking the window is the only lever that reduces both algorithmic delay and compute; load and network are exogenous.

| config | window | assumed rtf | remote | slack | m2e (s) | assumed quality |
|---|---|---|---|---|---|---|
| T1_W2 | 2.0 | 0.15 | yes | 0.60 | 2.62 | 0.95 |
| T1_W1 | 1.0 | 0.15 | yes | 0.40 | 1.42 | 0.88 |
| T2_W1 | 1.0 | 0.30 | no | 0.50 | 1.52 | 0.75 |
| T3_gate | 0.04 | 0.10 | no | 0.03 | 0.09 | 0.50 |
| T4_dfn | 0.04 | 0.10 | no | 0.03 | 0.09 | 0.35 |

All of these (rtf, slack, quality scores, jitter) are **assumptions** loosely based on the `docs/ARCHITECTURE.md` budgets, not measurements. A missed window is replaced by an always-on local DFN shadow (assumption: small CPU cost; quality 0.35; counted as a glitch).

## Controller (`adaptive`)
- Telemetry: per-resource (local/remote) load = max(median of last 12 observations, mean of last 2), log-variance, heartbeat RTT + loss flag (assumed always available, also while the remote is idle).
- Risk of tier *i* = lost-share + P(compute+net > slack) under a lognormal with spread ≥ 0.25.
- Down: risk > 0.20 or 2 consecutive misses → first tier whose risk ≤ 0.20 (fast). Up: one tier at a time, candidate risk < 0.05 sustained 14 s (blips < 0.12 do not reset the clock); a failed upgrade (down within 30 s) doubles the dwell (max 160 s). Min 4 s between switches.
- Starts on T2 (bounded local), earns Tier 1 after a 10 s warmup.
- **Churn breaker / demo-safe rollback:** ≥ 4 switches in 60 s pins ≥ T2 for 60 s; `rollback()` pins T4 until a human unpins.

## Metrics and QoS constraints
Miss rate, mean quality (miss → 0.35), m2e mean/p95, tier switches/min, delay-jump seconds/min (Σ|Δm2e| on switches; a proxy for audible skip/stretch), fallback share. QoS (assumed): miss ≤ 5 %, switches ≤ 2/min, p95 m2e ≤ 3.0 s. "Audibly harmful churn" is proxied by switches and jump seconds; real thresholds need listening tests.

## Result (20 seeds × 180 s, 0.5 s ticks; 20 cases)
Scenarios: `steady` (warmup only), `overload` (×4.5 load 60–100 s), `remote_loss` (20 s full loss, 35 % burst loss, RTT spike), `flap` (load oscillating around the deadline), `combined`. All start with a decaying cold-start inflation.

| scenario | policy | miss % | quality | m2e mean s | m2e p95 s | switches/min | jump s/min | QoS pass |
|---|---|---|---|---|---|---|---|---|
| steady | fixed1s | 3.0 | 0.864 | 1.42 | 1.42 | 0.00 | 0.00 | 100% |
| steady | fixed2s | 4.4 | 0.924 | 2.62 | 2.62 | 0.00 | 0.00 | 90% |
| steady | naive | 2.9 | 0.923 | 2.54 | 2.62 | 7.07 | 8.65 | 0% |
| steady | adaptive | 0.5 | 0.673 | 0.98 | 2.05 | 1.58 | 1.71 | 90% |
| overload | fixed1s | 25.0 | 0.748 | 1.42 | 1.42 | 0.00 | 0.00 | 0% |
| overload | fixed2s | 26.5 | 0.791 | 2.62 | 2.62 | 0.00 | 0.00 | 0% |
| overload | naive | 14.2 | 0.806 | 2.14 | 2.62 | 34.13 | 46.19 | 0% |
| overload | adaptive | 2.8 | 0.468 | 0.29 | 0.93 | 1.67 | 1.43 | 85% |
| remote_loss | fixed1s | 25.3 | 0.746 | 1.42 | 1.42 | 0.00 | 0.00 | 0% |
| remote_loss | fixed2s | 26.1 | 0.793 | 2.62 | 2.62 | 0.00 | 0.00 | 0% |
| remote_loss | naive | 16.9 | 0.817 | 2.26 | 2.62 | 40.47 | 21.81 | 0% |
| remote_loss | adaptive | 0.6 | 0.546 | 0.39 | 1.16 | 1.35 | 1.63 | 100% |
| flap | fixed1s | 49.0 | 0.620 | 1.42 | 1.42 | 0.00 | 0.00 | 0% |
| flap | fixed2s | 58.3 | 0.600 | 2.62 | 2.62 | 0.00 | 0.00 | 0% |
| flap | naive | 35.6 | 0.613 | 1.52 | 2.62 | 81.70 | 100.52 | 0% |
| flap | adaptive | 2.8 | 0.397 | 0.09 | 0.09 | 1.03 | 0.48 | 100% |
| combined | fixed1s | 38.0 | 0.679 | 1.42 | 1.42 | 0.00 | 0.00 | 0% |
| combined | fixed2s | 37.7 | 0.724 | 2.62 | 2.62 | 0.00 | 0.00 | 0% |
| combined | naive | 21.6 | 0.755 | 2.01 | 2.62 | 51.83 | 41.63 | 0% |
| combined | adaptive | 0.6 | 0.490 | 0.25 | 0.95 | 1.50 | 1.22 | 100% |

Reading (simulated only):
- Fixed 1 s / 2 s miss 25–58 % under overload, loss and flapping; adaptive 0.5–2.8 %. Naive adaptive (no hysteresis) cuts misses but switches 7–82 times/min, audibly unusable. Adaptive stays at 1.0–1.7 switches/min.
- **Cost:** adaptive spends much time on weak tiers (quality 0.40–0.67 vs 0.86–0.92 for fixed in `steady`) because it climbs slowly. Dwell 8 s raised steady quality (0.72) but overload misses rose to 17 % (tried once, not kept); 14 s is a middle point, not an optimum, tuned on the same synthetic seeds. No real heldout audio or source groups were used.
- Fixed policies still miss 3–4 % in `steady` due to warmup.
- `flap` stays at 2.8 % misses for adaptive.

## Demo-safe rollback
Before the demo, pin T3/T4 (`rollback()`) or use a fixed config with measured slack; the Natural fast path (≤100 ms) is the safe default. Auto-upgrade into Tier 1 only when risk < 5 % for the dwell and nobody has pinned.

## Reproduce / consume
```
.venv/bin/python experiments/theory/T-252/latency_controller.py --seeds 20 --out "$HEARING_RESULTS_DIR/overnight_prep/T-252"
```
Writes `cases/<scenario>__<policy>.json` (atomic, per case, skipped on resume when the config hash — constants + seeds + sha256 of the script — matches) and `summary.json` (config hash, seeds, planned/ran/skipped counts). Caps: ≤64 cases, ≤1 MiB output, seeds ≤50, deterministic end. Inputs: synthetic only (0 input files). The config hash is in `summary.json` and changes with any edit to the script. To consume real data, replace `make_trace` with recorded `(t, load, rtt, lost)` rows. Tests: `tests/test_theory_latency_controller.py`.

## Open questions
Listening threshold for delay jumps; whether time-stretch crossfades make tier switches inaudible; real RTT/loss statistics; whether one shared local load multiplier is valid when DFN and TSE contend differently.
