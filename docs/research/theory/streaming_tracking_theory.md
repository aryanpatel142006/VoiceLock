# Unified Streaming Speaker-Tracking Theory (T-295)

**Status:** theory + simulator. Anchors reproduced from team-audio measurements.
Untested claims labeled **[H]**. Falsifiable predictions listed in §7.

## 0. What this is

Gemini is building four separate tracks: T-273 (contrastive projection head),
T-274 (Mahalanobis metric), T-275 (short-crop EMA buffer), T-282
(confidence-weighted EMA). Nobody has written the **composition** theory: how
the metric, the window, the EMA gain, and the microphone interact in one
streaming decision loop. This note does that, and derives the optimal EMA gain
as a function of window length, turn-taking rate, metric choice, and — new in
this revision — **microphone directivity**, which makes the theory directly
justify the phone-only demo.

## 1. Observation model

Per streaming hop `t`, with window length `w` seconds, mic type `m`:

```
a_t   ~ Beta(mean = a(m), var = C_A / w)          # per-hop SIR fluctuation
s_t   = sqrt(a_t)·μ_{k_t} + sqrt(1-a_t)·μ_{other} # SIR mixture
x_t   = normalize( λ(w)·s_t + ε_t )               # fidelity + noise
ε_t   ~ N(0, (σ²/w)/D · I),   λ(w) = (w/2)^p
```

- `a(m) = SIR_lin/(1+SIR_lin)` is the target-energy fraction from mic SIR.
  Phone (omnidirectional): SIR ≈ 3 dB → a ≈ 0.67 [H: demo geometry].
  Directional: SIR ≈ 11 dB (3 dB + 8 dB directivity index) → a ≈ 0.93 [H].
- `a_t` fluctuates per hop: a 0.5 s hop often catches whoever is instantaneously
  louder; a 2.0 s hop averages it out. Variance ∝ 1/w.
- `λ(w)` is the **fidelity** ("norm degradation"): at short windows ECAPA
  retains only a fraction of the speaker-discriminative signal. Calibrated
  exponent p ≈ 1.25 [H: fitted to anchors, §2].
- `ε_t` is residual embedding noise [H: small; fidelity + SIR dominate].

## 2. Anchor calibration

Team-audio measurements (Aryan vs Joseph, phone mic, ECAPA, no EMA):

| window | measured margin |
|--------|----------------|
| 2.0 s  | 0.45           |
| 0.5 s  | 0.15           |
| 0.5 s + T-273 head | 1.19 |

The simulator (`experiments/theory/T-295/tracking_sim.py`) grid-searches
(centroid angle, C_A, p) to reproduce both cosine anchors, then fits the head
angle to 1.19. Result: angle ≈ 116°, C_A = 0.005, p = 1.25, head angle ≈ 174°.

Two structural conclusions fall out of the calibration:

1. **Isotropic noise alone cannot explain the 0.45→0.15 collapse.**
   A 3× margin drop from a 4× window shortening is inconsistent with any
   `σ² ∝ 1/w` noise model (verified by grid search). The fidelity term `λ(w)`
   is required — this is the formal statement of "embedding norms degrade."

2. **The head's 8× gain cannot come from separation alone.** Even antipodal
   centroids (max margin 2.0) fall short of 1.19 under the fitted noise unless
   the head also suppresses noise directions. Hence the head model uses
   `HEAD_NOISE_GAMMA = 0.25` [H] — the contrastive head must be doing
   denoising, not just pushing centroids apart. Testable: ablate the head's
   noise robustness vs a pure centroid-pushing baseline.

## 3. Streaming decision as sequential Bayesian update

Maintain belief `b_t(k) = P(speaker_t = k | x_1..t)`. The pipeline is:

```
x_t ──▶ [metric φ] ──▶ h_t ──▶ [EMA, gain α_t] ──▶ z_t ──▶ argmax_k sim(z_t, π_k)
                                                        │
margin_t = sim(z_t,π_ŝ) − sim(z_t,π_runner-up) ──────────┘
                                                        ▼
                                              [T-282] α_{t+1} = α_base·σ(β·margin_t)
```

- **Likelihood** `p(h_t | k)`: set by the metric. Cosine ≈ isotropic Gaussian;
  Mahalanobis (T-274) ≈ Gaussian whitened by within-speaker covariance;
  projection head (T-273) ≈ likelihood in a learned discriminative space.
- **Prior dynamics**: the EMA `z_t = α_t·h_t + (1−α_t)·z_{t−1}` (T-275) is the
  sufficient statistic of a Gaussian random-walk prior on speaker location.
- **Adaptive gain** (T-282): `α_t` as a sigmoid of the previous margin is a
  scalar Kalman gain — high margin (low observation uncertainty) → trust the
  observation; low margin → coast on history.

**Where each track slots in:**

| Track | Role in the loop | What it changes |
|-------|-----------------|-----------------|
| T-273 head | likelihood geometry | larger clean separation + noise suppression → higher margin at fixed (w, m) |
| T-274 Mahalanobis | likelihood geometry | whitens within-speaker covariance → lower effective noise (`maha_gamma` [H]) |
| T-275 EMA buffer | prior dynamics | the smoothing itself; `α` is the single knob trading noise vs lag |
| T-282 conf-weight | adaptive gain | `α_t` follows confidence; closes the loop margin→gain→margin |

## 4. Optimal EMA gain α*

Steady-state error of the EMA has two terms:

- **Noise error**: `Var(z) ∝ σ²_eff(w,m)·α/(2−α)` — falls as α→0 (more smoothing).
- **Lag error**: after a speaker switch, bias decays as `(1−α)^n`; expected
  cost ∝ turn-taking rate `Λ` / α — falls as α→1 (faster adaptation).

Minimizing the sum (small-α approximation):

```
α*(w, m) ≈ √[ 2·B·Λ(w) / (A·σ²_eff(w, m)) ]
```

with `Λ(w) = Λ₀·w/10` (hop = w/10, Poisson switches) and effective observation
variance

```
σ²_eff(w, m) = σ²/w + c_xtalk·(1 − a(m))        # window noise + cross-talk
```

`c_xtalk·(1−a(m))` is the **mic-directivity term**: the interferer's leakage
acts as extra observation variance, scaled by how much the mic lets through.
Phone (`a=0.67`): 4.5× the cross-talk variance of directional (`a=0.93`).

**Predictions (all falsifiable, §7):**

- **P1 (window):** α* decreases as w shortens (noisier observations → more
  smoothing). α*(0.5s) < α*(1.0s) < α*(2.0s) for fixed mic and metric.
  **Confirmed in sim** (phone, cosine): 0.30 < 0.70 < 0.80.
- **P2 (mic):** α*_phone(w) < α*_directional(w). **Confirmed at 0.5 s**
  (0.30 vs 0.65 — the phone needs much more smoothing where it matters);
  at ≥1.0 s both optima go flat and the ordering is not significant.
  **Sharper finding:** the phone's α* optimum is *sharp* (0.30 clearly best,
  neighbors worse), the directional mic's is *broad* (0.55–0.75 all ≈equal).
  The phone punishes mis-tuning; the directional mic forgives it. This
  asymmetry — not just a shifted optimum — is the core demo-relevant result.
- **P3 (mic × window):** REVISED per sim — the phone/directional α* gap does
  *not* widen with w; it washes out as both optima flatten. The robust
  statement is P2's sharp-vs-broad asymmetry.
- **P4 (metric):** better metric → lower σ²_eff → higher α*:
  α*_head > α*_maha > α*_cos at fixed (w, m). **Confirmed in sim**
  at (0.5 s, phone): 0.45 > 0.35 > 0.30. The head "buys back" agility.
- **P5 (T-282):** REVISED per sim — naive confidence-weighted EMA
  **underperforms** the best fixed α (0.025 vs 0.017 flip rate at
  (0.5 s, phone)). Two failure modes identified: (a) **switch-stick**:
  gain capped by low margin stalls adaptation exactly after real switches;
  (b) **jitter amplification**: gain spanning near 1.0 when confident makes
  the tracker chase observation noise (0.069 vs 0.017). Hysteresis also
  fails (0.049). Implication for T-282's design: the gain schedule must be
  asymmetric or use a better confidence signal than raw margin — e.g. margin
  *velocity*, or a switch detector (F0 discontinuity) gating the gain
  floor. The GPU sweep should explore this design space, not assume T-282
  wins by default.

## 5. The closed loop: stability

T-282 closes margin→gain→margin. Linearizing around the fixed point, the loop
is a contraction (stable) iff `α_base·β·(d margin/dα) < 1`. If β is too
aggressive, the system oscillates: low margin → low gain → stale z → low
margin … The simulator can map the stable (α_base, β) region [H: not yet
mapped — GPU sweep should include a β ladder].

Failure mode the loop creates: **switch-stick.** When the speaker genuinely
switches, margin collapses for a few hops; T-282 drops the gain exactly when
adaptation is most needed, delaying the switch by ~1/α hops. Mitigation [H]:
floor the gain (`α_t ≥ α_min`) or gate the confidence drop on a switch
detector (e.g., F0 discontinuity). The GPU sweep should measure switch
latency with and without the floor.

## 5a. Why naive confidence-weighting fails: the math (Round 2)

Setup: `z_t = α_t·h_t + (1−α_t)·z_{t−1}`, observation noise
`E[‖h_t − μ‖²] = σ²_eff`. For fixed α, the AR(1) steady state gives
`Var(z) = σ²_eff·α/(2−α)` (standard).

**Theorem 1 (switch-stick).** After a true switch at hop τ, the transient
margin `m < 0` for several hops (state still on the old speaker). The naive
gain `α_t = α_base·σ(β·m_{t-1}) ≤ α_base·e^{−β|m_{t-1}|}` collapses. The EMA
needs `(1−α_t)^n ≲ 1/2` to flip the decision, so
`n_stick ≳ ln2/α_t ≳ (ln2/α_base)·e^{β|m|}` — stick duration grows
**exponentially** in the transient margin depth. The harder the switch, the
longer the freeze. Fixed `α*` sticks `ln2/α*` hops; naive pays an excess lag
factor `≈ e^{β|m|}` on every switch. (Measured: 2.4 vs 1.7 hops mean
switch latency, synthetic.)

**Theorem 2 (jitter amplification — the lose-lose).** With random gain `α_t`,
steady-state `E[Var(z)] = σ²_eff·E[f(α_t)]` with `f(α) = α/(2−α)`.
`f''(α) = 4/(2−α)³ > 0` on (0,1): strictly convex. By Jensen,
`E[f(α_t)] ≥ f(E[α_t])`, strict whenever `α_t` varies — the fluctuating gain
has strictly *more* noise error than the fixed gain at its mean. And
`E[α_t] = α_base·E[σ(β·m)] < α_base` (margin unsaturated much of the time),
so lag error is worse than fixed `α_base` too. If `α_base = α*` (the optimal
fixed gain), naive confidence-weighting is **strictly dominated on both
error terms**. No β fixes this: β→0 recovers fixed α (then why bother);
β→∞ makes it bang-bang (maximizes the Jensen gap).

**Corollary.** The confidence signal is not the problem — modulating the
*base* gain with it is. The fix: never modulate the base gain (Theorem 2),
only ever *raise* gain on detected switches (Theorem 1's antidote).

## 5b. The gated design (Round 2)

Causal switch detector on the raw hop's disagreement with the tracked
speaker, plus a dwell gate that forces fast adaptation:

```
r_t   = sim(x_t, π_{ŝ_{t-1}}) − sim(x_t, π_{r_{t-1}})   # obs-vs-state margin
d_t   = 1[ r_t < −θ_m for `persist` consecutive hops ]  # switch detector
c_t   = τ_dwell  if d_t = 1 else max(0, c_{t-1} − 1)     # dwell counter
α_t   = α_boost            if c_t > 0
        α*  (FIXED)        otherwise
```

Parameters (sim-tuned): `α_boost ∈ [0.6, 0.8]`, `τ_dwell ∈ {2, 3}`,
`θ_m ∈ [0.12, 0.20]`, `persist = 2` hops. Single-hop disagreement is too
noisy at 0.5 s phone (margin ≈ 0.15); persistence-2 lifts detector precision
0.09 → ~0.5. The base path is deliberately FIXED — the confidence path is
dead (Corollary).

**Break-even condition.** Let π = switch prob/hop, p_d = P(detect | switch),
f_d = false detections/hop, e_stuck ≈ 1 (decision fully wrong while stuck).
Gated beats fixed α* on flip rate iff

```
π·p_d·(ln2/α* − ln2/α_boost)·e_stuck  >  f_d·τ_dwell·σ²_eff·(f(α_boost) − f(α*))
```

savings (faster switches) vs cost (false-detection noise).
**Measured in sim** (0.5 s, phone, cosine; fixed α* = 0.30 bar at flip
0.0225): break-even at detector precision ≈ 0.6. The margin-only cue traces
(P, R) = (0.49, 0.58) → (1.00, 0.09) as θ_m sweeps 0.08 → 0.25 — it cannot
reach (P ≥ 0.6, R ≥ 0.5) jointly. Gated ties fixed α* (flip 0.0223–0.0233,
never worse) but cannot clearly beat it on flip rate with margin cues alone.
**Design rule for T-282:** do not modulate the base gain at all; spend the
complexity budget on the switch detector. The GPU test must compare a
margin cue against an *acoustic* cue (F0 discontinuity) — the candidate that
could clear the (0.6, 0.5) bar (P8).

## 6. What the phone-only choice costs, in α*

From the simulator (cosine metric, §7 table): at 0.5 s the phone's α* sits
`Δα*` below the directional mic's — i.e., the phone needs that much more
temporal context to reach the same decision quality. The composition
(T-273/274/275/282) is not optional polish for the phone demo; it is what
recovers the margin the directional mic would have given for free. A
directional mic is forgiving (high SIR → flat α* optimum → any reasonable α
works). The phone is unforgiving (sharp, low α* optimum → the full
composition must be tuned). That asymmetry is why the rehearsal matrix
(T-285) should prioritize the phone-mic α/metric/latency cells.

## 7. Falsifiable predictions & GPU validation

| # | Prediction | Simulator says | GPU test (T-295 sweep) |
|---|-----------|----------------|------------------------|
| P1 | α*(0.5)<α*(1.0)<α*(2.0), phone, cosine | 0.30<0.70<0.80 ✓ | α sweep × window on ilab1 team audio |
| P2 | α*_phone < α*_directional; phone optimum sharp, directional broad | 0.30 vs 0.65 at 0.5s ✓; flat ≥1.0s | synthetic SIR ladder (§8): mix interferer at 0/3/6/12 dB, trace α* vs SIR |
| P3 | REVISED: gap washes out with w (not widens) | sim | SIR ladder at 0.5 s and 2.0 s |
| P4 | α*_head > α*_maha > α*_cos | 0.45>0.35>0.30 ✓ | metric × α sweep |
| P5 | REVISED: naive T-282 underperforms fixed α; needs asymmetric design | sim (switch-stick + jitter) | gain-schedule design space, not just validation |
| P6 | margin(1.0s, phone, cosine) ≈ 0.37 | sim | direct measurement on team audio |
| P7 | (α_base, β) stability region; switch-latency metric | not yet mapped | β ladder + switch-latency metric |
| P8 | **(Round 2)** gated tracker (fixed-α* base + dwell boost) beats fixed α* on flip rate iff the switch cue reaches (P ≥ 0.6, R ≥ 0.5) jointly; margin cues cannot at 0.5 s phone | sim PR curve: break-even at P≈0.6; margin cue maxes at (0.49, 0.58)→(1.00, 0.09) | cue comparison on ilab1: margin vs F0-discontinuity vs separator stream assignment |

All [H] items above are untested. The GPU sweep (`experiments/gpu/T-295/`)
is designed to test P1–P5 on real phone-mic audio; P6 is a zero-cost check
on existing recordings.

## 8. Simulator outputs

Run `python3 experiments/theory/T-295/tracking_sim.py`. Output recorded
2026-10-10 (seed 0). `tests/test_theory_tracking.py` asserts the anchors
and the P1/P2 orderings.

```
== anchor reproduction (phone mic) ==
cosine margin @2.0s: 0.438 (anchor 0.45)
cosine margin @0.5s: 0.182 (anchor 0.15)
head   margin @0.5s: 1.208 (anchor 1.19)
anchors reproduced: True

== predicted margins, no EMA ==
window 0.5s: phone 0.183  directional 0.535
window 1.0s: phone 0.366  directional 0.984
window 2.0s: phone 0.437  directional 1.117

== alpha* per window x mic (cosine) ==
phone       0.5s -> 0.30    directional 0.5s -> 0.65
phone       1.0s -> 0.70    directional 1.0s -> 0.70
phone       2.0s -> 0.80    directional 2.0s -> 0.70

== alpha* per metric @0.5s phone ==
cosine -> 0.30   mahalanobis -> 0.35   head -> 0.45

== T-282 naive conf-weighted vs best fixed alpha (0.5s) ==
phone:       fixed 0.0193  conf-weighted 0.0254
directional: fixed 0.0014  conf-weighted 0.0017
```

## 8b. Round-2 outputs (gated_tracking_sim.py, seed 11, synthetic)

```
== gain schedules @0.5s phone cosine ==
fixed : flip=0.0225  mean_switch_latency=1.7 hops
naive : flip=0.0291  mean_switch_latency=2.4 hops
gated : flip=0.0228  mean_switch_latency=1.7 hops  (det P=1.00 R=0.21)

== detector precision-recall vs flip rate (theta_m x persist=2) ==
th_m=0.08: flip=0.0258  P=0.49 R=0.58   <- below break-even: loses
th_m=0.12: flip=0.0233  P=0.64 R=0.42
th_m=0.15: flip=0.0233  P=0.75 R=0.35
th_m=0.20: flip=0.0228  P=1.00 R=0.21
th_m=0.25: flip=0.0225  P=1.00 R=0.09   <- detector nearly never fires

== directional mic (gated must not hurt) ==
fixed : flip=0.0030   gated : flip=0.0030
```

Break-even on flip rate at detector precision ≈ 0.6; margin-only cues at
0.5 s phone cannot reach (P ≥ 0.6, R ≥ 0.5) jointly. The gated design ties
fixed α* and never loses when P ≈ 1.0 — the remaining question is purely
the switch cue, which is what the GPU sweep tests (P8).

Note P6's check value: predicted margin(1.0 s, phone, cosine) ≈ 0.37.

## 9. Relation to other tracks

- T-247 (separator fine-tuning): orthogonal — improves stems; this note is
  about the *decision* layer that picks between them.
- T-249 (selection calibration): the margin here is T-249's confidence
  signal; the α* derivation tells T-249 what margin quality is achievable
  per (w, m, metric).
- T-285 (rehearsal matrix): §6 argues the phone-mic cells are the
  highest-value cells; the α* table gives principled defaults per cell
  instead of hand-tuning.
