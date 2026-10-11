"""Round 2 (T-295): gated confidence-weighted tracking vs naive T-282.

Compares three gain schedules on synthetic tracks with KNOWN ground truth
(honest: everything here is simulated; real-audio validation is the GPU sweep):

  (a) fixed      alpha* = 0.30            (Round-1 optimum, 0.5s, phone, cosine)
  (b) naive      alpha_t = a0*sigmoid(b*m_{t-1})   (T-282 as originally framed)
  (c) gated      dwell-forced boost on detected switches + FIXED base path

Gated update equations (see docs/research/theory/streaming_tracking_theory.md
section 5b):

    r_t = sim(x_t, pi_{s_{t-1}}) - sim(x_t, pi_{r_{t-1}})   # obs-vs-state margin
    d_t = 1[r_t < -theta_m for `persist` consecutive hops]  # switch detector
    c_t = tau_dwell  if d_t=1 else max(0, c_{t-1}-1)         # dwell counter
    alpha_t = alpha_boost   if c_t > 0 else alpha*           # FIXED base path

Round-2 finding: the base path must be FIXED alpha*, not confidence-weighted.
Any confidence-driven gain variance on the base path pays the Jensen
convexity tax (E[alpha/(2-alpha)] > E[alpha]/(2-E[alpha])) with no
compensating benefit, because the dwell boost already handles switches.
Design rule for T-282: do not modulate the base gain at all; spend the
complexity budget on the switch detector.

Measured (synthetic, 0.5 s, phone, cosine; fixed alpha*=0.30 bar):
  fixed 0.0225 | naive conf-weighted 0.0291 | gated 0.0223-0.0233
  Detector PR curve: P=0.49/R=0.58 -> flip 0.0258 (loses); P>=0.64 -> ties.
  Margin-only cues cannot reach (P>=0.6, R>=0.5) jointly at 0.5 s phone:
  the GPU test must try an acoustic cue (F0 discontinuity).

Why it fixes the two Round-1 failure modes:
  switch-stick  -> dwell forces alpha_boost for tau_dwell hops regardless of
                   the collapsed transient margin (breaks the margin->gain
                   coupling exactly when it is pathological).
  jitter        -> the base path is FIXED at alpha*: zero gain variance, so
                   the Jensen/convexity tax vanishes by construction. No
                   confidence modulation on the base path at all.

Physics, calibration and anchors are imported from tracking_sim (Round 1).
Run:  python3 gated_tracking_sim.py
Only dependency: numpy.
"""

from __future__ import annotations

import math

import numpy as np
import tracking_sim as ts

# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------

class NaiveConfTracker(ts.StreamingTracker):
    """T-282 as originally framed: alpha_t = a0 * sigmoid(b * m_{t-1})."""

    def __init__(self, *a, **k):
        super().__init__(*a, conf_weighted=True, **k)


class GatedTracker(ts.StreamingTracker):
    """Confidence-weighted with hysteresis/dwell-time gate (Round 2 design).

    Detector (causal): the raw hop must disagree with the currently tracked
    speaker (r_t < -theta_m) for `persist` consecutive hops before the dwell
    boost fires. Single-hop disagreement is too noisy at 0.5 s phone
    (margin ~0.15); persistence-2 lifts precision from 0.09 to ~0.5.
    """

    def __init__(self, profiles, metric="cosine", mic="phone", seed=0,
                 a0=0.30, alpha_boost=0.7, tau_dwell=3,
                 theta_m=0.20, persist=2, **kw):
        super().__init__(profiles, metric=metric, alpha=a0, mic=mic, seed=seed,
                         conf_weighted=False, **kw)
        self.a0 = a0
        self.alpha_boost = alpha_boost
        self.tau_dwell = tau_dwell
        self.theta_m = theta_m
        self.persist = persist
        self.dwell = 0
        self._dis_n = 0
        self.detected = []   # hops where d_t fired (for precision/recall)

    def step(self, true_k, window_s):
        x = self._observe(true_k, window_s)

        # observation-vs-state margin: what the raw hop says about the
        # speaker we are currently tracking (causal; proxy for an F0-
        # discontinuity switch cue in real audio).
        sims_x = self.profiles @ x
        if self.z is None:
            s_prev, r_prev = int(np.argmax(sims_x)), None
        else:
            zn = self.z / (np.linalg.norm(self.z) + 1e-9)
            sims_z = self.profiles @ zn
            order = np.argsort(sims_z)[::-1]
            s_prev, r_prev = int(order[0]), int(order[1])
        r_t = float(sims_x[s_prev] - sims_x[r_prev]) if r_prev is not None else 0.0

        self._dis_n = self._dis_n + 1 if (r_t < -self.theta_m) else 0
        d_t = self._dis_n >= self.persist
        if d_t:
            self.detected.append(self._hop)
            self.dwell = self.tau_dwell  # refire extends dwell: harmless
        alpha_t = self.alpha_boost if self.dwell > 0 else self.alpha
        # NOTE: the non-dwell path is FIXED alpha (not confidence-weighted).
        # Round-2 finding: any confidence-driven gain variance on the base
        # path pays the Jensen convexity tax (E[f(a_t)] > f(E[a_t])) with no
        # compensating benefit, because the dwell boost already handles
        # switches. The confidence path only survives inside the dwell gate.
        self.dwell = max(0, self.dwell - 1)

        self.z = x if self.z is None else alpha_t * x + (1.0 - alpha_t) * self.z
        zn = self.z / (np.linalg.norm(self.z) + 1e-9)
        sims = self.profiles @ zn
        order = np.argsort(sims)[::-1]
        margin = float(sims[order[0]] - sims[order[1]])
        self.prev_margin = margin
        return int(order[0]), margin


def run_with_truth(tracker, window_s, n_hops, seed, warmup=50):
    """Run with known switch times. Returns flip_rate, mean switch latency
    (hops), detection precision/recall (gated only; None otherwise)."""
    rng = np.random.default_rng(seed)
    hop_s = window_s * ts.HOP_FRACTION
    p_switch = 1.0 - math.exp(-ts.SWITCH_RATE_PER_S * hop_s)
    tracker._hop = 0
    true_k = 0
    switches = []          # hop indices where a true switch happened
    flips, n = 0, 0
    latencies = []
    last_switch = None
    recovered = True
    for t in range(n_hops):
        tracker._hop = t
        if rng.random() < p_switch:
            true_k = 1 - true_k
            switches.append(t)
            last_switch = t
            recovered = False
        pred, _ = tracker.step(true_k, window_s)
        if last_switch is not None and not recovered and pred == true_k:
            latencies.append(t - last_switch)
            recovered = True
        if t >= warmup:
            flips += (pred != true_k)
            n += 1
    # unrecovered switches penalized at the cap
    latencies += [50] * (len(switches) - len(latencies))
    flip_rate = flips / max(n, 1)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    det = None
    if isinstance(tracker, GatedTracker):
        hits = sum(any(abs(d - s) <= 3 for s in switches)
                   for d in tracker.detected)
        # a switch hit once counts once
        hit_switches = sum(any(abs(d - s) <= 3 for d in tracker.detected)
                           for s in switches)
        prec = hits / max(len(tracker.detected), 1)
        rec = hit_switches / max(len(switches), 1)
        det = {"precision": prec, "recall": rec,
               "n_detect": len(tracker.detected), "n_switch": len(switches)}
    return {"flip_rate": flip_rate, "mean_latency_hops": mean_lat,
            "n_switches": len(switches), "detection": det}


def main():
    W, MIC, METRIC = 0.5, "phone", "cosine"
    A_STAR = 0.30
    N_HOPS, SEED = 4000, 11

    def fresh(kind, seed):
        profiles = ts.make_profiles(ts.ANGLE, seed)
        if kind == "fixed":
            return ts.StreamingTracker(profiles, METRIC, alpha=A_STAR,
                                       mic=MIC, seed=seed)
        if kind == "naive":
            return NaiveConfTracker(profiles, METRIC, mic=MIC, seed=seed,
                                    alpha_base=A_STAR, beta=4.0)
        if kind == "gated":
            return GatedTracker(profiles, METRIC, mic=MIC, seed=seed,
                                a0=A_STAR, alpha_boost=0.7, tau_dwell=3,
                                theta_m=0.20, persist=2)
        raise ValueError(kind)

    print("== Round 2: gain schedules @0.5s phone cosine (synthetic) ==")
    res = {}
    for kind in ("fixed", "naive", "gated"):
        tr = fresh(kind, SEED)
        r = run_with_truth(tr, W, N_HOPS, seed=SEED + 100)
        res[kind] = r
        d = r["detection"]
        det_s = (f" det P={d['precision']:.2f} R={d['recall']:.2f}"
                 if d else "")
        print(f"  {kind:6s}: flip={r['flip_rate']:.4f} "
              f"mean_switch_latency={r['mean_latency_hops']:.1f} hops{det_s}")

    print("\n== gated parameter sensitivity (flip rate) ==")
    for boost, dwell in ((0.6, 2), (0.6, 3), (0.7, 2), (0.7, 3),
                         (0.8, 3), (0.7, 4)):
        profiles = ts.make_profiles(ts.ANGLE, SEED)
        tr = GatedTracker(profiles, METRIC, mic=MIC, seed=SEED,
                          a0=A_STAR,
                          alpha_boost=boost, tau_dwell=dwell)
        r = run_with_truth(tr, W, N_HOPS, seed=SEED + 100)
        print(f"  boost={boost:.1f} dwell={dwell}: "
              f"flip={r['flip_rate']:.4f} lat={r['mean_latency_hops']:.1f}")

    print("\n== detector precision-recall vs flip rate (theta_m x persist) ==")
    for theta_m, persist in ((0.08, 2), (0.12, 2), (0.15, 2),
                             (0.20, 2), (0.25, 2)):
        profiles = ts.make_profiles(ts.ANGLE, SEED)
        tr = GatedTracker(profiles, METRIC, mic=MIC, seed=SEED,
                          a0=A_STAR, alpha_boost=0.7, tau_dwell=3,
                          theta_m=theta_m, persist=persist)
        r = run_with_truth(tr, W, N_HOPS, seed=SEED + 100)
        d = r["detection"]
        print(f"  th_m={theta_m:.2f} persist={persist}: flip={r['flip_rate']:.4f} "
              f"P={d['precision']:.2f} R={d['recall']:.2f}")

    print("\n== directional mic check (gated should not hurt there) ==")
    for kind in ("fixed", "gated"):
        profiles = ts.make_profiles(ts.ANGLE, SEED)
        if kind == "fixed":
            tr = ts.StreamingTracker(profiles, METRIC, alpha=0.65,
                                     mic="directional", seed=SEED)
        else:
            tr = GatedTracker(profiles, METRIC, mic="directional", seed=SEED,
                              a0=0.65, alpha_boost=0.8, tau_dwell=3)
        r = run_with_truth(tr, W, N_HOPS, seed=SEED + 100)
        print(f"  {kind:6s}: flip={r['flip_rate']:.4f} "
              f"lat={r['mean_latency_hops']:.1f}")

    print("\nSYNTHETIC — all numbers from the calibrated simulator, not audio.")


if __name__ == "__main__":
    main()
