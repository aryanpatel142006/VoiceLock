"""T-252 offline latency/quality adaptive controller prototype (theoretical, synthetic traces).

No runtime imports, no audio, no network, no GPU. Every constant below is an ASSUMPTION
(not a measurement) and is hashed into the output so results are replayable.

    .venv/bin/python experiments/theory/T-252/latency_controller.py --seeds 20 --out <dir>

Cases = scenario x policy; each case is checkpointed atomically and skipped on resume when
its config hash matches. Stops deterministically after all cases (no repeats).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

TICK = 0.5  # control tick = hop of the heavy tier (s); one window result per tick
DURATION = 180.0
IO_S = 0.02  # capture + output buffers (s), assumption from ARCHITECTURE budget


@dataclass(frozen=True)
class Cfg:
    name: str
    window: float  # algorithmic window (s); heavy tiers must buffer a full window
    rtf: float  # assumed compute real-time factor at load 1.0
    remote: bool
    slack: float  # scheduled playout slack that absorbs compute+network (s)
    quality: float  # assumed relative quality 0..1


LADDER = [
    Cfg("T1_W2", 2.0, 0.15, True, 0.60, 0.95),
    Cfg("T1_W1", 1.0, 0.15, True, 0.40, 0.88),
    Cfg("T2_W1", 1.0, 0.30, False, 0.50, 0.75),
    Cfg("T3_gate", 0.04, 0.10, False, 0.03, 0.50),
    Cfg("T4_dfn", 0.04, 0.10, False, 0.03, 0.35),
]
FALLBACK_QUALITY = LADDER[-1].quality  # missed window is replaced by always-on local DFN shadow (assumption)
SAFE = len(LADDER) - 1


def m2e(c: Cfg) -> dict:
    """Mouth-to-ear split: algorithmic (window+IO), scheduled compute+network slack."""
    return {"algorithmic": c.window + IO_S, "slack": c.slack, "total": c.window + IO_S + c.slack}


# ---------------------------------------------------------------- traces
def make_trace(scenario: str, seed: int) -> list[dict]:
    rng = random.Random(f"{scenario}:{seed}")
    n = int(DURATION / TICK)
    out = []
    for i in range(n):
        t = i * TICK
        load = 1.0
        rtt = 0.06
        lost = False
        if t < 10:  # warmup: cold model/JIT inflation decaying
            load *= 1 + 4 * math.exp(-t / 4)
        if scenario == "overload" and 60 <= t < 100:
            load *= 4.5
        if scenario == "remote_loss":
            if 60 <= t < 80:
                lost = True
            if 110 <= t < 140 and rng.random() < 0.35:
                lost = True
            if 140 <= t < 150:
                rtt = 0.5
        if scenario == "flap":  # load oscillating around the W2 deadline
            load *= 2.2 + 1.6 * math.sin(2 * math.pi * t / 14)
        if scenario == "combined":
            if 50 <= t < 80:
                load *= 4.0
            if 90 <= t < 105:
                lost = True
            if 130 <= t < 150:
                rtt = 0.4
        jit = math.exp(rng.gauss(0, 0.2))
        out.append({"t": t, "load": load * jit, "rtt": rtt * jit, "lost": lost})
    return out


SCENARIOS = ["steady", "overload", "remote_loss", "flap", "combined"]


def window_latency(c: Cfg, s: dict) -> float:
    """Compute + network time for one window (inf if remote response lost)."""
    if c.remote and s["lost"]:
        return math.inf
    return c.rtf * c.window * s["load"] + (s["rtt"] if c.remote else 0.0)


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------- policies
class Fixed:
    def __init__(self, idx: int, name: str):
        self.idx, self.name = idx, name

    def reset(self):
        pass

    def observe(self, cfg_idx, lat, s, t):
        pass

    def pick(self, t):
        return self.idx

    def rollback(self):
        pass


class Naive:
    """Adaptive without hysteresis: step down on any miss, step up on any hit."""

    name = "naive"

    def reset(self):
        self.i = 0
        self.last = None

    def observe(self, cfg_idx, lat, s, t):
        miss = lat > LADDER[cfg_idx].slack
        self.i = min(self.i + 1, SAFE) if miss else max(self.i - 1, 0)

    def pick(self, t):
        return self.i

    def rollback(self):
        pass


class Adaptive:
    """Risk-based controller. Telemetry: per-resource load EWMA (local/remote), heartbeat RTT+loss
    for the remote (assumed cheap and always available), dwell/hysteresis, upgrade backoff,
    churn breaker that pins a safe tier (demo-safe rollback)."""

    name = "adaptive"
    DOWN_RISK = 0.20
    UP_RISK = 0.05
    UP_RESET_RISK = 0.12  # hysteresis band: blips between UP_RISK and this do not restart the dwell clock
    DWELL_UP = 14.0  # s of sustained low risk before upgrading
    DWELL_MIN = 4.0  # s minimum between any switches
    BREAKER_N, BREAKER_WIN, BREAKER_HOLD = 4, 60.0, 60.0
    WARMUP_HOLD = 10.0  # do not trust load estimates before this; start on local safe-ish tier

    def reset(self):
        self.i = SAFE - 2  # start on T2 (local, bounded), earn Tier 1 after warmup
        self.load = {False: 1.0, True: 1.0}
        self.var = {False: 0.04, True: 0.04}
        self.hist = {False: deque(maxlen=12), True: deque(maxlen=12)}
        self.rtt = 0.06
        self.lost = deque(maxlen=6)
        self.last_switch = -1e9
        self.good_since = None
        self.up_backoff = self.DWELL_UP
        self.switches: deque = deque()
        self.pinned_until = -1.0
        self.pin_idx = SAFE
        self.last_up_t = -1e9
        self.consec_miss = 0

    def risk(self, idx: int) -> float:
        c = LADDER[idx]
        lost = sum(self.lost) / len(self.lost) if self.lost and c.remote else 0.0
        mu = c.rtf * c.window * self.load[c.remote] + (self.rtt if c.remote else 0.0)
        sd = max(0.25, math.sqrt(self.var[c.remote]))  # relative spread
        p_late = 1 - phi((math.log(c.slack) - math.log(max(mu, 1e-6))) / sd)
        return min(1.0, lost + (1 - lost) * p_late)

    def observe(self, cfg_idx, lat, s, t):
        c = LADDER[cfg_idx]
        # heartbeat telemetry (assumed always available, also while the remote is idle): loss flag + RTT
        self.lost.append(1.0 if s["lost"] else 0.0)
        if not s["lost"]:
            self.rtt = 0.7 * self.rtt + 0.3 * s["rtt"]
        if math.isfinite(lat):
            m = max(1e-3, (lat - (s["rtt"] if c.remote else 0.0)) / (c.rtf * c.window))
            h = self.hist[c.remote]
            h.append(math.log(m))
            med = statistics.median(h)
            recent = statistics.fmean(list(h)[-2:])  # fast attack on sudden load
            self.load[c.remote] = math.exp(max(med, recent))
            self.var[c.remote] = statistics.pvariance(h) if len(h) > 3 else 0.04
        miss = lat > c.slack
        self.consec_miss = self.consec_miss + 1 if miss else 0

    def rollback(self):
        self.pinned_until = float("inf")

    def pick(self, t):
        if t < self.pinned_until:
            return self.pin_idx
        if self.pinned_until == float("inf"):
            return self.pin_idx
        i = self.i
        if t - self.last_switch < self.DWELL_MIN:
            return i
        r = self.risk(i)
        new = i
        if r > self.DOWN_RISK or self.consec_miss >= 2:
            new = min(i + 1, SAFE)
            while new < SAFE and self.risk(new) > self.DOWN_RISK:
                new += 1
            if t - self.last_up_t < 30:  # a failed upgrade: back off exponentially
                self.up_backoff = min(self.up_backoff * 2, 160.0)
            self.good_since = None
        elif i > 0 and t >= self.WARMUP_HOLD:
            cand = i - 1
            rc = self.risk(cand)
            if rc < self.UP_RESET_RISK and r < self.UP_RESET_RISK:
                if self.good_since is None and rc < self.UP_RISK:
                    self.good_since = t
                if self.good_since is not None and t - self.good_since >= self.up_backoff:
                    new = cand
                    self.good_since = None
                    self.last_up_t = t
            else:
                self.good_since = None
        if new != i:
            self.i = new
            self.last_switch = t
            self.switches.append(t)
            while self.switches and t - self.switches[0] > self.BREAKER_WIN:
                self.switches.popleft()
            if len(self.switches) >= self.BREAKER_N:  # churn breaker
                self.pinned_until = t + self.BREAKER_HOLD
                self.pin_idx = max(self.i, 2)
                self.i = self.pin_idx
                self.switches.clear()
        return self.i


def make_policies():
    return {
        "fixed1s": Fixed(1, "fixed1s"),
        "fixed2s": Fixed(0, "fixed2s"),
        "naive": Naive(),
        "adaptive": Adaptive(),
    }


# ---------------------------------------------------------------- simulation
def simulate(policy, trace: list[dict]) -> dict:
    policy.reset()
    prev = None
    misses = quality_sum = jump_s = 0.0
    switches = 0
    delays, tier_time = [], {}
    fallback_ticks = 0
    for s in trace:
        t = s["t"]
        idx = policy.pick(t)
        c = LADDER[idx]
        lat = window_latency(c, s)
        miss = lat > c.slack
        if prev is not None and idx != prev:
            switches += 1
            jump_s += abs(m2e(c)["total"] - m2e(LADDER[prev])["total"])
        prev = idx
        policy.observe(idx, lat, s, t)
        misses += miss
        quality_sum += FALLBACK_QUALITY if miss else c.quality
        delays.append(m2e(c)["total"])
        tier_time[c.name] = tier_time.get(c.name, 0) + TICK
        fallback_ticks += idx >= 3
    n = len(trace)
    mins = n * TICK / 60
    delays_sorted = sorted(delays)
    return {
        "miss_rate": misses / n,
        "mean_quality": quality_sum / n,
        "m2e_mean_s": statistics.fmean(delays),
        "m2e_p95_s": delays_sorted[int(0.95 * (n - 1))],
        "switches_per_min": switches / mins,
        "delay_jump_s_per_min": jump_s / mins,
        "fallback_share": fallback_ticks / n,
        "tier_seconds": tier_time,
    }


QOS = {"miss_rate_max": 0.05, "switches_per_min_max": 2.0, "m2e_p95_max_s": 3.0}


def qos_pass(m: dict) -> bool:
    return (m["miss_rate"] <= QOS["miss_rate_max"] and m["switches_per_min"] <= QOS["switches_per_min_max"]
            and m["m2e_p95_s"] <= QOS["m2e_p95_max_s"])


def run_case(scenario: str, pname: str, seeds: list[int]) -> dict:
    runs = [simulate(make_policies()[pname], make_trace(scenario, s)) for s in seeds]
    keys = [k for k in runs[0] if k != "tier_seconds"]
    mean = {k: statistics.fmean(r[k] for r in runs) for k in keys}
    return {"scenario": scenario, "policy": pname, "n_seeds": len(seeds), "mean": mean,
            "qos_pass_share": sum(qos_pass(r) for r in runs) / len(runs),
            "worst_miss_rate": max(r["miss_rate"] for r in runs)}


def config_hash(seeds: list[int]) -> str:
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    blob = json.dumps({"code_sha256": code, "ladder": [c.__dict__ for c in LADDER], "tick": TICK, "dur": DURATION, "io": IO_S,
                       "qos": QOS, "seeds": seeds, "scenarios": SCENARIOS,
                       "adaptive": {k: getattr(Adaptive, k) for k in
                                    ("DOWN_RISK", "UP_RISK", "UP_RESET_RISK", "DWELL_UP", "DWELL_MIN", "BREAKER_N",
                                     "BREAKER_WIN", "BREAKER_HOLD", "WARMUP_HOLD")}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def atomic_write(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


MAX_CASES = 64
MAX_BYTES = 1 << 20


def run(out: Path, n_seeds: int, base_seed: int = 252) -> dict:
    seeds = [base_seed + i for i in range(n_seeds)]
    h = config_hash(seeds)
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    plan = [(s, p) for s in SCENARIOS for p in make_policies()]
    assert len(plan) <= MAX_CASES
    results, ran, skipped = [], 0, 0
    for sc, pn in plan:
        f = cases_dir / f"{sc}__{pn}.json"
        if f.exists():
            try:
                d = json.loads(f.read_text())
                if d.get("config_hash") == h and d.get("status") == "complete":
                    results.append(d["result"])
                    skipped += 1
                    continue
            except (OSError, ValueError):
                pass
        res = run_case(sc, pn, seeds)
        atomic_write(f, {"config_hash": h, "status": "complete", "result": res})
        results.append(res)
        ran += 1
    summary = {"task": "T-252", "config_hash": h, "seeds": seeds, "inputs": "synthetic only (no input files)",
               "ladder_m2e": {c.name: m2e(c) for c in LADDER}, "qos": QOS,
               "cases_planned": len(plan), "cases_ran": ran, "cases_skipped_resume": skipped,
               "status": "complete", "results": results}
    atomic_write(out / "summary.json", summary)
    assert sum(p.stat().st_size for p in out.rglob("*.json")) < MAX_BYTES
    return summary


def table(summary: dict) -> str:
    rows = ["| scenario | policy | miss % | quality | m2e mean s | m2e p95 s | switches/min | jump s/min | QoS pass |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in summary["results"]:
        m = r["mean"]
        rows.append(f"| {r['scenario']} | {r['policy']} | {100*m['miss_rate']:.1f} | {m['mean_quality']:.3f} | "
                    f"{m['m2e_mean_s']:.2f} | {m['m2e_p95_s']:.2f} | {m['switches_per_min']:.2f} | "
                    f"{m['delay_jump_s_per_min']:.2f} | {100*r['qos_pass_share']:.0f}% |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.seeds < 1 or a.seeds > 50:
        ap.error("--seeds must be 1..50")
    out = a.out or Path(os.environ.get("HEARING_RESULTS_DIR", tempfile.gettempdir())) / "overnight_prep" / "T-252"
    s = run(out, a.seeds)
    print(table(s))
    print(f"\nconfig_hash={s['config_hash']} ran={s['cases_ran']} skipped={s['cases_skipped_resume']} out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
