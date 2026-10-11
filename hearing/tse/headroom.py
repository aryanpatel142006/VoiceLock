"""Compute headroom of the Mac's enrolled tier, before a demo (``scripts/demo.sh --check``).

Builds the ``enrolled`` tier of ``config/tiers.yaml`` exactly as the ladder does, loads it, runs
one cold window, then times ``--windows`` windows of noise against a noise enrollment (the
compute of a window barely depends on the content). Utilisation = compute p50 / the configured
hop; headroom = 1 - utilisation, the same number as ``telemetry.tiers.compute_headroom``. Exits
1 when the headroom is below ``--min`` (default 0.30): the tier would overrun once a browser,
Whisper and MediaPipe share the machine. Exits 0 with a note when the tier is not built here.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

MIN_HEADROOM = 0.30


def measure(windows: int = 6, tier: str = "enrolled") -> dict:
    from hearing.fusion.tiers import LadderConfig, TierLadder, iter_stages
    from hearing.tse.stage import FocusHQ

    cfg = LadderConfig.load()
    if tier not in {t.name for t in cfg.tiers}:
        return {"tier": tier, "built": False, "reason": "not configured"}
    try:
        ladder = TierLadder(cfg.only([tier]))
    except ValueError as e:  # the tier's only pipeline did not build (devices, weights)
        return {"tier": tier, "built": False, "reason": str(e)}
    built = ladder.built.get(tier)
    if built is None:
        return {"tier": tier, "built": False, "reason": ladder.unavailable.get(tier, "not configured")}
    stage = next((st for st in iter_stages(built.pipeline) if isinstance(st, FocusHQ)), None)
    if not isinstance(stage, FocusHQ):
        return {"tier": tier, "built": False, "reason": "no tse stage in the tier"}
    stage.load()
    if not stage.loaded:
        return {"tier": tier, "built": False, "reason": stage.error or "not loaded"}
    rng = np.random.default_rng(0)

    def clip(seconds: float) -> np.ndarray:
        return (0.1 * rng.standard_normal(round(seconds * 16000))).astype(np.float32)

    enroll, other = clip(2.0), clip(2.0)
    times = []
    for i in range(windows + 1):
        mix = clip(stage.window_s)
        args = (mix, None, None, enroll if "enroll" in stage.cand.needs else None,
                other if "other" in stage.cand.needs else None,
                np.full(len(mix) // 160, 0.9, np.float32) if "asd" in stage.cand.needs else None)
        t0 = time.perf_counter()
        y, err = stage._extract(*args)[:2]
        if y is None:
            return {"tier": tier, "built": False, "reason": f"extract: {err}"}
        if i:  # the first window is cold (kernels, enrollment embeddings)
            times.append(1000.0 * (time.perf_counter() - t0))
    stage.close()
    p50, p95 = np.percentile(times, [50, 95]).tolist()
    hop_ms = 1000.0 * stage.hops[0] / 16000
    util = p50 / hop_ms
    return {"tier": tier, "built": True, "candidate": stage.candidate, "device": stage.device,
            "window_s": stage.window_s, "hop_s": hop_ms / 1000.0, "compute_p50_ms": round(p50, 1),
            "compute_p95_ms": round(p95, 1), "utilisation": round(util, 3), "headroom": round(1.0 - util, 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--min", type=float, default=MIN_HEADROOM)
    args = ap.parse_args(argv)
    r = measure(args.windows)
    if not r["built"]:
        print(f"headroom check: {r['tier']} tier not run on this host ({r['reason']}); nothing to check")
        return 0
    print(f"headroom check: {r['candidate']} on {r['device']}, {r['window_s']:g} s window every {r['hop_s']:g} s: "
          f"compute p50 {r['compute_p50_ms']:.0f} / p95 {r['compute_p95_ms']:.0f} ms, "
          f"utilisation {r['utilisation']:.0%}, headroom {r['headroom']:.0%} (need >= {args.min:.0%})")
    if r["headroom"] < args.min:
        print("FAIL: not enough compute headroom for the enrolled tier; close other apps (or the load "
              "is not idle) and run again. Live, the tier widens its hop to cope.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
