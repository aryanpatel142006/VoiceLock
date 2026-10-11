"""The serving machine's load, for the app's "laptop is busy" pre-flight warning (T-485).

``host_load()`` is the 1-minute load average over the core count, cached for a second (telemetry asks at 10 Hz).
``busy`` is a load of at least one runnable thread per core: at the 4:19 live test agent jobs held the Mac there and
42 of 64 enrolled windows overran. ``None`` where the OS has no load average (Windows).
"""

from __future__ import annotations

import os
import time

BUSY_PER_CPU = 1.0  # 1-minute load average per core at which the app warns
_CACHE_S = 1.0
_cache: tuple[float, dict | None] = (-1e9, None)


def measure(load1: float, cpus: int) -> dict:
    """The JSON shape from a load average and a core count."""
    cpus = max(1, int(cpus or 1))
    per_cpu = load1 / cpus
    return {"load1": round(load1, 2), "cpus": cpus, "per_cpu": round(per_cpu, 2), "busy": per_cpu >= BUSY_PER_CPU}


def host_load(now: float | None = None) -> dict | None:
    """``{"load1": 9.8, "cpus": 12, "per_cpu": 0.82, "busy": false}`` (cached ``_CACHE_S``), or None."""
    global _cache
    now = time.monotonic() if now is None else now
    if now - _cache[0] < _CACHE_S:
        return _cache[1]
    try:
        value = measure(os.getloadavg()[0], os.cpu_count() or 1)
    except (AttributeError, OSError):
        value = None
    _cache = (now, value)
    return value
