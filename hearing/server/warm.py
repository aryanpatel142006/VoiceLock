"""Load the live models once at server start, off the request path (T-072).

Every model the first session needs (MediaPipe face assets, Light-ASD, the apparent-face
estimator, the enrolled tier's MossFormer2 separator and ECAPA) is loaded lazily through
:func:`hearing.models.load`. On a cold server the first session's vision worker spends its first
~2-3 s loading them and drops the camera frames meanwhile (laptop-a replay: 60 of the first ~100
frames). ``hearing serve`` calls :func:`start` so those loads happen in a background thread while
the server waits for its first client; the model cache is shared and locked, so a session that
asks for a model being warmed simply waits for it. Only models whose weights are already on disk
are warmed (``available()``): this never downloads.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time

import numpy as np

from hearing import models

log = logging.getLogger(__name__)
_state = "off"  # off (never started: --no-warm, tests) | warming | ready; GET /health "warm" (T-088)

# (module whose available() gates it, model names for hearing.models.load, needs TORCH_LOCK)
WARM = (
    ("mediapipe", (), False),  # the import alone is ~0.5 s; each session builds its own graphs
    ("hearing.vision.faces", ("mediapipe_face_detector", "mediapipe_face_landmarker"), False),
    ("hearing.vision.light_asd", ("light_asd",), True),
    ("hearing.synth.apparent_face", ("apparent_face",), False),
    ("hearing.tse.blind", ("MossFormer2_SS_16K",), True),
    ("hearing.tse.enrolled", ("spkrec-ecapa-voxceleb",), True),
)


def _available(module) -> bool:
    check = getattr(module, "available", None)
    if not callable(check):
        return True
    try:
        return bool(check())
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        log.debug("%s.available() failed: %s", module.__name__, e)
        return False


def warm(plan=WARM) -> dict[str, str]:
    """Load every available model in ``plan``; returns ``{name: "ok N ms" | "skipped" | error}``."""
    from hearing.tse.gpu import TORCH_LOCK

    done: dict[str, str] = {}
    for module_name, names, torch_lock in plan:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            done.update({n: f"skipped: {e}" for n in names})
            continue
        if not _available(module):
            done.update({n: "skipped: not available" for n in names})
            continue
        for name in names:
            t0 = time.perf_counter()
            try:
                if torch_lock:
                    with TORCH_LOCK:  # MPS is not thread-safe: never load beside a running window
                        models.load(name)
                else:
                    models.load(name)
            except (ImportError, KeyError, OSError, RuntimeError, ValueError) as e:
                done[name] = f"{type(e).__name__}: {e}"
                continue
            done[name] = f"ok {1000.0 * (time.perf_counter() - t0):.0f} ms"
    if plan is WARM:  # T-120: Name Call's spotter; loading it inside a session put it ~2 s behind the mic
        try:
            from hearing.synth import asr

            if asr.local_available():
                key = asr._whisper_loader(asr.whisper_model_name())
                t0 = time.perf_counter()
                models.load(key)
                done[key] = f"ok {1000.0 * (time.perf_counter() - t0):.0f} ms"
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            done["faster-whisper"] = f"{type(e).__name__}: {e}"
    return done


def warm_inference() -> str:
    """One enrolled extraction on noise, so the first real window does not pay the MPS kernel
    compile (laptop-a: first window 1.7-2.3 s cold, 0.2 s warm; T-088). Kernels are per process,
    so the session's own candidate benefits. Returns ``ok N ms`` / ``skipped`` / the error."""
    from hearing.tse import enrolled
    from hearing.tse.base import get_candidate
    from hearing.tse.gpu import TORCH_LOCK

    if not _available(enrolled):
        return "skipped: not available"
    rng = np.random.default_rng(0)
    clip = (0.05 * rng.standard_normal(2 * enrolled.SR)).astype(np.float32)
    mix = (0.1 * rng.standard_normal(enrolled.SR)).astype(np.float32)
    t0 = time.perf_counter()
    try:
        with TORCH_LOCK:
            get_candidate("enrolled").extract(mix, enrolled.SR, enroll=clip, other=clip[::-1].copy())
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return f"ok {1000.0 * (time.perf_counter() - t0):.0f} ms"


def warm_enrolled() -> str:
    """T-339 (``HEARING_WARM_ENROLLED=1``): embed every saved voice profile once (the pairs a session
    preloads, :func:`hearing.tse.enroll.profile_pairs`) into the shared cache
    :data:`hearing.tse.enrolled.WARM_PROFILES`, and run one 1.0 s window through the enrolled candidate
    with the first pair, so the closed-set (LDA) pick path is compiled too. Sessions then find the
    embeddings by clip key instead of a ~1 s ECAPA job each. ``ok N ms (P profiles)`` / ``skipped`` / error."""
    from hearing.tse import enrolled
    from hearing.tse.base import get_candidate
    from hearing.tse.enroll import list_profiles, profile_pairs
    from hearing.tse.gpu import TORCH_LOCK

    if not enrolled.warm_enabled():
        return "skipped: HEARING_WARM_ENROLLED is not 1"
    if not _available(enrolled):
        return "skipped: not available"
    if not list_profiles():
        return "skipped: no voice profiles"
    t0 = time.perf_counter()
    try:
        pairs = profile_pairs()
        cand = get_candidate("enrolled")
        mix = (0.1 * np.random.default_rng(0).standard_normal(enrolled.SR)).astype(np.float32)
        with TORCH_LOCK:
            for target, other in pairs:
                cand.prepare(target, other)
            cand.extract(mix, enrolled.SR, enroll=pairs[0][0], other=pairs[0][1])
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return f"ok {1000.0 * (time.perf_counter() - t0):.0f} ms ({len(pairs)} profiles)"


def warm_all() -> dict[str, str]:
    """What :func:`start` runs: :func:`warm`, :func:`warm_inference` and, with
    ``HEARING_WARM_ENROLLED=1`` only (T-339), :func:`warm_enrolled`."""
    from hearing.tse.enrolled import warm_enabled

    done = warm()
    done["enrolled_inference"] = warm_inference()
    if warm_enabled():
        done["enrolled_profiles"] = warm_enrolled()
    return done


def status() -> str:
    """``off`` / ``warming`` / ``ready``: whether :func:`start` finished (``demo.sh --judge`` waits for it)."""
    return _state


def start() -> threading.Thread:
    """Run :func:`warm` on a daemon thread and log what it loaded."""
    global _state
    _state = "warming"

    def run():
        global _state
        t0 = time.perf_counter()
        try:
            done = warm_all()
        finally:
            _state = "ready"
        # uvicorn configures only its own loggers: print, so the serve log shows it
        print(f"hearing: models warmed in {time.perf_counter() - t0:.1f} s: {done}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=run, name="warm-models", daemon=True)
    thread.start()
    return thread
