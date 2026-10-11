"""Mixer helpers and stages: the one place the suppression slider meets the audio.

The slider ``s`` in [0, 1] (``ctx.suppression``) is mapped to model knobs here only
(hearing-stack skill):

* The room's attenuation in dB: :func:`room_atten_db` (T-093). Equal slider steps are
  equal loudness steps of the room: ``30 * s`` dB (0 -> -30 dB over 0 -> 100), with a
  soft knee from ``s = 0.9`` that fades the last -27 dB of room smoothly to silence at
  ``s = 1`` ("only the target"). :func:`room_gain` is the same curve as a linear gain.
* DeepFilterNet ``atten_lim_db``: :func:`strength_to_db` (= :func:`room_atten_db`, so
  DFN's noisy mix-back ``10 ** (-lim / 20)`` equals :func:`room_gain`).
* Residual mix ``out = target + room_gain(s) * residual``: the ``mixer`` stage.
* The ``gate`` ducks non-target time to :func:`room_gain`.

Slider moves never click: the mixer ramps its gain linearly over ``SLIDER_RAMP_MS``
(30 ms), DFN ramps its mix-back over the same time, and the gate's envelope is
slope-limited (``docs/DESIGN.md#suppression-slider``).

Residual side channel (``ctx.telemetry.extras``, per chunk, consumed by ``mixer``):

* ``extras["residual"]``: an upstream stage puts the residual for the *current* chunk
  here: float32, the same length as the chunk, delay-aligned with the samples the
  mixer receives (the target). The mixer pops it and outputs
  ``target + g * residual`` with ``g = room_gain(s)``.
* ``extras["aligned_mix"]``: written by the ``split(delay_ms=D)`` stage, which passes
  its input through unchanged and stores a copy delayed by ``D`` ms. With ``D`` equal
  to the algorithmic latency of the stages between ``split`` and ``mixer`` the copy
  lines up with their output, and the mixer forms ``residual = aligned_mix - target``,
  i.e. ``out = (1 - g) * target + g * aligned_mix``. That is how
  ``split(delay_ms=32)>gtcrn>mixer`` and ``split(delay_ms=...)>tse>mixer`` work.
* Neither key present: the mixer passes the chunk through unchanged. This is the
  Natural fast path ``dfn>gate>mixer``, where DFN3 already applies the slider via
  ``atten_lim_db``.

* ``extras["tse_confidence"]`` (T-134, with a residual): how sure the extractor is that it picked
  the right voice, 0..1. It replaces ``max_duck_db``: the residual goes down at most
  :func:`confidence_floor_db` = ``UNSURE_DB + c * (SURE_DB - UNSURE_DB)`` dB (6 dB unsure, the
  near-mix, to 40 dB sure), never below the slider's own gain, ramped like a slider move.

``residual`` wins when both keys are present. The mixer pops both keys every chunk so
a residual is never reused for a later chunk, and it records
``extras["mixer"] = {"residual": bool}``. It does not clip (the output edge clips).
Stages between ``split`` and ``mixer`` should output the *full-strength* target: one
that also applies the slider itself (``gtcrn`` mixes dry/wet by ``s``) ends up at an
effective strength of ``s**2`` (endpoints 0 and 1 stay exact).
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np

from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.types import AudioChunk, StageContext

ROOM_RANGE_DB = 30.0  # room attenuation at the start of the knee is ROOM_RANGE_DB * KNEE_START
KNEE_START = 0.9  # above this the knee fades the remaining room to silence at s = 1
DFN_MAX_DB = 100.0  # above this DFN's mix-back is < 1e-5: the same as no limit
SLIDER_RAMP_MS = 30.0  # every slider-driven gain moves at most 0 -> 1 over this (>= 20 ms)
DUCK_FLOOR_DB = -30.0  # the gate never ducks the room below this (Natural: the room stays present)
RESIDUAL_KEY = "residual"
CONFIDENCE_KEY = "tse_confidence"  # T-134: the tse stage's per-chunk pick confidence in [0, 1]
CONFIDENCE_ENV = "HEARING_TSE_CONFIDENCE"  # "0": ignore the confidence (max_duck_db as before; A/B arm)
UNSURE_DB = 6.0  # residual floor at confidence 0: about the mix (a wrong pick is barely louder)
SURE_DB = 40.0  # residual floor at confidence 1
# T-180 A/B knobs (env, read per chunk; unset = the T-134 behaviour above):
UNSURE_ENV = "HEARING_UNSURE_DB"  # K3: the residual floor at confidence 0 in dB (e.g. 12 instead of UNSURE_DB)
GATED_ENV = "HEARING_GATED_FLOOR_DB"  # K2: floor for a gated window (confidence exactly 0), e.g. 20 (pre-T-134)
OVERLAP_KEY = "tse_overlap"  # T-337: share (0..1) of the chunk's windows whose two separated streams were both active
OVERLAP_ENV = "HEARING_OVERLAP_FLOOR"  # "1": the gated floor only on overlap windows, solo gated windows keep UNSURE_DB
ALIGNED_MIX_KEY = "aligned_mix"


def _check(strength: float) -> float:
    s = float(strength)
    if not math.isfinite(s) or not 0.0 <= s <= 1.0:
        raise ValueError(f"suppression must be in [0, 1], got {strength!r}")
    return s


def room_atten_db(strength: float) -> float:
    """Suppression slider in [0, 1] -> how far the room is pushed down, in dB (>= 0).

    ``30 * s`` dB (perceptually linear: each 10 slider points is 3 dB), times a raised-cosine
    soft knee above ``s = 0.9`` that takes the room from -27 dB to silence (``inf``) at
    ``s = 1``: 0 -> 0, 0.25 -> 7.5, 0.5 -> 15, 0.75 -> 22.5, 0.9 -> 27, 0.95 -> 34.5, 1 -> inf.
    """
    s = _check(strength)
    if s <= 0.0:
        return 0.0
    if s >= 1.0:
        return math.inf
    db = ROOM_RANGE_DB * s
    if s > KNEE_START:
        taper = 0.5 * (1.0 + math.cos(math.pi * (s - KNEE_START) / (1.0 - KNEE_START)))
        if taper <= 0.0:
            return math.inf
        db -= 20.0 * math.log10(taper)
    return db


def room_gain(strength: float) -> float:
    """Linear gain of the room at slider ``strength``: ``10 ** (-room_atten_db / 20)``; 0 at 1."""
    db = room_atten_db(strength)
    return 0.0 if math.isinf(db) else float(10.0 ** (-db / 20.0))


def strength_to_db(strength: float) -> float | None:
    """Suppression slider in [0, 1] -> DeepFilterNet attenuation limit in dB.

    * ``strength <= 0``: ``0.0`` dB, i.e. bypass (the noisy signal passes unchanged).
    * ``0 < strength < 1``: :func:`room_atten_db` (``30 * s`` dB with the soft knee above 0.9).
    * ``strength >= 1`` (or a knee value past ``DFN_MAX_DB``): ``None``, the model's full
      suppression.
    """
    db = room_atten_db(strength)
    if db <= 0.0:
        return 0.0
    if math.isinf(db) or db >= DFN_MAX_DB:
        return None
    return db


def duck_gain(strength: float, floor_db: float = DUCK_FLOOR_DB) -> float:
    """Suppression slider in [0, 1] -> linear gain the gate applies to the room while it ducks.

    ``max(room_gain(s), 10 ** (floor_db / 20))``: the T-093 slider curve (-3 dB per 10 points)
    down to the floor (-30 dB = 0.0316, reached inside the knee at s ~ 0.92), so s = 1 is "only
    the target" without going to digital silence (T-096). Out-of-range values are clamped;
    non-finite ones fail open (s = 0, gain 1).
    """
    s = float(strength)
    s = min(max(s, 0.0), 1.0) if math.isfinite(s) else 0.0
    return max(room_gain(s), 10.0 ** (float(floor_db) / 20.0))


def atten_lim_to_mix(limit_db: float | None) -> float:
    """Attenuation limit -> linear share of the *noisy* signal mixed back (DF convention).

    ``out = mix * noisy + (1 - mix) * enhanced``; ``None`` -> 0.0, ``0 dB`` -> 1.0 (bypass).
    """
    if limit_db is None:
        return 0.0
    lim = abs(float(limit_db))
    if lim < 0.01:
        return 1.0
    return float(10.0 ** (-lim / 20.0))


def _side_channel(ctx: StageContext, key: str, n: int) -> np.ndarray | None:
    value = ctx.telemetry.extras.pop(key, None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).ravel()
    if len(arr) != n:
        raise ValueError(f"mixer: extras[{key!r}] has {len(arr)} samples, chunk has {n}")
    return arr


def _env_db(name: str) -> float | None:
    """A non-negative dB value from the environment; None when unset, empty or invalid."""
    raw = os.environ.get(name, "").strip()
    try:
        v = float(raw) if raw else None
    except ValueError:
        return None
    return v if v is not None and math.isfinite(v) and v >= 0 else None


def confidence_floor_db(confidence: float, overlap: float | None = None) -> float:
    """Pick confidence in [0, 1] -> how far (dB) the residual may be pushed down (6 .. 40).

    T-180 knobs: ``HEARING_UNSURE_DB`` replaces the 6 dB at confidence 0 (K3); ``HEARING_GATED_FLOOR_DB``
    sets the floor of a window at confidence exactly 0, i.e. one the extractor gated (K2: 20 restores
    the pre-T-134 ``max_duck_db=20`` there). Both unset: ``UNSURE_DB + c * (SURE_DB - UNSURE_DB)``.
    T-337: with ``HEARING_OVERLAP_FLOOR=1`` the gated floor applies only where ``overlap`` (the tse stage's
    per-stream VAD, ``hearing.fusion.stream_vad``) is >= 0.5 or unknown (None); a gated chunk of solo speech
    keeps the confidence-0 floor (6 dB)."""
    c = float(confidence)
    c = min(max(c, 0.0), 1.0) if math.isfinite(c) else 0.0
    gated = _env_db(GATED_ENV)
    solo = overlap is not None and overlap < 0.5 and os.environ.get(OVERLAP_ENV, "").strip() == "1"
    if c == 0.0 and gated is not None and not solo:
        return gated
    unsure = _env_db(UNSURE_ENV)
    unsure = UNSURE_DB if unsure is None else min(unsure, SURE_DB)
    return unsure + c * (SURE_DB - unsure)


@registry.register("mixer")
class Mixer:
    """``out = target + room_gain(s) * residual``; pass-through when no residual is supplied.

    The residual gain follows the slider with a linear ramp of at most a full swing per
    ``SLIDER_RAMP_MS``, carried across chunks, so a slider jump never clicks. ``max_duck_db``
    bounds how far the residual goes down (``None`` = to ``room_gain(slider)``, silence at 100):
    a TSE window that misses the target then plays it 20 dB down instead of a hole (T-095).
    """

    name = "mixer"
    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def __init__(self, ramp_ms: float = SLIDER_RAMP_MS, max_duck_db: float | None = None):
        if not math.isfinite(float(ramp_ms)) or ramp_ms < 0:
            raise ValueError(f"mixer ramp_ms must be finite and >= 0, got {ramp_ms!r}")
        if max_duck_db is not None and not max_duck_db >= 0:
            raise ValueError(f"mixer max_duck_db must be >= 0, got {max_duck_db!r}")
        self.ramp_ms = float(ramp_ms)
        self.max_duck_db = None if max_duck_db is None else float(max_duck_db)
        self._floor = 0.0 if self.max_duck_db is None else 10.0 ** (-self.max_duck_db / 20.0)
        if self.max_duck_db is not None:
            self.name = f"mixer(max_duck_db={self.max_duck_db:g})"
        n = self.ramp_ms * self.sample_rate / 1000.0
        self._step = 1.0 if n <= 1.0 else 1.0 / n
        self.reset()

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        target = np.asarray(chunk.samples, dtype=np.float32)
        n = len(target)
        residual = _side_channel(ctx, RESIDUAL_KEY, n)
        aligned = _side_channel(ctx, ALIGNED_MIX_KEY, n)
        if residual is None and aligned is not None:
            residual = aligned - target
        conf = ctx.telemetry.extras.pop(CONFIDENCE_KEY, None)
        overlap = ctx.telemetry.extras.pop(OVERLAP_KEY, None)
        if os.environ.get(CONFIDENCE_ENV, "1").strip() == "0":
            conf = None
        floor = self._floor if conf is None else 10.0 ** (-confidence_floor_db(conf, overlap) / 20.0)
        goal = max(room_gain(min(max(float(ctx.suppression), 0.0), 1.0)), floor)
        ctx.telemetry.extras["mixer"] = {"residual": residual is not None}
        if conf is not None:
            ctx.telemetry.extras["mixer"]["floor_db"] = round(confidence_floor_db(conf, overlap), 1)
        if residual is None:
            self._keep = None  # nothing to mix: the next residual starts at the slider
            return chunk
        keep = self._ramp(goal, n)
        return replace(chunk, samples=(target + keep * residual).astype(np.float32, copy=False))

    def _ramp(self, goal: float, n: int) -> np.ndarray | np.float32:
        """Per-sample residual gain moving from the carried value toward ``goal``."""
        g = goal if self._keep is None else self._keep
        self._keep = goal
        if g == goal or n == 0:
            return np.float32(goal)
        step = self._step if goal > g else -self._step
        ramp = g + step * np.arange(1, n + 1, dtype=np.float64)
        ramp = np.minimum(ramp, goal) if step > 0 else np.maximum(ramp, goal)
        self._keep = float(ramp[-1])
        return ramp.astype(np.float32)

    def reset(self) -> None:
        self._keep: float | None = None


@registry.register("split")
class Split:
    """Pass the chunk through; store it delayed by ``delay_ms`` as ``extras["aligned_mix"]``.

    Set ``delay_ms`` to the algorithmic latency of the stages before the ``mixer`` so
    the copy lines up with their output. ``latency_ms`` is 0: the main path is not
    delayed. Any chunk length works (sample FIFO), and the first ``delay_ms`` of the
    copy is silence.
    """

    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def __init__(self, delay_ms: float = 0.0):
        if not math.isfinite(float(delay_ms)) or delay_ms < 0:
            raise ValueError(f"split delay_ms must be finite and >= 0, got {delay_ms!r}")
        self.delay_ms = float(delay_ms)
        self.name = f"split(delay_ms={self.delay_ms:g})"
        self._delay = round(self.delay_ms * self.sample_rate / 1000.0)
        self.reset()

    def reset(self) -> None:
        self._fifo = np.zeros(self._delay, dtype=np.float32)

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        x = np.asarray(chunk.samples, dtype=np.float32)
        buf = np.concatenate((self._fifo, x))
        ctx.telemetry.extras[ALIGNED_MIX_KEY] = buf[: len(x)].copy()
        self._fifo = buf[len(x) :].copy()
        return chunk
