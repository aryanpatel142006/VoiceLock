"""Output leveller with slew-limited gain (T-484): no loud/quiet pumping for sensory-overload listeners.

Live test Oct 11 (sessions 20261011T075314Z-14, 081558Z-1, 081909Z-2): the enrolled tier flipped between
isolated (target slot at full level, AGC makeup on top) and gated (target muted, room 20 dB down) every few
windows, and every flip moved the output 10-20 dB within ~30 ms. :class:`Smoother` sits after the AGC and
does two things only, both by attenuating (it never raises anything, so it cannot lift room noise or a
wrong voice):

* **Cap** (``cap_db``): the short-term level (400 ms RMS, a one-pole power envelope with ``SHORT_MS``) may
  exceed the long-term speech level (running mean in dB over voiced blocks, then a ``LONG_MS`` envelope) by at most ``cap_db``;
  the surplus is the desired cut (at most ``max_cut_db``).
* **Slew** (``down_db_per_s`` / ``up_db_per_s``): the applied gain moves toward the desired cut at most
  6 dB per 300 ms down (20 dB/s) and 6 dB per 600 ms back up (10 dB/s), so no correction is a jump.

Unvoiced 10 ms blocks (under ``floor_db`` or more than ``VOICED_UNDER_DB`` under the long-term level) hold both
the long-term level and the gain. Like :class:`hearing.audio.agc.Agc` the gain moves once per 10 ms block
and is ramped linearly over the next block: any chunk size gives the same output, zero latency.

Opt-in on the server: ``HEARING_SMOOTH=1`` (``HEARING_SMOOTH_CAP_DB`` moves the cap). Telemetry ``smooth``:
``{"gain_db": -1.2, "short_db": -18.0, "long_db": -21.0, "cap_db": 3.0}`` or null when off.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np

from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.types import AudioChunk, StageContext

ENV = "HEARING_SMOOTH"  # "1": the session's output goes through the smoother (after the AGC)
CAP_ENV = "HEARING_SMOOTH_CAP_DB"
BLOCK_MS = 10.0
SHORT_MS = 200.0  # power envelope time constant: about a 400 ms RMS window
LONG_MS = 8000.0  # long-term speech level (dB envelope over voiced blocks)
CAP_DB = 3.0  # short-term level may exceed the long-term level by this much
MAX_CUT_DB = 12.0
DOWN_DB_PER_S = 20.0  # 6 dB per 300 ms
UP_DB_PER_S = 10.0  # 6 dB per 600 ms
FLOOR_DB = -50.0  # blocks under this are silence (hold)
VOICED_UNDER_DB = 15.0  # short-term this far under the long-term level is a pause or the room (hold)
_EPS = 1e-12


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() in ("1", "on", "true")


def cap_from_env() -> float:
    try:
        v = float(os.environ.get(CAP_ENV, "") or CAP_DB)
    except ValueError:
        return CAP_DB
    return v if math.isfinite(v) and v >= 0 else CAP_DB


class Smoother:
    """Streaming cap-and-slew leveller on mono float32 at ``sample_rate``. See the module doc."""

    def __init__(self, sample_rate: int = PIPELINE_SR, *, cap_db: float = CAP_DB, max_cut_db: float = MAX_CUT_DB,
                 down_db_per_s: float = DOWN_DB_PER_S, up_db_per_s: float = UP_DB_PER_S,
                 floor_db: float = FLOOR_DB, short_ms: float = SHORT_MS, long_ms: float = LONG_MS):
        if not (down_db_per_s > 0 and up_db_per_s > 0):
            raise ValueError("smoother slew rates must be > 0 dB/s")
        self.sample_rate = int(sample_rate)
        self.block = max(1, round(self.sample_rate * BLOCK_MS / 1000.0))
        self.cap_db, self.max_cut_db, self.floor_db = float(cap_db), float(max_cut_db), float(floor_db)
        self.down_step = float(down_db_per_s) * BLOCK_MS / 1000.0
        self.up_step = float(up_db_per_s) * BLOCK_MS / 1000.0
        self._short = math.exp(-BLOCK_MS / short_ms)
        self._long = math.exp(-BLOCK_MS / long_ms)
        self.reset()

    def reset(self) -> None:
        self.short_pow = 0.0
        self.long_db: float | None = None
        self.gain_db = 0.0
        self._n = 0  # voiced blocks so far
        self._g0 = self._g1 = 1.0
        self._pos, self._acc = 0, 0.0

    @property
    def short_db(self) -> float:
        return 10.0 * math.log10(self.short_pow + _EPS)

    def _end_block(self) -> None:
        # the level is measured on the input to the smoother, so its own cut never feeds back into the reference
        p = self._acc / self.block
        self.short_pow = self._short * self.short_pow + (1.0 - self._short) * p
        s, level = self.short_db, 10.0 * math.log10(p + _EPS)
        # voiced by the block itself (10 ms), so a pause holds at once instead of riding the envelope's decay
        voiced = level >= self.floor_db and (self.long_db is None or level >= self.long_db - VOICED_UNDER_DB)
        if voiced:
            self._n += 1
            c = min(self._long, 1.0 - 1.0 / self._n)  # running mean until LONG_MS of speech, then the envelope
            self.long_db = s if self.long_db is None else c * self.long_db + (1.0 - c) * s
            want = -min(max(s - (self.long_db + self.cap_db), 0.0), self.max_cut_db)
            if want < self.gain_db:
                self.gain_db = max(want, self.gain_db - self.down_step)
            else:
                self.gain_db = min(want, self.gain_db + self.up_step)
        self._g0, self._g1 = self._g1, 10.0 ** (self.gain_db / 20.0)
        self._pos, self._acc = 0, 0.0

    def process_array(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.empty_like(x)
        i = 0
        while i < len(x):
            n = min(len(x) - i, self.block - self._pos)
            seg = x[i:i + n]
            ramp = self._g0 + (self._g1 - self._g0) * (np.arange(self._pos + 1, self._pos + n + 1) / self.block)
            y[i:i + n] = seg * ramp
            self._acc += float(np.dot(seg, seg))
            self._pos += n
            i += n
            if self._pos == self.block:
                self._end_block()
        return y

    def status(self) -> dict:
        return {"gain_db": round(self.gain_db, 1), "short_db": round(self.short_db, 1),
                "long_db": None if self.long_db is None else round(self.long_db, 1), "cap_db": self.cap_db}


@registry.register("smooth")
class SmoothStage:
    """``smooth(cap_db=3, max_cut_db=12, down_db_per_s=20, up_db_per_s=10)``."""

    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def __init__(self, cap_db: float = CAP_DB, max_cut_db: float = MAX_CUT_DB, down_db_per_s: float = DOWN_DB_PER_S,
                 up_db_per_s: float = UP_DB_PER_S):
        self.sm = Smoother(PIPELINE_SR, cap_db=cap_db, max_cut_db=max_cut_db, down_db_per_s=down_db_per_s,
                           up_db_per_s=up_db_per_s)
        self.name = f"smooth({cap_db:g} dB)"

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        return replace(chunk, samples=self.sm.process_array(chunk.samples))

    def reset(self) -> None:
        self.sm.reset()
