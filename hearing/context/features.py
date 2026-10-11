"""The 1 Hz context vector: how loud and how crowded the room is, and how sure we are of the target.

:class:`ContextTracker` is fed every raw 10 ms input hop (before any enhancement, so it
measures the room, not our output) plus the current :class:`~hearing.types.StageContext`,
and every ``window_hops`` hops (1 s of audio) :meth:`ContextTracker.push` returns a
:class:`ContextVector`. The session sends it as the wire ``context`` event and feeds it to
the preference model (``hearing.context.prefs``).

Coarse estimates, by design:

* ``noise_db``: noise floor in dBFS, the 10th percentile of per-hop energy over the last
  ``noise_hops`` hops (3 s; minimum statistics, so speech pauses expose the floor).
* ``snr_db``: speech level (90th percentile of per-hop energy over the last second)
  minus the noise floor, clamped to [0, 60].
* ``speakers``: the most face tracks speaking at once (``p_speaking >= 0.5``) in the
  window. Without tracks (no camera / no vision stage) it falls back to audio only:
  1 if the second contained speech-like level variation (``snr_db >= speech_snr_db``)
  else 0; audio alone cannot count talkers here.
* ``overlap``: fraction of hops with two or more tracks speaking (0 without tracks).
* ``target_conf``: mean target confidence over the window (0 while no target).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from hearing.audio.io import HOP, PIPELINE_SR
from hearing.types import StageContext

SPEAKING = 0.5
FLOOR_DB = -100.0
FEATURES = ("noise_db", "snr_db", "speakers", "overlap", "target_conf")


@dataclass(frozen=True)
class ContextVector:
    noise_db: float
    snr_db: float
    speakers: int
    overlap: float
    target_conf: float
    mode: str

    def values(self) -> list[float]:
        """The classifier inputs, in :data:`FEATURES` order (``mode`` is not one of them)."""
        return [float(getattr(self, k)) for k in FEATURES]

    def message(self, **extra) -> dict:
        """The wire ``context`` event (docs/CONTRACTS.md#events)."""
        return {"type": "context", **asdict(self), **extra}


def hop_db(samples: np.ndarray) -> float:
    """Energy of one hop in dBFS (a full-scale sine is about -3 dB)."""
    x = np.asarray(samples, dtype=np.float64)
    if not len(x):
        return FLOOR_DB
    return max(FLOOR_DB, 10.0 * float(np.log10(np.mean(x * x) + 1e-12)))


class ContextTracker:
    def __init__(self, window_s: float = 1.0, noise_s: float = 3.0, speech_snr_db: float = 10.0,
                 sample_rate: int = PIPELINE_SR, hop: int = HOP):
        hops_per_s = sample_rate / hop
        self.window_hops = max(1, round(window_s * hops_per_s))
        self.noise_hops = max(self.window_hops, round(noise_s * hops_per_s))
        self.speech_snr_db = speech_snr_db
        self.reset()

    def reset(self) -> None:
        self._noise: deque[float] = deque(maxlen=self.noise_hops)
        self._levels: list[float] = []
        self._talkers: list[int] = []
        self._conf: list[float] = []
        self._seen_tracks = False
        self.last: ContextVector | None = None

    def push(self, samples: np.ndarray, ctx: StageContext, mode: str) -> ContextVector | None:
        """Account one raw input hop; returns a vector when a window completes."""
        level = hop_db(samples)
        self._noise.append(level)
        self._levels.append(level)
        self._talkers.append(sum(1 for t in ctx.tracks if t.p_speaking >= SPEAKING))
        target = ctx.target
        self._conf.append(float(target.confidence) if target.track_id is not None else 0.0)
        self._seen_tracks |= bool(ctx.tracks)
        if len(self._levels) < self.window_hops:
            return None
        self.last = self._finish(mode)
        return self.last

    def _finish(self, mode: str) -> ContextVector:
        noise = float(np.percentile(self._noise, 10))
        speech = float(np.percentile(self._levels, 90))
        snr = float(np.clip(speech - noise, 0.0, 60.0))
        talkers = np.asarray(self._talkers)
        if self._seen_tracks:
            speakers = int(talkers.max())
            overlap = float(np.mean(talkers >= 2))
        else:
            speakers = int(snr >= self.speech_snr_db)
            overlap = 0.0
        vector = ContextVector(
            noise_db=round(noise, 1),
            snr_db=round(snr, 1),
            speakers=speakers,
            overlap=round(overlap, 2),
            target_conf=round(float(np.mean(self._conf)), 2),
            mode=mode,
        )
        self._levels.clear()
        self._talkers.clear()
        self._conf.clear()
        self._seen_tracks = False
        return vector
