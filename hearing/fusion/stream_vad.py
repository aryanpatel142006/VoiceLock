"""Per-stream speech activity on the separator's two streams: is a window overlap or solo speech? (T-337)

No new dependency: frame energy against the room's running noise floor plus a share-of-the-mix check and a
spectral-flatness check, on 20 ms frames every 10 ms.

A stream is **active** in a frame when all three hold:

* its energy is >= ``ABOVE_FLOOR_DB`` (9 dB) over the running noise floor: the ``FLOOR_PCT`` = 10th
  percentile of the mixture's frame energies over the last ``FLOOR_S`` = 4 s of windows seen, capped at
  ``FLOOR_RANGE_DB`` (25 dB) under their 99th percentile (sustained speech without a pause must not lift
  the floor to the speech level);
* it carries a real share of the frame: energy >= the mixture's frame energy - ``SHARE_DB`` (15 dB), so most of the
  separator's leakage into the silent stream (measured 15-30 dB under the mix on t141 solo windows) does not count;
* it is speech-like: spectral flatness (geometric / arithmetic mean of the power spectrum, 100-4000 Hz)
  <= ``MAX_FLATNESS`` (0.4); broadband noise is near 1, voiced speech well under 0.3.

A window is **overlap** when both streams are active together in >= ``MIN_BOTH_S`` (0.08 s) of frames and
those frames make up >= ``MIN_BOTH_SHARE`` (15 %) of the window's active frames (frames where either stream
is active); otherwise solo (or silence). The decision is per window, i.e. smoothed over the window/hop.

The mixer uses it under ``HEARING_OVERLAP_FLOOR=1`` (with ``HEARING_GATED_FLOOR_DB``): a gated window that is
overlap keeps the hard gated floor, a solo gated window drops back to the 6 dB unsure floor
(``hearing.fusion.mixer.confidence_floor_db``).
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np

SR = 16000
FRAME = 320  # 20 ms
HOP = 160  # 10 ms
ABOVE_FLOOR_DB = 9.0
SHARE_DB = 15.0
MAX_FLATNESS = 0.4
FLOOR_S = 4.0
FLOOR_PCT = 10.0
FLOOR_RANGE_DB = 25.0
MIN_BOTH_S = 0.08
MIN_BOTH_SHARE = 0.15
OVERLAP_ENV = "HEARING_OVERLAP_FLOOR"  # "1": the overlap-aware gated floor (unset = off)
_EPS = 1e-10
_BAND = (100.0, 4000.0)


def enabled() -> bool:
    return os.environ.get(OVERLAP_ENV, "").strip() == "1"


def _frames(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64).ravel()
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def frame_db(x: np.ndarray) -> np.ndarray:
    """Mean-square energy (dB) of 20 ms frames every 10 ms."""
    f = _frames(x)
    return 10.0 * np.log10(np.mean(f * f, axis=1) + _EPS)


def flatness(x: np.ndarray) -> np.ndarray:
    """Spectral flatness (0 tonal .. 1 white) of each frame over 100-4000 Hz."""
    f = _frames(x) * np.hanning(FRAME)[None, :]
    p = np.abs(np.fft.rfft(f, axis=1)) ** 2 + _EPS
    hz = np.fft.rfftfreq(FRAME, 1.0 / SR)
    p = p[:, (hz >= _BAND[0]) & (hz <= _BAND[1])]
    return np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)


class StreamVad:
    """Stateful: keeps the mixture's recent frame energies for the running noise floor."""

    def __init__(self, floor_s: float = FLOOR_S):
        self._hist: deque = deque(maxlen=max(1, round(floor_s * SR / HOP)))

    def reset(self) -> None:
        self._hist.clear()

    def activity(self, mix: np.ndarray, streams) -> np.ndarray:
        """``(len(streams), frames)`` bool: each stream's speech activity per 10 ms frame."""
        em = frame_db(mix)
        self._hist.extend(em.tolist())
        hist = np.fromiter(self._hist, float)
        floor = min(float(np.percentile(hist, FLOOR_PCT)), float(np.percentile(hist, 99.0)) - FLOOR_RANGE_DB)
        out = []
        for y in streams:
            e = frame_db(y)[: len(em)]
            act = (e >= floor + ABOVE_FLOOR_DB) & (e >= em[: len(e)] - SHARE_DB) & (flatness(y)[: len(e)] <= MAX_FLATNESS)
            out.append(np.pad(act, (0, len(em) - len(act))))
        return np.stack(out) if out else np.zeros((0, len(em)), bool)

    def window(self, mix: np.ndarray, streams) -> dict:
        """``{"overlap": bool, "both_s", "any_s", "solo_s"}`` for one window (two streams)."""
        act = self.activity(mix, streams)
        if len(act) < 2:
            return {"overlap": False, "both_s": 0.0, "any_s": float(act.any(axis=0).sum() * HOP / SR) if len(act) else 0.0,
                    "solo_s": 0.0}
        both = act[0] & act[1]
        anyf = act[0] | act[1]
        nb, na = int(both.sum()), int(anyf.sum())
        overlap = nb * HOP / SR >= MIN_BOTH_S and nb >= MIN_BOTH_SHARE * max(na, 1)
        return {"overlap": bool(overlap), "both_s": nb * HOP / SR, "any_s": na * HOP / SR,
                "solo_s": (na - nb) * HOP / SR}
