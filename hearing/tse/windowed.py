"""Sliding-window wrapper that turns an offline TSE candidate into a delayed streaming one.

The mix is cut into windows of ``window_s`` every ``hop_s`` (zero-padded at the end so every
window has the same length), the aligned lip frames and ``p_speaking`` values are cut on the
25 fps video grid, ``extract`` runs per window, and the outputs are overlap-added with a
raised-cosine crossfade normalised by the summed weights, so an identity candidate reproduces
the input exactly. In the live loop a window can only be processed once it is complete, so the
output lags the input by ``window_s`` (the algorithmic latency) plus the compute time per window.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from hearing.tse.base import LIPS_FPS, SR, TSECandidate, conform_output


def _round(x: float) -> int:
    return math.floor(x + 0.5)


def frame_span(start: int, n_samples: int, sr: int, fps: float) -> tuple[int, int]:
    """Video frames ``[f0, f0 + nf)`` for audio samples ``[start, start + n_samples)``.

    Nearest-frame alignment (error at most half a frame, 20 ms at 25 fps) and a constant frame
    count ``round(n_samples / sr * fps)`` per window, so models with fixed-size inputs see the
    same shape for every window.
    """
    return _round(start / sr * fps), _round(n_samples / sr * fps)


def _slice_frames(arr: np.ndarray, f0: int, nf: int, pad: str) -> np.ndarray:
    """``arr[f0:f0+nf]`` padded past the end by repeating the last frame or with zeros."""
    out = arr[f0 : f0 + nf]
    missing = nf - out.shape[0]
    if missing <= 0:
        return out
    if pad == "edge" and arr.shape[0] > 0:
        filler = np.repeat(arr[-1:], missing, axis=0)
    else:
        filler = np.zeros((missing, *arr.shape[1:]), dtype=arr.dtype)
    return np.concatenate([out, filler], axis=0)


def crossfade_window(n: int, hop: int) -> np.ndarray:
    """Weights for one window of ``n`` samples placed every ``hop`` samples (float64, all > 0).

    For ``hop >= n / 2`` a flat top with raised-cosine (sin²/cos²) ramps over the ``n - hop``
    overlap: consecutive windows sum to exactly 1, and ``hop = n / 2`` is a Hann window. For
    denser hops a strictly positive Hann window; the extractor normalises by the summed weights.
    """
    overlap = n - hop
    if overlap <= n // 2:
        w = np.ones(n)
        if overlap > 0:
            ramp = np.sin(0.5 * np.pi * (np.arange(overlap) + 0.5) / overlap) ** 2
            w[:overlap] = ramp
            w[n - overlap :] = ramp[::-1]
        return w
    return np.hanning(n + 2)[1:-1]


@dataclass(frozen=True)
class WindowStats:
    """Latency and speed of a :class:`SlidingWindowExtractor` since its last ``reset_stats``."""

    window_s: float
    hop_s: float
    algorithmic_latency_ms: float  # output lags input by one window
    compute_p50_ms: float  # per window
    compute_p95_ms: float
    compute_max_ms: float
    n_windows: int
    audio_s: float
    compute_s: float
    rtf: float  # compute_s / audio_s (nan before any audio)

    def as_dict(self) -> dict:
        return asdict(self)


class SlidingWindowExtractor:
    """Run ``candidate`` over sliding windows; itself satisfies :class:`TSECandidate`."""

    def __init__(
        self,
        candidate: TSECandidate,
        window_s: float,
        hop_s: float | None = None,
        sr: int = SR,
    ):
        hop_s = window_s / 2 if hop_s is None else hop_s
        if window_s <= 0 or not 0 < hop_s <= window_s:
            raise ValueError(
                f"need 0 < hop_s <= window_s, got window {window_s}, hop {hop_s}"
            )
        self.candidate = candidate
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        self.sr = sr
        self.window = _round(window_s * sr)
        self.hop = _round(hop_s * sr)
        self.name = f"{candidate.name}@{window_s:g}s"
        self.licence = candidate.licence
        self.needs = candidate.needs
        self.oracle = bool(getattr(candidate, "oracle", False))
        self._weights = crossfade_window(self.window, self.hop)
        self.reset_stats()

    @property
    def device(self) -> str | None:
        return getattr(self.candidate, "device", None)

    def load(self) -> None:
        self.candidate.load()

    def reset_stats(self) -> None:
        self._times_ms: list[float] = []
        self._audio_s = 0.0

    def window_starts(self, n: int) -> list[int]:
        """Start sample of every window for an ``n``-sample input (the last one may run past n)."""
        if n <= self.window:
            return [0]
        return [
            i * self.hop for i in range(math.ceil((n - self.window) / self.hop) + 1)
        ]

    def extract(
        self,
        mix: np.ndarray,
        sr: int,
        *,
        lips: np.ndarray | None = None,
        lips_fps: float = LIPS_FPS,
        p_speaking: np.ndarray | None = None,
        enroll: np.ndarray | None = None,
    ) -> np.ndarray:
        if sr != self.sr:
            raise ValueError(f"{self.name}: expected {self.sr} Hz audio, got {sr}")
        mix = np.asarray(mix, dtype=np.float32)
        n = mix.shape[0]
        starts = self.window_starts(n)
        padded = starts[-1] + self.window
        mix_p = np.pad(mix, (0, padded - n))
        enroll_p = None
        if self.oracle and enroll is not None:
            enroll = np.asarray(enroll, dtype=np.float32)[:padded]
            enroll_p = np.pad(enroll, (0, padded - enroll.shape[0]))
        out = np.zeros(padded)
        wsum = np.zeros(padded)
        for i, s in enumerate(starts):
            f0, nf = frame_span(s, self.window, sr, lips_fps)
            seg_lips = None if lips is None else _slice_frames(lips, f0, nf, "edge")
            seg_p = (
                None
                if p_speaking is None
                else _slice_frames(p_speaking, f0, nf, "zero")
            )
            seg_enroll = (
                enroll_p[s : s + self.window] if enroll_p is not None else enroll
            )
            t0 = time.perf_counter()
            y = self.candidate.extract(
                mix_p[s : s + self.window],
                sr,
                lips=seg_lips,
                lips_fps=lips_fps,
                p_speaking=seg_p,
                enroll=seg_enroll,
            )
            y = conform_output(y, self.window, self.candidate.name)
            self._times_ms.append((time.perf_counter() - t0) * 1000.0)
            w = self._weights
            if len(starts) == 1:  # nothing to crossfade with
                w = np.ones(self.window)
            elif i == 0 or i == len(starts) - 1:
                w = w.copy()
                if i == 0:
                    w[: self.window // 2] = 1.0
                else:
                    w[self.window // 2 :] = 1.0
            out[s : s + self.window] += w * y
            wsum[s : s + self.window] += w
        self._audio_s += n / sr
        return (out / np.maximum(wsum, 1e-12))[:n].astype(np.float32)

    def stats(self) -> WindowStats:
        t = np.asarray(self._times_ms)
        p50, p95, mx = np.percentile(t, [50, 95, 100]) if t.size else (math.nan,) * 3
        compute_s = sum(self._times_ms) / 1000.0
        return WindowStats(
            window_s=self.window_s,
            hop_s=self.hop_s,
            algorithmic_latency_ms=self.window_s * 1000.0,
            compute_p50_ms=float(p50),
            compute_p95_ms=float(p95),
            compute_max_ms=float(mx),
            n_windows=int(t.size),
            audio_s=self._audio_s,
            compute_s=compute_s,
            rtf=compute_s / self._audio_s if self._audio_s > 0 else math.nan,
        )
