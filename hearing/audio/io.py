"""Load/save WAV, resample, frame into hops and convert to/from the PCM16 wire format."""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import soundfile as sf

try:  # soxr is the primary resampler; scipy's resample_poly is the fallback.
    import soxr
except ImportError:  # pragma: no cover - soxr is a core dependency
    soxr = None

PIPELINE_SR = 16000
HOP = 160  # 10 ms at 16 kHz
_PCM16_SCALE = 32768.0


def load(path: str | os.PathLike, sr: int | None = PIPELINE_SR) -> tuple[np.ndarray, int]:
    """Read any soundfile-supported file as float32 mono at ``sr`` (native rate if None).

    Multi-channel files are downmixed by averaging channels; a differing file rate
    is resampled with :func:`resample`.
    """
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32) if x.shape[1] > 1 else x[:, 0]
    x = np.ascontiguousarray(x, dtype=np.float32)
    if sr is None or sr == file_sr:
        return x, int(file_sr)
    return resample(x, int(file_sr), sr), sr


def save(path: str | os.PathLike, x: np.ndarray, sr: int) -> None:
    """Write mono float32 as a PCM16 WAV; out-of-range values are clipped."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"save expects mono 1-D audio, got shape {x.shape}")
    sf.write(path, np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float32 ``x``; identity (same array) when the rates match.

    Uses soxr (very high quality) and falls back to scipy ``resample_poly``.
    Output length is ``round(len(x) * sr_out / sr_in)`` within one sample.
    """
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError(f"Sample rates must be positive, got {sr_in} -> {sr_out}")
    if sr_in == sr_out:
        return x
    x = np.asarray(x, dtype=np.float32)
    if soxr is not None:
        return np.ascontiguousarray(soxr.resample(x, sr_in, sr_out, quality="VHQ"), dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    y = resample_poly(x.astype(np.float64), sr_out // g, sr_in // g)
    return np.ascontiguousarray(y, dtype=np.float32)


def frame(x: np.ndarray, hop: int = HOP, pad: bool = True) -> Iterator[np.ndarray]:
    """Yield consecutive ``hop``-sized float32 views of ``x``.

    With ``pad=True`` the final partial chunk is zero-padded to ``hop`` samples (a
    copy); with ``pad=False`` it is dropped. Full chunks are views, not copies.
    """
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    full = n // hop
    for i in range(full):
        yield x[i * hop : (i + 1) * hop]
    rest = n - full * hop
    if rest and pad:
        last = np.zeros(hop, dtype=np.float32)
        last[:rest] = x[full * hop :]
        yield last


def pcm16_to_float(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """PCM16 little-endian bytes or an int16 array -> float32 in [-1, 1)."""
    if isinstance(data, np.ndarray):
        ints = data.astype("<i2", copy=False)
    else:
        ints = np.frombuffer(data, dtype="<i2")
    return (ints.astype(np.float32) / _PCM16_SCALE).astype(np.float32, copy=False)


def float_to_pcm16(x: np.ndarray) -> bytes:
    """float32 -> PCM16 little-endian bytes, clipping to the int16 range.

    Values on the int16 grid (``k / 32768``) round-trip exactly.
    """
    x = np.asarray(x, dtype=np.float32)
    scaled = np.rint(x.astype(np.float64) * _PCM16_SCALE)
    return np.clip(scaled, -32768, 32767).astype("<i2").tobytes()


def duration_s(x: np.ndarray, sr: int) -> float:
    """Length of ``x`` in seconds."""
    return len(x) / float(sr)
