"""Streaming buffers: a sample ring buffer, a hop chunker and an AudioChunk generator."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np

from hearing.audio.io import HOP
from hearing.types import AudioChunk

Overflow = Literal["raise", "drop_oldest"]


class RingBuffer:
    """Fixed-capacity FIFO of float32 samples.

    Overflow policy (chosen at construction):
      * ``"raise"`` (default): a push that would exceed ``capacity`` raises
        ``OverflowError`` and leaves the buffer untouched. Use this between stages,
        where losing samples is a bug.
      * ``"drop_oldest"``: the oldest samples are discarded so the newest ``capacity``
        samples survive. Use this at the capture edge, where staying live beats
        completeness.
    Pulling more than ``available`` raises ``ValueError``.
    """

    def __init__(self, capacity: int, overflow: Overflow = "raise"):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if overflow not in ("raise", "drop_oldest"):
            raise ValueError(f"overflow must be 'raise' or 'drop_oldest', got {overflow!r}")
        self.capacity = int(capacity)
        self.overflow = overflow
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._read = 0  # index of the oldest sample
        self._count = 0

    @property
    def available(self) -> int:
        return self._count

    def clear(self) -> None:
        self._read = 0
        self._count = 0

    def push(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).ravel()
        n = len(x)
        if n == 0:
            return
        free = self.capacity - self._count
        if n > free:
            if self.overflow == "raise":
                raise OverflowError(
                    f"RingBuffer overflow: {n} samples pushed, {free} free of {self.capacity}"
                )
            if n > self.capacity:  # only the newest ``capacity`` samples can survive
                x, n = x[-self.capacity :], self.capacity
            drop = n - (self.capacity - self._count)
            self._read = (self._read + drop) % self.capacity
            self._count -= drop
        write = (self._read + self._count) % self.capacity
        first = min(n, self.capacity - write)
        self._buf[write : write + first] = x[:first]
        self._buf[: n - first] = x[first:]
        self._count += n

    def pull(self, n: int) -> np.ndarray:
        """Remove and return the oldest ``n`` samples as a new float32 array."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n > self._count:
            raise ValueError(f"RingBuffer underflow: {n} requested, {self._count} available")
        out = np.empty(n, dtype=np.float32)
        first = min(n, self.capacity - self._read)
        out[:first] = self._buf[self._read : self._read + first]
        out[first:] = self._buf[: n - first]
        self._read = (self._read + n) % self.capacity
        self._count -= n
        return out

    def peek(self, n: int) -> np.ndarray:
        """Like :meth:`pull` without consuming."""
        read, count = self._read, self._count
        out = self.pull(n)
        self._read, self._count = read, count
        return out


class Chunker:
    """Re-blocks arbitrary-length arrays into exact ``hop``-sized float32 chunks.

    Leftover samples are kept for the next :meth:`push`; :meth:`flush` returns the
    zero-padded remainder (or ``None``) and clears it.
    """

    def __init__(self, hop: int = HOP):
        if hop <= 0:
            raise ValueError(f"hop must be positive, got {hop}")
        self.hop = int(hop)
        self._rest = np.zeros(0, dtype=np.float32)

    @property
    def remainder(self) -> int:
        return len(self._rest)

    def reset(self) -> None:
        self._rest = np.zeros(0, dtype=np.float32)

    def push(self, x: np.ndarray) -> Iterator[np.ndarray]:
        x = np.asarray(x, dtype=np.float32).ravel()
        data = np.concatenate([self._rest, x]) if len(self._rest) else x
        full = len(data) // self.hop
        self._rest = np.array(data[full * self.hop :], dtype=np.float32, copy=True)
        for i in range(full):
            yield np.array(data[i * self.hop : (i + 1) * self.hop], dtype=np.float32, copy=True)

    def flush(self) -> np.ndarray | None:
        if not len(self._rest):
            return None
        out = np.zeros(self.hop, dtype=np.float32)
        out[: len(self._rest)] = self._rest
        self.reset()
        return out


def chunk_stream(
    x: np.ndarray, sr: int, hop_ms: float = 10.0, t0: float = 0.0
) -> Iterator[AudioChunk]:
    """Yield ``AudioChunk``s of ``hop_ms`` with ``seq`` 0, 1, 2… and ``capture_ts`` ``t0 + seq*hop``.

    The last chunk is zero-padded to a full hop so every chunk has the same length.
    """
    hop_exact = sr * hop_ms / 1000.0
    hop = round(hop_exact)
    if hop <= 0 or abs(hop_exact - hop) > 1e-6:
        raise ValueError(f"hop_ms={hop_ms} is not a whole number of samples at {sr} Hz")
    x = np.asarray(x, dtype=np.float32)
    hop_s = hop / sr
    for seq, i in enumerate(range(0, len(x), hop)):
        samples = x[i : i + hop]
        if len(samples) < hop:
            padded = np.zeros(hop, dtype=np.float32)
            padded[: len(samples)] = samples
            samples = padded
        yield AudioChunk(samples=samples, sample_rate=sr, seq=seq, capture_ts=t0 + seq * hop_s)
