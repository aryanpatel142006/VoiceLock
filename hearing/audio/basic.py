"""Reference stages proving the Stage protocol: passthrough, gain(db), delay(ms).

They are deliberately trivial so the M0 benchmark runner has something to measure and
so ``hearing.testing.assert_streaming_equivalent`` has known-good and known-bad cases.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.types import AudioChunk, StageContext


@registry.register("passthrough")
class Passthrough:
    """Returns the chunk unchanged."""

    name = "passthrough"
    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        return chunk

    def reset(self) -> None:
        pass


@registry.register("gain")
class Gain:
    """Scales samples by ``db`` decibels (no clipping; the output edge clips)."""

    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def __init__(self, db: float = 0.0):
        self.db = float(db)
        self.name = f"gain(db={self.db:g})"
        self._scale = np.float32(10.0 ** (self.db / 20.0))

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        samples = np.asarray(chunk.samples, dtype=np.float32) * self._scale
        return replace(chunk, samples=samples.astype(np.float32, copy=False))

    def reset(self) -> None:
        pass


@registry.register("delay")
class Delay:
    """Delays the stream by ``ms`` using a sample FIFO; reports it as ``latency_ms``.

    Output chunks keep the input chunk's length, so the stream stays aligned and the
    first ``ms`` of output is silence.
    """

    sample_rate = PIPELINE_SR

    def __init__(self, ms: float = 0.0):
        if ms < 0:
            raise ValueError(f"delay ms must be non-negative, got {ms}")
        self.ms = float(ms)
        self.latency_ms = self.ms
        self.name = f"delay(ms={self.ms:g})"
        self._n = round(self.sample_rate * self.ms / 1000.0)
        self._state = np.zeros(self._n, dtype=np.float32)

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        x = np.asarray(chunk.samples, dtype=np.float32)
        if self._n == 0:
            return replace(chunk, samples=x)
        joined = np.concatenate([self._state, x])
        out = joined[: len(x)]
        self._state = joined[len(x) :].astype(np.float32, copy=True)
        return replace(chunk, samples=np.array(out, dtype=np.float32, copy=True))

    def reset(self) -> None:
        self._state = np.zeros(self._n, dtype=np.float32)
