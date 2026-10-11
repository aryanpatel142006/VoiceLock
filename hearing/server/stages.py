"""Stages the session server needs that no processing package owns yet."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.types import AudioChunk, StageContext


@registry.register("silence")
class Silence:
    """Outputs zeros of the input length: Quiet mode, and Synthetic/Text until synth exists."""

    name = "silence"
    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        return replace(chunk, samples=np.zeros(len(chunk.samples), dtype=np.float32))

    def reset(self) -> None:
        pass
