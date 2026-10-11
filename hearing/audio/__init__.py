"""Audio primitives: file I/O, resampling, framing, streaming buffers, reference stages.

Pipeline format: mono float32 in [-1, 1] at 16 kHz, framed in 10 ms (160-sample) hops.
Wire format: PCM16 little endian. See docs/CONTRACTS.md and the hearing-stack skill.
"""

from hearing.audio.io import (
    duration_s,
    float_to_pcm16,
    frame,
    load,
    pcm16_to_float,
    resample,
    save,
)
from hearing.audio.stream import Chunker, RingBuffer, chunk_stream

__all__ = [
    "Chunker",
    "RingBuffer",
    "chunk_stream",
    "duration_s",
    "float_to_pcm16",
    "frame",
    "load",
    "pcm16_to_float",
    "resample",
    "save",
]
