"""Focus HQ target-speaker extraction: candidate contract, registry, windowing and mocks."""

from hearing.tse import (
    mock,  # noqa: F401  (registers identity and oracle)
    stage,  # noqa: F401  (registers the "tse" pipeline stage)
)
from hearing.tse.base import (
    LIPS_FPS,
    LIPS_SIZE,
    NEEDS,
    SR,
    TSECandidate,
    candidates,
    conform_output,
    get_candidate,
    register_candidate,
)
from hearing.tse.windowed import SlidingWindowExtractor, WindowStats, frame_span

__all__ = [
    "LIPS_FPS",
    "LIPS_SIZE",
    "NEEDS",
    "SR",
    "SlidingWindowExtractor",
    "TSECandidate",
    "WindowStats",
    "candidates",
    "conform_output",
    "frame_span",
    "get_candidate",
    "register_candidate",
]
