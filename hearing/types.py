"""Shared in-process media and stage contracts (see docs/CONTRACTS.md)."""

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

Mode = Literal["natural", "focus_hq", "synthetic", "text", "quiet", "auto", "name_call"]


@dataclass
class AudioChunk:
    """Mono float32 samples in [-1, 1]; producers own format validation."""

    samples: np.ndarray
    sample_rate: int
    seq: int
    capture_ts: float


@dataclass
class VideoFrame:
    """An HxWx3 uint8 RGB image and its capture metadata."""

    image: np.ndarray
    seq: int
    capture_ts: float


@dataclass
class FaceTrack:
    track_id: int
    box: tuple[float, float, float, float]  # Normalised x, y, width, height.
    landmarks: np.ndarray | None
    p_speaking: float
    last_seen_ts: float


@dataclass
class TargetState:
    track_id: int | None
    confidence: float
    locked: bool
    since_ts: float
    # Bumped on every deliberate switch to a different face (lock, or a challenger that won while
    # a target was held); never by an automatic re-detection. AutoEnroller keys on it (T-083).
    switch_seq: int = 0


@dataclass
class Telemetry:
    seq: int = 0
    tier: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)
    device: str = ""
    extras: dict = field(default_factory=dict)


@dataclass
class StageContext:
    mode: Mode
    suppression: float  # 0 = unchanged room, 1 = maximum suppression.
    target: TargetState
    tracks: list[FaceTrack]
    telemetry: Telemetry

    @classmethod
    def default(cls) -> "StageContext":
        """Independent empty context; device/tier are unset until configured."""
        return cls(
            mode="natural",
            suppression=0.5,
            target=TargetState(None, 0.0, False, 0.0),
            tracks=[],
            telemetry=Telemetry(),
        )


@runtime_checkable
class Stage(Protocol):
    name: str
    sample_rate: int  # External input AND output rate; resample internally.
    latency_ms: float  # Algorithmic lookahead/windowing, not compute time.

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk: ...

    def reset(self) -> None: ...
