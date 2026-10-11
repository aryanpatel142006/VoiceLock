"""Lip crops for Focus HQ: cut from tracked faces and shared with every TSE stage of a session.

The video path (the session's vision engine, or a bench replaying a fixture) cuts one
``LIPS_SIZE`` x ``LIPS_SIZE`` grayscale mouth crop per face per frame with :func:`lip_crop` and
appends it to the session's :class:`LipFeed`, which lives in
``ctx.telemetry.extras["lip_feed"]`` (:func:`feed_of`). TSE stages read the feed without
consuming it (several tiers may run at once): each keeps a cursor and takes the entries of the
current target track. Entries carry the frame's capture timestamp on the client clock, the same
clock as ``AudioChunk.capture_ts``, so a stage can place them on its audio sample grid.

Crop geometry: centred on the mean of the inner-lip midpoints and mouth corners (FaceMesh
13/14/78/308) when landmarks exist, else 20 % of the face height below the box centre; side
``LIP_SCALE`` x the larger face dimension. ``LIP_SCALE = 0.5`` matches the fixed mouth box the
bake-off fixtures use (``eval.bakeoff_fixtures.mouth_box_from_face``: 50 % of the face width).
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass

import numpy as np

from hearing.tse.base import LIPS_SIZE
from hearing.types import StageContext

FEED_KEY = "lip_feed"
LIP_SCALE = 0.5  # crop side / max(face width, face height) in pixels
KEEP_S = 6.0  # history kept: longest window (2 s) + hop + vision lag, with margin


@dataclass(frozen=True)
class LipFrame:
    index: int  # feed-wide, strictly increasing: a stage's read cursor
    ts: float  # capture_ts of the video frame (client clock, seconds)
    track_id: int
    crop: np.ndarray  # LIPS_SIZE x LIPS_SIZE uint8 grayscale
    p_speaking: float


def lip_crop(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    landmarks: np.ndarray | None = None,
    size: int = LIPS_SIZE,
) -> np.ndarray:
    """``size`` x ``size`` uint8 grayscale mouth crop of an RGB frame for a normalised face box."""
    from hearing.vision.light_asd import crop_geometry, cut_crop

    h, w = image.shape[:2]
    cx, cy, side, _ = crop_geometry(box, landmarks, w, h, crop_scale=LIP_SCALE)
    return cut_crop(image, cx, cy, side, size)


class LipFeed:
    """Recent lip crops of every tracked face, oldest first, at most ``keep_s`` seconds."""

    def __init__(self, keep_s: float = KEEP_S):
        self.keep_s = float(keep_s)
        self._frames: deque[LipFrame] = deque()
        self._next = itertools.count()

    def append(self, ts: float, track_id: int, crop: np.ndarray, p_speaking: float = 0.0) -> None:
        crop = np.asarray(crop)
        if crop.ndim != 2 or crop.dtype != np.uint8:
            raise ValueError(f"lip crop must be 2-D uint8, got {crop.shape} {crop.dtype}")
        self._frames.append(LipFrame(next(self._next), float(ts), int(track_id), crop,
                                     float(p_speaking)))
        newest = self._frames[-1].ts
        while self._frames and newest - self._frames[0].ts > self.keep_s:
            self._frames.popleft()

    def since(self, cursor: int) -> list[LipFrame]:
        """Frames with ``index >= cursor`` (a stage passes ``last index + 1``)."""
        return [f for f in self._frames if f.index >= cursor]

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


def feed_of(ctx: StageContext, create: bool = False) -> LipFeed | None:
    """The session's feed in ``ctx.telemetry.extras``; with ``create`` one is added if missing."""
    feed = ctx.telemetry.extras.get(FEED_KEY)
    if feed is None and create:
        feed = ctx.telemetry.extras[FEED_KEY] = LipFeed()
    return feed
