"""Voice match: apparent gender/age estimates -> a TTS voice bucket from ``config/voices.yaml``.

This module holds no ML models. It defines the contract the face and voice adapters
(``hearing.synth.apparent_face`` / ``hearing.synth.apparent_voice``) implement, offline
mocks of both, the confidence-weighted fusion rule and the per-track ``VoiceMatcher``
the session consumes.

Privacy: apparent gender and age exist **only** to pick a TTS voice. Estimates live in
memory per track and are dropped on ``forget``/``reset``; they are never logged, printed,
persisted, sent over the wire or put in ``Telemetry.extras``. The only thing that leaves
this module is the chosen bucket (shown to the user in the ``voice`` telemetry field
``{bucket, source, override}``), and the user can always override it, including with
``neutral``. Nothing here emits log records about estimates or choices; keep it that way.

Fusion: estimates below ``min_confidence`` are ignored; the rest are averaged weighted by
confidence. Gender is ``female`` when the fused ``p_female >= 0.65``, ``male`` when it is
``<= 0.35`` and otherwise ``neutral`` (which carries no age suffix); age is the argmax of
(young, adult, senior). No usable estimate -> ``neutral`` with source ``default``.

Same-bucket signal (T-070): :class:`SameBucket` tells the tier ladder whether another face seen in the
last :data:`SAME_BUCKET_RECENT_S` shares the target's voice group (the gender half of its bucket), because
the Enrolled tier (ECAPA voice matching) confuses two same-gender voices
(``docs/decisions/association.md``). It writes ``ctx.telemetry.extras["voice_buckets"]`` (track id ->
bucket, the same value the ``voice`` field would show for that face; in-memory only, never serialized)
and ``ctx.telemetry.extras["same_bucket"]`` (bool). A ``neutral`` (unknown) bucket on either side -> False.

``HEARING_VOICE_MATCH=mock`` makes ``get_face_estimator`` / ``get_voice_estimator`` return
the mocks instead of the real adapters.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from hearing.audio.io import PIPELINE_SR
from hearing.synth.tts import bucket_names

if TYPE_CHECKING:
    from hearing.types import FaceTrack, StageContext

AGES = ("young", "adult", "senior")
GENDER_THRESHOLD = 0.65  # fused p_female >= this -> female, <= 1 - this -> male
DEFAULT_MIN_CONFIDENCE = 0.6
EMA_ALPHA = 0.3
MIN_VOICE_SECONDS = 1.0
SAME_BUCKET_PERSIST_S = 1.0  # raw match must hold this long before the flag turns on
SAME_BUCKET_LATCH_S = 5.0  # once on, stays on at least this long after the last match
SAME_BUCKET_KEY = "same_bucket"  # ctx.telemetry.extras: bool, read by hearing.fusion.tiers
VOICE_BUCKETS_KEY = "voice_buckets"  # ctx.telemetry.extras: {track_id: bucket}, internal only
SAME_BUCKET_RECENT_S = 3.0  # another face counts while it was seen within this many seconds

Source = Literal["face+voice", "face", "voice", "default", "override"]


@dataclass(frozen=True)
class ApparentEstimate:
    """One apparent gender/age estimate. In-memory only: never log or store it."""

    p_female: float  # 0-1
    p_age: tuple[float, float, float]  # over AGES (young, adult, senior)
    confidence: float  # 0-1
    source: Literal["face", "voice"]


@dataclass(frozen=True)
class VoiceChoice:
    bucket: str  # a key of config/voices.yaml
    source: Source
    confidence: float


@runtime_checkable
class FaceEstimator(Protocol):
    def estimate(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> ApparentEstimate | None:
        """RGB uint8 frame + normalised (x, y, w, h) from ``FaceTrack.box``."""
        ...


@runtime_checkable
class VoiceEstimator(Protocol):
    def estimate(self, samples: np.ndarray) -> ApparentEstimate | None:
        """Float32 mono 16 kHz, any length >= 1 s; shorter returns None."""
        ...


# --- mocks ------------------------------------------------------------------


def _bucket_estimate(bucket: str, confidence: float, source: str) -> ApparentEstimate:
    """A deterministic estimate that ``fuse`` maps back to ``bucket``."""
    if bucket not in bucket_names():
        raise ValueError(f"Unknown voice bucket {bucket!r}; use one of {bucket_names()}")
    gender, _, age = bucket.partition("_")
    p_female = {"female": 0.9, "male": 0.1}.get(gender, 0.5)
    age = age if age in AGES else "adult"
    p_age = tuple(0.8 if a == age else 0.1 for a in AGES)
    return ApparentEstimate(p_female, p_age, float(confidence), source)


class MockFaceEstimator:
    """Offline face estimator: always the estimate for ``bucket``."""

    def __init__(self, bucket: str = "male_adult", confidence: float = 0.9):
        self._estimate = _bucket_estimate(bucket, confidence, "face")

    def estimate(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> ApparentEstimate | None:
        return self._estimate


class MockVoiceEstimator:
    """Offline voice estimator: the estimate for ``bucket`` once >= 1 s of audio is given."""

    def __init__(self, bucket: str = "male_adult", confidence: float = 0.9):
        self._estimate = _bucket_estimate(bucket, confidence, "voice")

    def estimate(self, samples: np.ndarray) -> ApparentEstimate | None:
        if len(samples) < MIN_VOICE_SECONDS * PIPELINE_SR:
            return None
        return self._estimate


# --- fusion -----------------------------------------------------------------


def fuse(
    face: ApparentEstimate | None,
    voice: ApparentEstimate | None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> VoiceChoice:
    """Confidence-weighted fusion of the face and voice estimates into a bucket."""
    used = [e for e in (face, voice) if e is not None and e.confidence >= min_confidence]
    total = sum(e.confidence for e in used)
    if not used or total <= 0:
        return VoiceChoice("neutral", "default", 0.0)
    p_female = sum(e.confidence * e.p_female for e in used) / total
    p_age = [sum(e.confidence * e.p_age[i] for e in used) / total for i in range(len(AGES))]
    confidence = sum(e.confidence**2 for e in used) / total
    source = "+".join(e.source for e in used)
    if p_female >= GENDER_THRESHOLD:
        gender = "female"
    elif p_female <= 1 - GENDER_THRESHOLD:
        gender = "male"
    else:
        return VoiceChoice("neutral", source, confidence)
    bucket = f"{gender}_{AGES[int(np.argmax(p_age))]}"
    if bucket not in bucket_names():  # voices.yaml lacks it: never return an unplayable bucket
        bucket = "neutral"
    return VoiceChoice(bucket, source, confidence)


def _ema(old: ApparentEstimate | None, new: ApparentEstimate, alpha: float) -> ApparentEstimate:
    if old is None:
        return new

    def mix(a: float, b: float) -> float:
        return (1 - alpha) * a + alpha * b

    return ApparentEstimate(
        p_female=mix(old.p_female, new.p_female),
        p_age=tuple(mix(a, b) for a, b in zip(old.p_age, new.p_age, strict=True)),
        confidence=mix(old.confidence, new.confidence),
        source=new.source,
    )


def check_override(override: str | None) -> None:
    """Raise ``ValueError`` listing the buckets if ``override`` is not one of them."""
    if override is not None and override not in bucket_names():
        raise ValueError(f"Unknown voice override {override!r}; use one of {bucket_names()}")


class VoiceMatcher:
    """Per-track smoothed face/voice estimates (memory only) -> ``VoiceChoice``."""

    def __init__(self, alpha: float = EMA_ALPHA, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.alpha = alpha
        self.min_confidence = min_confidence
        self._face: dict[int, ApparentEstimate] = {}
        self._voice: dict[int, ApparentEstimate] = {}

    def update(
        self,
        track_id: int,
        face: ApparentEstimate | None = None,
        voice: ApparentEstimate | None = None,
    ) -> None:
        if face is not None:
            self._face[track_id] = _ema(self._face.get(track_id), face, self.alpha)
        if voice is not None:
            self._voice[track_id] = _ema(self._voice.get(track_id), voice, self.alpha)

    def choice(self, track_id: int | None, override: str | None = None) -> VoiceChoice:
        check_override(override)
        if override is not None:
            return VoiceChoice(override, "override", 1.0)
        if track_id is None:
            return VoiceChoice("neutral", "default", 0.0)
        return fuse(self._face.get(track_id), self._voice.get(track_id), self.min_confidence)

    def forget(self, track_id: int) -> None:
        self._face.pop(track_id, None)
        self._voice.pop(track_id, None)

    def reset(self) -> None:
        self._face.clear()
        self._voice.clear()

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._face or track_id in self._voice


def to_telemetry(choice: VoiceChoice, override: str | None) -> dict:
    """The CONTRACTS ``voice`` telemetry field: ``{bucket, source, override}`` and nothing else."""
    return {"bucket": choice.bucket, "source": choice.source, "override": override}


def voice_group(bucket: str | None) -> str | None:
    """``male`` / ``female`` from a bucket name; None for ``neutral`` or anything unknown."""
    gender = (bucket or "").partition("_")[0]
    return gender if gender in ("male", "female") else None


class SameBucket:
    """Whether another recently seen face shares the target's voice group (module doc).

    ``update`` records each visible track's bucket and ``now`` (the caller's clock: the session's audio
    capture time, so video timestamps never matter); tracks not seen for ``recent_s`` are forgotten. ``same(target)`` is True only when the target and at least one other
    remembered face both have a known group and the groups match.
    """

    def __init__(self, recent_s: float = SAME_BUCKET_RECENT_S):
        self.recent_s = float(recent_s)
        self._seen: dict[int, tuple[str, float]] = {}  # track_id -> (bucket, last seen)
        self._now = 0.0
        self._latched = False  # hysteresis state behind publish() (T-091)
        self._latch_target: int | None = None
        self._raw_since: float | None = None  # raw same() true since
        self._last_match = 0.0

    def update(self, tracks: Iterable[FaceTrack], bucket_of: Callable[[int], str], now: float) -> None:
        self._now = now
        for t in tracks:  # a track in ctx.tracks is visible now (or held by the tracker for a moment)
            self._seen[t.track_id] = (bucket_of(t.track_id), now)
        self._seen = {k: v for k, v in self._seen.items() if now - v[1] <= self.recent_s}

    def buckets(self) -> dict[int, str]:
        return {k: v[0] for k, v in self._seen.items()}

    def same(self, target_id: int | None) -> bool:
        if target_id is None or target_id not in self._seen:
            return False
        group = voice_group(self._seen[target_id][0])
        return group is not None and any(
            voice_group(b) == group for k, (b, _) in self._seen.items() if k != target_id)

    def latched(self, target_id: int | None) -> bool:
        """Hysteresis over :meth:`same` (T-091): estimator buckets flip between neutral and male/female,
        and every flip would demote/promote the ladder. The match must persist ``MIN_PERSIST_S`` before
        the flag turns on; once on it stays on for at least ``LATCH_MIN_S`` and until ``recent_s``
        has passed without a match."""
        now, raw = self._now, self.same(target_id)
        if target_id != self._latch_target:
            self._latch_target, self._latched, self._raw_since = target_id, False, None
        if raw:
            self._last_match = now
            if self._raw_since is None:
                self._raw_since = now
            if not self._latched and now - self._raw_since >= SAME_BUCKET_PERSIST_S:
                self._latched = True
        else:
            self._raw_since = None
            if self._latched and now - self._last_match >= max(self.recent_s, SAME_BUCKET_LATCH_S):
                self._latched = False
        return self._latched

    def publish(self, ctx: StageContext) -> bool:
        """Write the per-track buckets and the (latched) flag for ``ctx.target`` into ``ctx.telemetry.extras``."""
        same = self.latched(ctx.target.track_id)
        ctx.telemetry.extras[VOICE_BUCKETS_KEY] = self.buckets()
        ctx.telemetry.extras[SAME_BUCKET_KEY] = same
        return same

    def reset(self) -> None:
        self._seen.clear()
        self._latched, self._latch_target, self._raw_since = False, None, None


# --- adapter factories ------------------------------------------------------


def _use_mock() -> bool:
    return os.environ.get("HEARING_VOICE_MATCH", "").strip().lower() == "mock"


def _load_adapter(module: str, cls: str):
    """``module.cls()`` or None when the adapter is missing or says its weights are unavailable.

    An adapter may expose a module-level ``available() -> bool``; its constructor may also
    raise ``ImportError``/``OSError``/``RuntimeError`` when weights or deps are missing.
    """
    try:
        mod = importlib.import_module(f"hearing.synth.{module}")
    except ImportError:
        return None
    available = getattr(mod, "available", None)
    if callable(available) and not available():
        return None
    try:
        return getattr(mod, cls)()
    except (ImportError, OSError, RuntimeError):
        return None


def get_face_estimator() -> FaceEstimator | None:
    """The real face adapter, the mock under ``HEARING_VOICE_MATCH=mock``, else None."""
    if _use_mock():
        return MockFaceEstimator()
    return _load_adapter("apparent_face", "FaceEstimator")


def get_voice_estimator() -> VoiceEstimator | None:
    """The real voice adapter, the mock under ``HEARING_VOICE_MATCH=mock``, else None."""
    if _use_mock():
        return MockVoiceEstimator()
    return _load_adapter("apparent_voice", "VoiceEstimator")
