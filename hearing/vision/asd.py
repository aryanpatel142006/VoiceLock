"""Active-speaker detection (ASD): who among the face tracks is talking.

Every backend implements :class:`SpeakerDetector` and is built with
:func:`get_asd`. A detector is called once per video frame (25 fps) with the
current tracks from :class:`hearing.vision.tracker.IouTracker` and the audio that
arrived since the previous frame; it fills ``FaceTrack.p_speaking`` in [0, 1].

Backends:

* ``"lipmotion"`` (:class:`LipMotionASD`): pure numpy, no weights. Per track, a
  250 ms ring of mouth openings (inner-lip distance / face height from the
  landmarks) is compared with the audio energy envelope.
* ``"light_asd"``: the Light-ASD model, registered by ``hearing.vision.light_asd``
  through :func:`register_asd` (its deterministic mock backend when weights or
  torch are missing).
* ``"mock"`` (:class:`MockASD`): scripted ``p_speaking`` per ``track_id``.
* ``"auto"``: ``light_asd`` when its module imports, reports ``available()``
  (weights + torch present) and loads, else ``lipmotion``. Keyword arguments
  go to whichever backend is built.

Landmark convention: tracks carry the landmarker's points as an ``(N, 2|3)``
array in any consistent unit (pixels or normalised). T-025 left the layout
open, so the defaults are MediaPipe FaceMesh indices (468/478 points): 13/14 for
the upper/lower inner lip and 10/152 for forehead/chin (face height). Other
landmarkers pass their own indices to :class:`LipMotionASD`.

ASD is a frame-rate helper, not an audio ``Stage``, so it is not registered in
:mod:`hearing.registry`.
"""

from __future__ import annotations

import importlib
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from hearing.types import AudioChunk, FaceTrack, VideoFrame

# MediaPipe FaceMesh indices (also valid for the 478-point face landmarker).
MP_UPPER_INNER_LIP = 13
MP_LOWER_INNER_LIP = 14
MP_FOREHEAD = 10
MP_CHIN = 152


@runtime_checkable
class SpeakerDetector(Protocol):
    name: str

    def update(
        self,
        tracks: list[FaceTrack],
        frame_ts: float,
        audio: AudioChunk | None,
        frame: VideoFrame | None = None,
    ) -> list[FaceTrack]:
        """Fill ``p_speaking`` on ``tracks`` (in place) and return the same list.

        ``audio`` is the audio received since the previous call (any length,
        possibly ``None``). ``frame`` is the video frame the tracks came from, for
        backends that look at pixels (Light-ASD); landmark-only backends ignore it.
        """
        ...

    def reset(self) -> None: ...


_FACTORIES: dict[str, Callable[..., SpeakerDetector]] = {}
# Modules that register optional backends on import; imported on first lookup.
_OPTIONAL_MODULES = {"light_asd": "hearing.vision.light_asd"}


def register_asd(name: str) -> Callable[[Callable[..., SpeakerDetector]], Callable[..., SpeakerDetector]]:
    """Decorator registering a detector class or factory under ``name``."""

    def deco(factory: Callable[..., SpeakerDetector]) -> Callable[..., SpeakerDetector]:
        if name in _FACTORIES or name == "auto":
            raise ValueError(f"ASD backend {name!r} is already registered or reserved")
        _FACTORIES[name] = factory
        return factory

    return deco


def _try_import_optional(name: str) -> None:
    module = _OPTIONAL_MODULES.get(name)
    if module is None or name in _FACTORIES:
        return
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:  # The module exists but one of its imports is missing.
            raise


def asd_names() -> list[str]:
    """Registered backend names (plus ``"auto"``)."""
    for name in _OPTIONAL_MODULES:
        try:
            _try_import_optional(name)
        except ImportError:
            pass
    return sorted([*_FACTORIES, "auto"])


def get_asd(name: str = "auto", **kwargs) -> SpeakerDetector:
    """Build a fresh detector. Unknown names raise ``KeyError`` listing known ones."""
    if name == "auto":
        try:
            _try_import_optional("light_asd")
            module = importlib.import_module(_OPTIONAL_MODULES["light_asd"])
            if "light_asd" in _FACTORIES and module.available():
                detector = _FACTORIES["light_asd"](backend="torch", **kwargs)
                _ = detector.backend  # Load now, so a broken checkpoint falls back here.
                return detector
        except (ImportError, OSError, RuntimeError, KeyError, ValueError, TypeError):
            pass  # Weights/runtime missing or lipmotion-only kwargs: the fallback needs neither.
        return _FACTORIES["lipmotion"](**kwargs)
    _try_import_optional(name)
    if name not in _FACTORIES:
        raise KeyError(f"Unknown ASD backend {name!r}; known: {', '.join(asd_names())}")
    return _FACTORIES[name](**kwargs)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 1e-12 else 0.0


@dataclass
class _TrackState:
    ts: deque = field(default_factory=deque)  # Frame timestamps.
    opening: deque = field(default_factory=deque)  # Mouth opening / face height.
    energy: deque = field(default_factory=deque)  # Frame audio energy, dB above floor.
    p: float | None = None  # EMA-smoothed probability.
    last_ts: float | None = None


@register_asd("lipmotion")
class LipMotionASD:
    """Lip-opening × audio-energy active-speaker detector (no weights).

    Per frame and track, three features in [0, 1]:

    * ``motion``: std of the mouth opening over the window / ``motion_ref``.
    * ``coupling``: Pearson correlation (best lag within ``±max_lag_frames``,
      clipped at 0) of mouth opening and the audio energy envelope, weighted by
      ``motion`` and ``activity`` so landmark jitter and room noise cannot fake it.
    * ``activity``: loudest frame energy in the last ``activity_window_s`` (``None``:
      ``window_s``) above the noise floor, ``(dB - floor - margin_db) / range_db``.

    ``p = sigmoid(bias + w_motion*motion + w_coupling*coupling + w_activity*activity
    + w_motion_activity*motion*activity)``, then an EMA with time constant ``ema_s``. Tracks without (or with stale)
    landmarks get ``audio_only_max * activity`` and ``confidence`` 0; until any
    audio has arrived, lip motion alone is used (``video_only_*``).
    ``confidence[track_id]`` is the fraction of the window backed by landmarks.

    Defaults are the T-051 operating point on ``av_full`` (GRID two-face, 80 clips):
    ``window_s``/``activity_window_s``/``motion_ref`` from a grid sweep, then ``bias``/
    ``w_activity``/``w_motion_activity`` from a logistic fit of the per-frame features to
    the labels (so ``p`` is calibrated, 0.5 = even odds). Two weights are held, not fitted,
    because that set cannot see what they guard against: ``w_coupling`` = 5.0 (both faces
    talk at once there, but coupling is what rejects a moving mouth that does not match
    the audio) and ``w_motion`` = 0.5 (babble noise keeps ``activity`` up, so a free fit
    lets lip motion alone say "speaking"; capped, a mouth moving in silence stays at
    p ≈ 0.22). Leave-one-pair-out frame accuracy at SNR 0/5 dB is 0.83 (0.58 with the
    T-036 defaults); see ``docs/BENCHMARKS.md``.
    """

    name = "lipmotion"

    def __init__(
        self,
        window_s: float = 0.25,
        ema_s: float = 0.1,
        audio_frame_ms: float = 10.0,
        min_db: float = -65.0,
        floor_rise_db_per_s: float = 3.0,
        margin_db: float = 6.0,
        range_db: float = 12.0,
        motion_ref: float = 0.01,
        max_lag_frames: int = 1,
        min_frames: int = 4,
        bias: float = -1.75,
        w_motion: float = 0.5,
        w_coupling: float = 5.0,
        w_activity: float = -1.15,
        w_motion_activity: float = 3.1,
        audio_only_max: float = 0.45,
        video_only_bias: float = -3.0,
        video_only_w_motion: float = 5.0,
        stale_s: float = 0.1,
        activity_window_s: float | None = 1.0,
        upper_lip: int = MP_UPPER_INNER_LIP,
        lower_lip: int = MP_LOWER_INNER_LIP,
        face_top: int = MP_FOREHEAD,
        face_bottom: int = MP_CHIN,
    ) -> None:
        self.window_s = window_s
        self.ema_s = ema_s
        self.audio_frame_ms = audio_frame_ms
        self.min_db = min_db
        self.floor_rise_db_per_s = floor_rise_db_per_s
        self.margin_db = margin_db
        self.range_db = range_db
        self.motion_ref = motion_ref
        self.max_lag_frames = max_lag_frames
        self.min_frames = min_frames
        self.bias = bias
        self.w_motion = w_motion
        self.w_coupling = w_coupling
        self.w_activity = w_activity
        self.w_motion_activity = w_motion_activity
        self.audio_only_max = audio_only_max
        self.video_only_bias = video_only_bias
        self.video_only_w_motion = video_only_w_motion
        self.stale_s = stale_s
        self.activity_window_s = window_s if activity_window_s is None else activity_window_s
        self._idx = (upper_lip, lower_lip, face_top, face_bottom)
        self._min_points = max(self._idx) + 1
        self.confidence: dict[int, float] = {}
        self.features: dict[int, dict[str, float]] = {}  # Last features, for telemetry/debug.
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[int, _TrackState] = {}
        self._carry = np.zeros(0, dtype=np.float32)  # Samples short of a full 10 ms frame.
        self._pending: list[float] = []  # 10 ms energies (dB) since the last video frame.
        self.audio_energies: deque[float] = deque(maxlen=max(1, round(self.window_s * 1000 / self.audio_frame_ms)))
        self._floor: float | None = None
        self._have_audio = False
        self._frame_db: float | None = None  # Last per-video-frame energy (dB).
        self._audio_ts: deque[float] = deque()  # Frame timestamps for the activity window.
        self._audio_db: deque[float] = deque()  # Per-frame energy above floor (dB).
        self.confidence.clear()
        self.features.clear()

    # ------------------------------------------------------------------ audio
    def push_audio(self, chunk: AudioChunk) -> None:
        """Accumulate audio between video frames (``update(audio=...)`` calls this)."""
        x = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        hop = max(1, round(chunk.sample_rate * self.audio_frame_ms / 1000.0))
        if self._carry.size:
            x = np.concatenate([self._carry, x])
        n = (x.size // hop) * hop
        self._carry = x[n:].copy()
        if n == 0:
            return
        frames = x[:n].reshape(-1, hop).astype(np.float64)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        db = np.maximum(20.0 * np.log10(rms + 1e-12), self.min_db)
        rise = self.floor_rise_db_per_s * self.audio_frame_ms / 1000.0
        floor = self._floor
        for e in db.tolist():
            if floor is None or e < floor:
                floor = e  # Minimum tracking: follow dips at once, rise slowly.
            else:
                floor = min(floor + rise, e)
            self._pending.append(e)
            self.audio_energies.append(e)
        self._floor = floor
        self._have_audio = True

    def _frame_energy(self) -> float | None:
        """One envelope value per video frame: power mean of pending 10 ms energies."""
        if self._pending:
            p = np.asarray(self._pending)
            self._frame_db = float(10.0 * np.log10(np.mean(10.0 ** (p / 10.0))))
            self._pending.clear()
        if self._frame_db is None or self._floor is None:
            return None
        return max(self._frame_db - self._floor, 0.0)

    # ------------------------------------------------------------------ video
    def _opening(self, lm: np.ndarray | None) -> float | None:
        if lm is None:
            return None
        lm = np.asarray(lm)
        if lm.ndim != 2 or lm.shape[0] < self._min_points or lm.shape[1] < 2:
            return None
        up, lo, top, bot = (lm[i, :2].astype(np.float64) for i in self._idx)
        height = math.hypot(*(bot - top))
        if not height > 1e-9:
            return None
        return math.hypot(*(lo - up)) / height

    def update(
        self,
        tracks: list[FaceTrack],
        frame_ts: float,
        audio: AudioChunk | None,
        frame: VideoFrame | None = None,
    ) -> list[FaceTrack]:
        if audio is not None:
            self.push_audio(audio)
        energy = self._frame_energy()
        horizon = frame_ts - self.window_s
        if energy is not None:
            self._audio_ts.append(frame_ts)
            self._audio_db.append(energy)
        while self._audio_ts and self._audio_ts[0] <= frame_ts - self.activity_window_s:
            self._audio_ts.popleft()
            self._audio_db.popleft()
        loudest = max(self._audio_db) if self._audio_db else 0.0
        activity = min(max((loudest - self.margin_db) / self.range_db, 0.0), 1.0)

        live = {t.track_id for t in tracks}
        for tid in [tid for tid in self._tracks if tid not in live]:
            del self._tracks[tid]
            self.confidence.pop(tid, None)
            self.features.pop(tid, None)

        for track in tracks:
            st = self._tracks.get(track.track_id)
            if st is None:
                st = self._tracks[track.track_id] = _TrackState()
            fresh = track.last_seen_ts + self.stale_s >= frame_ts
            opening = self._opening(track.landmarks) if fresh else None
            if opening is not None:
                st.ts.append(frame_ts)
                st.opening.append(opening)
                st.energy.append(energy if energy is not None else 0.0)
            while st.ts and st.ts[0] <= horizon:
                st.ts.popleft()
                st.opening.popleft()
                st.energy.popleft()
            n = len(st.ts)
            expected = max(1.0, self.window_s * 25.0)
            self.confidence[track.track_id] = min(n / expected, 1.0) if opening is not None else 0.0

            if opening is None:
                raw = self.audio_only_max * activity if self._have_audio else 0.0
                feats = {"motion": 0.0, "coupling": 0.0, "activity": activity}
            else:
                op = np.fromiter(st.opening, dtype=np.float64, count=n)
                motion = min(float(op.std()) / self.motion_ref, 1.0) if n >= 2 else 0.0
                if not self._have_audio:
                    raw = _sigmoid(self.video_only_bias + self.video_only_w_motion * motion)
                    feats = {"motion": motion, "coupling": 0.0, "activity": 0.0}
                else:
                    corr = 0.0
                    if n >= self.min_frames:
                        en = np.fromiter(st.energy, dtype=np.float64, count=n)
                        for lag in range(-self.max_lag_frames, self.max_lag_frames + 1):
                            a = op[max(lag, 0) : n + min(lag, 0)]
                            b = en[max(-lag, 0) : n + min(-lag, 0)]
                            if a.size >= self.min_frames:
                                corr = max(corr, _pearson(a, b))
                    coupling = corr * motion * activity
                    logit = (
                        self.bias
                        + self.w_motion * motion
                        + self.w_coupling * coupling
                        + self.w_activity * activity
                        + self.w_motion_activity * motion * activity
                    )
                    raw = _sigmoid(logit)
                    feats = {"motion": motion, "coupling": coupling, "activity": activity}
            self.features[track.track_id] = feats

            if st.p is None or st.last_ts is None:
                st.p = raw
            else:
                dt = max(frame_ts - st.last_ts, 0.0)
                alpha = 1.0 - math.exp(-dt / self.ema_s) if self.ema_s > 0 else 1.0
                st.p += alpha * (raw - st.p)
            st.last_ts = frame_ts
            track.p_speaking = float(min(max(st.p, 0.0), 1.0))
        return tracks


@register_asd("mock")
class MockASD:
    """Scripted detector: ``p_speaking = script.get(track_id, default)``."""

    name = "mock"

    def __init__(self, script: dict[int, float] | None = None, default: float = 0.0) -> None:
        self.script = dict(script or {})
        self.default = default
        self.calls = 0

    def update(
        self,
        tracks: list[FaceTrack],
        frame_ts: float,
        audio: AudioChunk | None,
        frame: VideoFrame | None = None,
    ) -> list[FaceTrack]:
        self.calls += 1
        for t in tracks:
            t.p_speaking = float(self.script.get(t.track_id, self.default))
        return tracks

    def reset(self) -> None:
        self.calls = 0
