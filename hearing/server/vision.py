"""Video path of a session: JPEG frames -> faces -> tracks -> p(speaking) -> target.

Per client video frame (``kind`` 2, JPEG ≤ 640×360): decode to RGB (shrunk to fit
640×360 if larger), detect faces with landmarks, associate them into stable tracks
(:class:`~hearing.vision.tracker.IouTracker`), fill ``p_speaking`` with the ASD backend
(:func:`~hearing.vision.asd.get_asd`, fed the raw room audio captured up to the frame's
``capture_ts``) and pick the target (:class:`~hearing.vision.target.TargetSelector`).

Decode, detection, tracking and ASD run on one worker thread so the audio path never
waits (laptop-a: MediaPipe ≈ 10 ms, Light-ASD on MPS ≈ 27 ms for two faces, lip-motion
< 0.1 ms). Up to :data:`MAX_IN_FLIGHT` frames are in flight (one running, one queued behind it),
so a worker that takes ~38 ms against the camera's 40 ms keeps up through jitter instead of
dropping every frame that arrives a few ms early (T-072: 30-43 % dropped, 16 fps on laptop-a);
a frame that arrives with the queue full is dropped (counted in telemetry). The tracker and the
ASD are only touched by that thread.
Target selection is cheap and runs on the session thread when the session polls, so
``StageContext.tracks``/``target`` only change between audio hops.

Face detectors (``HEARING_FACES`` or :class:`~hearing.server.session.ServerConfig.faces`):

* ``"mediapipe"``: :class:`hearing.vision.faces.MediaPipeBackend` (BlazeFace boxes +
  Face Landmarker, 478 normalised landmarks, up to 4 faces; weights from
  ``python -m hearing.vision.faces --download``). Runs on the CPU delegate unless
  ``HEARING_FACES_DELEGATE=gpu``: the Metal path aborted the live server on laptop-a (T-060).
* ``"mock"``: :class:`MarkerFaces` finds pure-green rectangles (the test "faces") and
  reads a mouth opening from black rows inside them. Deterministic, no weights.
* ``"none"``: no detector; the session stays audio-only.
* ``"auto"`` (default): ``mediapipe`` when available, else ``none``.

No camera, no detector, or no frame for :data:`LIVE_S` seconds means the session is
audio-only: tracks and target are cleared, so the gate stage bypasses.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from hearing.audio.io import PIPELINE_SR
from hearing.server.protocol import error_message
from hearing.tse.gpu import TORCH_LOCK
from hearing.tse.lips import feed_of, lip_crop
from hearing.types import AudioChunk, FaceTrack, StageContext, TargetState, VideoFrame
from hearing.vision import faces
from hearing.vision.asd import SpeakerDetector, get_asd
from hearing.vision.target import TargetSelector
from hearing.vision.tracker import Detection, IouTracker

MAX_W, MAX_H = 640, 360  # docs/CONTRACTS.md: video in is at most 640x360
LIVE_S = 1.0  # no applied frame for this long (wall clock) -> audio-only
# T-138: with an explicit lock a short video gap (a backgrounded browser tab throttles capture to
# 1 fps; a GPU stall) keeps the tracker, its ids and the lock instead of going dark and back
LOCK_GAP_S = 3.0
FACE_EVERY_S = 1.0  # apparent-face estimate at most once per second per track
MAX_FACE_TRACKS = 4  # apparent-face estimates per frame at most
AUDIO_KEEP_S = 1.0  # raw audio waiting for the next video frame (ASD input)
LIP_TRACK_S = 0.3  # a dropped frame still gets lip crops at tracks seen this recently
MAX_IN_FLIGHT = 2  # frames submitted to the worker and not yet collected (running + queued)
# A GPU ASD waits this long for a Focus HQ / enrolled window to free the GPU, else skips the frame.
# T-072: the enrolled tier's MossFormer2 separation holds the GPU ~240-330 ms per 0.5 s hop on MPS;
# waiting 100 ms stalled the worker and dropped 30-37 % of frames (17-19 fps). 20 ms keeps the
# worker within the 40 ms frame budget; skipped frames keep the last p_speaking.
ASD_GPU_WAIT_S = 0.02
# T-083: while the GPU is busy the ASD runs on the CPU (Light-ASD, ~37 ms for two tracks), but at most
# once per this gap: every frame on the CPU halved the vision fps (22 -> 10; gap 120 ms still gave p95 225 ms) next to MediaPipe. The
# in-between frames keep p_speaking, which is then at most ~gap + a frame old (~ 200 ms).
ASD_CPU_GAP_S = 0.08
MAX_FACES = 4
DETECTORS = ("auto", "mediapipe", "mock", "none")

log = logging.getLogger(__name__)


def decode_jpeg(data: bytes, max_size: tuple[int, int] = (MAX_W, MAX_H)) -> np.ndarray:
    """JPEG bytes -> HxWx3 uint8 RGB, shrunk (aspect kept) to fit ``max_size``.

    Raises ``ValueError`` for anything Pillow cannot decode.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.draft("RGB", max_size)  # JPEG DCT scaling: decode big frames at a fraction
            rgb = im.convert("RGB")
            if rgb.width > max_size[0] or rgb.height > max_size[1]:
                rgb.thumbnail(max_size)
            return np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"cannot decode JPEG ({len(data)} bytes): {e}") from None


# ---------------------------------------------------------------- face detectors


class FaceDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray, ts: float) -> list[Detection]:
        """Faces in one RGB frame captured at ``ts`` (seconds, increasing)."""
        ...

    def close(self) -> None: ...


def mediapipe_available() -> bool:
    """``mediapipe`` importable and both pinned face models in ``$HEARING_MODELS_DIR``."""
    return importlib.util.find_spec("mediapipe") is not None and faces.weights_available()


class MediaPipeFaces:
    """:class:`hearing.vision.faces.MediaPipeBackend` in VIDEO mode, one per session.

    ``delegate`` defaults to :func:`hearing.vision.faces.default_delegate` (``cpu`` unless
    ``HEARING_FACES_DELEGATE=gpu``).
    """

    name = "mediapipe"

    def __init__(self, max_faces: int = MAX_FACES, delegate: str | None = None):
        self.delegate = delegate or faces.default_delegate()
        self._backend = faces.MediaPipeBackend(
            running_mode="video", max_faces=max_faces, delegate=self.delegate
        )
        log.info("faces: mediapipe on the %s delegate", self.delegate)

    def detect(self, image: np.ndarray, ts: float) -> list[Detection]:
        return self._backend.detect(image, ts)

    def close(self) -> None:
        self._backend.close()


class MarkerFaces:
    """Mock detector: every pure-green rectangle is a face, black rows inside are its mouth.

    A pixel is "green" when G ≥ 200 and R, B ≤ 80; "black" when all channels ≤ 60.
    Landmarks follow the MediaPipe FaceMesh layout (478×2, normalised) with only the
    points the lip-motion ASD reads set: 10/152 = box top/bottom, 13/14 = mouth top and
    bottom (both at the box centre when there is no mouth).
    """

    name = "mock"

    def __init__(self, min_px: int = 8):
        self.min_px = min_px

    def detect(self, image: np.ndarray, ts: float) -> list[Detection]:
        from scipy import ndimage

        img = np.asarray(image)
        h, w = img.shape[:2]
        r, g, b = (img[..., i].astype(np.int16) for i in range(3))
        dark = (r <= 60) & (g <= 60) & (b <= 60)
        green = (g >= 200) & (r <= 80) & (b <= 80)
        labels, _ = ndimage.label(green)  # a mouth hole does not split its face
        detections = []
        for sl in ndimage.find_objects(labels):
            if sl is None:
                continue
            ys, xs = sl
            bh, bw = ys.stop - ys.start, xs.stop - xs.start
            if bh < self.min_px or bw < self.min_px:
                continue
            rows = np.flatnonzero(dark[ys, xs].any(axis=1)) + ys.start
            cx = (xs.start + xs.stop) / 2 / w
            lm = np.zeros((478, 2), dtype=np.float32)
            lm[:, 0] = cx
            lm[10] = (cx, ys.start / h)
            lm[152] = (cx, ys.stop / h)
            if rows.size:
                lm[13] = (cx, rows[0] / h)
                lm[14] = (cx, (rows[-1] + 1) / h)
            else:
                lm[13] = lm[14] = (cx, (ys.start + ys.stop) / 2 / h)
            box = (xs.start / w, ys.start / h, bw / w, bh / h)
            detections.append(Detection(box, 1.0, lm))
        return detections

    def close(self) -> None:
        pass


def get_face_detector(name: str = "auto") -> FaceDetector | None:
    """A fresh detector, or None for ``"none"`` (and ``"auto"`` without MediaPipe).

    Unknown names raise ``ValueError``; an explicit ``"mediapipe"`` that cannot load
    raises ``RuntimeError``.
    """
    if name not in DETECTORS:
        raise ValueError(f"unknown face detector {name!r}; expected one of {DETECTORS}")
    if name == "mock":
        return MarkerFaces()
    if name == "mediapipe" or (name == "auto" and mediapipe_available()):
        return MediaPipeFaces()
    return None


# ---------------------------------------------------------------- engine


@dataclass
class _Result:
    """One frame out of the worker: tracks with ``p_speaking``, or why there are none."""

    seq: int
    ts: float
    gen: int = 0  # VisionEngine._gen at submit; older generations are discarded
    image: np.ndarray | None = None
    tracks: list[FaceTrack] = field(default_factory=list)
    lips: dict[int, np.ndarray] = field(default_factory=dict)  # track_id -> 88x88 mouth crop
    worker_ms: float = 0.0
    submitted: float = 0.0  # time.monotonic() at submit
    error: str | None = None  # this frame is unusable (bad JPEG)
    fatal: str | None = None  # no detector: the session is audio-only from now on


class VisionEngine:
    """The video side of one session. Not thread-safe: call it from the session thread.

    ``on_face(track_id, image, box)`` (the session's apparent-face hook) is called at most
    every :data:`FACE_EVERY_S` per track, for up to :data:`MAX_FACE_TRACKS` tracks.
    ``warmup()`` runs once on the worker with the detector load, before the first frame
    is applied, so model loads behind ``on_face`` (≈ 0.5 s) never stall the audio path.
    With ``lips`` (default) the worker also cuts each visible track's 88x88 mouth crop
    (:func:`hearing.tse.lips.lip_crop`, ≈ 0.1 ms per face) and :meth:`poll` appends them to the
    :class:`~hearing.tse.lips.LipFeed` in ``ctx`` (if the session put one there) for Focus HQ.
    A frame dropped because the worker is busy (the ASD waits for the GPU while Focus HQ runs)
    is still decoded on the session thread (≈ 2 ms) and cut at the last applied tracks' boxes and
    landmarks, so the lip stream stays at the camera rate: holes of 250 ms in it cost Focus HQ
    about 10 dB SI-SDRi (T-019).
    """

    def __init__(self, detector: str = "auto", asd: str = "auto", threaded: bool = True,
                 on_face: Callable[[int, np.ndarray, tuple], list[dict]] | None = None,
                 warmup: Callable[[], None] | None = None, lips: bool = True):
        if detector not in DETECTORS:
            raise ValueError(f"unknown face detector {detector!r}; expected one of {DETECTORS}")
        self.detector_name = detector
        self.threaded = threaded
        self.on_face = on_face
        self.warmup = warmup
        self.lips = lips
        self.asd_name = asd
        self.tracker = IouTracker()  # worker thread only
        self.asd: SpeakerDetector | None = None  # built on the worker (Light-ASD loads torch)
        self.selector = TargetSelector()
        self._gen = 0  # bumped by _go_dark: the worker resets tracker and ASD on the next frame
        self._worker_gen = 0
        self._detector: FaceDetector | None = None
        self._unavailable: str | None = None  # why there is no detector
        self._pool: ThreadPoolExecutor | None = None
        self._jobs: deque[Future] = deque()  # in submit order, at most MAX_IN_FLIGHT
        self._ready: list[_Result] = []
        self._audio: deque[tuple[float, np.ndarray]] = deque()
        self._audio_n = 0
        self._face_ts: dict[int, float] = {}
        self._applied_at: deque[float] = deque()  # monotonic times of applied frames (fps)
        self._last_applied: float | None = None  # monotonic time of the newest applied frame
        self._ms_sum = 0.0
        self._ms_n = 0
        self._ms = 0.0
        self._decode_reported = False
        self._feed = None  # the session's LipFeed, seen in poll()
        self._lip_tracks: list[FaceTrack] = []  # last applied tracks (boxes + landmarks)
        self.lip_only = 0  # dropped frames that still gave lip crops
        self.asd_skipped = 0  # frames whose GPU ASD gave way to Focus HQ (worker thread)
        self._last_p: dict[int, float] = {}  # worker thread only
        self.asd_cpu = 0  # frames whose ASD ran on the CPU because the GPU lock was busy (T-083)
        self._cpu_backend = None  # TorchBackend("cpu"), False once it failed to build
        self._last_update = time.monotonic()
        self._age_ms: deque[float] = deque(maxlen=250)  # last 250 gaps between fresh p_speaking updates (worker thread)
        self.live = False
        self.frames_in = 0
        self.frames_applied = 0
        self.dropped = 0
        self.last_ts: float | None = None  # capture_ts of the last applied frame

    # ------------------------------------------------------------ input

    def submit(self, seq: int, ts: float, jpeg: bytes) -> None:
        """Queue one JPEG frame; dropped if the worker is still busy with the previous one."""
        self.frames_in += 1
        if self._unavailable is not None:
            return
        self._collect()
        now = time.monotonic()
        if self.threaded and len(self._jobs) >= MAX_IN_FLIGHT:
            self.dropped += 1  # its audio stays queued for the next accepted frame
            self._lips_only(ts, jpeg)
            return
        args = (seq, ts, jpeg, now, self._gen, self._take_audio(ts, seq))
        if not self.threaded:
            self._ready.append(self._work(*args))
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="vision")
        self._jobs.append(self._pool.submit(self._work, *args))

    def _lips_only(self, ts: float, jpeg: bytes) -> None:
        """Lip crops of a dropped frame at the last applied tracks (no detection, no ASD)."""
        if not self.lips or self._feed is None:
            return
        recent = [t for t in self._lip_tracks if ts - t.last_seen_ts <= LIP_TRACK_S]
        if not recent:
            return
        try:
            image = decode_jpeg(jpeg)
        except (ValueError, ImportError):
            return
        for t in recent:
            try:
                self._feed.append(ts, t.track_id, lip_crop(image, t.box, t.landmarks), t.p_speaking)
            except ValueError as e:
                log.debug("lip crop of track %d failed: %s", t.track_id, e)
        self.lip_only += 1

    def push_audio(self, samples: np.ndarray, ts: float) -> None:
        """Raw room audio at 16 kHz captured at ``ts``; handed to the ASD with the next frame."""
        if self.frames_in == 0 or self._unavailable is not None or not len(samples):
            return
        self._audio.append((ts, np.asarray(samples, dtype=np.float32)))
        self._audio_n += len(samples)
        while self._audio_n > AUDIO_KEEP_S * PIPELINE_SR and len(self._audio) > 1:
            self._audio_n -= len(self._audio.popleft()[1])

    def lock(self, track_id: int) -> None:
        self.selector.lock(track_id)

    def unlock(self) -> None:
        self.selector.unlock()

    # ------------------------------------------------------------ worker

    def _work(self, seq: int, ts: float, jpeg: bytes, submitted: float, gen: int,
              audio: AudioChunk | None) -> _Result:
        t0 = time.perf_counter()
        result = _Result(seq, ts, gen, submitted=submitted)
        try:
            result.image = decode_jpeg(jpeg)
        except ValueError as e:
            result.error = str(e)
            return result
        except ImportError as e:  # Pillow missing: no video at all
            result.fatal = f"cannot decode JPEG without Pillow: {e}"
            return result
        if self._detector is None:
            try:
                self._detector = get_face_detector(self.detector_name)
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                result.fatal = f"face detector {self.detector_name!r} unavailable: {e}"
                return result
            if self._detector is None:
                result.fatal = ("no face detector (mediapipe or its weights missing: "
                                "python -m hearing.vision.faces --download)")
                return result
            with TORCH_LOCK:  # Light-ASD loads torch onto the GPU
                self.asd = self._build_asd()
            if self.warmup is not None:
                self.warmup()
            t0 = time.perf_counter()  # model loads are not per-frame cost
        try:
            detections = self._detector.detect(result.image, ts)
        except Exception as e:  # noqa: BLE001  (native detector error: degrade, keep audio)
            result.fatal = f"face detector failed: {type(e).__name__}: {e}"
            return result
        if gen != self._worker_gen:  # the camera went dark since the last frame
            self._worker_gen = gen
            self.tracker.reset()
            self.asd.reset()
        result.tracks = self.tracker.update(detections, ts)
        try:
            self._run_asd(result.tracks, ts, audio, VideoFrame(result.image, seq, ts))
        except Exception as e:  # noqa: BLE001  (torch/runtime error: keep faces, drop to lipmotion)
            log.warning("ASD %s failed (%s: %s); using lipmotion", self.asd.name,
                        type(e).__name__, e)
            self.asd = get_asd("lipmotion")
            self.asd.update(result.tracks, ts, audio, VideoFrame(result.image, seq, ts))
        if self.lips:
            for t in result.tracks:
                if t.last_seen_ts == ts:  # seen in this frame (not a coasting track)
                    try:
                        result.lips[t.track_id] = lip_crop(result.image, t.box, t.landmarks)
                    except ValueError as e:
                        log.debug("lip crop of track %d failed: %s", t.track_id, e)
        result.worker_ms = 1000.0 * (time.perf_counter() - t0)
        return result

    def _run_asd(self, tracks: list[FaceTrack], ts: float, audio: AudioChunk | None,
                 frame: VideoFrame) -> None:
        """ASD on the worker. A GPU model shares :data:`TORCH_LOCK` with Focus HQ (MPS is not
        thread-safe); if the GPU stays busy for :data:`ASD_GPU_WAIT_S` the frame keeps the last
        ``p_speaking`` per track instead, so a long Focus HQ window never stalls vision long
        enough to go dark (and drop the user's lock)."""
        backend = getattr(self.asd, "backend", None)
        fresh = True  # p_speaking was recomputed on this frame
        if getattr(backend, "device", "cpu") == "cpu":
            self.asd.update(tracks, ts, audio, frame)
        elif TORCH_LOCK.acquire(timeout=ASD_GPU_WAIT_S):
            try:
                self.asd.update(tracks, ts, audio, frame)
            finally:
                TORCH_LOCK.release()
        elif time.monotonic() - self._last_update >= ASD_CPU_GAP_S and self._update_on_cpu(
                tracks, ts, audio, frame):
            self.asd_cpu += 1
        else:
            self.asd_skipped += 1
            fresh = False
            push = getattr(self.asd, "push_audio", None)
            if audio is not None and callable(push):
                push(audio)  # keep the ASD's audio features continuous
            for t in tracks:
                t.p_speaking = self._last_p.get(t.track_id, 0.0)
        if fresh:
            now = time.monotonic()
            self._age_ms.append(1000.0 * (now - self._last_update))
            self._last_update = now
        self._last_p = {t.track_id: float(t.p_speaking) for t in tracks}

    def _update_on_cpu(self, tracks: list[FaceTrack], ts: float, audio: AudioChunk | None,
                       frame: VideoFrame) -> bool:
        """The GPU is busy with a Focus HQ window: run Light-ASD (~1M params) on the CPU instead,
        which needs no :data:`TORCH_LOCK`, so ``p_speaking`` stays fresh (T-083). False when the ASD
        has no torch backend to copy (lipmotion, mocks) or the CPU copy cannot be built."""
        gpu = getattr(self.asd, "backend", None)
        if os.environ.get("HEARING_ASD_CPU", "1") == "0":  # off: the pre-T-083 skip (benchmarks)
            return False
        if getattr(self.asd, "name", "") != "light_asd" or not hasattr(gpu, "net"):
            return False
        if self._cpu_backend is None:
            try:
                from hearing.vision.light_asd import TorchBackend

                self._cpu_backend = TorchBackend("cpu")
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                log.warning("Light-ASD CPU fallback unavailable (%s); skipping busy frames", e)
                self._cpu_backend = False
                return False
        if self._cpu_backend is False:
            return False
        self.asd._backend = self._cpu_backend
        try:
            self.asd.update(tracks, ts, audio, frame)
        finally:
            self.asd._backend = gpu
        return True

    def asd_age_ms(self) -> tuple[float, float]:
        """(p50, p95) over the last 250 fresh ASD updates of the age ``p_speaking`` had when it was
        replaced: the gap between consecutive recomputations, in ms."""
        ages = sorted(self._age_ms)
        if not ages:
            return 0.0, 0.0
        return ages[len(ages) // 2], ages[min(len(ages) - 1, int(0.95 * len(ages)))]

    def _build_asd(self) -> SpeakerDetector:
        """The configured ASD; ``lipmotion`` (no weights) when it cannot load."""
        try:
            return get_asd(self.asd_name)
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as e:
            log.warning("ASD %r unavailable (%s); using lipmotion", self.asd_name, e)
            return get_asd("lipmotion")

    def _collect(self, timeout: float = 0.0) -> None:
        """Move finished frames (in order) to ``_ready``, waiting up to ``timeout`` in total."""
        deadline = time.monotonic() + timeout
        while self._jobs:
            try:
                self._ready.append(self._jobs[0].result(max(0.0, deadline - time.monotonic())))
            except TimeoutError:
                return
            self._jobs.popleft()

    # ------------------------------------------------------------ session thread

    def poll(self, ctx: StageContext, timeout: float = 0.0) -> list[dict]:
        """Apply finished frames to ``ctx`` (tracks, target); returns ``error`` events."""
        self._collect(timeout)
        self._feed = feed_of(ctx)
        events: list[dict] = []
        ready, self._ready = self._ready, []
        for result in ready:
            events.extend(self._apply(result, ctx))
        dark_s = LOCK_GAP_S if self.selector.locked else LIVE_S
        if self.live and (self._last_applied is None or time.monotonic() - self._last_applied > dark_s):
            self._go_dark(ctx)
        return events

    def _apply(self, r: _Result, ctx: StageContext) -> list[dict]:
        if r.fatal is not None:
            self._unavailable = r.fatal
            self._go_dark(ctx)
            self._audio.clear()
            self._audio_n = 0
            return [error_message("vision", r.fatal, degraded_to="audio-only")]
        if r.error is not None:
            if self._decode_reported:
                return []
            self._decode_reported = True  # once per session; telemetry counts the rest
            return [error_message("vision", r.error)]
        if r.gen != self._gen:  # detected before the camera went dark: stale ids
            return []
        t0 = time.perf_counter()
        tracks = r.tracks
        ctx.target = self.selector.update(tracks, r.ts)
        ctx.tracks = tracks
        self._lip_tracks = tracks
        feed = feed_of(ctx)
        if feed is not None:
            p = {t.track_id: float(t.p_speaking) for t in tracks}
            for track_id, crop in r.lips.items():
                feed.append(r.ts, track_id, crop, p.get(track_id, 0.0))
        events = self._estimate_faces(r, tracks)
        now = time.monotonic()
        self._applied_at.append(now)
        self._last_applied = now
        while self._applied_at and now - self._applied_at[0] > 1.0:
            self._applied_at.popleft()
        self._ms_sum += r.worker_ms + 1000.0 * (time.perf_counter() - t0)
        self._ms_n += 1
        self.live = True
        self.frames_applied += 1
        self.last_ts = r.ts
        return events

    def _take_audio(self, ts: float, seq: int) -> AudioChunk | None:
        """Audio captured up to the frame's ``ts``: what the ASD envelope pairs with it."""
        parts = []
        while self._audio and self._audio[0][0] <= ts:
            parts.append(self._audio.popleft()[1])
        if not parts:
            return None
        x = np.concatenate(parts)
        self._audio_n -= len(x)
        return AudioChunk(x, PIPELINE_SR, seq, ts)

    def _estimate_faces(self, r: _Result, tracks: list[FaceTrack]) -> list[dict]:
        if self.on_face is None:
            return []
        live = {t.track_id for t in tracks}
        self._face_ts = {k: v for k, v in self._face_ts.items() if k in live}
        events = []
        due = [t for t in tracks
               if t.last_seen_ts == r.ts and r.ts - self._face_ts.get(t.track_id, -1e9) >= FACE_EVERY_S]
        for t in due[:MAX_FACE_TRACKS]:
            self._face_ts[t.track_id] = r.ts
            events.extend(self.on_face(t.track_id, r.image, t.box))
        return events

    def _go_dark(self, ctx: StageContext) -> None:
        """Camera gone: forget faces and the target so the gate bypasses (audio-only tier)."""
        self.live = False
        self._applied_at.clear()
        self._last_applied = None
        self._gen += 1  # the worker resets tracker and ASD before its next frame
        # track ids restart too: an explicit lock is kept and re-attached by position (T-107: a
        # vision stall under load used to reset it and the other person was then picked freely)
        self.selector.forget_ids()
        self._face_ts.clear()
        self._lip_tracks = []
        feed = feed_of(ctx)
        if feed is not None:
            feed.clear()  # track ids restart with the tracker
        ctx.tracks = []
        ctx.target = TargetState(None, 0.0, self.selector.locked, ctx.target.since_ts,
                                 ctx.target.switch_seq)

    # ------------------------------------------------------------ telemetry

    @property
    def status(self) -> str:
        """``live`` (frames applied in the last second), ``off`` (no camera) or ``unavailable``."""
        if self._unavailable is not None:
            return "unavailable"
        return "live" if self.live else "off"

    def latency_ms(self) -> float:
        """Mean decode + detect + track + ASD + target time per frame since the last call."""
        if self._ms_n:
            self._ms = self._ms_sum / self._ms_n
            self._ms_sum, self._ms_n = 0.0, 0
        return self._ms

    def telemetry(self) -> dict:
        detector = self._detector.name if self._detector is not None else self.detector_name
        return {
            "status": self.status,
            "detector": detector,
            "asd": self.asd.name if self.asd is not None else self.asd_name,
            "fps": float(len(self._applied_at)),
            "frames": self.frames_in,
            "dropped": self.dropped,
            "lip_only": self.lip_only,
            "asd_skipped": self.asd_skipped,
            "asd_cpu": self.asd_cpu,
            "asd_age_ms": [round(v, 1) for v in self.asd_age_ms()],
            "reason": self._unavailable,
        }

    def wait(self, timeout: float = 1.0) -> None:
        """Block until the in-flight frames are done (tests, offline runs); apply them with ``poll``."""
        self._collect(timeout)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._jobs.clear()
        if self._detector is not None:
            self._detector.close()
            self._detector = None
