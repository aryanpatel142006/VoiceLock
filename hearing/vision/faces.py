"""Face front end: MediaPipe BlazeFace + Face Landmarker adapters feeding the IoU tracker.

``FacePipeline(backend).update(frame)`` runs detect → :class:`IouTracker` and
returns :class:`hearing.types.FaceTrack`s with stable ids. Frames are RGB uint8
HxWx3, at most 640×360 (the backend does not resize). Boxes are normalised
``(x, y, w, h)``; ``Detection.landmarks`` / ``FaceTrack.landmarks`` are the 478
normalised Face Landmarker points as a ``(478, 3)`` float32 array (x, y in image
fractions, z relative depth). ``LIP_INDICES`` / ``lip_box`` / ``mouth_open``
serve the ASD and lip-crop work.

Weights: call ``download_models()`` (or ``python -m hearing.vision.faces
--download``) once; it installs the two pinned files into
``$HEARING_MODELS_DIR/mediapipe/``. ``MediaPipeBackend`` never downloads and
raises a clear error naming the fix when weights or ``mediapipe`` are missing.
``MockBackend`` replays caller-scripted boxes so tests and the server run
offline. Benchmark: ``python -m hearing.vision.faces --bench``.

Delegate: CPU (XNNPACK) everywhere by default, ≈ 9 ms per 640×360 frame on laptop-a.
``HEARING_FACES_DELEGATE=gpu`` opts into MediaPipe's Metal path, which aborted the live
server on laptop-a (T-060: ``CVPixelBuffer`` creation -6662 then a fatal CHECK in
``gpu_buffer_storage_cv_pixel_buffer.cc``, uncatchable from Python).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np

from hearing import models
from hearing.types import FaceTrack, VideoFrame
from hearing.vision.tracker import Box, Detection, IouTracker, iou_matrix

_STORAGE = "https://storage.googleapis.com/mediapipe-models"
DETECTOR_FILE = "blaze_face_short_range.tflite"
LANDMARKER_FILE = "face_landmarker.task"
# Pinned to version 1 of each official float16 model (identical to "latest" on Oct 4 2026).
MODEL_FILES: dict[str, tuple[str, str]] = {
    DETECTOR_FILE: (
        f"{_STORAGE}/face_detector/blaze_face_short_range/float16/1/{DETECTOR_FILE}",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
    LANDMARKER_FILE: (
        f"{_STORAGE}/face_landmarker/face_landmarker/float16/1/{LANDMARKER_FILE}",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}

NUM_LANDMARKS = 478
# Every point of MediaPipe's FACE_LANDMARKS_LIPS connections (outer and inner contour).
LIP_INDICES: tuple[int, ...] = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
    185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375,
    402, 405, 409, 415,
)  # fmt: skip
LIP_OUTER: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39,
    40, 185,
)  # fmt: skip
LIP_INNER: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81,
    80, 191,
)  # fmt: skip
# Inner-lip midpoints (upper, lower) and corners (left, right in image space).
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308

DELEGATES = ("cpu", "gpu")
DELEGATE_ENV = "HEARING_FACES_DELEGATE"


def default_delegate() -> str:
    """``$HEARING_FACES_DELEGATE`` (``cpu`` or ``gpu``), else ``cpu`` on every platform.

    An unknown value warns and falls back to ``cpu``: a typo must not cost the camera.
    """
    value = os.environ.get(DELEGATE_ENV, "").strip().lower()
    if not value or value == "auto":
        return "cpu"
    if value not in DELEGATES:
        print(f"faces: {DELEGATE_ENV}={value!r} is not cpu or gpu; using cpu", file=sys.stderr)
        return "cpu"
    return value


# ------------------------------------------------------------------ geometry
def _check_landmarks(landmarks: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks)
    if pts.ndim != 2 or pts.shape[0] != NUM_LANDMARKS or pts.shape[1] < 2:
        raise ValueError(f"expected ({NUM_LANDMARKS}, 2|3) landmarks, got {pts.shape}")
    return pts


def lip_box(landmarks: np.ndarray, margin: float = 0.0) -> Box:
    """Normalised ``(x, y, w, h)`` around the lips, grown by ``margin`` × size per side."""
    pts = _check_landmarks(landmarks)[list(LIP_INDICES), :2].astype(np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(x0 - mx, 0.0), max(y0 - my, 0.0)
    x1, y1 = min(x1 + mx, 1.0), min(y1 + my, 1.0)
    return (float(x0), float(y0), float(max(x1 - x0, 0.0)), float(max(y1 - y0, 0.0)))


def mouth_open(landmarks: np.ndarray, aspect: float = 1.0) -> float:
    """Inner-lip gap divided by mouth width (0 closed, ~0.3-0.6 wide open).

    ``aspect`` is image width / height, so normalised x and y compare in pixels.
    """
    pts = _check_landmarks(landmarks)[:, :2].astype(np.float64)
    scale = np.array([aspect, 1.0])
    gap = np.linalg.norm((pts[MOUTH_TOP] - pts[MOUTH_BOTTOM]) * scale)
    width = np.linalg.norm((pts[MOUTH_LEFT] - pts[MOUTH_RIGHT]) * scale)
    return float(gap / width) if width > 0.0 else 0.0


def landmarks_box(landmarks: np.ndarray) -> Box:
    """Normalised ``(x, y, w, h)`` bounding all landmarks, clipped to the image."""
    pts = np.clip(np.asarray(landmarks)[:, :2].astype(np.float64), 0.0, 1.0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def mock_landmarks(box: Box, mouth_open: float = 0.0) -> np.ndarray:
    """Deterministic ``(478, 3)`` stand-in landmarks inside ``box`` for offline tests.

    Non-lip points sit on a grid over the box; lips form two ellipses centred at
    75 % of the box height; :func:`mouth_open` of the result equals ``mouth_open``.
    """
    x, y, w, h = (float(v) for v in box)
    n = NUM_LANDMARKS
    side = int(np.ceil(np.sqrt(n)))
    grid = np.arange(n)
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = x + w * (grid % side + 0.5) / side
    pts[:, 1] = y + h * (grid // side + 0.5) / side
    cx, cy, half_w = x + 0.5 * w, y + 0.75 * h, 0.2 * w
    gap = float(mouth_open) * 1.6 * half_w  # Inner width is 2 × 0.8 × half_w.
    for ring, rx, ry in ((LIP_OUTER, half_w, gap / 2 + 0.04 * h), (LIP_INNER, 0.8 * half_w, gap / 2)):
        angles = np.linspace(np.pi, -np.pi, len(ring), endpoint=False)
        pts[list(ring), 0] = cx + rx * np.cos(angles)
        pts[list(ring), 1] = cy - ry * np.sin(angles)
    pts[MOUTH_LEFT, :2] = (cx - 0.8 * half_w, cy)
    pts[MOUTH_RIGHT, :2] = (cx + 0.8 * half_w, cy)
    pts[MOUTH_TOP, :2] = (cx, cy - gap / 2)
    pts[MOUTH_BOTTOM, :2] = (cx, cy + gap / 2)
    return pts


# ------------------------------------------------------------------- weights
def model_dir() -> Path:
    return models.cache_dir() / "mediapipe"


def model_path(name: str) -> Path:
    if name not in MODEL_FILES:
        raise KeyError(f"unknown MediaPipe face model {name!r}; known: {sorted(MODEL_FILES)}")
    return model_dir() / name


def _verify(name: str, data: bytes) -> None:
    if hashlib.sha256(data).hexdigest() != MODEL_FILES[name][1]:
        raise ValueError(
            f"MediaPipe {name} checksum mismatch; delete {model_path(name)} and run "
            "hearing.vision.faces.download_models()"
        )


def download_models() -> dict[str, Path]:
    """Explicit network adapter: fetch both pinned files, verify SHA-256, install atomically."""
    out: dict[str, Path] = {}
    for name, (url, _sha) in MODEL_FILES.items():
        path = model_path(name)
        if path.exists():
            _verify(name, path.read_bytes())
            out[name] = path
            continue
        import httpx  # httpx verifies against certifi; python.org builds lack a CA bundle.

        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        data = response.content
        _verify(name, data)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        out[name] = path
    return out


def weights_available() -> bool:
    return all(model_path(name).exists() for name in MODEL_FILES)


def _read_verified(name: str) -> bytes:
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: provision with `python -m hearing.vision.faces --download` "
            "(hearing.vision.faces.download_models())"
        )
    data = path.read_bytes()
    _verify(name, data)
    return data


# Loaders return the verified model bytes: immutable and shareable, while every
# MediaPipeBackend builds its own (stateful, single-threaded) task graphs from them.
@models.register_loader("mediapipe_face_detector")
def _load_detector_asset() -> bytes:
    return _read_verified(DETECTOR_FILE)


@models.register_loader("mediapipe_face_landmarker")
def _load_landmarker_asset() -> bytes:
    return _read_verified(LANDMARKER_FILE)


# ------------------------------------------------------------------ backends
@runtime_checkable
class FaceBackend(Protocol):
    """Per-frame face detector. ``ts`` (capture seconds) is optional; video-mode
    backends use it as their monotonic clock."""

    def detect(self, image: np.ndarray, ts: float | None = None) -> list[Detection]: ...

    def reset(self) -> None: ...


def _check_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {getattr(image, 'shape', type(image))}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image, got {image.dtype}")
    return image


Script = Sequence[Sequence["Box | Detection"]] | Callable[[int], Sequence["Box | Detection"]]


class MockBackend:
    """Deterministic offline backend replaying caller-scripted detections.

    ``script`` is either a sequence (one entry per ``detect`` call, each a list of
    boxes or :class:`Detection`s) or a callable ``frame_index -> list``. Past the
    end of a sequence it returns ``[]`` unless ``loop``. Plain boxes get score
    1.0 and, when ``landmarks`` is true, :func:`mock_landmarks`. The image is
    validated but otherwise ignored.
    """

    device = "mock"

    def __init__(self, script: Script = (), *, loop: bool = False, landmarks: bool = False):
        self.script = script
        self.loop = bool(loop)
        self.landmarks = bool(landmarks)
        self._index = 0

    def detect(self, image: np.ndarray, ts: float | None = None) -> list[Detection]:
        _check_image(image)
        i = self._index
        self._index += 1
        if callable(self.script):
            entries = self.script(i)
        else:
            if not self.script or (i >= len(self.script) and not self.loop):
                return []
            entries = self.script[i % len(self.script)]
        out = []
        for entry in entries:
            if isinstance(entry, Detection):
                out.append(entry)
            else:
                box = tuple(float(v) for v in entry)
                lms = mock_landmarks(box) if self.landmarks else None
                out.append(Detection(box=box, score=1.0, landmarks=lms))
        return out

    def reset(self) -> None:
        self._index = 0


def _match_landmarks(
    det_boxes: list[Box], lm_boxes: list[Box], threshold: float = 0.3
) -> dict[int, int]:
    """Greedy detector-box ↔ landmark-face matching by IoU; returns {det: face}."""
    if not det_boxes or not lm_boxes:
        return {}
    iou = iou_matrix(np.asarray(det_boxes), np.asarray(lm_boxes))
    pairs: dict[int, int] = {}
    used: set[int] = set()
    for flat in np.argsort(iou, axis=None)[::-1]:
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < threshold:
            break
        if r not in pairs and c not in used:
            pairs[r] = c
            used.add(c)
    return pairs


class MediaPipeBackend:
    """MediaPipe Tasks ``FaceDetector`` (BlazeFace short range) + ``FaceLandmarker``.

    Detector boxes and scores are authoritative; each gets the landmarks of the
    Face Landmarker face that overlaps it most (IoU ≥ 0.3). A landmarker face
    with no detector box (the landmarker's own tracker held it, e.g. a turned
    head) is still returned with its landmark bounding box and ``score=0.0``.

    ``running_mode="video"`` (default) lets the landmarker track between frames
    and needs monotonic timestamps: ``ts`` when given (forced strictly
    increasing), else +40 ms per call. ``delegate="auto"`` resolves through
    :func:`default_delegate` (CPU unless ``HEARING_FACES_DELEGATE=gpu``); the GPU
    path feeds RGBA frames to Metal and can abort the whole process (T-060).
    Instances are not thread-safe: use one per session.
    """

    def __init__(
        self,
        running_mode: str = "video",
        max_faces: int = 4,
        min_detection_confidence: float = 0.5,
        landmarks: bool = True,
        delegate: str = "auto",
    ) -> None:
        if running_mode not in ("video", "image"):
            raise ValueError("running_mode must be 'video' or 'image'")
        if delegate not in ("auto", *DELEGATES):
            raise ValueError("delegate must be 'auto', 'cpu' or 'gpu'")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe faces require mediapipe; install hearing[ml] (pip install mediapipe)"
            ) from exc
        detector_bytes = models.load("mediapipe_face_detector")
        landmarker_bytes = models.load("mediapipe_face_landmarker") if landmarks else None
        if delegate == "auto":
            delegate = default_delegate()
        self.delegate = delegate
        self.device = f"mediapipe-{delegate}"
        self.running_mode = running_mode
        self.with_landmarks = bool(landmarks)
        self._mp = mp
        self._rgba = delegate == "gpu"  # The Metal path only accepts 4-channel frames.
        d = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
        mode = vision.RunningMode.VIDEO if running_mode == "video" else vision.RunningMode.IMAGE
        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_buffer=detector_bytes, delegate=d),
                running_mode=mode,
                min_detection_confidence=min_detection_confidence,
            )
        )
        self._landmarker = None
        if landmarks:
            self._landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_buffer=landmarker_bytes, delegate=d),
                    running_mode=mode,
                    num_faces=max_faces,
                    min_face_detection_confidence=min_detection_confidence,
                )
            )
        self._last_ms = -1
        self._buffer: np.ndarray | None = None

    def _to_mp_image(self, image: np.ndarray):
        mp = self._mp
        if not self._rgba:
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        if self._buffer is None or self._buffer.shape[:2] != image.shape[:2]:
            self._buffer = np.full(image.shape[:2] + (4,), 255, dtype=np.uint8)
        self._buffer[..., :3] = image
        return mp.Image(image_format=mp.ImageFormat.SRGBA, data=self._buffer)

    def _next_ms(self, ts: float | None) -> int:
        ms = self._last_ms + 40 if ts is None else round(ts * 1000.0)
        ms = max(ms, self._last_ms + 1)
        self._last_ms = ms
        return ms

    def detect(self, image: np.ndarray, ts: float | None = None) -> list[Detection]:
        image = _check_image(image)
        h, w = image.shape[:2]
        mp_image = self._to_mp_image(image)
        if self.running_mode == "video":
            ms = self._next_ms(ts)
            det_result = self._detector.detect_for_video(mp_image, ms)
            lm_result = self._landmarker.detect_for_video(mp_image, ms) if self._landmarker else None
        else:
            det_result = self._detector.detect(mp_image)
            lm_result = self._landmarker.detect(mp_image) if self._landmarker else None

        det_boxes: list[Box] = []
        scores: list[float] = []
        for d in det_result.detections:
            bb = d.bounding_box
            x0, y0 = min(max(bb.origin_x / w, 0.0), 1.0), min(max(bb.origin_y / h, 0.0), 1.0)
            x1 = min(max((bb.origin_x + bb.width) / w, 0.0), 1.0)
            y1 = min(max((bb.origin_y + bb.height) / h, 0.0), 1.0)
            det_boxes.append((x0, y0, x1 - x0, y1 - y0))
            scores.append(float(d.categories[0].score) if d.categories else 1.0)

        faces = []
        if lm_result is not None:
            for face in lm_result.face_landmarks:
                faces.append(np.array([(p.x, p.y, p.z) for p in face], dtype=np.float32))
        lm_boxes = [landmarks_box(f) for f in faces]
        pairs = _match_landmarks(det_boxes, lm_boxes)

        out = [
            Detection(box=b, score=s, landmarks=faces[pairs[i]] if i in pairs else None)
            for i, (b, s) in enumerate(zip(det_boxes, scores, strict=True))
        ]
        matched = set(pairs.values())
        out += [
            Detection(box=lm_boxes[j], score=0.0, landmarks=faces[j])
            for j in range(len(faces))
            if j not in matched
        ]
        out.sort(key=lambda d: (d.box[0], d.box[1]))
        return out

    def reset(self) -> None:
        """Timestamps stay monotonic across sessions; the landmarker re-acquires faces."""

    def close(self) -> None:
        for task in (self._detector, self._landmarker):
            if task is not None:
                task.close()
        self._detector = self._landmarker = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def get_backend(name: str = "auto", **kwargs) -> FaceBackend:
    """``"mediapipe"`` (raises when unavailable), ``"mock"`` (no faces unless scripted),
    or ``"auto"``: MediaPipe if weights and package are present, else an empty mock."""
    if name == "mock":
        return MockBackend(**kwargs)
    if name == "mediapipe":
        return MediaPipeBackend(**kwargs)
    if name == "auto":
        try:
            return MediaPipeBackend(**kwargs)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"faces: MediaPipe unavailable ({exc}); using MockBackend", file=sys.stderr)
            return MockBackend()
    raise ValueError(f"unknown face backend {name!r}; use 'mediapipe', 'mock' or 'auto'")


# ------------------------------------------------------------------ pipeline
class FacePipeline:
    """detect → IoU tracker. ``last_ms`` is the compute time of the latest update."""

    def __init__(self, backend: FaceBackend, tracker: IouTracker | None = None) -> None:
        self.backend = backend
        self.tracker = tracker if tracker is not None else IouTracker()
        self.last_ms = 0.0

    def update(self, frame: VideoFrame) -> list[FaceTrack]:
        start = time.perf_counter()
        detections = self.backend.detect(frame.image, frame.capture_ts)
        tracks = self.tracker.update(detections, frame.capture_ts)
        self.last_ms = (time.perf_counter() - start) * 1000.0
        return tracks

    @property
    def tracks(self) -> list[FaceTrack]:
        return self.tracker.tracks

    def reset(self) -> None:
        self.tracker.reset()
        self.backend.reset()
        self.last_ms = 0.0


# ----------------------------------------------------------------- benchmark
FRAME_W, FRAME_H, FPS = 640, 360, 25.0


def _fill_ellipse(img: np.ndarray, cx: float, cy: float, rx: float, ry: float, color) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(int(cy - ry), 0), min(int(cy + ry) + 1, h)
    x0, x1 = max(int(cx - rx), 0), min(int(cx + rx) + 1, w)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    img[y0:y1, x0:x1][mask] = color


def draw_face(img: np.ndarray, cx: float, cy: float, scale: float = 1.0) -> None:
    """Draw a cartoon frontal face (BlazeFace detects it) centred at ``(cx, cy)`` pixels."""
    s = scale
    _fill_ellipse(img, cx, cy, 70 * s, 95 * s, (224, 172, 105))  # Face.
    for side in (-1, 1):
        ex = cx + side * 27.5 * s
        _fill_ellipse(img, ex, cy - 45 * s, 15 * s, 3 * s, (60, 40, 20))  # Brow.
        _fill_ellipse(img, ex, cy - 25 * s, 12.5 * s, 10 * s, (255, 255, 255))  # Eye.
        _fill_ellipse(img, ex, cy - 25 * s, 6 * s, 6 * s, (40, 30, 20))  # Pupil.
    _fill_ellipse(img, cx, cy + 13 * s, 8 * s, 15 * s, (190, 130, 90))  # Nose.
    _fill_ellipse(img, cx, cy + 50 * s, 25 * s, 10 * s, (150, 50, 50))  # Mouth.


def synthetic_frame(i: int, faces: int = 2) -> np.ndarray:
    """640×360 frame with ``faces`` drawn faces drifting sideways (deterministic in ``i``)."""
    img = np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    img[:] = (60, 80, 100)
    for k in range(faces):
        base = FRAME_W * (k + 1) / (faces + 1)
        dx = 40.0 * np.sin(2 * np.pi * (i / 100.0 + k / 3.0))
        draw_face(img, base + dx, FRAME_H / 2, scale=0.9 if faces > 1 else 1.0)
    return img


def _fit(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= FRAME_W and h <= FRAME_H:
        return np.ascontiguousarray(image)
    import cv2

    f = min(FRAME_W / w, FRAME_H / h)
    return cv2.resize(image, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)


def _read_capture(source, n: int) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video/camera input needs OpenCV (installed with mediapipe)") from exc
    cap = cv2.VideoCapture(source)
    frames: list[np.ndarray] = []
    try:
        while cap.isOpened() and len(frames) < n:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(_fit(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return frames


def _fixture_video() -> Path | None:
    root = Path(os.environ.get("HEARING_FIXTURES_DIR") or "eval/fixtures/generated")
    videos = sorted(root.glob("**/*.mp4")) if root.exists() else []
    grid = [v for v in videos if "grid" in str(v).lower()]
    return (grid or videos or [None])[0]


def _bench(args) -> int:
    import json

    from hearing import device

    if args.video:
        source, frames = f"video {args.video}", _read_capture(args.video, args.frames)
    elif args.camera is not None:
        source, frames = f"camera {args.camera}", _read_capture(args.camera, args.frames)
    elif args.synthetic:
        source, frames = "", []
    else:
        fixture = _fixture_video()
        frames = _read_capture(str(fixture), args.frames) if fixture else []
        source = f"fixture video {fixture}" if frames else ""
    if not frames:
        if args.video or args.camera is not None:
            print("faces bench: no frames from the requested source", file=sys.stderr)
            return 1
        source = f"synthetic: {args.faces} drawn face(s) drifting at 640x360"
        frames = [synthetic_frame(i, args.faces) for i in range(min(args.frames, 100))]

    backend = (
        MockBackend(lambda i: [(0.4, 0.3, 0.2, 0.35)], landmarks=True)
        if args.backend == "mock"
        else MediaPipeBackend(running_mode=args.mode, max_faces=4, delegate=args.delegate)
    )
    pipe = FacePipeline(backend)
    warmup = 10
    times, counts, with_lms = [], [], 0
    for i in range(args.frames + warmup):
        frame = VideoFrame(frames[i % len(frames)], seq=i, capture_ts=i / FPS)
        tracks = pipe.update(frame)
        if i >= warmup:
            times.append(pipe.last_ms)
            counts.append(len(tracks))
            with_lms += sum(t.landmarks is not None for t in tracks)
    ms = np.asarray(times)
    h, w = frames[0].shape[:2]
    result = {
        "backend": getattr(backend, "device", args.backend),
        "running_mode": args.mode,
        "source": source,
        "resolution": f"{w}x{h}",
        "frames": len(ms),
        "fps": round(1000.0 * len(ms) / ms.sum(), 1),
        "p50_ms": round(float(np.percentile(ms, 50)), 2),
        "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "faces_mean": round(float(np.mean(counts)), 2),
        "faces_max": int(max(counts)),
        "tracks_with_landmarks": with_lms,
        "track_ids": sorted({t.track_id for t in pipe.tracks}),
        "device": device.describe(),
    }
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m hearing.vision.faces", description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch the pinned weights")
    parser.add_argument("--bench", action="store_true", help="time detect+landmark+track")
    parser.add_argument("--video", help="video file to read frames from")
    parser.add_argument("--camera", type=int, help="camera index (macOS asks for permission)")
    parser.add_argument("--synthetic", action="store_true", help="force the drawn-face source (skip fixture auto-pick)")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--faces", type=int, default=2, help="faces in the synthetic source")
    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument("--backend", choices=("mediapipe", "mock"), default="mediapipe")
    parser.add_argument("--delegate", choices=("auto", *DELEGATES), default="auto",
                        help=f"MediaPipe delegate; auto = ${DELEGATE_ENV} or cpu")
    args = parser.parse_args(argv)
    if args.download:
        for name, path in download_models().items():
            print(f"{name}: {path}")
    if args.bench:
        return _bench(args)
    if not args.download:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
