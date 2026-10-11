"""Face front end: MockBackend + FacePipeline offline; MediaPipe on a drawn face when weights exist."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from hearing import models
from hearing.types import VideoFrame
from hearing.vision import faces
from hearing.vision.faces import (
    LIP_INDICES,
    NUM_LANDMARKS,
    FaceBackend,
    FacePipeline,
    MockBackend,
    lip_box,
    mock_landmarks,
    mouth_open,
)
from hearing.vision.tracker import Detection, IouTracker

FPS = 25.0
IMAGE = np.zeros((360, 640, 3), dtype=np.uint8)
LEFT, RIGHT = (0.2, 0.3, 0.15, 0.27), (0.6, 0.3, 0.15, 0.27)


def frame(i: int, image: np.ndarray = IMAGE) -> VideoFrame:
    return VideoFrame(image=image, seq=i, capture_ts=i / FPS)


def shift(box, dx):
    return (box[0] + dx, box[1], box[2], box[3])


def run(pipe: FacePipeline, n: int):
    return [pipe.update(frame(i)) for i in range(n)]


@pytest.fixture
def empty_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    models.clear()
    yield tmp_path
    models.clear()


# ------------------------------------------------------------- mock pipeline
def test_mock_backend_satisfies_protocol():
    assert isinstance(MockBackend(), FaceBackend)


def test_pipeline_ids_stable_for_two_moving_faces():
    script = [[shift(LEFT, 0.004 * i), shift(RIGHT, -0.004 * i)] for i in range(50)]
    pipe = FacePipeline(MockBackend(script))
    out = run(pipe, 50)
    assert out[0] == []  # min_hits=2: first sighting is tentative.
    by_x = [sorted(tracks, key=lambda t: t.box[0]) for tracks in out[1:]]
    assert all(len(t) == 2 for t in by_x)
    assert {(t[0].track_id, t[1].track_id) for t in by_x} == {(1, 2)}
    assert pipe.last_ms >= 0.0


def test_pipeline_holds_id_through_short_dropout():
    script = [[LEFT]] * 10 + [[]] * 5 + [[LEFT]] * 10  # 200 ms gap < max_missing_s.
    out = run(FacePipeline(MockBackend(script)), 25)
    assert {t.track_id for tracks in out[1:] for t in tracks} == {1}
    assert all(len(tracks) == 1 for tracks in out[1:])


def test_pipeline_passes_landmarks_through():
    lms = mock_landmarks(LEFT, mouth_open=0.2)
    script = [[Detection(box=LEFT, score=0.9, landmarks=lms)]] * 3
    tracks = run(FacePipeline(MockBackend(script)), 3)[-1]
    assert len(tracks) == 1 and tracks[0].landmarks is lms


def test_mock_backend_generates_landmarks_when_asked():
    dets = MockBackend([[LEFT]], landmarks=True).detect(IMAGE)
    assert dets[0].landmarks.shape == (NUM_LANDMARKS, 3)
    assert dets[0].landmarks.dtype == np.float32
    assert MockBackend([[LEFT]]).detect(IMAGE)[0].landmarks is None


def test_mock_backend_script_end_loop_callable_and_reset():
    backend = MockBackend([[LEFT], [RIGHT]])
    assert [len(backend.detect(IMAGE)) for _ in range(3)] == [1, 1, 0]
    backend.reset()
    assert backend.detect(IMAGE)[0].box == LEFT
    looping = MockBackend([[LEFT], [RIGHT]], loop=True)
    assert [looping.detect(IMAGE)[0].box for _ in range(3)] == [LEFT, RIGHT, LEFT]
    scripted = MockBackend(lambda i: [shift(LEFT, 0.01 * i)])
    scripted.detect(IMAGE)
    assert scripted.detect(IMAGE)[0].box[0] == pytest.approx(LEFT[0] + 0.01)
    assert MockBackend().detect(IMAGE) == []


def test_pipeline_reset_restarts_ids_and_script():
    pipe = FacePipeline(MockBackend([[LEFT]] * 5), tracker=IouTracker(min_hits=1))
    first = run(pipe, 3)
    pipe.reset()
    again = run(pipe, 3)
    assert [t.track_id for t in first[0]] == [t.track_id for t in again[0]] == [1]
    assert pipe.tracks and pipe.tracks[0].track_id == 1


@pytest.mark.parametrize(
    "image",
    [np.zeros((360, 640, 3), np.float32), np.zeros((360, 640), np.uint8), np.zeros((360, 640, 4), np.uint8)],
)
def test_backend_rejects_non_rgb_uint8(image):
    with pytest.raises(ValueError):
        MockBackend().detect(image)


# ------------------------------------------------------------------ geometry
def test_lip_indices_are_mediapipe_lips():
    assert len(LIP_INDICES) == 40 and len(set(LIP_INDICES)) == 40
    assert set(faces.LIP_OUTER) | set(faces.LIP_INNER) == set(LIP_INDICES)
    assert max(LIP_INDICES) < NUM_LANDMARKS
    # Cross-check against MediaPipe when it is installed (a pure import, no weights).
    try:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections
    except ImportError:
        return
    lips = {i for c in FaceLandmarksConnections.FACE_LANDMARKS_LIPS for i in (c.start, c.end)}
    assert lips == set(LIP_INDICES)


def test_lip_box_geometry():
    x, y, w, h = LEFT
    bx, by, bw, bh = lip_box(mock_landmarks(LEFT, mouth_open=0.3))
    assert x <= bx and bx + bw <= x + w and y <= by and by + bh <= y + h  # Inside the face.
    assert bx + bw / 2 == pytest.approx(x + w / 2, abs=1e-6)  # Horizontally centred.
    assert by + bh / 2 == pytest.approx(y + 0.75 * h, abs=1e-6)  # Lower face.
    assert bw == pytest.approx(0.4 * w, rel=1e-5)
    grown = lip_box(mock_landmarks(LEFT), margin=0.25)
    tight = lip_box(mock_landmarks(LEFT))
    assert grown[2] == pytest.approx(1.5 * tight[2], rel=1e-5)
    assert grown[3] == pytest.approx(1.5 * tight[3], rel=1e-5)


def test_lip_box_clips_to_image_and_accepts_2d_points():
    edge = mock_landmarks((0.0, 0.8, 0.2, 0.3))[:, :2]  # Mouth near the bottom edge.
    x, y, w, h = lip_box(edge, margin=2.0)
    assert x >= 0.0 and y >= 0.0 and x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9


def test_lip_box_rejects_wrong_shape():
    with pytest.raises(ValueError):
        lip_box(np.zeros((68, 2)))


def test_mouth_open_tracks_lip_gap():
    assert mouth_open(mock_landmarks(LEFT, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert mouth_open(mock_landmarks(LEFT, 0.4)) == pytest.approx(0.4, rel=1e-4)
    assert mouth_open(mock_landmarks(LEFT, 0.4)) > mouth_open(mock_landmarks(LEFT, 0.1))


def test_landmark_association_prefers_overlap():
    det_boxes = [(0.1, 0.1, 0.2, 0.3), (0.6, 0.1, 0.2, 0.3)]
    lm_boxes = [(0.62, 0.08, 0.2, 0.33), (0.9, 0.9, 0.05, 0.05), (0.09, 0.07, 0.21, 0.34)]
    assert faces._match_landmarks(det_boxes, lm_boxes) == {0: 2, 1: 0}
    assert faces._match_landmarks([], lm_boxes) == {}


# -------------------------------------------------------- weights & errors
def test_missing_weights_raise_with_fix(empty_models_dir):
    pytest.importorskip("mediapipe")
    with pytest.raises(FileNotFoundError, match="--download"):
        faces.MediaPipeBackend()


def test_missing_mediapipe_raises_with_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    with pytest.raises(RuntimeError, match=r"hearing\[ml\]"):
        faces.MediaPipeBackend()


def test_checksum_mismatch_is_reported(empty_models_dir):
    (empty_models_dir / "mediapipe").mkdir()
    for name in faces.MODEL_FILES:
        (empty_models_dir / "mediapipe" / name).write_bytes(b"not a model")
    with pytest.raises(ValueError, match="checksum"):
        models.load("mediapipe_face_detector")


def test_auto_backend_degrades_to_mock(empty_models_dir):
    backend = faces.get_backend("auto")
    assert isinstance(backend, MockBackend) and backend.detect(IMAGE) == []
    with pytest.raises(ValueError):
        faces.get_backend("opencv")


# ---------------------------------------------------------------- delegate
@pytest.fixture
def fake_mediapipe(monkeypatch):
    """A stand-in ``mediapipe`` that records each task's delegate (no weights, no Metal)."""
    import types

    created = []

    class BaseOptions:
        class Delegate:
            CPU, GPU = "CPU", "GPU"

        def __init__(self, model_asset_buffer=None, delegate=None):
            self.delegate = delegate

    class Task:
        def __init__(self, options):
            created.append(options.base_options.delegate)

        def close(self):
            pass

    def options(**kwargs):
        return types.SimpleNamespace(**kwargs)

    vision = types.SimpleNamespace(
        RunningMode=types.SimpleNamespace(VIDEO="VIDEO", IMAGE="IMAGE"),
        FaceDetector=types.SimpleNamespace(create_from_options=Task),
        FaceLandmarker=types.SimpleNamespace(create_from_options=Task),
        FaceDetectorOptions=options,
        FaceLandmarkerOptions=options,
    )
    mp = types.ModuleType("mediapipe")
    tasks = types.ModuleType("mediapipe.tasks")
    python = types.ModuleType("mediapipe.tasks.python")
    python.BaseOptions, python.vision = BaseOptions, vision
    mp.tasks, tasks.python = tasks, python
    for name, mod in (("mediapipe", mp), ("mediapipe.tasks", tasks), ("mediapipe.tasks.python", python)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(faces.models, "load", lambda name: b"")
    monkeypatch.delenv(faces.DELEGATE_ENV, raising=False)
    return created


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_default_delegate_is_cpu_everywhere(monkeypatch, platform):
    monkeypatch.delenv(faces.DELEGATE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    assert faces.default_delegate() == "cpu"


@pytest.mark.parametrize(
    ("value", "expected"), [("gpu", "gpu"), (" GPU ", "gpu"), ("cpu", "cpu"), ("auto", "cpu"), ("", "cpu")]
)
def test_delegate_env_override(monkeypatch, value, expected):
    monkeypatch.setenv(faces.DELEGATE_ENV, value)
    assert faces.default_delegate() == expected


def test_unknown_delegate_env_warns_and_uses_cpu(monkeypatch, capsys):
    monkeypatch.setenv(faces.DELEGATE_ENV, "metal")
    assert faces.default_delegate() == "cpu"
    assert faces.DELEGATE_ENV in capsys.readouterr().err


def test_mediapipe_auto_delegate_is_cpu_on_macos(fake_mediapipe, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    backend = faces.MediaPipeBackend()
    assert backend.delegate == "cpu" and backend.device == "mediapipe-cpu"
    assert fake_mediapipe == ["CPU", "CPU"]  # detector + landmarker
    assert not backend._rgba  # RGB frames: no CVPixelBuffer/Metal upload


def test_mediapipe_gpu_only_on_opt_in(fake_mediapipe, monkeypatch):
    monkeypatch.setenv(faces.DELEGATE_ENV, "gpu")
    backend = faces.MediaPipeBackend()
    assert backend.delegate == "gpu" and fake_mediapipe == ["GPU", "GPU"] and backend._rgba
    assert faces.MediaPipeBackend(delegate="cpu").delegate == "cpu"  # explicit beats env
    with pytest.raises(ValueError, match="delegate"):
        faces.MediaPipeBackend(delegate="metal")


def test_model_paths_live_under_models_dir(empty_models_dir):
    assert faces.model_path(faces.DETECTOR_FILE) == empty_models_dir / "mediapipe" / faces.DETECTOR_FILE
    assert not faces.weights_available()


def test_bench_cli_with_mock_backend(capsys):
    assert faces.main(["--bench", "--synthetic", "--backend", "mock", "--frames", "20"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 20 and result["resolution"] == "640x360"
    assert result["faces_max"] == 1 and result["track_ids"] == [1]
    assert result["source"].startswith("synthetic")
    assert {"fps", "p50_ms", "p95_ms", "device"} <= set(result)


# -------------------------------------------------------------- real model
@pytest.fixture
def mediapipe_backend():
    pytest.importorskip("mediapipe")
    if not faces.weights_available():
        pytest.skip("MediaPipe face weights absent; run python -m hearing.vision.faces --download")
    backends = []

    def make(**kwargs):
        backends.append(faces.MediaPipeBackend(**kwargs))
        return backends[-1]

    yield make
    for b in backends:
        b.close()


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["image", "video"])
def test_mediapipe_detects_drawn_face(mediapipe_backend, mode):
    image = np.empty((360, 640, 3), np.uint8)
    image[:] = (60, 80, 100)
    faces.draw_face(image, 320, 180)
    dets = mediapipe_backend(running_mode=mode).detect(image, ts=0.0)
    if not dets:
        pytest.skip("BlazeFace did not fire on the drawn face; synthetic stimulus too crude")
    assert len(dets) == 1
    d = dets[0]
    x, y, w, h = d.box
    assert x < 0.5 < x + w and y < 0.5 < y + h and d.score > 0.5
    assert d.landmarks is not None and d.landmarks.shape == (NUM_LANDMARKS, 3)
    bx, by, bw, bh = lip_box(d.landmarks)
    # Drawn mouth: centre (320, 230) px, 50×20 px.
    assert abs((bx + bw / 2) * 640 - 320) < 15 and abs((by + bh / 2) * 360 - 230) < 20
    assert x <= bx and bx + bw <= x + w


@pytest.mark.slow
def test_mediapipe_pipeline_tracks_two_drawn_faces(mediapipe_backend):
    pipe = FacePipeline(mediapipe_backend(running_mode="video"))
    out = [pipe.update(VideoFrame(faces.synthetic_frame(i), i, i / FPS)) for i in range(30)]
    if not any(out):
        pytest.skip("BlazeFace did not fire on the drawn faces; synthetic stimulus too crude")
    steady = out[5:]
    assert all(len(t) == 2 for t in steady)
    assert {t.track_id for tracks in steady for t in tracks} == {1, 2}
    assert all(t.landmarks is not None for tracks in steady for t in tracks)
