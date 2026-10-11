"""Light-ASD adapter (T-037): crops, window bookkeeping, interface; real weights in the slow test."""

import math

import numpy as np
import pytest

from hearing import models
from hearing.testing import make_test_signal
from hearing.types import AudioChunk, FaceTrack, VideoFrame
from hearing.vision import asd, light_asd
from hearing.vision.asd import LipMotionASD, SpeakerDetector, get_asd
from hearing.vision.light_asd import (
    PAD_VALUE,
    LightASD,
    MockBackend,
    StreamingMFCC,
    crop_geometry,
    cut_crop,
    synthetic_face_frame,
)

SR = 16000
FPS = 25
SPF = SR // FPS
BOX = (0.35, 0.2, 0.3, 0.5)
W, H = 640, 360


class Recorder:
    """Backend recording the shapes it is called with; logit 0 → p 0.5."""

    device = "rec"
    needs_audio = True

    def __init__(self):
        self.calls = []

    def scores(self, video, mfcc):
        self.calls.append((video.shape, None if mfcc is None else mfcc.shape))
        return np.zeros(video.shape[0], np.float32)


def speech(seconds: float, seed: int = 1) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    x = make_test_signal(seconds, kind="speechlike", seed=seed)[: t.size]
    return (x * 0.5 * (1 - np.cos(2 * np.pi * 4 * t))).astype(np.float32)


def run_clip(det, moving: bool, frames: int = 50, audio: np.ndarray | None = None):
    audio = speech(frames / FPS) if audio is None else audio
    ps = []
    for i in range(frames):
        ts = i / FPS
        opening = 0.5 * (1 - math.cos(2 * np.pi * 4 * ts)) if moving else 0.3
        image = synthetic_face_frame([BOX], [opening])
        tracks = [FaceTrack(1, BOX, None, 0.0, ts)]
        chunk = AudioChunk(audio[i * SPF : (i + 1) * SPF], SR, i, ts)
        det.update(tracks, ts, chunk, VideoFrame(image, i, ts))
        ps.append(tracks[0].p_speaking)
    return np.array(ps)


# --------------------------------------------------------------------- interface
def test_get_asd_light_asd_returns_detector():
    det = get_asd("light_asd", backend="mock")
    assert isinstance(det, LightASD)
    assert isinstance(det, SpeakerDetector)
    assert det.name == "light_asd" and det.is_mock
    assert "light_asd" in asd.asd_names()
    assert get_asd("light_asd", backend="mock") is not det
    with pytest.raises(ValueError):
        LightASD(backend="bogus")
    with pytest.raises(ValueError):
        LightASD(window=0)


def test_without_weights_auto_is_mock_torch_raises_and_get_asd_auto_is_lipmotion(monkeypatch):
    monkeypatch.setattr(light_asd, "available", lambda: False)
    assert LightASD().is_mock
    assert get_asd("light_asd").is_mock
    with pytest.raises(RuntimeError, match="download_model"):
        LightASD(backend="torch")
    assert isinstance(get_asd("auto"), LipMotionASD)


def test_get_asd_auto_prefers_light_asd_when_available(monkeypatch):
    monkeypatch.setattr(light_asd, "available", lambda: True)
    monkeypatch.setitem(models._loaders, "light_asd", MockBackend)
    models.clear()
    try:
        det = get_asd("auto", window=10)
        assert isinstance(det, LightASD) and det.window == 10
        assert isinstance(get_asd("auto", window_s=0.4), LipMotionASD)  # Lipmotion-only kwarg.

        def broken():
            raise FileNotFoundError("no weights")

        monkeypatch.setitem(models._loaders, "light_asd", broken)
        models.clear()
        assert isinstance(get_asd("auto"), LipMotionASD)
    finally:
        models.clear()


# --------------------------------------------------------------------- crops
def test_crop_geometry_falls_back_to_lower_face_box():
    cx, cy, side, used = crop_geometry(BOX, None, W, H)
    face = max(0.3 * W, 0.5 * H)  # 192 px.
    assert not used
    assert cx == pytest.approx(0.5 * W)
    assert cy == pytest.approx(0.45 * H + 0.2 * face)
    assert side == pytest.approx(0.7 * face)
    # Too few landmarks for the lip indices: still the box fallback.
    assert not crop_geometry(BOX, np.zeros((10, 2)), W, H)[3]


@pytest.mark.parametrize("pixels", [False, True])
def test_crop_geometry_centres_on_mouth_landmarks(pixels):
    lm = np.full((478, 3), 0.5)
    lm[13, :2], lm[14, :2] = (0.50, 0.60), (0.50, 0.64)
    lm[61, :2], lm[291, :2] = (0.45, 0.62), (0.55, 0.62)
    if pixels:
        lm[:, 0] *= W
        lm[:, 1] *= H
    cx, cy, side, used = crop_geometry(BOX, lm, W, H)
    assert used
    assert cx == pytest.approx(0.5 * W)
    assert cy == pytest.approx(0.62 * H)
    assert side == pytest.approx(0.7 * 192)


def test_cut_crop_shape_grayscale_and_padding():
    image = np.zeros((H, W, 3), np.uint8)
    image[:, : W // 2] = (255, 0, 0)  # Left half red, right half blue.
    image[:, W // 2 :] = (0, 0, 255)
    crop = cut_crop(image, W / 2, H / 2, 100.0)
    assert crop.shape == (112, 112) and crop.dtype == np.uint8
    assert crop[56, 10] == round(0.299 * 255)
    assert crop[56, 100] == round(0.114 * 255)
    # Half the crop hangs off the left edge: those pixels are the upstream pad value.
    edge = cut_crop(image, 0.0, H / 2, 100.0)
    assert (edge[:, :50] == PAD_VALUE).all()
    assert edge[56, 80] == round(0.299 * 255)
    assert (cut_crop(image, -500.0, -500.0, 50.0) == PAD_VALUE).all()


def test_cut_crop_identity_scale_matches_pixels():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (H, W), dtype=np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    crop = cut_crop(image, 300.0, 150.0, 112.0)  # 1 px per output pixel, integer offset.
    np.testing.assert_array_equal(crop, gray[94:206, 244:356])


# --------------------------------------------------------------------- audio
def test_streaming_mfcc_matches_whole_clip_upstream_call():
    psf = pytest.importorskip("python_speech_features")
    x = speech(1.3)
    whole = psf.mfcc(x.astype(np.float64) * 32768.0, SR, numcep=13, winlen=0.025, winstep=0.010)
    stream = StreamingMFCC()
    chunks, pos, rng = [], 0, np.random.default_rng(3)
    while pos < x.size:
        n = int(rng.integers(1, 900))
        chunks.append(stream.push(x[pos : pos + n]))
        pos += n
    got = np.concatenate(chunks)
    n = 1 + (x.size - 400) // 160  # Whole frames only; upstream zero-pads one more.
    assert got.shape == (n, 13)
    np.testing.assert_allclose(got, whole[:n], rtol=1e-4, atol=1e-3)


# --------------------------------------------------------------------- window bookkeeping
def test_window_bookkeeping_and_matching_audio_slice():
    rec = Recorder()
    det = LightASD(window=5, backend=rec)
    run_clip(det, moving=True, frames=8)
    assert det.window_frames(1) == 5
    assert det.mfcc_frames == 20
    # Frame 0: 640 samples give 2 MFCC rows (< 4) → visual-only head. Frame N then has
    # 4N + 2 rows, so T = min(crops, rows // 4) and the audio slice is always 4T rows.
    assert rec.calls[0] == ((1, 1, 112, 112), None)
    assert rec.calls[1] == ((1, 1, 112, 112), (4, 13))
    assert rec.calls[-1] == ((1, 5, 112, 112), (20, 13))
    assert all(a[0] == 4 * v[1] for v, a in rec.calls[1:])


def test_visual_only_head_until_audio_arrives():
    rec = Recorder()
    det = LightASD(window=4, backend=rec)
    image = synthetic_face_frame([BOX], [0.3])
    for i in range(3):
        det.update([FaceTrack(1, BOX, None, 0.0, i / FPS)], i / FPS, None, VideoFrame(image, i, i / FPS))
    assert [c[1] for c in rec.calls] == [None, None, None]
    assert rec.calls[-1][0] == (1, 3, 112, 112)


def test_two_tracks_with_different_windows_are_grouped():
    rec = Recorder()
    det = LightASD(window=5, backend=rec)
    boxes = [(0.05, 0.2, 0.4, 0.5), (0.55, 0.2, 0.4, 0.5)]
    audio = speech(1.0)
    for i in range(7):
        ts = i / FPS
        tracks = [FaceTrack(1, boxes[0], None, 0.0, ts)]
        if i >= 4:
            tracks.append(FaceTrack(2, boxes[1], None, 0.0, ts))
        image = synthetic_face_frame(boxes[: len(tracks)], [0.3] * len(tracks))
        rec.calls.clear()
        det.update(tracks, ts, AudioChunk(audio[i * SPF : (i + 1) * SPF], SR, i, ts), VideoFrame(image, i, ts))
    assert sorted(rec.calls) == [((1, 3, 112, 112), (12, 13)), ((1, 5, 112, 112), (20, 13))]
    assert all(t.p_speaking == pytest.approx(0.5) for t in tracks)


def test_dropped_stale_and_frameless_tracks_hold_or_reset():
    det = LightASD(window=5, backend=MockBackend())
    ps = run_clip(det, moving=True, frames=10)
    held = ps[-1]
    assert held > 0.5
    # No frame: p holds, window unchanged.
    t = FaceTrack(1, BOX, None, 0.0, 0.4)
    det.update([t], 0.4, None, None)
    assert t.p_speaking == pytest.approx(held) and det.window_frames(1) == 5
    # Stale track (last seen long ago): holds too.
    t = FaceTrack(1, BOX, None, 0.0, 0.0)
    det.update([t], 0.44, None, VideoFrame(synthetic_face_frame([BOX], [0.3]), 11, 0.44))
    assert t.p_speaking == pytest.approx(held) and det.window_frames(1) == 5
    # A gap longer than max_gap_s restarts the window.
    t = FaceTrack(1, BOX, None, 0.0, 2.0)
    det.update([t], 2.0, None, VideoFrame(synthetic_face_frame([BOX], [0.3]), 50, 2.0))
    assert det.window_frames(1) == 1
    # Track gone: its state is dropped.
    det.update([], 2.04, None, None)
    assert det.window_frames(1) == 0 and 1 not in det.scores
    det.reset()
    assert det.mfcc_frames == 0


def test_every_n_frames_runs_model_on_alternate_updates():
    rec = Recorder()
    det = LightASD(window=5, backend=rec, every=2)
    run_clip(det, moving=True, frames=10)
    assert len(rec.calls) == 5
    assert det.window_frames(1) == 5  # Crops are still cut every frame.


# --------------------------------------------------------------------- mock scoring
def test_mock_scores_moving_mouth_above_static_and_is_deterministic():
    moving = run_clip(LightASD(backend="mock"), moving=True)
    static = run_clip(LightASD(backend="mock"), moving=False)
    assert moving[10:].min() > 0.5 > static[10:].max()
    np.testing.assert_array_equal(moving, run_clip(LightASD(backend="mock"), moving=True))


def test_benchmark_runs_with_mock():
    out = light_asd.benchmark(MockBackend(), n_tracks=2, window=5, frames=3)
    assert out["device"] == "mock" and out["tracks"] == 2
    assert out["update_ms_p50"] >= 0


# --------------------------------------------------------------------- real weights
@pytest.mark.slow
def test_real_weights_moving_mouth_scores_above_static():
    if not light_asd.available():
        pytest.skip(f"Light-ASD weights missing at {light_asd.model_path()} (run download_model())")
    det = LightASD(backend="torch", calibration=(1.0, 0.0))  # Raw separation; T-051 calibration is tested elsewhere.
    moving = run_clip(det, moving=True, frames=50)
    det.reset()
    static = run_clip(det, moving=False, frames=50)
    assert not det.is_mock
    assert moving[25:].mean() > 0.8
    assert static[25:].mean() < 0.3
    assert moving[25:].mean() - static[25:].mean() > 0.5
