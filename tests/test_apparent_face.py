"""Face apparent gender/age adapter: stub-session checks offline, real weights opt-in."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hearing import models
from hearing.synth import apparent_face, voice_match
from hearing.synth.apparent_face import (
    INPUT_SIZE,
    FaceEstimator,
    face_pixels,
    to_estimate,
)

H, W = 360, 640
RED, BLUE = (200, 0, 0), (0, 0, 200)


class StubSession:
    """Records the blob and returns a fixed ``[female, male, age/100]`` row."""

    def __init__(self, pred=(2.0, -2.0, 0.35)):
        self.pred = np.array([pred], np.float32)
        self.blobs = []

    def get_inputs(self):
        return [SimpleNamespace(name="data")]

    def run(self, outputs, feed):
        self.blobs.append(feed["data"])
        return [self.pred]


def frame_with_face(px_box):
    """Blue frame with a red rectangle at pixel ``(x0, y0, x1, y1)``."""
    image = np.empty((H, W, 3), np.uint8)
    image[:] = BLUE
    x0, y0, x1, y1 = px_box
    image[y0:y1, x0:x1] = RED
    return image


def norm(px_box):
    x0, y0, x1, y1 = px_box
    return (x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H)


def check_ranges(estimate):
    assert estimate.source == "face"
    assert 0 <= estimate.p_female <= 1
    assert 0 <= estimate.confidence <= 1
    assert len(estimate.p_age) == len(voice_match.AGES)
    assert all(0 <= p <= 1 for p in estimate.p_age)
    assert sum(estimate.p_age) == pytest.approx(1)


def test_protocol_and_lazy_construction():
    estimator = FaceEstimator()  # no weights, no session, no network
    assert isinstance(estimator, voice_match.FaceEstimator)


def test_crop_centred_rgb_with_margin():
    px = (300, 120, 380, 220)
    session = StubSession()
    estimate = FaceEstimator(session).estimate(frame_with_face(px), norm(px))
    check_ranges(estimate)
    (blob,) = session.blobs
    assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE) and blob.dtype == np.float32
    centre = blob[0, :, INPUT_SIZE // 2, INPUT_SIZE // 2]
    np.testing.assert_allclose(centre, RED)  # RGB order, 0-255, no normalisation
    corner = blob[0, :, 1, 1]  # 1.5x crop around the face: the margin is background
    np.testing.assert_allclose(corner, BLUE)
    red_rows = (blob[0, 0, :, INPUT_SIZE // 2] > 100).sum()
    assert red_rows == pytest.approx(INPUT_SIZE / 1.5, abs=2)  # taller side fills 2/3


def test_box_clamped_at_frame_edges_and_padded_with_zeros():
    session = StubSession()
    estimator = FaceEstimator(session)
    px = (0, 100, 64, 180)
    image = frame_with_face(px)
    # Box hangs off the left edge; it is clamped to the visible part.
    off = (-40 / W, 100 / H, 104 / W, 80 / H)
    assert face_pixels(image.shape, off) == pytest.approx((0, 100, 64, 180))
    check_ranges(estimator.estimate(image, off))
    blob = session.blobs[-1][0]
    np.testing.assert_array_equal(blob[:, :, :5], 0)  # outside the frame -> zero padding
    np.testing.assert_allclose(blob[:, INPUT_SIZE // 2, INPUT_SIZE // 2], RED)
    # Bottom-right corner, overhanging both edges.
    corner = (600 / W, 320 / H, 0.2, 0.2)
    assert face_pixels(image.shape, corner) == pytest.approx((600, 320, 640, 360))
    check_ranges(estimator.estimate(image, corner))
    np.testing.assert_array_equal(session.blobs[-1][0][:, -5:, -5:], 0)


@pytest.mark.parametrize(
    "box",
    [
        (0.5, 0.5, 20 / W, 0.3),  # 20 px wide
        (0.5, 0.5, 0.3, 31 / H),  # 31 px tall
        (-0.2, 0.5, 0.2 + 10 / W, 0.3),  # only 10 px visible after clamping
        (1.2, 0.5, 0.2, 0.2),  # entirely outside the frame
        (0.5, 0.5, 0.0, 0.2),
        (float("nan"), 0.5, 0.2, 0.2),
    ],
)
def test_tiny_or_empty_crop_returns_none_without_inference(box):
    session = StubSession()
    assert FaceEstimator(session).estimate(frame_with_face((0, 0, 1, 1)), box) is None
    assert session.blobs == []


@pytest.mark.parametrize(
    "pred",
    [(2, -2, 0.2), (-3, 3, 0.45), (0, 0, 0.75), (300, -300, -1), (-1e9, 1e9, 5), (0.1, -0.1, 0)],
)
def test_estimate_fields_in_range(pred):
    px = (200, 100, 300, 220)
    estimate = FaceEstimator(StubSession(pred)).estimate(frame_with_face(px), norm(px))
    check_ranges(estimate)


def test_output_mapping():
    female = to_estimate(np.array([3.0, -3.0, 0.20]), face_px=120)
    male = to_estimate(np.array([-3.0, 3.0, 0.40]), face_px=120)
    unsure = to_estimate(np.array([0.0, 0.0, 0.75]), face_px=120)
    assert female.p_female > 0.99 and male.p_female < 0.01 and unsure.p_female == 0.5
    assert [int(np.argmax(e.p_age)) for e in (female, male, unsure)] == [0, 1, 2]
    assert unsure.confidence == 0  # no gender margin -> no confidence
    assert voice_match.fuse(female, None).bucket == "female_young"
    assert voice_match.fuse(male, None).bucket == "male_adult"
    small = to_estimate(np.array([3.0, -3.0, 0.4]), face_px=32)
    assert small.confidence < female.confidence  # smaller crops are less trusted


def test_weights_available_and_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("HEARING_VOICE_MATCH", raising=False)
    assert not apparent_face.weights_available()
    assert voice_match.get_face_estimator() is None  # never downloads
    assert not any(tmp_path.iterdir())
    path = apparent_face.model_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"placeholder")  # factory only checks presence; loading is lazy
    assert apparent_face.weights_available()
    pytest.importorskip("onnxruntime")
    assert isinstance(voice_match.get_face_estimator(), apparent_face.FaceEstimator)


def test_import_touches_no_network_or_runtime():
    code = (
        "import socket, sys\n"
        "def deny(*a, **k): raise RuntimeError('network at import')\n"
        "socket.socket = deny; socket.create_connection = deny\n"
        "for m in ('onnxruntime', 'httpx', 'huggingface_hub'): sys.modules[m] = None\n"
        "import hearing.synth.apparent_face as af\n"
        "af.FaceEstimator()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


# --- real weights (download ~1.3 MB once into HEARING_MODELS_DIR) -----------


@pytest.fixture(scope="module")
def weights():
    import httpx

    try:
        return apparent_face.download_model()
    except (OSError, httpx.HTTPError) as exc:  # offline or Hub unavailable
        pytest.skip(f"genderage.onnx not downloadable: {exc}")


def synthetic_face():
    """A drawn face (skin oval, eyes, mouth) on grey, 640x360."""
    yy, xx = np.mgrid[:H, :W]
    image = np.full((H, W, 3), 110, np.uint8)
    image[((xx - 320) / 60) ** 2 + ((yy - 180) / 80) ** 2 <= 1] = (224, 172, 140)
    for ex in (297, 343):
        image[((xx - ex) / 9) ** 2 + ((yy - 160) / 5) ** 2 <= 1] = (40, 30, 30)
    image[((xx - 320) / 22) ** 2 + ((yy - 215) / 6) ** 2 <= 1] = (150, 60, 60)
    return image, (260 / W, 100 / H, 120 / W, 160 / H)


@pytest.mark.slow
@pytest.mark.network
def test_real_inference_on_synthetic_face(weights):
    models.clear()
    estimator = voice_match.get_face_estimator()
    assert isinstance(estimator, apparent_face.FaceEstimator)
    image, box = synthetic_face()
    estimate = estimator.estimate(image, box)
    check_ranges(estimate)
    assert estimator.estimate(image, box) == estimate  # deterministic


@pytest.mark.slow
@pytest.mark.network
def test_real_face_gender(weights):
    """Public-domain NASA portrait (scikit-image ``astronaut``): apparent female."""
    import io

    import httpx

    PIL = pytest.importorskip("PIL.Image")
    url = (
        "https://raw.githubusercontent.com/scikit-image/scikit-image/"
        "v0.24.0/skimage/data/astronaut.png"
    )
    try:
        data = httpx.get(url, timeout=30, follow_redirects=True).raise_for_status().content
    except httpx.HTTPError as exc:
        pytest.skip(f"test image not downloadable: {exc}")
    image = np.asarray(PIL.open(io.BytesIO(data)).convert("RGB"))
    box = (177 / 512, 62 / 512, 95 / 512, 105 / 512)  # OpenCV Haar detection, hand-checked
    estimate = FaceEstimator().estimate(image, box)
    check_ranges(estimate)
    assert estimate.p_female > 0.9 and estimate.confidence > 0.6
