import numpy as np
import pytest

from hearing.tse import TSECandidate, candidates, get_candidate
from hearing.tse.dolphin import (
    LICENCE,
    LIP_MEAN,
    LIP_STD,
    DolphinCandidate,
    MockBackend,
    download_model,
    num_frames,
    prepare_lips,
)


def synthetic_lips(seconds, fps=25.0, size=88):
    t = round(seconds * fps)
    return (np.arange(t * size * size) % 256).astype(np.uint8).reshape(t, size, size)


def test_registered_and_conforms():
    assert "dolphin" in candidates()
    c = get_candidate("dolphin")
    assert isinstance(c, TSECandidate)
    assert c.name == "dolphin" and c.needs == frozenset({"lips"})
    assert "non-commercial" in c.licence == LICENCE


def test_mock_extract_returns_input_length_and_lip_tensor():
    rng = np.random.default_rng(0)
    target = rng.normal(0, 0.1, 16000).astype(np.float32)
    backend = MockBackend(target[:15950])  # off by < LENGTH_TOLERANCE: padded
    c = DolphinCandidate(backend)
    c.load()
    mix = target + rng.normal(0, 0.1, 16000).astype(np.float32)
    y = c.extract(mix, 16000, lips=synthetic_lips(1.0), lips_fps=25.0)
    assert y.shape == (16000,) and y.dtype == np.float32
    np.testing.assert_array_equal(y[:15950], target[:15950])
    assert backend.last_mouth.shape == (1, 1, 25, 88, 88)
    assert c.device == "cpu"


@pytest.mark.parametrize("seconds", [0.5, 1.0, 2.0])
def test_lip_tensor_shape_and_normalisation(seconds):
    n = int(seconds * 16000)
    lips = synthetic_lips(seconds)
    mouth = prepare_lips(lips, n, 25.0)
    assert mouth.shape == (1, 1, num_frames(n), 88, 88) == (1, 1, int(seconds * 25), 88, 88)
    assert mouth.dtype == np.float32
    np.testing.assert_allclose(mouth[0, 0], (lips / 255.0 - LIP_MEAN) / LIP_STD, rtol=1e-5)


def test_lip_tensor_resizes_resamples_and_pads():
    lips = np.full((10, 96, 96), 255, np.uint8)  # 12.5 fps, larger crops, 0.8 s for a 1 s mix
    mouth = prepare_lips(lips, 16000, 12.5)
    assert mouth.shape == (1, 1, 25, 88, 88)
    np.testing.assert_allclose(mouth, (1.0 - LIP_MEAN) / LIP_STD, rtol=1e-5)


def test_missing_inputs_raise():
    c = DolphinCandidate(MockBackend())
    with pytest.raises(ValueError, match="lips"):
        c.extract(np.zeros(16000, np.float32), 16000)
    with pytest.raises(ValueError):
        c.extract(np.zeros(16000, np.float32), 8000, lips=synthetic_lips(1.0))
    with pytest.raises(ValueError):
        c.extract(np.zeros(16000, np.float32), 16000, lips=np.zeros((25, 88), np.uint8))


def test_missing_weights_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    from hearing.tse.dolphin import _load_network, available

    assert not available()
    with pytest.raises(RuntimeError, match="download_model"):
        _load_network()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
@pytest.mark.network
def test_real_dolphin_window():
    download_model()
    c = DolphinCandidate()
    c.load()
    mix = np.random.default_rng(0).normal(0, 0.05, 16000).astype(np.float32)
    y = c.extract(mix, 16000, lips=synthetic_lips(1.0), lips_fps=25.0)
    assert y.shape == (16000,) and np.isfinite(y).all() and np.abs(y).max() > 0
    assert c.device in {"cuda", "mps", "cpu"}
