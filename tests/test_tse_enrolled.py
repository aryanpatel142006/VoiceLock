import numpy as np
import pytest

from hearing.tse import TSECandidate, get_candidate
from hearing.tse.blind import BlindSeparator
from hearing.tse.blind import MockBackend as SeparatorMock
from hearing.tse.enrolled import EnrolledCandidate, MockBackend, download_model


def test_cosine_selection_and_similarity():
    streams = np.array([np.ones(1600), -np.ones(1600)], dtype=np.float32)

    # Amplitude-independent cosine should prefer the second embedding.
    class Embedder:
        device = "cpu"

        def embed(self, audio):
            return np.array([10.0, 1.0]) if audio.mean() > 0 else np.array([0.0, 2.0])

    c = EnrolledCandidate(BlindSeparator(SeparatorMock(streams)), Embedder(), clean_enroll=False)
    y = c.extract(streams.sum(axis=0), 16000, enroll=-np.ones(3200))
    np.testing.assert_array_equal(y, streams[1])
    assert c.similarity == pytest.approx(1.0)
    assert c.selected_stream == 1


def test_offline_mock_contract_and_validation():
    assert isinstance(get_candidate("enrolled"), TSECandidate)
    c = EnrolledCandidate(BlindSeparator(SeparatorMock()), MockBackend())
    c.load()
    with pytest.raises(ValueError):
        c.extract(np.ones(1600), 16000)
    with pytest.raises(ValueError):
        c.extract(np.ones(1600), 16000, enroll=np.array([]))
    y = c.extract(np.ones(1600), 16000, enroll=np.ones(1600))
    assert y.shape == (1600,) and np.isfinite(c.similarity)


@pytest.mark.slow
@pytest.mark.network
def test_real_enrolled_window():
    from hearing.tse.blind import download_model as download_separator

    download_separator()
    download_model()
    c = EnrolledCandidate()
    c.load()
    x = np.random.default_rng(0).normal(0, 0.02, 16000).astype(np.float32)
    y = c.extract(x, 16000, enroll=x)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert np.isfinite(c.similarity)


def test_missing_ecapa_weights_never_downloads(tmp_path, monkeypatch):
    from hearing.tse.enrolled import _load_encoder

    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="download_model"):
        _load_encoder()
    assert list(tmp_path.iterdir()) == []
