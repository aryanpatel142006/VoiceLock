import numpy as np
import pytest

from hearing.tse import TSECandidate, get_candidate
from hearing.tse.blind import (
    BlindAssocCandidate,
    BlindSeparator,
    MockBackend,
    download_model,
)


def signals():
    rng = np.random.default_rng(42)
    t = np.arange(50) * 0.04
    envelopes = np.array(
        [0.5 + 0.4 * np.sin(2 * np.pi * t), 0.5 + 0.4 * np.cos(3 * np.pi * t)]
    )
    streams = np.repeat(np.sqrt(envelopes), 640, axis=1) * rng.choice(
        [-1.0, 1.0], (2, 32000)
    )
    return streams.astype(np.float32), envelopes


def test_association_selects_oracle_and_holds_ambiguous_choice():
    streams, env = signals()
    candidate = BlindAssocCandidate(BlindSeparator(MockBackend(streams)))
    mix = streams.sum(axis=0)
    np.testing.assert_array_equal(
        candidate.extract(mix, 16000, p_speaking=env[1]), streams[1]
    )
    # A small advantage for stream 0 (gap < 0.1) must retain stream 1.
    np.testing.assert_array_equal(
        candidate.extract(mix, 16000, p_speaking=0.52 * env[0] + 0.48 * env[1]),
        streams[1],
    )
    np.testing.assert_array_equal(
        candidate.extract(mix, 16000, p_speaking=env[0]), streams[0]
    )


def test_low_rate_probabilities_and_reset():
    streams, env = signals()
    c = BlindAssocCandidate(BlindSeparator(MockBackend(streams)))
    mix = streams.sum(axis=0)
    np.testing.assert_array_equal(
        c.extract(mix, 16000, p_speaking=env[1, ::2], lips_fps=12.5), streams[1]
    )
    c.reset()
    np.testing.assert_array_equal(
        c.extract(mix, 16000, p_speaking=np.ones(50)), streams[0]
    )


def test_contract_and_missing_inputs():
    assert isinstance(get_candidate("blind_assoc"), TSECandidate)
    c = BlindAssocCandidate(BlindSeparator(MockBackend()))
    with pytest.raises(ValueError):
        c.extract(np.zeros(16000), 8000, p_speaking=np.ones(25))
    with pytest.raises(ValueError):
        c.extract(np.zeros(16000), 16000)


def test_missing_weights_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    from hearing.tse.blind import _load_separator

    with pytest.raises(RuntimeError, match="download_model"):
        _load_separator()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
@pytest.mark.network
def test_real_blind_window():
    download_model()
    c = BlindAssocCandidate()
    c.load()
    y = c.extract(
        np.random.default_rng(0).normal(0, 0.02, 16000).astype(np.float32),
        16000,
        p_speaking=np.linspace(0, 1, 25),
    )
    assert y.shape == (16000,) and np.isfinite(y).all()
