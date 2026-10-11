"""T-336 team head: features, fit/save/load, and the HEARING_TEAM_HEAD knob (default off = unchanged)."""

import numpy as np
import pytest

from hearing.tse import enroll
from hearing.tse.blind import BlindSeparator
from hearing.tse.blind import MockBackend as SeparatorMock
from hearing.tse.enrolled import EnrolledCandidate
from hearing.tse.team_head import (
    ENV,
    FEATURES,
    TeamHead,
    fit_logistic,
    from_env,
    stream_features,
)

N = 16000


class Embedder:
    """Stream 0 (positive mean) sits near the target profile, stream 1 near the other."""

    device = "cpu"

    def embed(self, audio):
        m = float(np.mean(audio))
        base = np.array([1.0, 0.1, 0.05]) if m > 0 else np.array([0.1, 1.0, 0.05])
        return base + 0.02 * np.array([np.std(audio), m, len(audio) % 7])


def _profile(sign, seed):
    rng = np.random.default_rng(seed)
    return enroll.profile_clip([sign * (0.5 + 0.1 * rng.random(N // 2)).astype(np.float32) for _ in range(3)])


def _run(streams):
    c = EnrolledCandidate(BlindSeparator(SeparatorMock(streams)), Embedder(), clean_enroll=False, f0_veto=False)
    target, other = _profile(1.0, 0), _profile(-1.0, 1)
    c.extract(streams.sum(axis=0), N, enroll=target, other=other)
    y = c.extract(streams.sum(axis=0), N, enroll=target, other=other)
    return c, y


def _streams(a=0.3, b=0.3):
    return np.stack([np.full(N, a, np.float32), np.full(N, -b, np.float32)])


def _head(threshold):
    return TeamHead(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), [1.0, 0, 0, 0, 0], 0.0, threshold)


def test_stream_features_shape_and_rival():
    x = stream_features([0.8, -0.6], [0.5, 0.1], [0.1, 0.4], [4.0, 1.0])
    assert x.shape == (2, len(FEATURES))
    np.testing.assert_allclose(x[:, 1], [-0.6, 0.8])
    np.testing.assert_allclose(x[:, 4], [0.0, 10 * np.log10(0.25)])


def test_fit_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(FEATURES)))
    y = (x[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(int)
    head = TeamHead(*fit_logistic(x, y), threshold=0.6)
    assert np.mean((head.prob(x) >= 0.5) == y) > 0.9
    head.save(tmp_path / "h.json")
    back = TeamHead.load(tmp_path / "h.json")
    np.testing.assert_allclose(back.prob(x), head.prob(x))
    assert back.threshold == 0.6
    assert np.all((back.scores(x) > 0) == (back.prob(x) >= 0.6))


def test_knob_unset_is_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert from_env() is None
    monkeypatch.setenv(ENV, "  ")
    assert from_env() is None


def test_unset_knob_leaves_extract_bit_for_bit(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    c, y = _run(_streams())
    assert c.team_head is None and c.head_p is None
    assert c.scoring == "lda"
    ref = EnrolledCandidate(BlindSeparator(SeparatorMock(_streams())), Embedder(), clean_enroll=False, f0_veto=False)
    ref.team_head = None
    target, other = _profile(1.0, 0), _profile(-1.0, 1)
    for _ in range(2):
        y_ref = ref.extract(_streams().sum(axis=0), N, enroll=target, other=other)
    np.testing.assert_array_equal(y, y_ref)
    assert (c.scores, c.confidence, c.gated, c.selected_stream) == (ref.scores, ref.confidence, ref.gated, ref.selected_stream)


def test_knob_set_gates_by_head_probability(monkeypatch, tmp_path):
    path = tmp_path / "head.json"
    _head(0.5).save(path)
    monkeypatch.setenv(ENV, str(path))
    c, y = _run(_streams())
    assert c.team_head is not None and c.head_p is not None
    assert c.selected_stream == 0 and not c.gated
    assert c.confidence == pytest.approx(c.head_p[0])
    np.testing.assert_array_equal(y, _streams()[0])
    _head(0.9999).save(path)  # a strict reject threshold gates the same window
    c, y = _run(_streams())
    assert c.gated and c.confidence == 0.0
    assert not np.any(y)


def test_head_off_without_closed_set(monkeypatch, tmp_path):
    path = tmp_path / "head.json"
    _head(0.9999).save(path)
    monkeypatch.setenv(ENV, str(path))
    streams = _streams()
    c = EnrolledCandidate(BlindSeparator(SeparatorMock(streams)), Embedder(), clean_enroll=False, f0_veto=False)
    y = c.extract(streams.sum(axis=0), N, enroll=np.full(N, 0.5, np.float32))
    assert c.scoring == "cosine" and c.head_p is None
    np.testing.assert_array_equal(y, streams[0])
