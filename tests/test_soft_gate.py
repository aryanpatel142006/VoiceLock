"""T-370: ``HEARING_GATED_SOFT`` (default off) plays a gated overlap window's best stream at -(1 - p) * D dB."""
import numpy as np
import pytest

from hearing.tse.enrolled import soft_db, soft_gain_db

SR = 16000
T = np.arange(SR) / SR


def _tone(hz):
    return (0.3 * np.sin(2 * np.pi * hz * T)).astype(np.float32)


class ToneEmbedder:
    """Embedding = energy at 300 Hz and at 500 Hz, so a 300 Hz stream matches a 300 Hz enrollment."""

    device = "cpu"

    def embed(self, audio):
        a = np.asarray(audio, np.float64)
        return np.array([abs(np.dot(a, np.sin(2 * np.pi * hz * T[: len(a)]))) for hz in (300, 500)]) + 1e-6


def _candidate(streams, contrast=5.0):
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.blind import MockBackend as SeparatorMock
    from hearing.tse.enrolled import EnrolledCandidate

    # contrast far above any score: every two-profile window is gated
    return EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack(streams))), ToneEmbedder(), f0_veto=False,
                             contrast=contrast, clean_enroll=False)


def test_knob_reading(monkeypatch):
    monkeypatch.delenv("HEARING_GATED_SOFT", raising=False)
    monkeypatch.delenv("HEARING_GATED_SOFT_DB", raising=False)
    assert soft_db() is None
    monkeypatch.setenv("HEARING_GATED_SOFT", "0")
    assert soft_db() is None
    monkeypatch.setenv("HEARING_GATED_SOFT", "1")
    assert soft_db() == 12.0
    monkeypatch.setenv("HEARING_GATED_SOFT_DB", "6")
    assert soft_db() == 6.0
    monkeypatch.setenv("HEARING_GATED_SOFT_DB", "junk")
    assert soft_db() == 12.0


def test_gain_interpolates_in_db():
    assert soft_gain_db(0.0, 0.0, 12.0) == pytest.approx(0.0)  # at the contrast: p = 1, full level
    assert soft_gain_db(-0.25, 0.0, 12.0) == pytest.approx(-6.0)  # p = 0.5
    assert soft_gain_db(-2.0, 0.0, 12.0) == pytest.approx(-12.0)  # p clipped at 0: Gmin
    assert soft_gain_db(-2.0, 0.0, 12.0, p=0.75) == pytest.approx(-3.0)  # the team head's p wins


def test_default_off_keeps_the_hard_gate(monkeypatch):
    monkeypatch.delenv("HEARING_GATED_SOFT", raising=False)
    a, b = _tone(300), _tone(500)
    cand = _candidate([a, b])
    y = cand.extract(a + b, SR, enroll=a, other=b)
    assert cand.gated and cand.confidence == 0.0 and not cand.soft_gated
    assert not np.any(y)


def test_overlap_window_plays_best_stream_at_soft_gain(monkeypatch):
    monkeypatch.setenv("HEARING_GATED_SOFT", "1")
    monkeypatch.setenv("HEARING_GATED_SOFT_DB", "12")
    a, b = _tone(300), _tone(500)
    cand = _candidate([b, a])
    y = cand.extract(a + b, SR, enroll=a, other=b)
    assert cand.gated and cand.confidence == 0.0  # mixer still sees a gated window (room at the gated floor)
    assert cand.soft_gated and cand.selected_stream == 1
    assert cand.soft_gain_db == pytest.approx(-12.0)  # score far under the contrast: p = 0
    np.testing.assert_allclose(y, a * 10 ** (-12 / 20), rtol=1e-5, atol=1e-7)


def test_solo_confident_window_stays_gated(monkeypatch):
    monkeypatch.setenv("HEARING_GATED_SOFT", "1")
    b = _tone(500)
    cand = _candidate([b, np.zeros_like(b)])  # one active stream, a large score margin: a true gate
    y = cand.extract(b, SR, enroll=_tone(300), other=b)
    assert cand.gated and not cand.soft_gated
    assert not np.any(y)
