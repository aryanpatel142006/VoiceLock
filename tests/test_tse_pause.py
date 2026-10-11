"""Pause handling of the Enrolled tier (T-069): hold or duck when no stream matches the target and the
camera says the target is quiet, and the stage's ``asd=`` side input."""

import math

import numpy as np
import pytest

from hearing.tse.blind import BlindSeparator
from hearing.tse.enrolled import EnrolledCandidate
from hearing.tse.stage import FocusHQ
from hearing.types import AudioChunk, FaceTrack, StageContext, TargetState

SR = 16000
HOP = 160


class SignEmbedder:
    """Embedding = (positive share, negative share): a "voice" is the sign of the samples."""

    device = "cpu"

    def embed(self, audio):
        return np.array([np.mean(audio > 0) + 1e-3, np.mean(audio < 0) + 1e-3])


class Streams:
    device = "cpu"

    def __init__(self, fn):
        self.fn = fn

    def separate(self, mix):
        return self.fn(mix)


def _scene():
    """Window 1: the target stream (all positive) matches clearly. Later windows: the streams are swapped
    and both are sign-balanced, so neither matches the enrolled "positive" voice (a target pause)."""
    rng = np.random.default_rng(1)
    pos = np.abs(rng.standard_normal(3 * SR)).astype(np.float32)
    other = rng.standard_normal(3 * SR).astype(np.float32)
    mix = pos + other
    calls = []

    def sep(window):
        start = int(np.flatnonzero(mix == window[0])[0])
        t, o = pos[start : start + len(window)], other[start : start + len(window)]
        calls.append(start)
        if len(calls) == 1:
            return np.stack([t, o])
        return np.stack([o, t - np.abs(t).mean()])

    return mix, sep


def _cand(sep, **kw):
    return EnrolledCandidate(BlindSeparator(Streams(sep)), SignEmbedder(), margin=0.0, clean_enroll=False, **kw)


ENROLL = np.ones(SR, np.float32)


def _windows(c, mix, n=2, **kw):
    out = []
    for k in range(n):
        out.append(c.extract(mix[k * SR // 2 : k * SR // 2 + SR], SR, enroll=ENROLL, **kw))
    return out


def test_pause_hold_keeps_the_previous_stream_when_nothing_matches():
    mix, sep = _scene()
    plain = _cand(sep)
    _windows(plain, mix)
    assert plain.selected_stream == 0 and not plain.paused  # the residual stream wins the argmax
    mix, sep = _scene()
    c = _cand(sep, pause_floor=0.95)
    _, y1 = _windows(c, mix)
    assert not c.gated and c.paused and c.held and c.selected_stream == 1  # followed the waveform
    assert max(c.similarities) < 0.95 and y1.any()


def test_a_matching_window_is_not_a_pause():
    mix, sep = _scene()
    c = _cand(sep, pause_floor=0.95)
    c.extract(mix[:SR], SR, enroll=ENROLL)
    assert not c.paused and c.selected_stream == 0 and c.similarity > 0.95


def test_pause_hold_is_limited_to_pause_max_s():
    mix, sep = _scene()
    c = _cand(sep, pause_floor=0.95, pause_max_s=0.5)
    _windows(c, mix, n=3)
    assert not c.paused and c.selected_stream == 0  # third window: 1.0 s of pause > 0.5 s, back to argmax
    mix, sep = _scene()
    c = _cand(sep, pause_floor=0.95, pause_max_s=1.0)
    _windows(c, mix, n=3)
    assert c.paused and c.selected_stream == 1  # still following the target's waveform


@pytest.mark.parametrize(("asd", "paused"), [(0.1, True), (0.9, False), (math.nan, True)])
def test_pause_p_asks_the_camera(asd, paused):
    mix, sep = _scene()
    c = _cand(sep, pause_floor=0.95, pause_p=0.5)
    assert "asd" in c.needs and "asd" not in EnrolledCandidate.needs
    _windows(c, mix, asd=np.full(100, asd, np.float32))
    assert c.paused is paused  # NaN: no target face visible counts as quiet
    assert c.selected_stream == (1 if paused else 0)


def test_pause_p_without_camera_evidence_is_not_a_pause():
    mix, sep = _scene()
    c = _cand(sep, pause_floor=0.95, pause_p=0.5)
    _windows(c, mix)  # asd=None
    assert not c.paused and c.selected_stream == 0


def test_pause_duck_returns_silence():
    mix, sep = _scene()
    c = _cand(sep, pause_floor=0.95, pause_action="duck")
    _, y1 = _windows(c, mix)
    assert c.paused and c.gated and not y1.any()


def test_pause_loud_plays_the_stream_with_the_most_energy():
    rng = np.random.default_rng(2)
    quiet, loud = rng.standard_normal((2, SR)).astype(np.float32) * np.float32([[0.1], [0.5]])

    def sep(window):
        return np.stack([quiet, loud])  # both sign-balanced: neither matches the "positive" enrollment

    c = _cand(sep, pause_floor=0.95, pause_action="loud")
    y = c.extract(quiet + loud, SR, enroll=ENROLL)
    assert c.paused and not c.gated and c.selected_stream == 1 and np.array_equal(y, loud)
    c = _cand(sep, pause_floor=0.95, pause_action="loud", pause_p=0.5)
    c.extract(quiet + loud, SR, enroll=ENROLL, asd=np.full(100, 0.9, np.float32))
    assert not c.paused  # the camera says the target talks: not a pause, argmax decides


def test_pause_options_are_validated_and_off_by_default():
    _, sep = _scene()
    with pytest.raises(ValueError, match="pause_action"):
        _cand(sep, pause_floor=0.3, pause_action="mute")
    c = _cand(sep)
    assert c.pause_floor is None and c.needs == frozenset({"enroll", "other"})


def _ctx(target, p):
    ctx = StageContext.default()
    ctx.target = TargetState(target, 1.0, target is not None, 0.0)
    ctx.tracks = [FaceTrack(1, (0, 0, 0.2, 0.2), None, p, 0.0), FaceTrack(2, (0.6, 0, 0.2, 0.2), None, 0.5, 0.0)]
    return ctx


def test_stage_passes_the_target_asd_trace_only_to_candidates_that_ask():
    seen, plain_calls = [], []

    class Asd:
        name, licence, device = "a", "test", "cpu"
        needs = frozenset({"asd"})

        def load(self):
            pass

        def extract(self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None, asd=None):
            seen.append(asd)
            return mix

    class Plain(Asd):
        needs = frozenset()

        def extract(self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None):
            plain_calls.append(len(mix))
            return mix

    for cand in (Asd(), Plain()):
        st = FocusHQ(_candidate=cand, window_s=0.1, hop_s=0.05, threaded=False, residual=False)
        x = np.full(HOP, 0.1, np.float32)
        for i in range(30):
            target = None if i >= 20 else 1
            st.process(AudioChunk(x, SR, i, i * 0.01), _ctx(target, i / 100))
        assert all(w.status == "ok" for w in st._recent)
    np.testing.assert_allclose(seen[0], np.arange(10) / 100, atol=1e-6)  # per 10 ms, the target's p
    np.testing.assert_allclose(seen[2], np.arange(10, 20) / 100, atol=1e-6)
    assert np.isnan(seen[-1]).all()  # no target: NaN
    assert len(plain_calls) == len(seen)
