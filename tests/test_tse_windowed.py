import math

import numpy as np
import pytest

from hearing.tse import (
    SlidingWindowExtractor,
    TSECandidate,
    candidates,
    conform_output,
    frame_span,
    get_candidate,
    register_candidate,
)
from hearing.tse.mock import IdentityCandidate, OracleCandidate
from hearing.tse.windowed import crossfade_window

SR = 16000
FPS = 25.0


def _mix(seconds: float, seed: int = 0) -> np.ndarray:
    return (
        np.random.default_rng(seed)
        .uniform(-0.9, 0.9, round(seconds * SR))
        .astype(np.float32)
    )


class Recorder:
    """Identity candidate that records every call's inputs."""

    name = "recorder"
    licence = "test"
    needs = frozenset({"lips", "p_speaking", "enroll"})

    def __init__(self):
        self.calls = []

    def load(self) -> None:
        pass

    def extract(self, mix, sr, *, lips, lips_fps, p_speaking, enroll):
        self.calls.append(
            {
                "mix": mix.copy(),
                "lips": lips,
                "p": p_speaking,
                "enroll": enroll,
                "fps": lips_fps,
            }
        )
        return mix


class Gain(Recorder):
    name = "gain"

    def extract(self, mix, sr, **kw):
        return 0.5 * mix


class WrongLength(Recorder):
    name = "wrong-length"

    def extract(self, mix, sr, **kw):
        return mix[: len(mix) // 2]


def test_mocks_satisfy_protocol_and_are_registered():
    for cls in (IdentityCandidate, OracleCandidate):
        assert isinstance(cls(), TSECandidate)
    assert {"identity", "oracle"} <= set(candidates())
    assert isinstance(get_candidate("identity"), IdentityCandidate)
    assert get_candidate("oracle").needs == frozenset({"enroll"})
    assert isinstance(SlidingWindowExtractor(IdentityCandidate(), 1.0), TSECandidate)


def test_registry_rejects_duplicates_and_unknown_names():
    with pytest.raises(ValueError):
        register_candidate("identity")(Recorder)
    with pytest.raises(KeyError):
        get_candidate("no-such-candidate")


@pytest.mark.parametrize("window_s", [0.5, 1.0, 2.0])
def test_identity_overlap_add_reproduces_input(window_s):
    x = _mix(5.3)
    ext = SlidingWindowExtractor(IdentityCandidate(), window_s, window_s / 2)
    y = ext.extract(x, SR)
    assert y.dtype == np.float32 and y.shape == x.shape
    first = round(window_s * SR)
    assert np.max(np.abs(y[first:] - x[first:])) < 1e-5
    assert np.max(np.abs(y - x)) < 1e-5  # the flat-edged first window is exact too


@pytest.mark.parametrize(
    ("window_s", "hop_s"), [(1.0, 0.25), (1.0, 0.8), (1.0, 1.0), (2.0, 0.5)]
)
def test_identity_for_other_hops(window_s, hop_s):
    x = _mix(4.1, seed=1)
    y = SlidingWindowExtractor(IdentityCandidate(), window_s, hop_s).extract(x, SR)
    assert np.max(np.abs(y - x)) < 1e-5


def test_input_shorter_than_one_window():
    x = _mix(0.3)
    y = SlidingWindowExtractor(IdentityCandidate(), 1.0).extract(x, SR)
    assert np.max(np.abs(y - x)) < 1e-5


def test_linear_candidate_is_preserved():
    x = _mix(3.0)
    y = SlidingWindowExtractor(Gain(), 1.0).extract(x, SR)
    assert np.max(np.abs(y - 0.5 * x)) < 1e-5


@pytest.mark.parametrize("hop", [8000, 10000, 16000])
def test_crossfade_sums_to_one_for_half_or_sparser_hops(hop):
    n = 16000
    w = crossfade_window(n, hop)
    assert np.all(w > 0)
    total = np.zeros(n + 4 * hop)
    for k in range(5):
        total[k * hop : k * hop + n] += w
    assert np.allclose(total[n : 4 * hop], 1.0, atol=1e-12)
    assert np.allclose(
        crossfade_window(n, n // 2), np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    )  # Hann


def test_frame_span():
    assert frame_span(0, 16000, SR, FPS) == (0, 25)
    assert frame_span(8000, 16000, SR, FPS) == (13, 25)  # 12.5 rounds half up
    assert frame_span(4000, 8000, SR, FPS) == (6, 13)


@pytest.mark.parametrize("window_s", [0.5, 1.0, 2.0])
def test_extract_gets_aligned_lips_and_p_speaking(window_s):
    seconds = 4.0
    x = _mix(seconds)
    n_frames = math.ceil(seconds * FPS)
    # Frame i of the lips is filled with i, p_speaking[i] = i / 1000: slices reveal their offset.
    lips = np.broadcast_to(
        np.arange(n_frames, dtype=np.uint8)[:, None, None], (n_frames, 88, 88)
    ).copy()
    p = (np.arange(n_frames) / 1000).astype(np.float32)
    enroll = _mix(3.0, seed=7)
    rec = Recorder()
    ext = SlidingWindowExtractor(rec, window_s)
    ext.extract(x, SR, lips=lips, lips_fps=FPS, p_speaking=p, enroll=enroll)
    starts = ext.window_starts(len(x))
    assert len(rec.calls) == len(starts) > 1
    win = round(window_s * SR)
    for s, call in zip(starts, rec.calls):
        f0, nf = frame_span(s, win, SR, FPS)
        assert nf == math.floor(window_s * FPS + 0.5)  # 13, 25, 50 frames
        assert call["mix"].shape == (win,)
        seg = x[s : s + win]
        assert np.array_equal(call["mix"][: len(seg)], seg)
        assert call["lips"].shape == (nf, 88, 88) and call["lips"].dtype == np.uint8
        assert call["p"].shape == (nf,)
        assert call["lips"][0, 0, 0] == min(f0, n_frames - 1)
        assert call["p"][0] == pytest.approx(min(f0, n_frames - 1) / 1000)
        # Audio sample s sits inside the first lip frame of the slice.
        assert abs(s / SR - f0 / FPS) <= 0.5 / FPS + 1e-9
        assert call["fps"] == FPS
        assert call["enroll"] is enroll  # enrollment clips are passed whole


def test_lips_and_p_speaking_are_padded_past_the_end():
    x = _mix(2.2)
    lips = np.full((40, 88, 88), 9, np.uint8)  # 1.6 s of video only
    p = np.ones(40, np.float32)
    rec = Recorder()
    SlidingWindowExtractor(rec, 1.0).extract(
        x, SR, lips=lips, lips_fps=FPS, p_speaking=p, enroll=None
    )
    last = rec.calls[-1]
    assert last["lips"].shape == (25, 88, 88) and np.all(
        last["lips"] == 9
    )  # edge-repeated
    assert last["p"].shape == (25,) and last["p"][-1] == 0.0  # no speech past the end
    assert all(c["lips"] is not None for c in rec.calls)


def test_oracle_gets_window_aligned_clean_target():
    x = _mix(4.0)
    target = _mix(4.0, seed=3)
    for window_s in (0.5, 1.0, 2.0):
        y = SlidingWindowExtractor(OracleCandidate(), window_s).extract(
            x, SR, enroll=target
        )
        assert np.max(np.abs(y - target)) < 1e-5


def test_stats_report_latency_and_rtf():
    x = _mix(4.0)
    ext = SlidingWindowExtractor(IdentityCandidate(), 1.0)
    ext.extract(x, SR)
    st = ext.stats()
    assert st.algorithmic_latency_ms == 1000.0
    assert st.n_windows == len(ext.window_starts(len(x))) == 7
    assert st.audio_s == pytest.approx(4.0)
    assert 0 <= st.compute_p50_ms <= st.compute_p95_ms <= st.compute_max_ms
    assert 0 <= st.rtf < 1
    assert st.as_dict()["window_s"] == 1.0
    ext.reset_stats()
    assert ext.stats().n_windows == 0 and math.isnan(ext.stats().rtf)


def test_contract_violations_raise():
    with pytest.raises(ValueError):
        SlidingWindowExtractor(IdentityCandidate(), 1.0, 1.5)
    with pytest.raises(ValueError):
        SlidingWindowExtractor(IdentityCandidate(), 1.0).extract(_mix(1.0), 48000)
    with pytest.raises(ValueError):
        SlidingWindowExtractor(WrongLength(), 1.0).extract(_mix(2.0), SR)


def test_conform_output():
    assert conform_output(np.ones(15990), 16000).shape == (16000,)
    assert conform_output(np.ones((1, 16010)), 16000).shape == (16000,)
    with pytest.raises(ValueError):
        conform_output(np.ones(15000), 16000)
    with pytest.raises(ValueError):
        conform_output(np.ones((2, 16000)), 16000)
