"""T-337: HEARING_OVERLAP_FLOOR=1 keeps the hard gated floor only on overlap windows (default off)."""

import numpy as np
import pytest

from hearing.fusion.mixer import OVERLAP_KEY, Mixer, confidence_floor_db
from hearing.tse.stage import FocusHQ
from hearing.types import AudioChunk, StageContext
from tests.test_stream_vad import SR, room, voice

HOP = 160
KNOBS = ("HEARING_GATED_FLOOR_DB", "HEARING_OVERLAP_FLOOR", "HEARING_UNSURE_DB", "HEARING_TSE_CONFIDENCE")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)


def _mix_overlap(overlap, conf=0.0):
    ctx_extra = {} if overlap is None else {OVERLAP_KEY: overlap}
    m = Mixer()
    ctx = StageContext.default()
    ctx.suppression = 1.0
    ctx.telemetry.extras.update({"residual": np.ones(480, np.float32), "tse_confidence": conf, **ctx_extra})
    out = m.process(AudioChunk(np.zeros(480, np.float32), 0, 0.0, SR), ctx).samples
    assert OVERLAP_KEY not in ctx.telemetry.extras  # popped every chunk
    return out[-1], ctx.telemetry.extras["mixer"]["floor_db"]


def test_floor_rule(monkeypatch):
    monkeypatch.setenv("HEARING_GATED_FLOOR_DB", "20")
    assert confidence_floor_db(0.0, 0.0) == pytest.approx(20.0)  # knob off: overlap ignored
    monkeypatch.setenv("HEARING_OVERLAP_FLOOR", "1")
    assert confidence_floor_db(0.0, 1.0) == pytest.approx(20.0)  # overlap: hard
    assert confidence_floor_db(0.0, 0.0) == pytest.approx(6.0)  # solo: the unsure floor
    assert confidence_floor_db(0.0, None) == pytest.approx(20.0)  # unknown: hard (floor-20 behaviour)
    assert confidence_floor_db(0.5, 0.0) == pytest.approx(23.0)  # ungated windows: the T-134 line
    assert _mix_overlap(0.0) == (pytest.approx(10 ** (-6 / 20), rel=1e-4), 6.0)
    assert _mix_overlap(1.0) == (pytest.approx(0.1, rel=1e-4), 20.0)
    monkeypatch.delenv("HEARING_GATED_FLOOR_DB")
    assert confidence_floor_db(0.0, 1.0) == pytest.approx(6.0)  # without a gated floor the knob does nothing


class _Sep:
    device = "cpu"

    def __init__(self, solo):
        self.solo = solo

    def separate(self, mix, sr):
        n = len(mix)
        a = voice(110.0, s=n / SR + 0.01)[:n] + room(n / SR + 0.01, seed=1)[:n]
        b = room(n / SR + 0.01, seed=2)[:n] if self.solo else voice(170.0, s=n / SR + 0.01)[:n]
        return [a, b]


class _GatedCand:
    name = "gatedfake"
    licence = "test"
    needs = frozenset()
    device = "cpu"

    def __init__(self, solo):
        self.separator = _Sep(solo)
        self.confidence = None

    def load(self):
        pass

    def extract(self, mix, sr, **_):
        self.separator.separate(mix, sr)
        self.confidence = 0.0
        return np.zeros_like(mix)


def _run(solo, seconds=3.0):
    st = FocusHQ(_candidate=_GatedCand(solo), window_s=1.0, hop_s=0.5, threaded=False)
    mixer = Mixer()
    ctx = StageContext.default()
    ctx.suppression = 1.0
    x = voice(110.0, s=seconds) + voice(170.0, s=seconds)
    ys, ovl = [], []
    for i in range(0, len(x) - HOP + 1, HOP):
        chunk = st.process(AudioChunk(x[i : i + HOP], SR, i // HOP, i / SR), ctx)
        ovl.append(ctx.telemetry.extras.get(OVERLAP_KEY))
        ys.append(mixer.process(chunk, ctx).samples)
    return np.concatenate(ys), ovl, st


def test_knob_unset_is_bit_identical(monkeypatch):
    """Default off: no overlap side channel, no separator wrapper, output equal to the knob-off floor-20 run."""
    monkeypatch.setenv("HEARING_GATED_FLOOR_DB", "20")
    base, ovl, st = _run(solo=True)
    assert all(o is None for o in ovl) and "overlap" not in st.health()
    assert "separate" not in vars(st.cand.separator)  # not wrapped
    monkeypatch.setenv("HEARING_OVERLAP_FLOOR", "0")
    again, _, _ = _run(solo=True)
    np.testing.assert_array_equal(base, again)
    monkeypatch.delenv("HEARING_GATED_FLOOR_DB")
    monkeypatch.delenv("HEARING_OVERLAP_FLOOR")
    unset, _, _ = _run(solo=True)
    tail = slice(-SR, None)
    assert np.sqrt(np.mean(unset[tail] ** 2)) > 4 * np.sqrt(np.mean(base[tail] ** 2))  # 6 dB vs 20 dB floor


@pytest.mark.parametrize("solo,share,db", [(True, 0.0, 6.0), (False, 1.0, 20.0)])
def test_stage_reports_overlap_and_mixer_picks_the_floor(monkeypatch, solo, share, db):
    monkeypatch.setenv("HEARING_GATED_FLOOR_DB", "20")
    monkeypatch.setenv("HEARING_OVERLAP_FLOOR", "1")
    out, ovl, _ = _run(solo=solo)
    seen = [o for o in ovl if o is not None]
    assert seen and seen[-1] == pytest.approx(share)
    x = voice(110.0, s=3.0) + voice(170.0, s=3.0)
    tail = slice(-SR // 2, None)
    gain = np.sqrt(np.mean(out[tail] ** 2)) / np.sqrt(np.mean(x[tail] ** 2))
    assert 20 * np.log10(gain) == pytest.approx(-db, abs=0.5)
