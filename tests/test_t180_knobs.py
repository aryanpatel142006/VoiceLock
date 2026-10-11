"""T-180: A/B knobs for gated / unsure windows (all off by default, set by env)."""

import numpy as np
import pytest

from hearing.fusion.mixer import Mixer, confidence_floor_db
from hearing.latency_preset import apply_postfilter
from tests.test_tse_confidence import SR, _mix

ENROLLED = 'tse(candidate="enrolled", window_s=1.0, hop_s=0.5, threaded=true)>mixer(max_duck_db=20)'


def test_defaults_unchanged(monkeypatch):
    for k in ("HEARING_GATED_FLOOR_DB", "HEARING_UNSURE_DB", "HEARING_TSE_POSTFILTER"):
        monkeypatch.delenv(k, raising=False)
    assert confidence_floor_db(0.0) == pytest.approx(6.0)
    assert confidence_floor_db(1.0) == pytest.approx(40.0)
    assert apply_postfilter(ENROLLED) == ENROLLED


def test_k2_gated_floor_only_at_confidence_zero(monkeypatch):
    monkeypatch.setenv("HEARING_GATED_FLOOR_DB", "20")
    assert confidence_floor_db(0.0) == pytest.approx(20.0)
    assert confidence_floor_db(0.5) == pytest.approx(23.0)  # an ungated window keeps the T-134 line
    out, ctx = _mix(Mixer(max_duck_db=20), 0.0)
    assert out[-1] == pytest.approx(0.1, rel=1e-4)  # the mix at -20 dB, not -6 dB
    assert ctx.telemetry.extras["mixer"]["floor_db"] == pytest.approx(20.0)


def test_k3_unsure_db(monkeypatch):
    monkeypatch.setenv("HEARING_UNSURE_DB", "12")
    assert confidence_floor_db(0.0) == pytest.approx(12.0)
    assert confidence_floor_db(0.5) == pytest.approx(26.0)
    assert confidence_floor_db(1.0) == pytest.approx(40.0)
    out, _ = _mix(Mixer(), 0.0)
    assert out[-1] == pytest.approx(10 ** (-12 / 20), rel=1e-4)
    monkeypatch.setenv("HEARING_UNSURE_DB", "junk")  # invalid -> default
    assert confidence_floor_db(0.0) == pytest.approx(6.0)


def _gated_candidate():
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.blind import MockBackend as SeparatorMock
    from hearing.tse.enrolled import EnrolledCandidate

    class SignEmbedder:
        device = "cpu"

        def embed(self, audio):
            return np.array([np.mean(audio > 0) + 1e-3, np.mean(audio < 0) + 1e-3])

    a = np.ones(SR, np.float32)
    # contrast far above any score: every two-profile window is gated
    cand = EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack([a, -a]))), SignEmbedder(), f0_veto=False,
                             contrast=5.0, clean_enroll=False)
    return cand, a


def test_k5_gated_two_profile_window_plays_best_stream(monkeypatch):
    monkeypatch.delenv("HEARING_GATED_PLAY", raising=False)
    cand, a = _gated_candidate()
    y = cand.extract(a, SR, enroll=a, other=-a)
    assert cand.negative_active and cand.gated and cand.confidence == 0.0
    assert not np.any(y)  # default: silence
    monkeypatch.setenv("HEARING_GATED_PLAY", "stream")
    cand, a = _gated_candidate()
    y = cand.extract(a, SR, enroll=a, other=-a)
    assert cand.gated and cand.confidence == 0.0
    np.testing.assert_allclose(y, a)  # the best stream (target side), at confidence 0


def test_k5_needs_a_negative(monkeypatch):
    monkeypatch.setenv("HEARING_GATED_PLAY", "stream")
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.blind import MockBackend as SeparatorMock
    from hearing.tse.enrolled import EnrolledCandidate

    class Flat:
        device = "cpu"

        def embed(self, audio):
            return np.array([1.0, 0.0])

    a = np.ones(SR, np.float32)
    cand = EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack([a, -a]))), Flat(), f0_veto=False, floor=2.0)
    y = cand.extract(a, SR, enroll=a)  # one profile, gated by the floor: still silence
    assert cand.gated and not np.any(y)


def test_p2_postfilter_inserts_dfn(monkeypatch):
    monkeypatch.setenv("HEARING_TSE_POSTFILTER", "1")
    assert apply_postfilter(ENROLLED) == ENROLLED.replace(")>mixer", ")>dfn>mixer")
    assert apply_postfilter("dfn_ll") == "dfn_ll"
    already = 'tse(candidate="enrolled")>dfn>mixer'
    assert apply_postfilter(already) == already
    assert apply_postfilter(ENROLLED, on=False) == ENROLLED


def test_p2_reaches_the_tiers_ladder(monkeypatch):
    from hearing import registry
    from hearing.fusion import tiers

    seen = []
    real = registry.build_pipeline

    def record(text):
        seen.append(text)
        return real("mixer")

    monkeypatch.setenv("HEARING_TSE_POSTFILTER", "1")
    monkeypatch.setattr(tiers.registry, "build_pipeline", record)
    cfg = {"tiers": [{"name": "enrolled", "pipeline": 'tse(candidate="enrolled")>mixer'},
                     {"name": "audio-only", "pipeline": "mixer"}]}
    tiers.TierLadder(cfg)
    assert any(">dfn>mixer" in t for t in seen) and "mixer" in seen
