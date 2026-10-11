"""Enrolled pick ``contrast`` floor and the pair-cosine telemetry (T-178)."""

import numpy as np

from hearing.tse.blind import BlindSeparator
from hearing.tse.blind import MockBackend as SeparatorMock
from hearing.tse.enrolled import CONTRAST_ENV, EnrolledCandidate

SR = 16000


class SignEmbedder:
    """Embedding = (positive share, negative share): a "voice" is the sign of the samples."""

    device = "cpu"

    def embed(self, audio):
        return np.array([np.mean(audio > 0) + 1e-3, np.mean(audio < 0) + 1e-3])


def _streams():
    # stream 0: 60 % positive samples (leans to the target), stream 1: all negative (the other voice)
    s0 = np.where(np.arange(SR) % 10 < 6, 1.0, -1.0).astype(np.float32)
    return np.stack([s0, -np.ones(SR, np.float32)])


def _cand(contrast, **kw):
    return EnrolledCandidate(BlindSeparator(SeparatorMock(_streams())), SignEmbedder(), margin=0.0, clean_enroll=False,
                             contrast=contrast, f0_veto=False, **kw)


TARGET, OTHER = np.ones(SR, np.float32), -np.ones(SR, np.float32)


def test_contrast_silences_a_window_whose_score_is_below_it():
    mix = _streams().sum(axis=0)  # the mock separator rescales its streams to sum to the mix
    # the picked stream scores cos(target) - cos(other) = (0.6 - 0.4) / |(0.6, 0.4)| ~= 0.277
    for contrast, gated in ((0.0, False), (0.1, False), (0.2, False), (0.3, True)):
        c = _cand(contrast)
        y = c.extract(mix, SR, enroll=TARGET, other=OTHER)
        assert c.selected_stream == 0
        assert 0.27 < c.scores[0] < 0.29
        assert c.gated is gated, contrast
        assert (not y.any()) is gated
        assert c.confidence == 0.0 if gated else c.confidence > 0.0


def test_contrast_needs_the_negative():
    c = _cand(0.9)  # no negative: the floor does not apply (only ``floor`` gates then)
    y = c.extract(np.ones(SR, np.float32), SR, enroll=TARGET)
    assert not c.gated and y.any()
    assert c.pair_cos is None and c.negative_active is False


def test_pair_cos_logged_and_same_voice_drops_the_negative():
    c = _cand(0.3)
    c.extract(np.ones(SR, np.float32), SR, enroll=TARGET, other=OTHER)
    assert c.pair_cos < 0.01 and c.negative_active
    near = np.where(np.arange(SR) % 10 < 9, 1.0, -1.0).astype(np.float32)  # cos ~0.99 with the target
    c = _cand(0.3)
    y = c.extract(np.ones(SR, np.float32), SR, enroll=TARGET, other=near)
    assert c.pair_cos > 0.7 and not c.negative_active
    assert not c.gated and y.any()  # the dropped negative also drops the contrast floor


def test_contrast_default_from_env(monkeypatch):
    monkeypatch.delenv(CONTRAST_ENV, raising=False)
    assert _cand(None).contrast == 0.0
    monkeypatch.setenv(CONTRAST_ENV, "0.2")
    assert _cand(None).contrast == 0.2
    assert _cand(0.1).contrast == 0.1  # an explicit value (pipeline spec) wins


def test_tse_stage_passes_contrast_and_reports_it():
    from hearing.tse.stage import FocusHQ

    st = FocusHQ(candidate="enrolled", contrast=0.2)
    assert st.cand.contrast == 0.2
    h = st.health()
    assert h["contrast"] == 0.2 and h["pair_cos"] is None


def test_team_preset_sets_the_contrast_env(monkeypatch):
    from hearing.server.team import apply_env

    for k in (CONTRAST_ENV, "HEARING_ASR", "HEARING_VOICE_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    assert CONTRAST_ENV not in apply_env({"asr": "local"})  # no key: the candidate's default (0) stays
    monkeypatch.delenv(CONTRAST_ENV, raising=False)
    assert apply_env({"asr": "local", "enrolled_contrast": 0.3})[CONTRAST_ENV] == "0.3"
