"""T-386: the enrolled tier plays the fine-tuned checkpoint's fixed slot under HEARING_PICK=fixed + HEARING_SEP_CKPT."""
import numpy as np

from hearing.tse.enroll import profile_clip, profile_name, split_pieces

SR = 16000
T = np.arange(SR) / SR


def _tone(hz, amp=0.3):
    return (amp * np.sin(2 * np.pi * hz * T)).astype(np.float32)


class ToneEmbedder:
    """Embedding = energy at 300 Hz and at 500 Hz, so a 300 Hz stream matches a 300 Hz enrollment."""

    device = "cpu"

    def embed(self, audio):
        a = np.asarray(audio, np.float64)
        return np.array([abs(np.dot(a, np.sin(2 * np.pi * hz * T[: len(a)]))) for hz in (300, 500)]) + 1e-6


def _candidate(streams):
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.blind import MockBackend as SeparatorMock
    from hearing.tse.enrolled import EnrolledCandidate

    return EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack(streams))), ToneEmbedder(), f0_veto=False,
                             clean_enroll=False, closed_set=False)


def _profiles():
    # aryan sounds like the 500 Hz tone, joseph like 300 Hz; ECAPA would pick stream 1 (500 Hz) for aryan
    return profile_clip(split_pieces(_tone(500, 0.31), SR), "aryan"), profile_clip(split_pieces(_tone(300, 0.29), SR))


def test_profile_name_registry():
    aryan, unnamed = _profiles()
    assert profile_name(aryan) == "aryan"
    assert profile_name(unnamed) is None
    assert profile_name(np.zeros(0, np.float32)) is None


def test_default_off_keeps_the_ecapa_pick(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARING_PICK", raising=False)
    monkeypatch.setenv("HEARING_SEP_CKPT", str(tmp_path / "x.pt"))  # ckpt alone: pick unchanged
    aryan, joseph = _profiles()
    lo, hi = _tone(300), _tone(500)
    cand = _candidate([lo, hi])
    y = cand.extract(lo + hi, SR, enroll=aryan, other=joseph)
    assert cand.selected_stream == 1 and not cand.fixed_pick
    np.testing.assert_allclose(y, hi)


def test_fixed_pick_needs_both_knobs(monkeypatch):
    monkeypatch.setenv("HEARING_PICK", "fixed")
    monkeypatch.delenv("HEARING_SEP_CKPT", raising=False)
    aryan, joseph = _profiles()
    lo, hi = _tone(300), _tone(500)
    cand = _candidate([lo, hi])
    cand.extract(lo + hi, SR, enroll=aryan, other=joseph)
    assert cand.selected_stream == 1 and not cand.fixed_pick


def test_fixed_pick_plays_the_slot_and_keeps_ecapa_for_confidence(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_PICK", "fixed")
    monkeypatch.setenv("HEARING_SEP_CKPT", str(tmp_path / "x.pt"))
    aryan, joseph = _profiles()
    lo, hi = _tone(300), _tone(500)
    # slot 0 (aryan's) carries the 500 Hz voice: the fixed slot and ECAPA agree, full confidence
    cand = _candidate([hi, lo])
    y = cand.extract(lo + hi, SR, enroll=aryan, other=joseph)
    assert cand.fixed_pick and cand.selected_stream == 0 and not cand.gated
    np.testing.assert_allclose(y, hi)
    assert cand.confidence > 0.5
    # slot 0 now scores on the other side of the contrast: still the slot is played or gated, never the argmax stream
    cand = _candidate([lo, hi])
    y = cand.extract(lo + hi, SR, enroll=aryan, other=joseph)
    assert cand.fixed_pick and cand.selected_stream == 0
    assert cand.gated and cand.confidence == 0.0
    assert not np.any(y)
    # an unnamed enrollment (a live calibration clip) keeps the ECAPA pick
    cand = _candidate([lo, hi])
    cand.extract(lo + hi, SR, enroll=hi.copy(), other=joseph)
    assert not cand.fixed_pick and cand.selected_stream == 1
