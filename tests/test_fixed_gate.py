"""T-402: the fixed-slot gate (HEARING_FIXED_GATE=1) decides from the slot itself; ECAPA only flags strangers."""
import json

import numpy as np
import pytest

from hearing.tse import fixed_gate
from hearing.tse.enroll import profile_clip, split_pieces

SR = 16000
T = np.arange(SR) / SR
HZ = (300, 500, 700)  # joseph, aryan, a stranger
# energy-only weights: target when the target slot is louder than the partner slot, partner when quieter
ENERGY_W = np.array([[0.5, 0, 0, 0, 0, 0], [-0.5, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -5.0]])


def _tone(hz, amp=0.3):
    return (amp * np.sin(2 * np.pi * hz * T)).astype(np.float32)


class ToneEmbedder:
    device = "cpu"

    def embed(self, audio):
        a = np.asarray(audio, np.float64)
        return np.array([abs(np.dot(a, np.sin(2 * np.pi * hz * T[: len(a)]))) for hz in HZ]) + 1e-9


def _candidate(streams):
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.blind import MockBackend as SeparatorMock
    from hearing.tse.enrolled import EnrolledCandidate

    return EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack(streams))), ToneEmbedder(), f0_veto=False,
                             clean_enroll=False, closed_set=False)


@pytest.fixture
def fixed(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_PICK", "fixed")
    monkeypatch.setenv("HEARING_SEP_CKPT", str(tmp_path / "x.pt"))
    monkeypatch.setattr(fixed_gate, "WEIGHTS", ENERGY_W)
    monkeypatch.delenv(fixed_gate.FIXED_GATE_HEAD_ENV, raising=False)
    joseph = profile_clip(split_pieces(_tone(300, 0.28), SR), "joseph")
    aryan = profile_clip(split_pieces(_tone(500, 0.32), SR), "aryan")
    return joseph, aryan


def test_knob_default_off_keeps_t386_contrast_gate(fixed, monkeypatch):
    monkeypatch.delenv(fixed_gate.FIXED_GATE_ENV, raising=False)
    joseph, aryan = fixed
    # joseph's slot (1) is loud but sounds closer to aryan: T-386's ECAPA contrast gate mutes it
    slot1 = _tone(500, 0.3) + _tone(300, 0.1)
    cand = _candidate([_tone(500, 0.01), slot1])
    y = cand.extract(slot1, SR, enroll=joseph, other=aryan)
    assert cand.fixed_pick and cand.selected_stream == 1 and cand.gated and not np.any(y)
    assert cand.fixed_gate_probs is None


def test_gate_plays_an_active_target_slot_the_contrast_gate_would_mute(fixed, monkeypatch):
    monkeypatch.setenv(fixed_gate.FIXED_GATE_ENV, "1")
    joseph, aryan = fixed
    slot1 = _tone(500, 0.3) + _tone(300, 0.1)
    cand = _candidate([_tone(500, 0.01), slot1])
    y = cand.extract(slot1, SR, enroll=joseph, other=aryan)
    assert cand.fixed_pick and cand.selected_stream == 1 and not cand.gated
    assert np.corrcoef(y, slot1)[0, 1] > 0.99  # the slot plays (the mock separator rescales it)
    assert cand.confidence >= fixed_gate.MIN_CONF
    assert cand.fixed_gate_probs[0] > 0.5


def test_gate_does_not_rescue_when_the_partner_slot_carries_the_speech(fixed, monkeypatch):
    monkeypatch.setenv(fixed_gate.FIXED_GATE_ENV, "1")
    joseph, aryan = fixed
    cand = _candidate([_tone(500, 0.3), _tone(500, 0.003)])  # aryan talks alone; joseph's slot is his leakage
    y = cand.extract(_tone(500, 0.3), SR, enroll=joseph, other=aryan)
    assert cand.selected_stream == 1 and cand.gated and cand.confidence == 0.0 and not np.any(y)


def test_stranger_in_the_target_slot_stays_ducked(fixed, monkeypatch):
    monkeypatch.setenv(fixed_gate.FIXED_GATE_ENV, "1")
    joseph, aryan = fixed
    cand = _candidate([_tone(500, 0.01), _tone(700, 0.3)])  # a third voice lands in joseph's slot
    cand.extract(_tone(700, 0.3), SR, enroll=joseph, other=aryan)
    assert cand.gated and cand.fixed_gate_stranger


def test_gate_never_mutes_a_window_the_contrast_gate_plays(fixed, monkeypatch):
    monkeypatch.setenv(fixed_gate.FIXED_GATE_ENV, "1")
    joseph, aryan = fixed
    cand = _candidate([_tone(500, 0.3), _tone(300, 0.003)])  # quiet slot, but it sounds like joseph: ECAPA plays it
    cand.extract(_tone(500, 0.3), SR, enroll=joseph, other=aryan)
    assert not cand.gated and cand.fixed_gate_probs[0] < 0.5


def test_hysteresis_holds_one_window_then_closes():
    g = fixed_gate.FixedGate(ENERGY_W)
    loud, quiet, mix = _tone(300, 0.3), _tone(300, 0.003), _tone(300, 0.3)
    assert g.step(loud, quiet, mix, 0.9, 0.1)[0]  # target slot louder: open
    assert g.step(quiet, loud, mix, 0.9, 0.1)[0]  # one window below CLOSE_P: held open
    assert not g.step(quiet, loud, mix, 0.9, 0.1)[0]  # second: closed
    assert not g.step(_tone(300, 0.03), _tone(300, 0.03), mix, 0.9, 0.1)[0]  # p = 0.5 - none: stays closed
    g.reset()
    assert not g.open and g.probs is None


def test_features_use_the_window_tail():
    head_loud = np.concatenate([np.full(800, 0.5), np.full(800, 0.01)]).astype(np.float32)
    x = fixed_gate.features(head_loud, head_loud, head_loud, 0.3, 0.1)
    assert x[0] == 0.0 and x[1] == 0.0 and x[-1] == 1.0
    assert x.shape == (len(fixed_gate.FEATURES),)


def test_weights_from_json(tmp_path, monkeypatch):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"weights": ENERGY_W.tolist()}))
    monkeypatch.setenv(fixed_gate.FIXED_GATE_HEAD_ENV, str(p))
    np.testing.assert_allclose(fixed_gate.load_weights(), ENERGY_W)
    p.write_text(json.dumps({"weights": [[0.0, 1.0]]}))
    with pytest.raises(ValueError):
        fixed_gate.load_weights()


def test_shipped_weights_prefer_target_for_a_loud_matching_slot():
    x = fixed_gate.features(_tone(300, 0.3), _tone(300, 0.01), _tone(300, 0.3), 0.6, 0.1)
    assert np.argmax(fixed_gate.softmax_probs(fixed_gate.WEIGHTS, x)) == 0
    x = fixed_gate.features(_tone(300, 0.01), _tone(300, 0.3), _tone(300, 0.3), 0.1, 0.6)
    assert np.argmax(fixed_gate.softmax_probs(fixed_gate.WEIGHTS, x)) != 0


def test_swap_rescue_plays_the_partner_slot_when_the_target_was_routed_there(fixed, monkeypatch):
    monkeypatch.setenv(fixed_gate.FIXED_GATE_ENV, "1")
    joseph, aryan = fixed
    # joseph talks alone but the net put him in aryan's slot (0); joseph's slot (1) holds faint aryan-like residue
    cand = _candidate([_tone(300, 0.3), _tone(500, 0.002)])
    y = cand.extract(_tone(300, 0.3), SR, enroll=joseph, other=aryan)
    assert cand.fixed_gate_swapped and cand.selected_stream == 0 and not cand.gated
    assert np.corrcoef(y, _tone(300))[0, 1] > 0.99 and cand.confidence == fixed_gate.MIN_CONF


def test_swap_rescue_never_plays_the_partner_talking(fixed, monkeypatch):
    monkeypatch.setenv(fixed_gate.FIXED_GATE_ENV, "1")
    joseph, aryan = fixed
    cand = _candidate([_tone(500, 0.3), _tone(500, 0.002)])  # aryan alone in his own slot: no swap
    y = cand.extract(_tone(500, 0.3), SR, enroll=joseph, other=aryan)
    assert not cand.fixed_gate_swapped and cand.gated and not np.any(y)


def test_no_rescue_while_the_partner_slot_outscores_the_target_slot(fixed, monkeypatch):
    monkeypatch.setenv(fixed_gate.FIXED_GATE_ENV, "1")
    joseph, aryan = fixed
    slot1 = _tone(500, 0.3) + _tone(300, 0.1)  # the energy gate would play joseph's slot ...
    cand = _candidate([_tone(300, 0.05), slot1])  # ... but aryan's slot sounds more like joseph than joseph's does
    y = cand.extract(slot1 + _tone(300, 0.05), SR, enroll=joseph, other=aryan)
    assert cand.fixed_gate_probs[0] > 0.5 and cand.gated and not np.any(y)
