"""T-134: pick confidence from the enrolled tier, F0 cue, closed-set LDA, and the mixer's dynamic floor."""

import numpy as np
import pytest

from hearing.fusion.mixer import (
    CONFIDENCE_KEY,
    RESIDUAL_KEY,
    Mixer,
    confidence_floor_db,
)
from hearing.tse.enrolled import f0_frames, fit_lda, median_f0
from hearing.types import AudioChunk, StageContext

SR = 16000


def _voice(f0, s=1.0, seed=0):
    t = np.arange(round(s * SR)) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    return (0.2 * x).astype(np.float32)


def test_f0_tracker_finds_the_pitch_and_rejects_noise():
    assert median_f0(_voice(120.0)) == pytest.approx(120.0, rel=0.05)
    assert median_f0(_voice(220.0)) == pytest.approx(220.0, rel=0.05)
    noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1
    assert median_f0(noise) is None
    assert np.isnan(f0_frames(np.zeros(SR, np.float32))).all()


def test_lda_maps_centroids_to_plus_and_minus_one():
    rng = np.random.default_rng(1)
    mt, mo = rng.standard_normal(16), rng.standard_normal(16)
    t = mt + 0.1 * rng.standard_normal((6, 16))
    o = mo + 0.1 * rng.standard_normal((6, 16))
    w, b = fit_lda(t, o)
    assert float(w @ t.mean(0) - b) == pytest.approx(1.0)
    assert float(w @ o.mean(0) - b) == pytest.approx(-1.0)


def _mix(mixer, conf, s=1.0, n=480):
    ctx = StageContext.default()
    ctx.suppression = s
    ctx.telemetry.extras[RESIDUAL_KEY] = np.ones(n, np.float32)
    if conf is not None:
        ctx.telemetry.extras[CONFIDENCE_KEY] = conf
    out = mixer.process(AudioChunk(np.zeros(n, np.float32), 0, 0.0, SR), ctx).samples
    return out, ctx


@pytest.mark.parametrize("conf,db", [(0.0, 6.0), (0.5, 23.0), (1.0, 40.0)])
def test_confidence_sets_the_residual_floor(conf, db):
    assert confidence_floor_db(conf) == pytest.approx(db)
    m = Mixer(max_duck_db=20)
    out, ctx = _mix(m, conf)
    assert out[-1] == pytest.approx(10 ** (-db / 20), rel=1e-4)
    assert ctx.telemetry.extras["mixer"]["floor_db"] == pytest.approx(db)
    assert CONFIDENCE_KEY not in ctx.telemetry.extras  # popped every chunk


def test_without_confidence_max_duck_db_still_holds():
    out, _ = _mix(Mixer(max_duck_db=20), None)
    assert out[-1] == pytest.approx(0.1, rel=1e-4)


def test_slider_still_wins_below_the_floor_and_moves_are_ramped():
    out, _ = _mix(Mixer(), 0.0, s=0.0)
    assert out[-1] == pytest.approx(1.0)  # slider 0: the room untouched, whatever the confidence
    m = Mixer()
    _mix(m, 1.0)
    out, _ = _mix(m, 0.0)  # sure -> unsure: no jump, a <= 30 ms ramp
    assert np.max(np.abs(np.diff(out))) <= 1.0 / (0.03 * SR) + 1e-6
    assert out[-1] == pytest.approx(10 ** (-6 / 20), rel=1e-4)


def test_enrolled_window_reports_confidence_from_the_score_margin():
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.blind import MockBackend as SeparatorMock
    from hearing.tse.enrolled import EnrolledCandidate

    class SignEmbedder:
        device = "cpu"

        def embed(self, audio):
            return np.array([np.mean(audio > 0) + 1e-3, np.mean(audio < 0) + 1e-3])

    a = np.ones(SR, np.float32)
    clear = EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack([a, -a]))), SignEmbedder(), f0_veto=False)
    clear.extract(a, SR, enroll=a)
    assert clear.confidence == pytest.approx(1.0) and clear.selected_stream == 0
    tie = EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack([a, a]))), SignEmbedder(), f0_veto=False)
    tie.extract(a, SR, enroll=a)
    assert tie.confidence == pytest.approx(0.0)


def test_confidence_env_off_restores_max_duck_db(monkeypatch):
    monkeypatch.setenv("HEARING_TSE_CONFIDENCE", "0")
    out, _ = _mix(Mixer(max_duck_db=20), 1.0)
    assert out[-1] == pytest.approx(0.1, rel=1e-4)


def test_pipeline_eval_profile_path_controls_and_isolated_state():
    from eval.pipeline_eval import _controls, _tick

    _, ctl = _controls(profile=True)
    assert ctl["mode"] == "focus_hq" and ctl["enroll"] == {"action": "profile", "name": "target", "other": "other"}
    tel = {"mode": "focus_hq", "tiers": {"active": "enrolled"}, "tse": {"confidence": 0.8}}
    assert _tick(1.0, tel)["state"] == "isolated"
    assert _tick(1.0, {**tel, "tiers": {"active": "audio-only"}})["state"] == "locking"
    assert _tick(1.0, {"mode": "name_call", "name_call": {"state": "listening"}})["state"] == "listening"


def test_pipeline_eval_realigns_when_the_playout_delay_steps():
    from eval.pipeline_eval import SR as ESR
    from eval.pipeline_eval import _align, _delay_segments

    rng = np.random.default_rng(0)
    ref = rng.standard_normal(6 * ESR).astype(np.float32)
    out = np.zeros(8 * ESR, np.float32)
    out[ESR // 2 : ESR // 2 + 3 * ESR] = ref[: 3 * ESR]  # 0.5 s delay for 3 s ...
    out[3 * ESR + ESR * 3 // 4 : 6 * ESR + ESR * 3 // 4] = ref[3 * ESR :]  # ... then 0.75 s
    ticks = [{"t": t / 10, "delay_ms": 500.0 if t < 30 else 750.0} for t in range(1, 60)]
    segs = _delay_segments(ticks, 0.1)
    assert [d for _, d in segs] == [0.5, 0.75]
    y, _ = _align(out, ref, 0.5, segs)
    assert np.corrcoef(y[3 * ESR + 1600 : 6 * ESR - 1600], ref[3 * ESR + 1600 : 6 * ESR - 1600])[0, 1] > 0.99
    assert np.corrcoef(y[ESR : 3 * ESR - 1600], ref[ESR : 3 * ESR - 1600])[0, 1] > 0.99


def test_profile_swap_from_the_cache_moves_the_closed_set_pieces_too():
    """T-136 cache hits must also restore the profile pieces the LDA and F0 cue read (T-134)."""
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.blind import MockBackend as SeparatorMock
    from hearing.tse.enroll import profile_clip
    from hearing.tse.enrolled import EnrolledCandidate

    class SeedEmbedder:
        device = "cpu"

        def embed(self, audio):
            return np.random.default_rng(int(abs(audio[:50]).sum() * 1e4) % 2**32).standard_normal(8)

    rng = np.random.default_rng(1347)  # not a seed other tests draw room clips from (_PROFILE_CLIPS is global)
    a = profile_clip([(0.1 * rng.standard_normal(SR)).astype(np.float32) for _ in range(3)])
    b = profile_clip([(0.1 * rng.standard_normal(SR)).astype(np.float32) for _ in range(3)])
    x = np.ones(SR, np.float32)
    cand = EnrolledCandidate(BlindSeparator(SeparatorMock(np.stack([x, -x]))), SeedEmbedder(), f0_veto=False)
    cand.prepare(a, b)
    cand.prepare(b, a)
    w_ba = cand._closed_set()[0].copy()
    cand.prepare(a, b)  # cache hit for both slots
    assert cand._pieces["target"][0] != cand._pieces["other"][0]
    assert np.allclose(cand._closed_set()[0], -w_ba)  # roles swapped: the LDA direction flips
