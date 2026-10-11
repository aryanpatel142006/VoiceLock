"""T-085: manual voice calibration (control.enroll) and the strict isolation preset (control.isolation)."""

import json

import numpy as np
import pytest

from hearing.audio.io import float_to_pcm16
from hearing.fusion.tiers import TierLadder
from hearing.latency_preset import STRICT_KWARGS, apply_isolation
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    ProtocolError,
    pack_frame,
    parse_message,
)
from hearing.server.session import ServerConfig, Session, load_stage_modules
from hearing.tse import stage as tse_stage
from hearing.tse.enroll import AutoEnroller
from hearing.types import FaceTrack, StageContext, TargetState

SR = 16000

LADDER = """
crossfade_ms: 50
upgrade_after_s: 0.2
tiers:
  - name: enrolled
    pipeline: 'tse(candidate="enrolled", window_s=0.4, hop_s=0.2, threaded=false, residual=false)'
    needs: [healthy]
  - name: audio-only
    pipeline: passthrough
"""


class _HalfEnrolled:
    name = "t085_enrolled"
    licence = "test"
    needs = frozenset({"enroll"})
    device = "cpu"

    def load(self):
        pass

    def extract(self, mix, sr, **kwargs):
        return 0.5 * np.asarray(mix, np.float32)


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A session on a two-tier ladder; ``kwargs`` records what each enrolled candidate was built with."""
    for k, v in {"HEARING_DEVICE": "cpu", "HEARING_ASR": "mock", "HEARING_TTS": "mock",
                 "HEARING_VOICE_MATCH": "mock"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)
    kwargs: list[dict] = []
    real = tse_stage.get_candidate

    def fake(name, **kw):
        if name != "enrolled":
            return real(name, **kw)
        kwargs.append(kw)
        return _HalfEnrolled()

    monkeypatch.setattr(tse_stage, "get_candidate", fake)
    load_stage_modules()
    path = tmp_path / "tiers.yaml"
    path.write_text(LADDER)
    s = Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": f'tiers(config="{path}")'},
                             synth_feed="mixture", faces="mock", vision_threaded=False), "cpu")
    yield s, kwargs
    s.close()


def _control(s: Session, **fields) -> list:
    """The control's error events (a control is answered with a telemetry snapshot too)."""
    return [m for m in s.handle_text(json.dumps({"type": "control", **fields})) if m.get("type") == "error"]


def _speech(n: int, seed: int = 0) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def _ctx(target: int | None, switch_seq: int = 0, p_target: float = 0.0, p_other: float = 0.9,
         ids: tuple[int, int] = (1, 2)) -> StageContext:
    ctx = StageContext.default()
    ctx.tracks = [FaceTrack(ids[0], (0.1, 0.3, 0.2, 0.4), None, p_target, 0.0),
                  FaceTrack(ids[1], (0.6, 0.3, 0.2, 0.4), None, p_other, 0.0)]
    ctx.target = TargetState(target, 1.0 if target is not None else 0.0, target is not None, 0.0, switch_seq)
    return ctx


# ------------------------------------------------------------------------------- AutoEnroller


def test_calibration_keeps_every_voiced_chunk_whatever_asd_says():
    e = AutoEnroller(2.0, SR)
    ctx = _ctx(1, p_target=0.0, p_other=0.95)  # ASD would never have let a chunk through
    e.calibrate(3.0)
    x = _speech(SR * 3)
    x[SR : SR + SR // 2] = 0  # 0.5 s of silence is not kept
    for i in range(0, len(x), 160):
        assert not e.ready
        e.update(x[i : i + 160], ctx)
    assert e.ready and e.calibrated and not e.calibrating
    assert len(e.clip()) == round(2.5 * SR)  # the whole voiced part, not cut to the 2 s auto length
    assert e.track_id == 1


def test_calibration_survives_redetection_and_drops_on_deliberate_switch():
    e = AutoEnroller(2.0, SR)
    e.calibrate(1.5)
    e.update(_speech(round(1.5 * SR)), _ctx(None))  # no face yet: binds to the first target
    assert e.calibrated and e.track_id is None
    e.update(_speech(160), _ctx(1))
    assert e.track_id == 1 and e.calibrated
    e.update(_speech(160), _ctx(7, ids=(7, 2)))  # the same face re-detected under a new id, no switch
    assert e.track_id == 7 and e.calibrated and e.ready
    e.update(_speech(160), _ctx(2, switch_seq=1))  # the wearer turned to someone else
    assert not e.calibrated and not e.ready and e.track_id == 2
    assert e.other_clip() is not None  # the calibrated voice is now the negative


def test_calibration_drops_when_the_target_moves_to_a_face_still_in_view():
    # No switch_seq bump (e.g. an oracle lock in eval), but the calibrated face is still visible:
    # that is another person, not a re-detection, so the clip must not follow the new target.
    e = AutoEnroller(2.0, SR)
    e.calibrate(1.5)
    e.update(_speech(round(1.5 * SR)), _ctx(1))
    assert e.calibrated and e.track_id == 1
    e.update(_speech(160), _ctx(2))
    assert not e.calibrated and e.track_id == 2


def test_calibration_with_too_little_speech_falls_back_and_clear_forgets():
    e = AutoEnroller(2.0, SR)
    ctx = _ctx(1, p_target=0.9, p_other=0.0)
    e.update(_speech(2 * SR), ctx)
    assert e.ready and not e.calibrated
    old = e.clip()
    e.calibrate(3.0)
    assert not e.ready and e.holding and np.array_equal(e.clip(), old)  # the old clip plays meanwhile
    e.update(np.zeros(3 * SR, np.float32), ctx)
    assert e.calibration_failed and not e.calibrated and np.array_equal(e.clip(), old)
    e.clear()
    assert e.clip() is None and e.other_clip() is None and not e.calibration_failed


# ------------------------------------------------------------------------------- session / wire


def test_control_enroll_and_isolation_validate():
    msg = parse_message({"type": "control", "enroll": {"action": "start"}, "isolation": "strict"})
    assert msg.enroll.action == "start" and msg.enroll.seconds == 3.0 and msg.isolation == "strict"
    for bad in ({"enroll": {"action": "go"}}, {"enroll": {"action": "start", "seconds": 0.2}},
                {"isolation": "paranoid"}):
        with pytest.raises(ProtocolError):
            parse_message({"type": "control", **bad})


def test_calibrate_button_enrolls_within_3_5_s_of_the_press(built):
    s, _ = built
    _control(s, mode="focus_hq")
    x = _speech(round(3.5 * SR))
    assert s.telemetry()["enrollment"]["state"] == "collecting"
    assert _control(s, enroll={"action": "start", "seconds": 3.0}) == []
    ready_at = None
    for i in range(len(x) // 320):  # 20 ms frames, as the console sends
        s.handle_binary(pack_frame(KIND_AUDIO_IN, i, i * 0.02, float_to_pcm16(x[i * 320 : (i + 1) * 320])))
        if i == 60:
            t = s.telemetry()
            assert t["enrollment"]["state"] == "calibrating"
            assert t["acquisition"]["detail"].startswith("calibrating 1.")
        if ready_at is None and s.enroller.ready:
            ready_at = (i + 1) * 0.02
    assert ready_at is not None and ready_at <= 3.5
    t = s.telemetry()
    assert t["enrollment"]["state"] == "calibrated" and t["enrollment"]["collected_s"] >= 3.0 - 0.02
    _control(s, enroll={"action": "clear"})
    assert s.telemetry()["enrollment"]["state"] == "collecting" and not s.enroller.ready


def test_strict_isolation_switches_the_ladder_and_keeps_the_calibration(built):
    s, kwargs = built
    _control(s, mode="focus_hq")
    s.handle_binary(pack_frame(KIND_AUDIO_IN, 0, 0.0, float_to_pcm16(np.zeros(320, np.float32))))
    assert s.telemetry()["isolation"] == "standard" and "isolation=" not in s._active_spec
    assert not any("pause_action" in kw for kw in kwargs)
    s.enroller.calibrate(1.0)
    s.enroller.update(_speech(SR), _ctx(None))
    assert _control(s, isolation="strict") == []
    s.handle_binary(pack_frame(KIND_AUDIO_IN, 1, 0.02, float_to_pcm16(np.zeros(320, np.float32))))
    assert 'isolation="strict"' in s._active_spec and s.telemetry()["isolation"] == "strict"
    assert any(all(kw.get(k) == v for k, v in STRICT_KWARGS.items()) for kw in kwargs)
    ladder = next(st for st in s._built[s._active_spec].stages if isinstance(st, TierLadder))
    assert ladder.built["enrolled"].pipeline.stages[0].enroller is s.enroller  # shared: no re-enrollment
    assert s.enroller.calibrated
    _control(s, isolation="standard")
    assert "isolation=" not in s.active_spec


def test_apply_isolation_rewrites_enrolled_tse_and_ladders_only():
    assert apply_isolation("dfn_ll>gate>mixer", "strict") == "dfn_ll>gate>mixer"
    assert apply_isolation("tiers", "standard") == "tiers"
    assert apply_isolation("tiers", "strict") == 'tiers(isolation="strict")'
    spec = apply_isolation('tse(candidate="enrolled", window_s=1.0)>mixer', "strict")
    assert 'pause_action="duck"' in spec and "pause_floor=0.25" in spec and "pause_p=0.5" in spec
    assert apply_isolation('tse(candidate="av_mossformer2")', "strict") == 'tse(candidate="av_mossformer2")'
    with pytest.raises(ValueError):
        apply_isolation("tiers", "loose")
