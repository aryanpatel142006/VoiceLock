"""T-088: the judge profile (control.profile = "judge") and automatic calibration."""

import json

import numpy as np
import pytest

from hearing.audio.io import float_to_pcm16
from hearing.server.judge import JUDGE_FIELDS, JUDGE_NATURAL_SUPPRESSION, AutoCalibrator
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    ProtocolError,
    pack_frame,
    parse_message,
)
from hearing.server.session import ServerConfig, Session, _nested, load_stage_modules
from hearing.tse.enroll import AutoEnroller
from hearing.types import FaceTrack, StageContext, TargetState
from hearing.vision.target import ATTENTION_PRESETS

SR = 16000
HOP = 160


def _ctx(target: int | None, p_target: float, p_other: float, locked: bool = True) -> StageContext:
    ctx = StageContext.default()
    ctx.tracks = [FaceTrack(1, (0.1, 0.3, 0.2, 0.4), None, p_target, 0.0),
                  FaceTrack(2, (0.6, 0.3, 0.2, 0.4), None, p_other, 0.0)]
    ctx.target = TargetState(target, 1.0 if target is not None else 0.0, locked, 0.0, 0)
    return ctx


def _speech(n: int, seed: int = 0) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def _run(cal: AutoCalibrator, e: AutoEnroller, ctx: StageContext, seconds: float, x=None) -> float | None:
    """Feed ``seconds`` of hops; returns when (s) a calibration was started, else None."""
    started = None
    x = _speech(round(seconds * SR)) if x is None else x
    for i in range(0, len(x), HOP):
        e.update(x[i : i + HOP], ctx)
        if cal.update(HOP / SR, ctx, e) and started is None:
            started = (i + HOP) / SR
    return started


def test_starts_calibration_after_1_s_of_the_target_speaking_alone():
    cal, e = AutoCalibrator(enabled=True), AutoEnroller(2.0, SR)
    ctx = _ctx(1, p_target=0.9, p_other=0.1)
    started = _run(cal, e, ctx, 1.5)
    assert started is not None and 0.95 <= started <= 1.05
    assert e.calibrating and cal.status()["state"] == "calibrating"
    assert e.calibration_progress[1] == pytest.approx(2.0)  # T-088: 2 s automatic window (manual: 3 s)
    _run(cal, e, ctx, 3.0)  # the window closes on voiced audio: calibrated, never restarted
    assert e.calibrated and cal.status()["state"] == "done" and cal.attempts == 1
    assert _run(cal, e, ctx, 3.0) is None


@pytest.mark.parametrize(("target", "p_target", "p_other", "locked"), [
    (1, 0.9, 0.9, True),   # both talking: not alone
    (1, 0.1, 0.1, True),   # nobody talking
    (None, 0.9, 0.1, False),  # no target
    (None, 0.9, 0.1, True),  # a candidate face, not selected yet
])
def test_waits_unless_the_selected_target_speaks_alone(target, p_target, p_other, locked):
    cal, e = AutoCalibrator(enabled=True), AutoEnroller(2.0, SR)
    assert _run(cal, e, _ctx(target, p_target, p_other, locked), 3.0) is None
    assert not e.calibrating


def test_short_asd_gaps_do_not_reset_the_solo_clock_but_long_ones_do():
    cal, e = AutoCalibrator(enabled=True), AutoEnroller(2.0, SR)
    talk, gap = _ctx(1, 0.9, 0.1), _ctx(1, 0.2, 0.1)
    assert _run(cal, e, talk, 0.6) is None
    assert _run(cal, e, gap, 0.2) is None  # a pause between words
    assert _run(cal, e, talk, 0.5) is not None
    cal, e = AutoCalibrator(enabled=True), AutoEnroller(2.0, SR)
    _run(cal, e, talk, 0.6)
    _run(cal, e, gap, 1.0)  # stopped talking
    assert _run(cal, e, talk, 0.6) is None


def test_off_by_default_and_a_failed_window_rearms():
    e = AutoEnroller(2.0, SR)
    assert _run(AutoCalibrator(), e, _ctx(1, 0.9, 0.1), 2.0) is None
    cal = AutoCalibrator(enabled=True)
    ctx = _ctx(1, 0.9, 0.1)
    _run(cal, e, ctx, 1.0)
    assert e.calibrating
    _run(cal, e, ctx, 2.5, x=np.zeros(5 * SR // 2, np.float32))  # silence: too little speech
    assert e.calibration_failed and cal.status()["state"] == "waiting"
    assert _run(cal, e, ctx, 1.2) is not None and cal.attempts == 2


def test_manual_calibration_counts_and_is_not_overridden():
    cal, e = AutoCalibrator(enabled=True), AutoEnroller(2.0, SR)
    e.calibrate(3.0)
    assert _run(cal, e, _ctx(1, 0.9, 0.1), 3.5) is None
    assert e.calibrated and cal.status()["state"] == "done"


# ------------------------------------------------------------------------------- session / wire


@pytest.fixture
def session(monkeypatch):
    for k, v in {"HEARING_DEVICE": "cpu", "HEARING_ASR": "mock", "HEARING_TTS": "mock",
                 "HEARING_VOICE_MATCH": "mock"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)
    load_stage_modules()
    s = Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": "passthrough"},
                             synth_feed="mixture", faces="mock", vision_threaded=False), "cpu")
    yield s
    s.close()


def test_profile_validates():
    assert parse_message({"type": "control", "profile": "judge"}).profile == "judge"
    with pytest.raises(ProtocolError):
        parse_message({"type": "control", "profile": "keynote"})


def test_judge_profile_locks_every_default_in_one_control(session):
    s = session
    t = s.telemetry()
    assert t["profile"] is None and t["auto_calibrate"]["state"] == "off"
    replies = s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    errors = [m for m in replies if m.get("type") == "error" and m.get("stage") != "hq_backend"]
    assert errors == []
    t = replies[-1]
    assert t["profile"] == "judge"
    assert t["mode"] == "focus_hq" and s.ctx.suppression == 1.0
    assert t["isolation"] == "strict" and t["latency_preset"] == "balanced"
    assert t["synth_feed_request"] == "auto" and t["hq_backend"] == "auto"
    assert t["attention"] == ATTENTION_PRESETS["sticky"]
    assert t["auto_calibrate"]["state"] == "waiting"
    assert set(JUDGE_FIELDS) >= {"mode", "suppression", "isolation", "synth_feed", "hq_backend"}


def test_judge_natural_defaults_to_slider_80(session):
    """T-103: under the judge profile Natural (camera lost) plays at s=80 unless the slider is sent."""
    s = session
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq", "suppression": 30}))
    s.handle_text(json.dumps({"type": "control", "mode": "natural"}))
    assert s.ctx.suppression == 0.3  # no profile: the slider is untouched
    s.handle_text(json.dumps({"type": "control", "profile": "judge", "mode": "natural"}))  # judge.js reconnect
    assert s.mode == "natural" and s.ctx.suppression == JUDGE_NATURAL_SUPPRESSION / 100 == 0.8
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    assert s.mode == "focus_hq" and s.ctx.suppression == 1.0
    s.handle_text(json.dumps({"type": "control", "mode": "natural"}))  # judge.js setCameraLost
    assert s.ctx.suppression == 0.8
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    s.handle_text(json.dumps({"type": "control", "mode": "natural", "suppression": 30}))
    assert s.ctx.suppression == 0.3


def test_judge_session_calibrates_on_its_own(session):
    s = session
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    s.ctx.tracks = [FaceTrack(1, (0.1, 0.3, 0.2, 0.4), None, 0.9, 0.0),
                    FaceTrack(2, (0.6, 0.3, 0.2, 0.4), None, 0.05, 0.0)]
    s.ctx.target = TargetState(1, 1.0, False, 0.0, 0)  # selected by attention, not tapped
    s._poll_vision = lambda timeout=0.0: None  # keep the hand-set tracks (no camera in this test)
    x = _speech(round(1.3 * SR))
    for i in range(len(x) // 320):
        s.handle_binary(pack_frame(KIND_AUDIO_IN, i, i * 0.02, float_to_pcm16(x[i * 320 : (i + 1) * 320])))
    t = s.telemetry()
    assert s.enroller.calibrating and t["auto_calibrate"]["state"] == "calibrating"
    assert t["acquisition"]["detail"].startswith("calibrating") or t["enrollment"]["state"] == "calibrating"


def test_release_turns_auto_calibration_off_until_a_saved_person(session, monkeypatch):
    """T-177: /judge Release (enroll clear) must not let the auto-calibrator re-enroll whoever talks
    next; picking saved person 1 or 2 (enroll profile) turns it back on."""
    s = session
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    assert s.telemetry()["auto_calibrate"]["state"] == "waiting"
    s.handle_text(json.dumps({"type": "control", "enroll": {"action": "clear"}}))
    assert s.telemetry()["auto_calibrate"]["state"] == "off"
    s.ctx.tracks = [FaceTrack(1, (0.1, 0.3, 0.2, 0.4), None, 0.9, 0.0)]
    s.ctx.target = TargetState(1, 1.0, False, 0.0, 0)
    s._poll_vision = lambda timeout=0.0: None
    x = _speech(round(1.3 * SR))
    for i in range(len(x) // 320):  # the target speaks alone for 1.3 s: no calibration window opens
        s.handle_binary(pack_frame(KIND_AUDIO_IN, i, i * 0.02, float_to_pcm16(x[i * 320 : (i + 1) * 320])))
    assert not s.enroller.calibrating and s.telemetry()["auto_calibrate"]["state"] == "off"
    calls = []
    monkeypatch.setattr(s, "load_voice_profile", lambda name, other=None: calls.append(name) or [])
    s.handle_text(json.dumps({"type": "control", "enroll": {"action": "profile", "name": "joseph"}}))
    assert calls == ["joseph"] and s.telemetry()["auto_calibrate"]["state"] != "off"
    s.handle_text(json.dumps({"type": "control", "enroll": {"action": "clear"}}))
    assert s.telemetry()["auto_calibrate"]["state"] == "off"
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))  # e.g. a reconnect re-sends the profile
    assert s.telemetry()["auto_calibrate"]["state"] != "off"  # set_profile turns it back on


def test_judge_ladder_moves_up_after_half_a_second(session):
    """T-088: the judge profile's ladder upgrades after JUDGE_UPGRADE_AFTER_S (0.5 s, the enrolled hop)."""
    from hearing.server.judge import JUDGE_UPGRADE_AFTER_S, judge_spec

    s = session
    s.config.pipelines["focus_hq"] = "tiers"
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    assert "upgrade_after_s=0.5" in s.active_spec and 'isolation="strict"' in s.active_spec
    assert JUDGE_UPGRADE_AFTER_S == 0.5
    assert judge_spec("tiers(upgrade_after_s=2, min_switch_s=10)") == (
        "tiers(upgrade_after_s=2, min_switch_s=10)")  # explicit wins
    assert judge_spec("dfn_ll>gate") == "dfn_ll>gate"


def test_judge_ladder_promotes_to_enrolled_soon_after_the_gate(session):
    """T-135: config/tiers.yaml holds every promotion 10 s after the last; under the judge the
    audio-only -> gate step at ~1 s kept the calibrated Enrolled tier out until ~11 s."""
    from hearing.fusion.tiers import TierPolicy
    from hearing.server.judge import JUDGE_MIN_SWITCH_S, judge_spec

    s = session
    s.config.pipelines["focus_hq"] = "tiers"
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    assert f"min_switch_s={JUDGE_MIN_SWITCH_S}" in s.active_spec
    assert judge_spec("tiers") == f"tiers(upgrade_after_s=0.5, min_switch_s={JUDGE_MIN_SWITCH_S})"
    pol = TierPolicy(("enrolled", "gate", "audio-only"), upgrade_after_s=0.5, min_switch_s=JUDGE_MIN_SWITCH_S)
    assert pol.choose([False, False, True], 0.0) == 2
    assert pol.choose([False, True, True], 0.4) == 2
    assert pol.choose([False, True, True], 0.9) == 1  # gate at 0.9 s
    pol.choose([True, True, True], 4.8)  # calibrated: enrolled eligible
    assert pol.choose([True, True, True], 5.3) == 0  # lock -> Isolated inside the 6 s budget


def test_judge_pins_the_tse_hop_once_isolated(session):
    """T-135 (Q-341): the hop adapts until the first isolated tick (a loaded machine still locks), then
    every tse stage keeps its hop (each later change was an audible dip)."""
    from hearing.server.judge import judge_spec

    s = session
    s.config.pipelines["focus_hq"] = 'tse(candidate="identity", window_s=1.0, hop_s=0.5)'
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    assert "adapt_hop" not in s.active_spec
    assert judge_spec('tse(candidate="enrolled", window_s=1.0)') == 'tse(candidate="enrolled", window_s=1.0)'
    pipe = s._pipeline([])
    tse = next(st for st in _nested(pipe) if hasattr(st, "pin_hop"))
    s._pin_hops(pipe, {"state": "locked-visual"})
    assert not tse.hop_pinned and len(tse.hops) == 3
    s._pin_hops(pipe, {"state": "locked"})  # isolated
    assert tse.hop_pinned and tse.health()["hop_pinned"]
    s._pin_hops(pipe, {"state": "searching"})  # pinned for good
    assert tse.hop_pinned


def test_judge_ladder_can_pin_the_hop_from_start_and_levels_the_output(session):
    """T-135: adapt_hop=false (HEARING_JUDGE_ADAPT_HOP=0, the A/B) gives one rung; the output AGC."""
    from hearing.fusion.tiers import TierLadder

    s = session
    s.config.pipelines["focus_hq"] = "tiers"
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    ladder = TierLadder({"tiers": [{"name": "a", "pipeline": 'tse(candidate="identity", window_s=1.0)'},
                                   {"name": "b", "pipeline": "passthrough"}]}, adapt_hop=False)
    tse = ladder.built["a"].pipeline.stages[0]
    assert tse.hops == (tse.hop,)  # one rung: the hop never changes
    # the output is pulled to -20 dBFS: a quiet voice comes out ~12 dB louder, never clipped
    t = np.arange(3 * SR) / SR  # a voice: a 4 Hz syllable envelope (a steady tone is room noise to the AGC, T-169)
    quiet = (0.014 * np.sin(2 * np.pi * 220 * t) * np.clip(np.sin(2 * np.pi * 2 * t), 0, None) ** 0.5).astype(np.float32)
    s.config.pipelines["focus_hq"] = "passthrough"
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    out = [s.handle_binary(pack_frame(KIND_AUDIO_IN, i, 0.0, float_to_pcm16(quiet[i * 320:(i + 1) * 320])))
           for i in range(len(quiet) // 320)]
    from hearing.audio.io import pcm16_to_float
    from hearing.server.protocol import unpack_frame
    y = np.concatenate([pcm16_to_float(unpack_frame(m).payload) for o in out for m in o if isinstance(m, bytes)])
    gain_db = 20 * np.log10(np.sqrt(np.mean(y[-SR:] ** 2)) / np.sqrt(np.mean(quiet[-SR:] ** 2)))
    assert 10.0 < gain_db < 13.0
    assert s.telemetry()["agc"]["gain_db"] > 10.0


def test_a_remote_stage_session_leaves_leveling_to_the_laptop(session):
    """T-129: the GPU session behind the laptop's remote stage gets profile=judge too; only the laptop's
    session runs the AGC, so the output is leveled once, not twice in series."""
    s = session
    s.handle_text(json.dumps({"type": "hello", "sample_rate": 16000, "frame_ms": 20, "video": False,
                              "client": "remote-stage"}))
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    assert s.profile == "judge"
    assert s.telemetry()["agc"] is None
    s.handle_text(json.dumps({"type": "hello", "sample_rate": 16000, "frame_ms": 20, "video": False,
                              "client": "judge"}))
    assert s.telemetry()["agc"] is not None


def _play(s: Session, x: np.ndarray) -> np.ndarray:
    from hearing.audio.io import pcm16_to_float
    from hearing.server.protocol import unpack_frame
    out = [s.handle_binary(pack_frame(KIND_AUDIO_IN, i, 0.0, float_to_pcm16(x[i * 320:(i + 1) * 320])))
           for i in range(len(x) // 320)]
    return np.concatenate([pcm16_to_float(unpack_frame(m).payload) for o in out for m in o if isinstance(m, bytes)])


def test_control_agc_false_turns_the_judge_output_agc_off(session):
    """T-169: the remote leg of the judge path runs without AGC (``control.agc: false``, sent by the
    remote stage); the laptop levels the output once. A profile re-sent later keeps it off."""
    s = session
    s.config.pipelines["focus_hq"] = "passthrough"
    t = np.arange(3 * SR) / SR  # a voice: a 4 Hz syllable envelope (a steady tone is room noise to the AGC, T-169)
    quiet = (0.014 * np.sin(2 * np.pi * 220 * t) * np.clip(np.sin(2 * np.pi * 2 * t), 0, None) ** 0.5).astype(np.float32)
    s.handle_text(json.dumps({"type": "control", "profile": "judge", "agc": False}))
    s.handle_text(json.dumps({"type": "control", "profile": "judge", "mode": "focus_hq"}))
    y = _play(s, quiet)
    gain_db = 20 * np.log10(np.sqrt(np.mean(y[-SR:] ** 2)) / np.sqrt(np.mean(quiet[-SR:] ** 2)))
    assert abs(gain_db) < 0.2
    assert s.telemetry()["agc"] is None
    s.handle_text(json.dumps({"type": "control", "agc": True}))
    y = _play(s, quiet)
    assert s.telemetry()["agc"]["gain_db"] > 10.0
