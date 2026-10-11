"""T-078: latency presets and mode switches keep the session's enrollment; switches dip across delays."""

import json

import numpy as np
import pytest

from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.fusion.tiers import TierLadder
from hearing.server import remote
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    pack_frame,
    unpack_frame,
)
from hearing.server.remote import RemoteStage
from hearing.server.session import ServerConfig, Session, load_stage_modules
from hearing.tse import stage as tse_stage
from hearing.types import FaceTrack, StageContext, TargetState

SR = 16000
LEVEL = 0.25
ENROLLED_LEVEL = 0.5 * LEVEL  # the mock enrolled candidate halves its input; audio-only passes it

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
    """Stands in for the enrolled candidate: needs an enrollment clip, returns half the mix."""

    name = "t078_enrolled"
    licence = "test"
    needs = frozenset({"enroll"})
    device = "cpu"

    def load(self):
        pass

    def extract(self, mix, sr, **kwargs):
        return 0.5 * np.asarray(mix, np.float32)


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_ASR", "mock")
    monkeypatch.setenv("HEARING_TTS", "mock")
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)
    real = tse_stage.get_candidate
    monkeypatch.setattr(tse_stage, "get_candidate",
                        lambda name, **kw: _HalfEnrolled() if name == "enrolled" else real(name, **kw))
    load_stage_modules()
    path = tmp_path / "tiers.yaml"
    path.write_text(LADDER)
    s = Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": f'tiers(config="{path}")'},
                             synth_feed="mixture", faces="mock", vision_threaded=False), "cpu")
    yield s
    s.close()


def _enroll(s: Session) -> None:
    """2 s of the target talking alone, as ASD would have seen it."""
    ctx = StageContext.default()
    ctx.tracks = [FaceTrack(1, (0.4, 0.3, 0.2, 0.4), None, 1.0, 0.0)]
    ctx.target = TargetState(1, 1.0, False, 0.0)
    s.enroller.update(np.full(round(2.0 * SR), 0.1, np.float32), ctx)
    assert s.enroller.ready


def _feed(s: Session, seconds: float, start: int, x=None) -> np.ndarray:
    """Constant (or ``x``) 20 ms frames from frame ``start``; the concatenated audio out."""
    n = round(seconds * 50)
    x = np.full(n * 320, LEVEL, np.float32) if x is None else x
    ys = []
    for i in range(n):
        frame = float_to_pcm16(x[i * 320 : (i + 1) * 320])
        for m in s.handle_binary(pack_frame(KIND_AUDIO_IN, start + i, (start + i) * 0.02, frame)):
            if isinstance(m, bytes) and m[0] == KIND_AUDIO_OUT:
                ys.append(pcm16_to_float(unpack_frame(m).payload))
    return np.concatenate(ys)


def _ladder(s: Session) -> TierLadder:
    return next(st for st in s._built[s._active_spec].stages if isinstance(st, TierLadder))


def _enrolled_tse(ladder: TierLadder):
    return ladder.built["enrolled"].pipeline.stages[0]


def _control(s: Session, **fields) -> None:
    s.handle_text(json.dumps({"type": "control", **fields}))


def test_preset_change_keeps_enrollment_and_never_drops_to_the_room(session):
    s = session
    _enroll(s)
    _control(s, mode="focus_hq")
    _feed(s, 1.5, 0)
    assert s.telemetry()["tier"] == "enrolled"
    old = _ladder(s)
    _control(s, latency_preset="accuracy")  # a new tiers(preset=...) ladder, 1.5 s windows
    y = _feed(s, 6.0, 75)
    new = _ladder(s)
    assert new is not old and "preset=" in s._active_spec
    tse = _enrolled_tse(new)
    assert tse.window_s == pytest.approx(1.5)
    assert tse.enroller is s.enroller and tse.health()["enrolled_s"] == pytest.approx(2.0)
    assert new.active == "enrolled" and new.status()["tiers"][0]["eligible"]
    assert s.telemetry()["transition"] is None
    # The old ladder kept playing the target until the new one isolated it too: the room (the
    # audio-only tier, full level) never played; the 0.4 s -> 1.5 s jump dipped instead.
    assert y.max() <= ENROLLED_LEVEL + 2e-3
    np.testing.assert_allclose(y[-SR:], ENROLLED_LEVEL, atol=2e-3)


def test_focus_hq_text_focus_hq_round_trip_keeps_enrollment(session):
    s = session
    _enroll(s)
    _control(s, mode="focus_hq")
    _feed(s, 1.5, 0)
    ladder = _ladder(s)
    assert ladder.active == "enrolled"
    _control(s, mode="text")
    _feed(s, 0.5, 75)
    assert "synth" in s._active_spec
    _control(s, mode="focus_hq")
    _feed(s, 0.02, 100)
    assert _ladder(s) is ladder and s.enroller.ready  # the ladder was reset, the clip was not
    assert _enrolled_tse(ladder).health()["enrolled_s"] == pytest.approx(2.0)
    y = _feed(s, 1.5, 101)
    assert ladder.active == "enrolled" and s.telemetry()["tier"] == "enrolled"
    np.testing.assert_allclose(y[-SR // 2 :], ENROLLED_LEVEL, atol=2e-3)


def test_switch_across_a_delay_jump_never_overlaps_two_moments(tmp_path, monkeypatch):
    """passthrough (0 ms) -> delay 1 s: the fade must dip, not mix now with one second ago.

    Tone A plays until 0.5 s after the switch, tone B after it. The switch fades once the new
    pipeline has filled its 1 s delay, when the old pipeline plays B and the new one plays A.
    Run once with only A and once with only B (the session is linear): no 10 ms block may carry
    both, i.e. at no instant does the listener hear the two moments on top of each other."""
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    load_stage_modules()
    t = np.arange(round(4.0 * SR)) / SR
    a = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    b = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    switch_s, change_s = 1.0, 1.5
    change = round(change_s * SR)
    only_a = np.concatenate([a[:change], np.zeros(len(t) - change, np.float32)])
    only_b = np.concatenate([np.zeros(change, np.float32), b[change:]])

    def run(x):
        s = Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": "delay(ms=1000)"},
                                 vision_threaded=False), "cpu")
        k = round(switch_s * 50)
        y = [_feed(s, switch_s, 0, x[: k * 320])]
        _control(s, mode="focus_hq")
        y.append(_feed(s, 4.0 - switch_s, k, x[k * 320 :]))
        kinds = s.telemetry()["transition"]
        s.close()
        return np.concatenate(y), kinds

    ya, kind = run(only_a)
    yb, _ = run(only_b)
    assert kind is None  # the switch finished
    def rms(y):  # per 10 ms block
        blocks = len(y) // 160
        return np.sqrt(np.mean(y[: blocks * 160].reshape(blocks, 160) ** 2, axis=1))

    both = np.minimum(rms(ya), rms(yb))
    assert both.max() < 1e-3, f"blocks with two moments: {np.flatnonzero(both >= 1e-3)}"
    # The fade did happen (A from the new pipeline after it, B from the old before it).
    fade = round((switch_s + 1.0) * SR)
    assert rms(yb[fade - 1600 : fade - 160]).min() > 0.1 and rms(ya[fade + 1600 : fade + 3200]).min() > 0.1


def test_remote_latency_includes_the_remote_pipelines_own_delay(monkeypatch):
    monkeypatch.delenv(remote.ENV_URL, raising=False)
    stage = RemoteStage(url="ws://127.0.0.1:9/ws/session", fallback="passthrough", delay_ms=200)
    try:
        assert stage.latency_ms == pytest.approx(200.0)  # nothing reported yet
        stage.client.remote_telemetry = {"latency_ms": {"algorithmic": 750.0}}  # GPU enrolled 0.5/0.25
        assert stage.remote_algorithmic_ms == pytest.approx(750.0)
        assert stage.latency_ms == pytest.approx(950.0)
        assert stage.stats()["remote_algorithmic_ms"] == pytest.approx(750.0)
        stage.client.remote_telemetry = {"latency_ms": {"algorithmic": None}}
        assert stage.latency_ms == pytest.approx(200.0)
    finally:
        stage.close()
