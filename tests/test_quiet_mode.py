"""T-149: Quiet = digital silence + captions only from the selected person, on Focus HQ's own pipeline."""

import json

import numpy as np
import pytest

from hearing import registry
from hearing.audio.io import HOP, PIPELINE_SR, float_to_pcm16, pcm16_to_float
from hearing.server.protocol import KIND_AUDIO_IN, pack_frame, unpack_frame
from hearing.server.session import QUIET_FADE_MS, ServerConfig, Session
from hearing.types import AudioChunk, StageContext

TIER = {"name": "enrolled"}  # what the fake ladder reports as its playing tier


@registry.register("test_quiet_tier")
class _FakeLadder:
    """Passes audio through and reports :data:`TIER` as the playing tier; counts its resets."""

    name = "test_quiet_tier"
    sample_rate = PIPELINE_SR
    latency_ms = 0.0
    resets = 0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        ctx.telemetry.tier = TIER["name"]
        return chunk

    def reset(self) -> None:
        type(self).resets += 1


@registry.register("test_quiet_boom")
class _Boom:
    name = "test_quiet_boom"
    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        raise RuntimeError("focus chain down")

    def reset(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_ASR", "mock")
    monkeypatch.setenv("HEARING_TTS", "mock")
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    monkeypatch.setenv("HEARING_MOCK_TRANSCRIPT", " ".join(f"w{i}" for i in range(400)))
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)
    TIER["name"] = "enrolled"


def _session(focus="test_quiet_tier", **kw) -> Session:
    pipelines = {"focus_hq": focus, "natural": "passthrough", **kw.pop("pipelines", {})}
    return Session(ServerConfig(pipelines=pipelines, vision_threaded=False, **kw), "cpu")


def _control(s: Session, **fields) -> list[dict]:
    return s.handle_text(json.dumps({"type": "control", **fields}))


def _run(s: Session, seconds: float, seq0: int = 0) -> tuple[np.ndarray, list[dict]]:
    """Feed a voiced 220 Hz tone in 20 ms frames; the output audio and the JSON events."""
    n = round(seconds * PIPELINE_SR)
    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / PIPELINE_SR)).astype(np.float32)
    frame, ys, events = 2 * HOP, [], []
    for i in range(n // frame):
        for m in s.handle_binary(pack_frame(KIND_AUDIO_IN, seq0 + i, 0.0, float_to_pcm16(x[i * frame:(i + 1) * frame]))):
            if isinstance(m, bytes):
                ys.append(pcm16_to_float(unpack_frame(m).payload))
            else:
                events.append(m)
    return (np.concatenate(ys) if ys else np.zeros(0, np.float32)), events


def _dbfs(y: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(y.astype(np.float64) ** 2))), 1e-10))


def test_quiet_runs_focus_pipeline_silently_and_captions_the_target():
    s = _session()
    _control(s, mode="focus_hq")
    y, _ = _run(s, 0.5)
    assert _dbfs(y[-PIPELINE_SR // 10:]) > -20  # Focus plays the target
    t = _control(s, mode="quiet")[-1]
    assert t["pipeline"] == "test_quiet_tier" and t["synth_feed"] == "focus"
    y, events = _run(s, 3.0, seq0=100)
    fade = round(QUIET_FADE_MS * PIPELINE_SR / 1000)
    assert np.all(y[fade:] == 0.0)  # digital silence after the ramp
    captions = [e for e in events if e["type"] == "caption"]
    assert captions and captions[0]["text"].lower().startswith("w0")
    t = s.telemetry()
    assert t["quiet"]["feed"] == "focus" and t["quiet"]["tier"] == "enrolled"
    s.close()


def test_toggles_keep_the_pipeline_target_and_enrollment():
    s = _session()
    _control(s, mode="focus_hq", target={"lock_track_id": 7})
    _run(s, 0.3)
    pipeline, resets = s._built[s.active_spec], _FakeLadder.resets
    s.enroller.update(np.full(PIPELINE_SR, 0.1, np.float32), s.ctx)
    for i in range(10):
        _control(s, mode="quiet")
        _run(s, 0.2, seq0=1000 * i)
        _control(s, mode="focus_hq")
        y, _ = _run(s, 0.2, seq0=1000 * i + 500)
        assert _dbfs(y[-HOP * 4:]) > -20  # back in Focus: playing again at once
        assert s.telemetry()["target"]["track_id"] == 7
    assert s._built[s.active_spec] is pipeline and _FakeLadder.resets == resets  # never rebuilt or reset
    s.close()


def test_quiet_pauses_captions_while_not_isolating_and_says_so():
    s = _session()
    _control(s, mode="quiet")
    _run(s, 0.5)
    assert s.telemetry()["quiet"]["feed"] == "focus"
    TIER["name"] = "gate"  # the ladder drops to the room: no captions of the other talker
    _, events = _run(s, 0.1, seq0=100)
    assert any(e["type"] == "error" and e["stage"] == "quiet" and e["degraded_to"] == "paused" for e in events)
    q = s.telemetry()["quiet"]
    assert q["feed"] == "paused" and "gate" in q["reason"]
    _, events = _run(s, 2.0, seq0=200)  # voiced input, but the ASR hears silence
    texts = {e["text"].lower() for e in events if e["type"] == "caption"}
    assert texts <= {"w0"}  # only the phrase heard while isolated closes; nothing new is captioned
    s.close()


def test_quiet_focus_chain_failure_stays_silent_and_is_visible():
    s = _session(focus="test_quiet_boom")
    _control(s, mode="quiet")
    y, events = _run(s, 0.5)
    assert np.all(y[round(QUIET_FADE_MS * PIPELINE_SR / 1000):] == 0.0)  # never the room
    q = s.telemetry()["quiet"]
    assert q["feed"] == "paused" and "failed" in q["reason"]
    assert any(e["type"] == "error" for e in events)
    s.close()


def test_explicit_quiet_pipeline_keeps_the_old_behaviour():
    s = _session(pipelines={"quiet": "silence"})
    t = _control(s, mode="quiet")[-1]
    assert t["pipeline"] == "silence" and t["quiet"] is None
    s.close()


def test_quiet_from_the_start_is_silent_from_the_first_sample():
    s = _session()
    _control(s, mode="quiet")
    y, _ = _run(s, 0.2)
    assert len(y) and not y.any()
    s.close()


def test_quiet_without_an_asr_engine_stays_silent_and_says_so_once(monkeypatch):
    s = _session()
    _control(s, mode="focus_hq")
    _run(s, 0.3)  # the Focus pipeline is built; only the captions synth can fail now

    def no_asr(spec, *a, **kw):
        raise RuntimeError("no ASR engine")

    monkeypatch.setattr(registry, "build_pipeline", no_asr)
    events = []
    for i in range(2):  # Quiet -> Focus -> Quiet: re-entering must not touch the missing synth either
        _control(s, mode="quiet")
        y, ev = _run(s, 0.5, seq0=1000 * i)
        events += ev
        assert np.all(y[round(QUIET_FADE_MS * PIPELINE_SR / 1000):] == 0.0)
        assert s.telemetry()["quiet"]["feed"] == "paused" and "ASR" in s.telemetry()["quiet"]["reason"]
        _control(s, mode="focus_hq")
        y, _ = _run(s, 0.2, seq0=1000 * i + 500)
        assert _dbfs(y[-HOP * 4:]) > -20
    errors = [e for e in events if e["type"] == "error" and e["stage"] == "quiet"]
    assert len(errors) == 1 and not any(e["type"] == "caption" for e in events)
    s.close()
