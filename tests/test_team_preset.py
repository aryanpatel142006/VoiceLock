"""T-130: the 'team' demo preset (config/demo-team.yaml on top of the judge profile)."""

import json
import time

import numpy as np
import pytest

from hearing.server import team
from hearing.server.protocol import parse_message
from hearing.server.session import ServerConfig, Session, load_stage_modules
from hearing.synth.asr import caption_vocab, vocab_only, whisper_caption_kwargs
from hearing.synth.voice_match import SAME_BUCKET_KEY
from hearing.types import FaceTrack, TargetState
from hearing.vision.target import ATTENTION_PRESETS


def _session(monkeypatch, preset: str | None):
    for k, v in {"HEARING_DEVICE": "cpu", "HEARING_ASR": "mock", "HEARING_TTS": "mock",
                 "HEARING_VOICE_MATCH": "mock"}.items():
        monkeypatch.setenv(k, v)
    for k in ("HEARING_PREFS_LOG", "HEARING_REMOTE_URL", "HEARING_PROFILES_DIR", "HEARING_VOICE_PROFILE", team.HQ_ENV):
        monkeypatch.delenv(k, raising=False)
    if preset is None:
        monkeypatch.delenv(team.PRESET_ENV, raising=False)
    else:
        monkeypatch.setenv(team.PRESET_ENV, preset)
    load_stage_modules()
    return Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": "tiers"},
                                synth_feed="mixture", faces="mock", vision_threaded=False), "cpu")


@pytest.fixture
def off(monkeypatch):
    s = _session(monkeypatch, None)
    yield s
    s.close()


@pytest.fixture
def on(monkeypatch):
    s = _session(monkeypatch, "team")
    yield s
    s.close()


def _judge(s: Session) -> dict:
    replies = s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    assert [m for m in replies if m.get("type") == "error"] == []
    return replies[-1]


def test_preset_off_is_the_judge_profile_unchanged(off):
    t = _judge(off)
    assert t["profile"] == "judge" and t["preset"] is None and off.preset is None
    assert t["attention"] == ATTENTION_PRESETS["sticky"] and t["synth_feed_request"] == "auto"
    assert t["voice"]["override"] is None and off._name == ("", [])
    # T-135: the judge profile itself promotes at most every 3 s (JUDGE_MIN_SWITCH_S); the preset reads the same value
    assert "min_switch_s=3.0" in off.active_spec and "upgrade_after_s=0.5" in off.active_spec
    assert off._same_bucket_guard and not off._lock_on_start and off._namecall_timers == {}
    assert whisper_caption_kwargs() == {}
    assert team.apply_env() == {}


def test_team_preset_pins_every_knob(on):
    t = _judge(on)
    preset = team.load_preset("team")
    assert t["profile"] == "judge" and t["preset"]["name"] == "team"
    pins = t["preset"]["pins"]
    assert pins["hq_backend"] == "local" and pins["name"] == "Aryan" and pins["match"] == "strict"
    assert t["attention"]["hold_lock_s"] == 60.0 and t["synth_feed_request"] == "focus"
    assert t["voice"]["override"] == preset["voice_override"]
    assert on._name == ("Aryan", ["Ariane", "Arion", "Arian"]) and on._name_match == "strict"
    assert on._namecall_timers == {"quiet_s": 10.0, "timeout_s": 180.0}
    assert "min_switch_s=3.0" in on.active_spec and "upgrade_after_s=0.5" in on.active_spec
    assert on._lock_on_start and on._same_bucket_guard  # the shipped pair is not a verified M/F pair
    assert t["mode"] == "focus_hq"  # the judge fields still apply
    # T-181 (T-129 Q-470): standard over the judge profile's strict; the drawer can still pick strict
    assert t["isolation"] == "standard" and pins["isolation"] == "standard" and on.isolation == "standard"
    assert pins["hq_reason"] == team.MAC_REASON


def test_explicit_team_profile_without_env(off):
    assert parse_message({"type": "control", "profile": "team"}).profile == "team"
    t = off.handle_text(json.dumps({"type": "control", "profile": "team"}))[-1]
    assert t["profile"] == "judge" and t["preset"]["name"] == "team"


def test_lock_on_start_locks_the_first_attention_pick(on):
    _judge(on)
    on.ctx.tracks = [FaceTrack(4, (0.4, 0.3, 0.2, 0.4), None, 0.9, 0.0)]
    on.ctx.target = TargetState(4, 1.0, False, 0.0, 0)
    on.vision.poll = lambda ctx, timeout=0.0: []
    locked = []
    on.vision.lock = locked.append
    on._poll_vision()
    assert locked == [4] and not on._lock_on_start
    on._poll_vision()
    assert locked == [4]  # once


def test_same_bucket_guard_only_off_for_a_verified_mf_pair(on):
    mf = {"pair": {"target": {"gender": "male"}, "other": {"gender": "female"}, "verified": True}}
    assert not team.same_bucket_guard(mf)
    assert team.same_bucket_guard({**mf, "pair": {**mf["pair"], "verified": False}})
    assert team.same_bucket_guard({"pair": {"target": {"gender": "male"}, "other": {"gender": "male"},
                                            "verified": True}})
    _judge(on)
    on._same_bucket_guard = False
    on.ctx.telemetry.extras[SAME_BUCKET_KEY] = True
    on.vision.poll = lambda ctx, timeout=0.0: []
    on._poll_vision()
    assert on.ctx.telemetry.extras[SAME_BUCKET_KEY] is False


def test_backend_is_picked_once_from_the_probe(monkeypatch):
    monkeypatch.delenv(team.HQ_ENV, raising=False)
    assert team.pick_backend([10, 12, 11, 13, 14], 60)[0] == "remote"
    assert team.pick_backend([10, 12, 11, 13, 95], 60)[0] == "local"
    assert team.pick_backend([10, None, 11], 60)[0] == "remote"  # T-154: a majority answered
    assert team.pick_backend([10, None, None], 60)[0] == "local"
    assert team.health_url("ws://127.0.0.1:18765/ws/session") == "http://127.0.0.1:18765/health"
    calls = []

    def probe(url, n):
        calls.append(url)
        return [5.0] * n

    gpu = {"hq_backend": "remote"}  # T-181: only the --gpu path probes
    team._PINNED.pop("ws://x:1/ws/session", None)
    assert team.pinned_backend("ws://x:1/ws/session", gpu, probe)[0] == "remote"
    assert team.pinned_backend("ws://x:1/ws/session", gpu, probe)[0] == "remote"
    assert calls == ["ws://x:1/ws/session"]  # no re-probe mid-session: pinned per server
    assert team.pinned_backend(None, gpu, probe)[0] == "local"
    team._PINNED.pop("ws://x:1/ws/session", None)


def test_team_preset_has_no_caption_prompt(monkeypatch):
    # T-139/Q-375: the prompt put 'Selective Hearing,' and 'Select' over real speech; captions only show spoken words
    preset = team.load_preset("team")
    assert "caption_prompt" not in preset
    for k in ("HEARING_ASR", "HEARING_ASR_TEMPERATURE", "HEARING_ASR_PROMPT", "HEARING_ASR_VOCAB"):
        monkeypatch.delenv(k, raising=False)
    applied = team.apply_env(preset)
    assert applied["HEARING_ASR"] == "local" and applied["HEARING_ASR_TEMPERATURE"] == "0.0"
    assert "HEARING_ASR_PROMPT" not in applied and "HEARING_ASR_VOCAB" not in applied
    assert whisper_caption_kwargs() == {"temperature": 0.0} and caption_vocab() == []


def test_caption_env_is_vocabulary_only(monkeypatch):
    preset = {**team.load_preset("team"), "caption_prompt": "Selective Hearing, HackRU, Rutgers, Focus HQ, Aryan, Joseph."}
    for k in ("HEARING_ASR", "HEARING_ASR_TEMPERATURE", "HEARING_ASR_PROMPT", "HEARING_ASR_VOCAB"):
        monkeypatch.delenv(k, raising=False)
    applied = team.apply_env(preset)
    assert applied["HEARING_ASR"] == "local" and applied["HEARING_ASR_TEMPERATURE"] == "0.0"
    kw = whisper_caption_kwargs()
    assert kw["temperature"] == 0.0
    words = [w.strip(" .") for w in kw["initial_prompt"].split(",")]
    assert {"Selective Hearing", "HackRU", "Rutgers", "Focus HQ", "Aryan"} <= set(words)
    assert len(kw["initial_prompt"]) < 100  # a vocabulary list, never the scripted lines
    vocab = caption_vocab()  # T-136's guard knows the prompt words: "Selective Hearing," alone near the floor is dropped
    assert "Selective Hearing" in vocab and vocab_only("Selective Hearing,", vocab)
    assert not vocab_only("see you at noon", vocab)
    monkeypatch.setenv("HEARING_ASR", "elevenlabs")
    assert "HEARING_ASR" not in team.apply_env(preset)  # a verified ElevenLabs setup is kept


def test_namecall_timers_are_per_instance():
    from hearing.server.namecall import QUIET_S, TIMEOUT_S, NameCaller

    n = NameCaller("Aryan", transcriber=lambda x: [], embedder=lambda x: None, cleaner=None, threaded=False,
                   quiet_s=10, timeout_s=180)
    assert (n.quiet_s, n.timeout_s) == (10.0, 180.0)
    d = NameCaller("Aryan", transcriber=lambda x: [], embedder=lambda x: None, cleaner=None, threaded=False)
    assert (d.quiet_s, d.timeout_s) == (QUIET_S, TIMEOUT_S)


# ------------------------------------------------------------ T-154: one blip no longer pins local

URL = "ws://127.0.0.1:9/ws/session"


class FakeProber:
    """``probe_rtts`` stand-in: each round answers ``fails`` failures (None) then 6 ms."""

    def __init__(self, fails: int):
        self.fails, self.rounds = fails, 0

    def __call__(self, url, n=5):
        self.rounds += 1
        return [None] * self.fails + [6.0] * (n - self.fails)


def test_majority_of_probes_pins_remote_and_a_failed_probe_is_retried():
    assert team.pick_backend([None, 6, 6, 6, 6], 60) == ("remote", "GPU /health 4/5 ok, RTT p95 6 ms < 60 ms")
    backend, reason = team.pick_backend([None, None, None, 6, 6], 60)
    assert backend == "local" and reason == "GPU /health failed 3/5"
    assert team.pick_backend([None, None, 6, 6, 6], 60)[0] == "remote"  # >= 3/5
    calls = []

    def once(url, timeout):  # the 2nd GET blips, its retry answers
        calls.append(url)
        return None if len(calls) == 2 else 6.0

    assert team.probe_rtts(URL, 5, once=once) == [6.0] * 5 and len(calls) == 6
    calls.clear()
    dead = team.probe_rtts(URL, 5, once=lambda url, timeout: calls.append(url))
    assert dead == [None] * 5 and len(calls) == 6  # 3 failures (each retried): no majority, stop waiting


def _team_session(monkeypatch, prober, hq_env: str = "remote"):  # remote: demo.sh --judge --team --gpu
    for k, v in {"HEARING_DEVICE": "cpu", "HEARING_ASR": "mock", "HEARING_TTS": "mock",
                 "HEARING_VOICE_MATCH": "mock", team.PRESET_ENV: "team", "HEARING_REMOTE_URL": URL,
                 team.HQ_ENV: hq_env}.items():
        monkeypatch.setenv(k, v)
    for k in ("HEARING_PREFS_LOG", "HEARING_PROFILES_DIR", "HEARING_VOICE_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(team, "probe_rtts", prober)
    monkeypatch.setattr(team, "_PINNED", {})
    monkeypatch.setattr(Session, "_make_reprobe", lambda self, url, preset: team.Reprobe(url, preset))  # no thread
    load_stage_modules()
    s = Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": "tiers"}, remote_url=URL,
                             synth_feed="mixture", faces="mock", vision_threaded=False), "cpu")
    s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
    return s


def test_one_failed_probe_of_five_pins_remote(monkeypatch):
    s = _team_session(monkeypatch, FakeProber(1))
    try:
        pins = s.telemetry()["preset"]["pins"]
        assert s.hq_backend == "remote" and pins["hq_backend"] == "remote" and s._hq_reprobe is None
        assert pins["hq_reason"] == "GPU /health 4/5 ok, RTT p95 6 ms < 60 ms"
    finally:
        s.close()


class CountingProber(FakeProber):
    """A healthy A100 (the tunnel is up): every probe round would pick remote."""

    def __init__(self):
        super().__init__(0)


@pytest.mark.parametrize("hq_env", ["", "local", "auto"])
def test_team_preset_pins_the_mac_with_the_tunnel_up(monkeypatch, hq_env):
    """T-181: T-129 measured the Mac ahead of every A100 variant with profiles: no probe, no re-probe, Mac pinned."""
    prober = CountingProber()
    s = _team_session(monkeypatch, prober, hq_env)
    try:
        pins = s.telemetry()["preset"]["pins"]
        assert prober.rounds == 0 and s._hq_reprobe is None
        assert s.hq_backend == "local" and pins["hq_backend"] == "local"
        assert pins["hq_reason"] == "Mac (measured: beats the A100 with profiles, T-129)" == team.MAC_REASON
        assert URL not in team._PINNED  # a later --gpu server still probes
    finally:
        s.close()


def test_gpu_flag_probes_and_pins_the_a100(monkeypatch):
    """T-181: ``demo.sh --judge --team --gpu`` exports HEARING_HQ_BACKEND=remote: probed and pinned as before."""
    prober = CountingProber()
    s = _team_session(monkeypatch, prober, "remote")
    try:
        pins = s.telemetry()["preset"]["pins"]
        assert prober.rounds == 1 and s.hq_backend == "remote" and pins["hq_backend"] == "remote"
        assert pins["hq_reason"] == "GPU /health 5/5 ok, RTT p95 6 ms < 60 ms"
    finally:
        s.close()


def test_wants_gpu_env_beats_the_yaml(monkeypatch):
    monkeypatch.delenv(team.HQ_ENV, raising=False)
    assert team.load_preset("team")["hq_backend"] == "local" and not team.wants_gpu(team.load_preset("team"))
    assert not team.wants_gpu({}) and team.wants_gpu({"hq_backend": "remote"})
    monkeypatch.setenv(team.HQ_ENV, "remote")
    assert team.wants_gpu({}) and team.wants_gpu({"hq_backend": "local"})
    monkeypatch.setenv(team.HQ_ENV, "local")
    assert not team.wants_gpu({"hq_backend": "remote"})


def test_demo_sh_pins_the_mac_unless_gpu():
    """T-181: ``--judge --team`` skips the GPU probe and says Mac; ``--gpu`` exports the remote pin and probes."""
    from pathlib import Path

    s = (Path(__file__).resolve().parents[1] / "scripts/demo.sh").read_text()
    assert 'if [ "$TEAM" = 1 ] && [ "$GPU" = 1 ]; then\n  export HEARING_HQ_BACKEND=remote' in s
    assert '[ "$TEAM" = 1 ] && [ "${HEARING_HQ_BACKEND:-}" != remote ] && TEAM_MAC=1' in s
    assert 'if [ "$TEAM_MAC" = 0 ]; then\n    PROBE_OUT=' in s  # the probe only runs off the Mac pin
    assert 'BACKEND="Mac (measured: beats the A100 with profiles, T-129)"' in s
    assert 'GPU_LABEL="iLab A100 (--gpu)"' in s


LOUD = np.full(160, 0.2, dtype=np.float32)
QUIET = np.zeros(160, dtype=np.float32)


def test_gpu_local_pin_reprobes_and_switches_at_a_pause_not_mid_utterance(monkeypatch):
    prober = FakeProber(3)
    s = _team_session(monkeypatch, prober)
    try:
        pins = s.telemetry()["preset"]["pins"]
        assert s.hq_backend == "local" and pins["hq_backend"] == "local"
        assert pins["hq_reason"] == "GPU /health failed 3/5; re-probing every 10 s"  # the drawer's text
        reprobe = s._hq_reprobe
        assert reprobe is not None and reprobe.every_s == 10.0
        assert reprobe.tick()[0] == "local" and reprobe.ready is None  # still 3/5 down
        s._update_hq_pin(QUIET)
        assert s.hq_backend == "local"
        prober.fails = 0  # the A100 answers 5/5
        assert reprobe.tick()[0] == "remote" and team._PINNED[URL][0] == "remote"  # later sessions start remote
        s._isolated = True  # Isolated, mid-utterance: no switch
        for _ in range(100):
            s._update_hq_pin(LOUD)
        assert s.hq_backend == "local" and s.telemetry()["preset"]["pins"]["hq_backend"] == "local"
        s._isolated = True
        for _ in range(int(team.PAUSE_S * 100) - 1):
            s._update_hq_pin(QUIET)
        assert s.hq_backend == "local"  # the pause is not long enough yet
        s._isolated = True
        s._update_hq_pin(QUIET)
        assert s.hq_backend == "remote" and s._hq_reprobe is None
        pins = s.telemetry()["preset"]["pins"]
        assert pins["hq_backend"] == "remote"
        assert pins["hq_reason"] == "GPU /health 5/5 ok, RTT p95 6 ms < 60 ms; switched from local at a pause"
    finally:
        s.close()


def test_switch_before_isolated_needs_no_pause(monkeypatch):
    prober = FakeProber(5)
    s = _team_session(monkeypatch, prober)
    try:
        assert s.hq_backend == "local"
        prober.fails = 0
        s._hq_reprobe.tick()
        s._isolated = False  # still acquiring: switching now cuts nothing
        s._update_hq_pin(LOUD)
        assert s.hq_backend == "remote" and "at a safe point" in s._preset_pins["hq_reason"]
    finally:
        s.close()


def test_reprobe_thread_stops_once_remote():
    prober = FakeProber(5)
    r = team.Reprobe(URL, {}, probe=prober, every_s=0.01)
    team._PINNED.pop(URL, None)
    r.start()
    deadline = time.monotonic() + 2
    while prober.rounds < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert r.ready is None and r.last[0] == "local"
    prober.fails = 0
    while r.ready is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert r.ready is not None and r.ready[0] == "remote"
    r._thread.join(timeout=1)
    assert not r._thread.is_alive()
    team._PINNED.pop(URL, None)


def test_audio_only_preset_drops_camera_frames(on, monkeypatch):
    """T-168: the decided demo has no camera: under the preset a video frame never reaches vision or the GPU."""
    from hearing.server.protocol import KIND_VIDEO_IN, pack_frame

    t = _judge(on)
    assert t["preset"]["pins"]["audio_only"] is True and on.audio_only
    forwarded, submitted = [], []
    monkeypatch.setattr(on, "_forward_video", lambda *a: forwarded.append(a))
    monkeypatch.setattr(on.vision, "submit", lambda *a: submitted.append(a))
    for i in range(3):
        assert on.handle_binary(pack_frame(KIND_VIDEO_IN, i, i * 0.04, b"\xff\xd8jpeg")) == []
    assert forwarded == [] and submitted == [] and on.video_frames == 0 and on.video_dropped == 3
    assert on.handle_text(json.dumps({"type": "control", "ab": "raw"}))[-1]["preset"]["video_dropped"] == 3


def test_camera_frames_still_flow_without_the_preset(off, monkeypatch):
    from hearing.server.protocol import KIND_VIDEO_IN, pack_frame

    _judge(off)
    forwarded = []
    monkeypatch.setattr(off, "_forward_video", lambda *a: forwarded.append(a))
    monkeypatch.setattr(off.vision, "submit", lambda *a: None)
    off.handle_binary(pack_frame(KIND_VIDEO_IN, 0, 0.0, b"\xff\xd8jpeg"))
    assert not off.audio_only and off.video_frames == 1 and len(forwarded) == 1
