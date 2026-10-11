"""Session server: /health, console serving, and the /ws/session wire protocol."""

import json
import time
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import hearing
from hearing import registry
from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.server.app import create_app
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    KIND_VIDEO_IN,
    ProtocolError,
    pack_frame,
    unpack_frame,
)
from hearing.server.resample import StreamResampler
from hearing.server.session import ServerConfig, Session, parse_pipelines
from hearing.testing import make_test_signal
from hearing.types import AudioChunk, StageContext

SR = 16000


@registry.register("test_server_boom")
class _Boom:
    """Raises on every chunk, to prove a failing stage degrades instead of killing the session."""

    name = "test_server_boom"
    sample_rate = SR
    latency_ms = 0.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        raise RuntimeError("boom")

    def reset(self) -> None:
        pass


@registry.register("test_server_half")
class _Half:
    """Halves the signal and reports 5 ms of algorithmic latency."""

    name = "test_server_half"
    sample_rate = SR
    latency_ms = 5.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        return replace(chunk, samples=(chunk.samples * np.float32(0.5)).astype(np.float32))

    def reset(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")


def _client(tmp_path=None, **config) -> TestClient:
    config.setdefault("console_dir", tmp_path)
    return TestClient(create_app(ServerConfig(**config)))


def _grid_signal(n: int, seed: int = 0) -> np.ndarray:
    """A speech-like signal snapped to the int16 grid so PCM16 round trips are exact."""
    x = make_test_signal(n / SR + 0.01, SR, seed=seed)[:n]
    return pcm16_to_float(float_to_pcm16(x))


def _recv(ws, want, limit: int = 400):
    """Read messages until ``want(msg)`` holds; ``msg`` is a dict (JSON) or bytes (binary)."""
    for _ in range(limit):
        raw = ws.receive()
        if raw["type"] == "websocket.close":
            raise AssertionError(f"socket closed: {raw}")
        msg = json.loads(raw["text"]) if raw.get("text") is not None else raw["bytes"]
        if want(msg):
            return msg
    raise AssertionError("expected message never arrived")


def _is(kind: str):
    return lambda m: isinstance(m, dict) and m.get("type") == kind


def _is_audio(m) -> bool:
    return isinstance(m, bytes) and m[0] == KIND_AUDIO_OUT


def _send_audio(ws, x: np.ndarray, seq: int, ts: float) -> None:
    ws.send_bytes(pack_frame(KIND_AUDIO_IN, seq, ts, float_to_pcm16(x)))


# ---------------------------------------------------------------- protocol


def test_frame_round_trip_is_little_endian_13_byte_header():
    data = pack_frame(KIND_AUDIO_IN, 123456, 12.5, b"\x01\x02")
    assert data[:5] == bytes([1]) + (123456).to_bytes(4, "little")
    assert len(data) == 13 + 2
    frame = unpack_frame(data)
    assert (frame.kind, frame.seq, frame.capture_ts, frame.payload) == (1, 123456, 12.5, b"\x01\x02")


@pytest.mark.parametrize("data", [b"", b"\x01\x00\x00", bytes([9]) + bytes(12)])
def test_unpack_rejects_short_frames_and_unknown_kinds(data):
    with pytest.raises(ProtocolError):
        unpack_frame(data)


def test_parse_pipelines_reads_mode_assignments_with_quoted_separators():
    got = parse_pipelines('natural = gain(db=-6) > delay(ms=10); quiet=passthrough;focus_hq=a(label="a;b")')
    assert got == {
        "natural": "gain(db=-6) > delay(ms=10)",
        "quiet": "passthrough",
        "focus_hq": 'a(label="a;b")',
    }
    with pytest.raises(ValueError):
        parse_pipelines("bogus_mode=passthrough")
    with pytest.raises(ValueError):
        parse_pipelines("natural")


# ---------------------------------------------------------------- resampler


@pytest.mark.parametrize(("sr_in", "sr_out"), [(48000, 16000), (16000, 48000), (44100, 16000),
                                               (16000, 44100), (16000, 16000)])
def test_stream_resampler_is_chunk_invariant_and_never_bursty(sr_in, sr_out):
    x = make_test_signal(0.3, sr_in, seed=2)
    whole = StreamResampler(sr_in, sr_out).process(x)
    r = StreamResampler(sr_in, sr_out)
    rng = np.random.default_rng(0)
    parts, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, sr_in // 50))
        parts.append(r.process(x[i : i + n]))
        end = min(i + n, len(x))
        expected = -(-end * sr_out // sr_in) + (-i * sr_out // sr_in)  # ceil(end) - ceil(i)
        assert len(parts[-1]) == expected  # output keeps pace with input, no bursts
        i += n
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-5)
    assert len(whole) == -(-len(x) * sr_out // sr_in)


@pytest.mark.parametrize(("sr_in", "sr_out"), [(48000, 16000), (16000, 48000), (44100, 16000)])
def test_stream_resampler_passes_speech_band_and_reports_small_delay(sr_in, sr_out):
    r = StreamResampler(sr_in, sr_out)
    assert 0 < r.delay_ms < 3
    t = np.arange(sr_in // 4) / sr_in
    y = r.process((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
    tail = y[len(y) // 2 :]
    assert np.sqrt(np.mean(tail**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)


def test_stream_resampler_rejects_aliases_when_downsampling():
    sr = 48000
    t = np.arange(sr // 4) / sr
    y = StreamResampler(sr, 16000).process((0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32))
    tail = y[len(y) // 2 :]
    assert 20 * np.log10(np.sqrt(np.mean(tail**2)) / (0.5 / np.sqrt(2))) < -40


# ---------------------------------------------------------------- http


def test_health_returns_documented_json(tmp_path):
    client = _client(tmp_path, pipelines={"natural": "passthrough>tse_hq"})
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["device"] == "cpu"
    assert body["version"] == hearing.__version__
    assert body["stages"]["passthrough"] == "ready"
    assert body["stages"]["tse_hq"] == "unavailable"  # configured but not registered
    assert set(body["stages"].values()) <= {"ready", "unavailable"}


def test_health_reports_the_tiers_config_hashes(tmp_path):
    # T-118: demo_regress compares the remote's loaded config/tiers-remote.yaml with the local file.
    import hashlib
    from pathlib import Path

    from hearing.server.app import config_hashes

    body = _client(tmp_path).get("/health").json()
    root = Path(hearing.__file__).resolve().parents[1]
    expect = hashlib.sha256((root / "config/tiers-remote.yaml").read_bytes()).hexdigest()[:12]
    assert body["tiers_config"]["tiers-remote.yaml"] == expect
    assert body["tiers_config"] == config_hashes(root)
    assert "tiers.yaml" in body["tiers_config"]


def test_root_serves_console_index(tmp_path):
    (tmp_path / "index.html").write_text("<h1>console</h1>")
    (tmp_path / "app.js").write_text("console.log(1)")
    client = _client(tmp_path)
    assert "<h1>console</h1>" in client.get("/").text
    assert client.get("/app.js").text == "console.log(1)"
    assert client.get("/health").json()["ok"] is True  # routes win over the static mount


def test_root_without_console_explains_instead_of_404(tmp_path):
    client = _client(tmp_path / "missing")
    r = client.get("/")
    assert r.status_code == 200
    assert "console" in r.text.lower()


# ---------------------------------------------------------------- websocket


def test_hello_gets_immediate_telemetry_with_documented_shape(tmp_path):
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "hello", "sample_rate": 16000, "frame_ms": 20,
                                 "video": False, "client": "test"}))
        t = _recv(ws, _is("telemetry"))
    assert {"type", "seq", "tier", "latency_ms", "faces", "target", "voice"} <= set(t)
    assert t["seq"] is None  # no audio yet
    assert isinstance(t["tier"], str) and t["tier"]
    assert t["faces"] == []
    assert t["target"] == {"track_id": None, "confidence": 0.0, "locked": False}
    assert set(t["voice"]) == {"bucket", "source", "override"}
    assert t["mode"] == "natural"
    assert t["device"] == "cpu"


def test_audio_round_trip_keeps_seq_and_timestamp(tmp_path):
    x = _grid_signal(320 * 5)
    got = []
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "hello", "sample_rate": 16000, "frame_ms": 20}))
        for i in range(5):
            _send_audio(ws, x[i * 320 : (i + 1) * 320], seq=100 + i, ts=1.0 + 0.02 * i)
            got.append(unpack_frame(_recv(ws, _is_audio)))
    assert [f.seq for f in got] == [100, 101, 102, 103, 104]
    assert [f.capture_ts for f in got] == pytest.approx([1.0 + 0.02 * i for i in range(5)])
    assert all(f.kind == KIND_AUDIO_OUT for f in got)
    out = np.concatenate([pcm16_to_float(f.payload) for f in got])
    np.testing.assert_array_equal(out, x)  # default natural pipeline is a passthrough


def test_audio_before_hello_uses_16k_defaults(tmp_path):
    x = _grid_signal(160)
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        _send_audio(ws, x, seq=7, ts=0.0)
        frame = unpack_frame(_recv(ws, _is_audio))
    np.testing.assert_array_equal(pcm16_to_float(frame.payload), x)


def test_telemetry_reports_stage_latency_and_correlation_seq(tmp_path):
    client = _client(tmp_path, pipelines={"natural": "test_server_half>passthrough"})
    x = _grid_signal(320)
    with client.websocket_connect("/ws/session") as ws:
        _send_audio(ws, x, seq=41, ts=0.0)
        out = unpack_frame(_recv(ws, _is_audio))
        t = _recv(ws, lambda m: _is("telemetry")(m) and m["seq"] == 41)
    np.testing.assert_allclose(pcm16_to_float(out.payload), x * 0.5, atol=1 / 32768)
    lat = t["latency_ms"]
    assert {"test_server_half", "passthrough", "algorithmic", "server", "total"} <= set(lat)
    assert all(isinstance(v, float) and v >= 0 for v in lat.values())
    assert lat["algorithmic"] == 5.0
    assert lat["resample"] == 0.0  # 16 kHz client
    assert lat["total"] >= lat["algorithmic"] + 20.0  # + the 20 ms client frame
    assert t["pipeline"] == "test_server_half>passthrough"


def test_telemetry_arrives_at_about_10_hz(tmp_path):
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        _recv(ws, _is("telemetry"))
        t0 = time.monotonic()
        for _ in range(5):
            _recv(ws, _is("telemetry"))
        rate = 5 / (time.monotonic() - t0)
    assert 5 <= rate <= 20


def test_control_switches_mode_and_quiet_outputs_silence(tmp_path):
    x = _grid_signal(320)
    # an explicit quiet pipeline plays as given (T-149: the default Quiet runs Focus HQ silently)
    with _client(tmp_path, pipelines={"quiet": "silence"}).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "control", "mode": "quiet", "suppression": 80}))
        t = _recv(ws, lambda m: _is("telemetry")(m) and m["mode"] == "quiet")
        _send_audio(ws, x, seq=1, ts=0.0)
        out = pcm16_to_float(unpack_frame(_recv(ws, _is_audio)).payload)
        ws.send_text(json.dumps({"type": "control", "mode": "natural"}))
        _send_audio(ws, x, seq=2, ts=0.02)
        back = pcm16_to_float(unpack_frame(_recv(ws, _is_audio)).payload)
    assert t["pipeline"] == "silence"
    assert len(out) == 320 and not out.any()
    np.testing.assert_array_equal(back, x)


def test_control_suppression_reaches_stage_context():
    session = Session(ServerConfig(), device="cpu")
    session.handle_text(json.dumps({"type": "control", "suppression": 80}))
    assert session.ctx.suppression == pytest.approx(0.8)
    session.handle_text(json.dumps({"type": "control", "mode": "auto"}))
    assert session.telemetry()["mode"] == "auto"
    assert session.effective_mode == "natural"  # until the context model exists


def test_control_target_lock_and_voice_override_show_in_telemetry(tmp_path):
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "control", "target": {"lock_track_id": 17},
                                 "voice_override": "female_adult"}))
        t = _recv(ws, lambda m: _is("telemetry")(m) and m["target"]["locked"])
        ws.send_text(json.dumps({"type": "control", "target": None, "voice_override": None}))
        u = _recv(ws, lambda m: _is("telemetry")(m) and not m["target"]["locked"])
    assert t["target"]["track_id"] == 17
    assert t["voice"] == {"bucket": "female_adult", "source": "override", "override": "female_adult"}
    assert u["target"]["track_id"] is None
    assert u["voice"]["override"] is None


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"type": "nope"}),
        json.dumps({"type": "control", "mode": "loud"}),
        json.dumps({"type": "control", "suppression": 150}),
        json.dumps({"type": "hello", "sample_rate": -1}),
    ],
)
def test_bad_json_messages_get_error_and_session_survives(tmp_path, text):
    x = _grid_signal(160)
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_text(text)
        err = _recv(ws, _is("error"))
        _send_audio(ws, x, seq=3, ts=0.0)
        frame = unpack_frame(_recv(ws, _is_audio))
    assert err["stage"] == "protocol"
    assert err["message"]
    assert "degraded_to" in err
    assert frame.seq == 3


def test_bad_binary_frame_gets_error(tmp_path):
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_bytes(b"\x01\x00")
        err = _recv(ws, _is("error"))
        ws.send_bytes(pack_frame(KIND_AUDIO_IN, 1, 0.0, b"\x00\x00\x00"))  # odd PCM16 length
        err2 = _recv(ws, _is("error"))
    assert err["stage"] == "protocol" and err2["stage"] == "protocol"


def test_failing_stage_degrades_to_fallback_and_audio_keeps_flowing(tmp_path):
    client = _client(tmp_path, pipelines={"natural": "test_server_boom"})
    x = _grid_signal(320)
    with client.websocket_connect("/ws/session") as ws:
        _send_audio(ws, x, seq=1, ts=0.0)
        err = _recv(ws, _is("error"))
        first = unpack_frame(_recv(ws, _is_audio))
        _send_audio(ws, x, seq=2, ts=0.02)
        second = unpack_frame(_recv(ws, _is_audio))
        t = _recv(ws, _is("telemetry"))
    assert err["stage"] == "test_server_boom"
    assert "boom" in err["message"]
    assert err["degraded_to"] == "passthrough"
    assert (first.seq, second.seq) == (1, 2)
    np.testing.assert_array_equal(pcm16_to_float(second.payload), x)
    assert t["pipeline"] == "passthrough"


def test_unknown_stage_in_config_degrades_at_first_use(tmp_path):
    client = _client(tmp_path, pipelines={"natural": "no_such_stage>gate"})
    x = _grid_signal(160)
    with client.websocket_connect("/ws/session") as ws:
        _send_audio(ws, x, seq=5, ts=0.0)
        err = _recv(ws, _is("error"))
        frame = unpack_frame(_recv(ws, _is_audio))
    assert err["stage"] == "pipeline"
    assert "no_such_stage" in err["message"]
    assert err["degraded_to"] == "passthrough"
    np.testing.assert_array_equal(pcm16_to_float(frame.payload), x)


def test_48k_client_is_resampled_both_ways(tmp_path):
    sr = 48000
    t = np.arange(sr // 2) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    n = 960  # 20 ms at 48 kHz
    out = []
    # passthrough, not the default dfn_ll: a steady tone is noise to DFN (the test checks resampling)
    with _client(tmp_path, pipelines={"natural": "passthrough"}).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "hello", "sample_rate": sr, "frame_ms": 20}))
        for i in range(len(x) // n):
            _send_audio(ws, x[i * n : (i + 1) * n], seq=i, ts=i * 0.02)
        deadline = len(x) // n - 1
        while True:
            frame = unpack_frame(_recv(ws, _is_audio))
            out.append(pcm16_to_float(frame.payload))
            if frame.seq == deadline:
                break
        t = _recv(ws, _is("telemetry"))
    y = np.concatenate(out)
    held = len(x) - len(y)  # at most one partial 10 ms hop waits in the chunker
    assert 0 <= held < 10 * sr / 1000
    assert 0 < t["latency_ms"]["resample"] < 6
    tail = y[len(y) // 2 :]
    assert np.sqrt(np.mean(tail**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.1)  # unit gain


def test_video_frames_are_accepted_silently(tmp_path):
    # protocol test: a passthrough pipeline, so the fast suite needs no DeepFilterNet weights (Q-1113, laptop-c)
    client = _client(tmp_path, pipelines={"natural": "passthrough"})
    with client.websocket_connect("/ws/session") as ws:
        ws.send_bytes(pack_frame(KIND_VIDEO_IN, 1, 0.0, b"\xff\xd8\xff\xd9"))
        ws.send_bytes(pack_frame(KIND_AUDIO_IN, 2, 0.0, float_to_pcm16(_grid_signal(160))))
        frame = unpack_frame(_recv(ws, lambda m: _is_audio(m) or _is("error")(m)))
    assert frame.seq == 2


def test_session_disambiguates_repeated_stage_names():
    session = Session(ServerConfig(pipelines={"natural": "passthrough>passthrough"}), device="cpu")
    session.handle_binary(pack_frame(KIND_AUDIO_IN, 0, 0.0, float_to_pcm16(_grid_signal(160))))
    lat = session.telemetry()["latency_ms"]
    assert {"passthrough", "passthrough#2"} <= set(lat)


# ---------------------------------------------------------------- cli


def test_cli_serve_builds_config_and_runs_uvicorn(monkeypatch, tmp_path):
    from hearing import cli

    calls = {}

    def fake_run(app, host, port, log_level):
        calls.update(app=app, host=host, port=port, log_level=log_level)

    monkeypatch.setattr("uvicorn.run", fake_run)
    rc = cli.main(["serve", "--no-warm", "--host", "0.0.0.0", "--port", "9001",
                   "--pipelines", "natural=gain(db=-6)", "--console-dir", str(tmp_path)])
    assert rc == 0
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 9001)
    config = calls["app"].state.config
    assert config.pipelines["natural"] == "gain(db=-6)"
    assert config.pipelines["quiet"] == "silence"  # unspecified modes keep defaults
    assert config.console_dir == tmp_path


def test_python_dash_m_hearing_help_lists_serve():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "hearing", "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0
    assert "serve" in r.stdout


def test_failing_fallback_degrades_to_raw_input():
    config = ServerConfig(pipelines={"natural": "test_server_boom"}, fallback="test_server_boom")
    session = Session(config, device="cpu")
    x = _grid_signal(160)
    replies = session.handle_binary(pack_frame(KIND_AUDIO_IN, 9, 0.0, float_to_pcm16(x)))
    errors = [r for r in replies if isinstance(r, dict)]
    frames = [unpack_frame(r) for r in replies if isinstance(r, bytes)]
    assert [e["degraded_to"] for e in errors] == ["raw"]
    np.testing.assert_array_equal(pcm16_to_float(frames[0].payload), x)
    assert session.telemetry()["pipeline"] == ""
