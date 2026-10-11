"""Remote GPU path: the ``remote`` stage against a real session server on loopback."""

import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from hearing import registry
from hearing.audio.io import HOP
from hearing.server import remote
from hearing.server.app import create_app
from hearing.server.remote import RemoteStage, health_url, probe
from hearing.server.session import ServerConfig, Session
from hearing.types import AudioChunk, StageContext

SR = 16000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """``hearing serve`` in a background thread, focus_hq pinned to passthrough (default: tiers)."""

    def __init__(self, port: int | None = None):
        self.port = port or _free_port()
        app = create_app(ServerConfig(pipelines={"focus_hq": "passthrough"}))
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws/session"

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 5
        while not self.server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.01)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture
def server():
    with _Server() as s:
        yield s


def _hops(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(0.3 * rng.standard_normal(HOP)).clip(-1, 1).astype(np.float32) for _ in range(n)]


def _run(stage, hops, ctx, pace_s=0.003):
    out = []
    for i, h in enumerate(hops):
        out.append(stage.process(AudioChunk(h, SR, i, i * 0.01), ctx).samples)
        time.sleep(pace_s)
    return out


def test_health_url():
    assert health_url("ws://h:8765/ws/session") == "http://h:8765/health"
    assert health_url("wss://pod-8000.proxy.runpod.net/ws/session") == "https://pod-8000.proxy.runpod.net/health"


def test_registered_and_parsed_from_spec():
    p = registry.build_pipeline('remote(url="ws://127.0.0.1:9/ws/session", fallback="gain(db=-6)", delay_ms=100)')
    stage = p.stages[0]
    assert isinstance(stage, RemoteStage)
    assert stage.latency_ms == pytest.approx(100.0) and p.latency_ms == pytest.approx(100.0)
    stage.close()


def test_remote_passthrough_is_input_delayed(server):
    stage = RemoteStage(url=server.url, delay_ms=100, fallback="gain(db=-20)")
    ctx = StageContext.default()
    assert stage.client.start().wait_connected(5)
    hops = _hops(300)
    out = _run(stage, hops, ctx)
    d = stage.delay_hops
    assert d == 10
    # After warm-up every hop is the remote passthrough of the hop d earlier (PCM16 round trip).
    tail = range(100, 300)
    assert stage.stats()["source"] == "remote"
    assert stage.remote_hops >= 190
    for i in tail:
        np.testing.assert_allclose(out[i], hops[i - d], atol=1.5 / 32768)
    assert ctx.telemetry.tier == "focus_hq"
    stats = ctx.telemetry.extras["remote"]
    assert stats["connected"] and stats["rtt_ms"]["n"] > 100
    assert 0 < stats["rtt_ms"]["p50"] < 100
    assert ctx.telemetry.latency_ms["remote_rtt"] == stats["rtt_ms"]["p50"]
    stage.close(wait=True)


def test_suppression_and_target_forwarded(server):
    stage = RemoteStage(url=server.url, delay_ms=50)
    ctx = StageContext.default()
    assert stage.client.start().wait_connected(5)
    ctx.suppression = 0.8
    ctx.target.track_id, ctx.target.locked = 17, True
    _run(stage, _hops(60), ctx)
    deadline = time.time() + 2
    while time.time() < deadline:
        tel = stage.client.remote_telemetry or {}
        if tel.get("mode") == "focus_hq" and tel.get("target", {}).get("track_id") == 17:
            break
        time.sleep(0.05)
    assert tel["mode"] == "focus_hq"
    assert tel["target"] == {"track_id": 17, "confidence": 0.0, "locked": True}
    stage.close(wait=True)


def test_latency_preset_reaches_remote_session_live(server):
    session = Session(ServerConfig(pipelines={"natural": "passthrough"}), device="cpu")
    stage = RemoteStage(url=server.url, delay_ms=50)
    try:
        assert stage.client.start().wait_connected(5)
        for preset in ("speed", "accuracy", "balanced"):
            session.handle_text('{"type":"control","latency_preset":"' + preset + '"}')
            assert session.mode == "natural"
            _run(stage, _hops(60), session.ctx)
            sent = remote.REMOTE_PRESETS.get(preset, preset)  # forwarded as is (T-086)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if (stage.client.remote_telemetry or {}).get("latency_preset") == sent:
                    break
                time.sleep(0.01)
            assert stage.client.remote_telemetry["latency_preset"] == sent
    finally:
        stage.close(wait=True)
        session.close()


def test_isolation_and_profile_reach_remote_session_live(server):
    """T-110: the remote ladder runs with the local session's strict isolation and judge profile."""
    session = Session(ServerConfig(pipelines={"natural": "passthrough"}), device="cpu")
    stage = RemoteStage(url=server.url, delay_ms=50)
    try:
        assert stage.client.start().wait_connected(5)
        sent, send = [], stage.client.send_control
        stage.client.send_control = lambda **fields: sent.append(fields) or send(**fields)
        session.handle_text('{"type":"control","isolation":"strict"}')
        _run(stage, _hops(30), session.ctx)
        session.handle_text('{"type":"control","profile":"judge","suppression":70}')
        _run(stage, _hops(30), session.ctx)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            tel = stage.client.remote_telemetry or {}
            if tel.get("profile") == "judge":
                break
            time.sleep(0.01)
        assert tel["isolation"] == "strict"
        assert tel["profile"] == "judge"
        # the profile's defaults (suppression 100) land first, so every field is re-sent with it
        assert sent[-1]["profile"] == "judge" and sent[-1]["suppression"] == 70.0
        assert sent[-1]["isolation"] == "strict"
        assert sent[-1]["agc"] is False  # T-169: the remote leg never levels; the laptop does, once
        assert tel["agc"] is None
    finally:
        stage.close(wait=True)
        session.close()


class _FakeClient:
    """Records the control messages a RemoteStage sends (T-169)."""

    connected = True
    remote_telemetry = None

    def __init__(self):
        self.sent: list[dict] = []

    def send_control(self, **fields) -> bool:
        self.sent.append(fields)
        return True


@pytest.mark.parametrize("profile", [None, "judge"])
def test_control_sent_to_the_remote_disables_its_agc(profile):
    """T-169: under the judge profile both sessions used to run the output AGC (+12 dB on room
    noise at the remote, more on the laptop); the remote is told ``agc: false`` with the first
    control and again with every profile change (the remote applies the profile's defaults first)."""
    stage = RemoteStage(url="ws://127.0.0.1:9/ws/session")
    stage.client.close()
    stage.client = fake = _FakeClient()
    ctx = StageContext.default()
    if profile:
        ctx.telemetry.extras["profile"] = profile
    stage._forward_control(ctx)
    assert fake.sent[0]["agc"] is False
    ctx.telemetry.extras["profile"] = "judge" if profile is None else "team"
    stage._forward_control(ctx)
    assert fake.sent[-1]["agc"] is False and "profile" in fake.sent[-1]


def test_git_rev_falls_back_to_revision_file(tmp_path):
    from hearing.server.app import read_git_rev

    assert read_git_rev(tmp_path) == "unknown"
    (tmp_path / "REVISION").write_text("abc123-dirty\n")
    assert read_git_rev(tmp_path) == "abc123-dirty"


def test_down_remote_falls_back_undelayed_without_blocking():
    stage = RemoteStage(url=f"ws://127.0.0.1:{_free_port()}/ws/session", fallback="gain(db=-6)",
                        connect_timeout=0.5, retry_s=0.2)
    ctx = StageContext.default()
    hops = _hops(50)
    t0 = time.perf_counter()
    out = _run(stage, hops, ctx, pace_s=0)
    per_hop_ms = 1000 * (time.perf_counter() - t0) / len(hops)
    assert per_hop_ms < 5, per_hop_ms
    gain = np.float32(10 ** (-6 / 20))
    for x, y in zip(hops, out, strict=True):
        np.testing.assert_allclose(y, x * gain, atol=1e-6)
    assert ctx.telemetry.tier == "audio-only"
    time.sleep(0.3)
    assert stage.client.last_error and "connect" in stage.client.last_error
    stage.close(wait=True)


def test_no_url_plays_fallback(monkeypatch):
    monkeypatch.delenv(remote.ENV_URL, raising=False)
    stage = RemoteStage(fallback="passthrough")
    assert stage.client is None
    x = _hops(1)[0]
    np.testing.assert_array_equal(stage.process(AudioChunk(x, SR, 0, 0.0), StageContext.default()).samples, x)


@pytest.mark.slow  # > 2 s (T-098: keeps verify_fast under 60 s)
def test_remote_dies_then_returns():
    port = _free_port()
    stage = RemoteStage(url=f"ws://127.0.0.1:{port}/ws/session", delay_ms=50, stall_ms=300,
                        connect_timeout=0.5, retry_s=0.1)
    ctx = StageContext.default()
    with _Server(port):
        assert stage.client.start().wait_connected(5)
        _run(stage, _hops(80), ctx)
        assert stage.stats()["source"] == "remote"
    out = _run(stage, _hops(100, seed=1), ctx, pace_s=0.01)  # remote gone: local after the stall
    assert stage.stats()["source"] == "local" and ctx.telemetry.tier == "audio-only"
    assert all(np.isfinite(o).all() and len(o) == HOP for o in out)
    with _Server(port):
        deadline = time.time() + 5
        while stage.stats()["source"] != "remote":
            assert time.time() < deadline, stage.stats()
            _run(stage, _hops(10, seed=2), ctx)
        assert stage.client.connects >= 2
    stage.close(wait=True)


def test_session_uses_remote_for_focus_hq(server):
    config = ServerConfig(pipelines={"focus_hq": f'remote(url="{server.url}", delay_ms=50)'})
    session = Session(config, "cpu")
    session.handle_text('{"type":"control","mode":"focus_hq"}')
    assert session.telemetry()["pipeline"].startswith("remote(")


def test_probe_reports_rtt(server):
    result = probe(server.url, seconds=0.5)
    assert result["ok"] and result["health"]["ok"]
    assert result["rtt_ms"]["n"] >= 40 and result["rtt_ms"]["p95"] is not None
    assert result["lost"] == 0


def test_probe_unreachable():
    result = probe(f"ws://127.0.0.1:{_free_port()}/ws/session", seconds=0.1, connect_timeout=0.3)
    assert result["ok"] is False and "health_error" in result


def test_dropped_stage_closes_its_connection(server):
    import gc

    stage = RemoteStage(url=server.url)
    client = stage.client
    assert client.start().wait_connected(5)
    del stage
    gc.collect()
    deadline = time.time() + 3
    while client.connected:
        assert time.time() < deadline, "connection left open after the session dropped its stage"
        time.sleep(0.02)


def test_declared_latency_follows_active_path(monkeypatch):
    monkeypatch.delenv(remote.ENV_URL, raising=False)
    s = RemoteStage(fallback="passthrough", delay_ms=200)  # no url: local only
    assert s.latency_ms == 0.0
    s = RemoteStage(url=f"ws://127.0.0.1:{_free_port()}/ws/session", fallback="passthrough",
                    delay_ms=200, connect_timeout=0.5, retry_s=0.2)
    assert s.latency_ms == pytest.approx(200.0)  # remote expected until proven down
    ctx = StageContext.default()
    _run(s, _hops(5), ctx, pace_s=0)
    assert s.latency_ms == 0.0 and ctx.telemetry.latency_ms["remote_stage"] == 0.0
    s.close(wait=True)


def test_declared_latency_remote_path(server):
    stage = RemoteStage(url=server.url, delay_ms=100)
    ctx = StageContext.default()
    assert stage.client.start().wait_connected(5)
    _run(stage, _hops(60), ctx)
    assert stage.stats()["source"] == "remote"
    assert stage.latency_ms == pytest.approx(100.0)
    assert ctx.telemetry.latency_ms["remote_stage"] == pytest.approx(100.0)
    stage.close(wait=True)


def test_tier_names_the_remote_ladders_active_tier():
    """A remote ``tiers`` ladder reports what it plays (T-075); a single remote pipeline is ``tier``."""
    stage = RemoteStage(url="ws://127.0.0.1:9/ws/session", fallback="passthrough")
    ctx = StageContext.default()
    stage._source = "remote"
    stage.client.remote_telemetry = {"tier": "enrolled", "tiers": {"active": "enrolled"}}
    stage._report(ctx)
    assert ctx.telemetry.tier == "enrolled" and stage.stats()["remote_tier"] == "enrolled"
    stage.client.remote_telemetry = {"tier": "av"}  # no ladder on the remote
    stage._report(ctx)
    assert ctx.telemetry.tier == "focus_hq" and stage.stats()["remote_tier"] is None
    stage._source = "delayed"  # a late hop filled locally is not the remote ladder's audio
    stage.client.remote_telemetry = {"tiers": {"active": "enrolled"}}
    stage._report(ctx)
    assert ctx.telemetry.tier == "focus_hq"
    stage.close()


def test_remote_tiers_report_the_remote_ladder_and_the_stream_delay():
    """T-108: while remote audio plays, telemetry's ``tiers`` is the remote ladder's, delayed by
    buffer + the remote's own delay (not the local fallback ladder's 1.5 s)."""
    stage = RemoteStage(url=f"ws://127.0.0.1:{_free_port()}/ws/session", delay_ms=200,
                        connect_timeout=0.2, retry_s=10)
    try:
        assert stage.remote_tiers() is None  # nothing remote has played
        stage.client.remote_telemetry = {"tier": "enrolled", "latency_ms": {"algorithmic": 750.0},
                                         "tiers": {"active": "enrolled", "delay_ms": 750.0, "tiers": []}}
        stage._source = "remote"
        tiers = stage.remote_tiers()
        assert tiers["active"] == "enrolled" and tiers["remote"] is True
        assert tiers["delay_ms"] == pytest.approx(950.0) == stage.latency_ms
        assert tiers["remote_delay_ms"] == pytest.approx(750.0) and tiers["buffer_ms"] == 200.0
    finally:
        stage.close(wait=True)


def test_probe_runs_as_a_module():
    """``python -m hearing.server.remote`` (scripts/gpu_up.sh probe, demo.sh --judge) used to die
    with "Stage already registered: 'remote'", so the judge always fell back to the Mac (T-108)."""
    import json
    import subprocess
    import sys

    url = f"ws://127.0.0.1:{_free_port()}/ws/session"
    r = subprocess.run([sys.executable, "-m", "hearing.server.remote", "--url", url, "--seconds", "0.1",
                        "--timeout", "0.3"], capture_output=True, text=True, timeout=60, check=False)
    assert "already registered" not in r.stderr
    assert r.returncode == 1 and json.loads(r.stdout)["ok"] is False
