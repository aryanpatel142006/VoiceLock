"""Where Focus HQ runs: ``control.hq_backend`` (local / remote / auto), the auto rule, the
switch crossfade, the remote-down fallback and camera frames forwarded to the remote."""

import io
import json
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from PIL import Image
from websockets.sync.server import serve

from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.server import remote
from hearing.server.app import create_app
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    KIND_VIDEO_IN,
    pack_frame,
    unpack_frame,
)
from hearing.server.remote import AutoBackend, RemoteStage, auto_rule
from hearing.server.session import ServerConfig, Session, load_stage_modules
from hearing.types import FaceTrack, StageContext, TargetState

FRAME = 320  # 20 ms at 16 kHz, like the console
LOCAL = "gain(db=-6)"  # the "local Focus HQ" in these tests: distinguishable from the remote echo
W, H = 640, 360


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EchoRemote:
    """A fake GPU session server: echoes ``audio in`` as ``audio out`` (until ``mute``), records
    control messages and video frames, and sends ``telemetry`` with scripted ``faces``."""

    def __init__(self):
        self.mute = threading.Event()
        self.controls: list[dict] = []
        self.videos: list = []
        self.faces: list[dict] = []
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws/session"

    def _handle(self, ws):
        for msg in ws:
            if isinstance(msg, str):
                self.controls.append(json.loads(msg))
                continue
            f = unpack_frame(msg)
            if f.kind == KIND_VIDEO_IN:
                self.videos.append(f)
                ws.send(json.dumps({"type": "telemetry", "faces": self.faces}))
            elif f.kind == KIND_AUDIO_IN and not self.mute.is_set():
                ws.send(pack_frame(KIND_AUDIO_OUT, f.seq, f.capture_ts, f.payload))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)


class RealRemote:
    """``hearing serve`` (focus_hq = passthrough) on loopback: real /health and /ws/session."""

    def __init__(self):
        self.port = _free_port()
        app = create_app(ServerConfig(pipelines={"focus_hq": "passthrough"}, faces="mock",
                                      asd="lipmotion"))
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                                    log_level="warning"))
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


def session_for(url: str | None, backend: str = "local", **config) -> Session:
    load_stage_modules()
    config.setdefault("faces", "none")
    return Session(ServerConfig(pipelines={"focus_hq": LOCAL}, remote_url=url, hq_backend=backend,
                                vision_threaded=False, **config), "cpu")


def control(session: Session, **fields) -> list:
    return session.handle_text(json.dumps({"type": "control", **fields}))


def noise(seq: int) -> np.ndarray:
    x = np.random.default_rng(seq).standard_normal(FRAME).astype(np.float32) * 0.2
    return pcm16_to_float(float_to_pcm16(x))


class Feeder:
    """Feeds 20 ms noise frames in (roughly) real time; keeps every reply's audio."""

    def __init__(self, session: Session):
        self.session, self.seq, self.out, self.events = session, 0, [], []

    def run(self, seconds: float, pace: float = 0.02, until=None) -> float:
        start = time.perf_counter()
        for _ in range(round(seconds / 0.02)):
            t0 = time.perf_counter()
            replies = self.session.handle_binary(
                pack_frame(KIND_AUDIO_IN, self.seq, self.seq * 0.02, float_to_pcm16(noise(self.seq))))
            self.seq += 1
            audio = [r for r in replies if isinstance(r, bytes)]
            assert len(audio) == 1, "every audio frame gets exactly one audio reply"
            self.out.append(pcm16_to_float(unpack_frame(audio[0]).payload))
            self.events += [r for r in replies if isinstance(r, dict)]
            if until is not None and until():
                return time.perf_counter() - start
            time.sleep(max(0.0, pace - (time.perf_counter() - t0)))
        return time.perf_counter() - start


def _dropouts(out: list[np.ndarray]) -> int:
    """10 ms blocks of digital silence in the output (the input is never silent)."""
    y = np.concatenate(out)
    blocks = y[: len(y) // 160 * 160].reshape(-1, 160)
    return int((np.abs(blocks).max(axis=1) < 1e-4).sum())


# ------------------------------------------------------------------ control message


def test_hq_backend_control_message_is_validated_and_echoed():
    s = session_for(None)
    t = s.telemetry()
    assert t["hq_backend"] == "local" and t["hq_active"] == "local" and t["remote_rtt_ms"] is None
    bad = control(s, hq_backend="gpu")
    assert bad[0]["type"] == "error" and bad[0]["stage"] == "protocol" and "hq_backend" in bad[0]["message"]
    replies = control(s, hq_backend="remote")  # no url: remote is impossible, say so
    err = next(r for r in replies if r["type"] == "error")
    assert err["stage"] == "hq_backend" and err["degraded_to"] == "local"
    assert "HEARING_REMOTE_URL" in err["message"]
    t = replies[-1]
    assert t["hq_backend"] == "remote" and t["hq_active"] == "local"
    control(s, mode="focus_hq")
    assert s.active_spec == LOCAL
    json.dumps(s.telemetry())


def test_config_default_and_env(monkeypatch):
    monkeypatch.delenv(remote.ENV_URL, raising=False)
    monkeypatch.delenv("HEARING_HQ_BACKEND", raising=False)
    cfg = ServerConfig()
    assert cfg.hq_backend == "auto" and cfg.hq_remote is None and "HEARING_REMOTE_URL" in cfg.hq_unavailable
    monkeypatch.setenv(remote.ENV_URL, "ws://gpu:8765/ws/session")
    monkeypatch.setenv("HEARING_HQ_BACKEND", "remote")
    cfg = ServerConfig(pipelines={"focus_hq": "tiers"})
    assert cfg.hq_backend == "remote"
    assert cfg.hq_remote.startswith('remote(url="ws://gpu:8765/ws/session", fallback="tiers"')
    assert "remote" in cfg.stage_names()
    # A focus_hq that is already remote is not wrapped twice.
    cfg = ServerConfig(pipelines={"focus_hq": 'remote(fallback="passthrough")'})
    assert cfg.hq_remote is None and "itself remote" in cfg.hq_unavailable
    with pytest.raises(ValueError, match="hq_backend"):
        ServerConfig(hq_backend="gpu")


# ------------------------------------------------------------------ auto rule


def test_auto_rule():
    assert auto_rule(True, 20.0) == (True, "remote healthy, RTT 20 ms < 150 ms")
    assert auto_rule(True, 150.0)[0] is False
    assert auto_rule(True, 149.9)[0] is True
    assert auto_rule(False, 5.0) == (False, "remote /health not green")
    assert auto_rule(None, 5.0)[0] is False
    assert auto_rule(True, None)[0] is False


def test_auto_hysteresis_drops_at_once_and_upgrades_after_a_while():
    a = AutoBackend(upgrade_after_s=2.0)
    assert a.update(True, 0.0) is False
    assert a.update(True, 1.9) is False
    assert a.update(True, 2.0) is True
    assert a.update(False, 2.1) is False  # first bad check: local
    assert a.update(True, 2.2) is False  # and the clock starts over
    assert a.update(True, 4.2) is True


class FakeMonitor:
    def __init__(self, ok=True, rtt=10.0):
        self.ok, self.rtt = ok, rtt

    def snapshot(self):
        return {"ok": self.ok, "rtt_ms": self.rtt, "error": None if self.ok else "refused"}

    def close(self):
        pass


def test_auto_switches_the_focus_hq_pipeline_by_health_and_rtt():
    s = session_for(f"ws://127.0.0.1:{_free_port()}/ws/session", "auto", hq_auto_upgrade_s=0.1)
    s._hq_monitor.close()
    s._hq_monitor = monitor = FakeMonitor(ok=True, rtt=40.0)
    control(s, mode="focus_hq")
    feed = Feeder(s)
    feed.run(0.3, pace=0.0)
    assert s.active_spec == LOCAL  # good, but not for upgrade_after_s yet (no wall time passed)
    time.sleep(0.15)
    feed.run(0.1, pace=0.0)
    assert s.active_spec == s.config.hq_remote
    t = s.telemetry()
    assert t["hq_backend"] == "auto" and t["pipeline"].startswith("remote(")
    assert t["remote_rtt_ms"] == 40.0
    monitor.rtt = 180.0  # too slow: straight back to local
    feed.run(0.1, pace=0.0)
    assert s.active_spec == LOCAL
    t = s.telemetry()
    assert t["hq_active"] == "local" and "180 ms >= 150 ms" in t["hq_reason"]
    monitor.rtt, monitor.ok = 10.0, False
    feed.run(0.1, pace=0.0)
    assert s.active_spec == LOCAL and "not green" in s.telemetry()["hq_reason"]
    s.close()


# ------------------------------------------------------------------ live switch, fallback


@pytest.mark.slow  # > 2 s (T-098: keeps verify_fast under 60 s)
def test_console_switch_changes_the_source_live_without_dropouts():
    with EchoRemote() as fake:
        s = session_for(fake.url, "local")
        control(s, mode="focus_hq", suppression=50)
        feed = Feeder(s)
        feed.run(0.4)
        assert s.telemetry()["hq_active"] == "local"
        control(s, hq_backend="remote")  # the console's GPU button
        feed.run(1.0)
        t = s.telemetry()
        assert t["hq_backend"] == "remote" and t["hq_active"] == "remote", t["hq_reason"]
        assert t["tier"] == "focus_hq" and 0 < t["remote_rtt_ms"] < 100
        stage = s._remote_stage()
        assert stage.remote_hops > 30 and stage.source == "remote"
        control(s, hq_backend="local")
        feed.run(0.4)
        t = s.telemetry()
        assert t["hq_active"] == "local" and t["pipeline"] == LOCAL
        # Every frame came back full length; the switches were crossfades, never silence.
        assert all(len(y) == FRAME for y in feed.out)
        assert _dropouts(feed.out[1:]) == 0
        # The remote echo is the input (rms 0.2), the local path is -6 dB (rms 0.1): both were heard.
        rms = [float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) for y in feed.out]
        assert any(abs(r - 0.2) < 0.03 for r in rms) and any(abs(r - 0.1) < 0.015 for r in rms)
        s.close()


def test_remote_down_degrades_to_local_within_one_second():
    with EchoRemote() as fake:
        s = session_for(fake.url, "remote")
        control(s, mode="focus_hq")
        feed = Feeder(s)
        feed.run(1.0)
        assert s.telemetry()["hq_active"] == "remote"
        fake.mute.set()  # connected but silent: the worst case (a dead socket is immediate)
        elapsed = feed.run(2.0, until=lambda: s.hq_status()["active"] == "local")
        assert s.hq_status()["active"] == "local"
        assert elapsed < 1.0, elapsed
        t = s.telemetry()
        assert "remote down" in t["hq_reason"] and t["tier"] == "audio-only"
        assert _dropouts(feed.out[1:]) == 0
        s.close()


# ------------------------------------------------------------------ video forwarding


def _jpeg(boxes) -> bytes:
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    for x, y, bw, bh in boxes:
        img[round(y * H) : round((y + bh) * H), round(x * W) : round((x + bw) * W)] = (0, 255, 0)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=90)
    return buf.getvalue()


LEFT, RIGHT = (0.05, 0.3, 0.16, 0.4), (0.6, 0.3, 0.16, 0.4)


@pytest.mark.slow  # > 2 s (T-098: keeps verify_fast under 60 s)
def test_video_frames_and_the_mapped_target_reach_the_remote():
    with EchoRemote() as fake:
        # The remote numbers the same two faces differently from the laptop.
        fake.faces = [{"track_id": 41, "box": list(LEFT), "p_speaking": 0.0},
                      {"track_id": 42, "box": list(RIGHT), "p_speaking": 0.0}]
        s = session_for(fake.url, "remote", faces="mock", asd="lipmotion")
        control(s, mode="focus_hq")
        feed = Feeder(s)
        for i in range(25):
            s.handle_binary(pack_frame(KIND_VIDEO_IN, i, i * 0.04, _jpeg([LEFT, RIGHT])))
            feed.run(0.04)
        local = {t.track_id: t.box for t in s.ctx.tracks}
        assert len(local) == 2
        right_id = min(local, key=lambda k: abs(local[k][0] - RIGHT[0]))
        control(s, target={"lock_track_id": right_id})
        for i in range(25, 40):
            s.handle_binary(pack_frame(KIND_VIDEO_IN, i, i * 0.04, _jpeg([LEFT, RIGHT])))
            feed.run(0.04)
        time.sleep(0.1)
        assert len(fake.videos) >= 20  # forwarded while the remote plays (minus pre-connect)
        assert fake.videos[-1].payload[:2] == b"\xff\xd8"  # JPEG as sent
        hello = next(c for c in fake.controls if c["type"] == "hello")
        assert hello["video"] is True
        locks = [c["target"] for c in fake.controls if c.get("target")]
        assert locks and locks[-1] == {"lock_track_id": 42}  # the remote's id for that face
        s.close()


def test_target_mapping_waits_for_remote_faces_and_falls_back_to_the_id_without_camera():
    stage = RemoteStage(url=f"ws://127.0.0.1:{_free_port()}/ws/session")
    ctx = StageContext.default()
    ctx.target = TargetState(7, 1.0, True, 0.0)
    assert stage._remote_target(ctx) == {"lock_track_id": 7}  # audio-only: ids are all we have
    ctx.tracks = [FaceTrack(7, RIGHT, None, 0.0, 0.0)]
    assert stage._remote_target(ctx) is remote._PENDING  # remote faces not seen yet
    stage.client.remote_telemetry = {"faces": [{"track_id": 3, "box": [0.0, 0.0, 0.1, 0.1]}]}
    assert stage._remote_target(ctx) is remote._PENDING  # no overlap: never lock a stranger
    stage.client.remote_telemetry["faces"].append({"track_id": 9, "box": [0.61, 0.31, 0.16, 0.4]})
    assert stage._remote_target(ctx) == {"lock_track_id": 9}
    ctx.target = TargetState(7, 1.0, False, 0.0)
    assert stage._remote_target(ctx) is None
    stage.close(wait=True)


def test_auto_with_a_real_remote_moves_focus_hq_to_the_gpu():
    with RealRemote() as server:
        s = session_for(server.url, "auto", hq_auto_upgrade_s=0.2)
        assert s._hq_monitor.wait_checked(3)
        control(s, mode="focus_hq")
        feed = Feeder(s)
        feed.run(3.0, until=lambda: s.hq_status()["active"] == "remote" and s._remote_stage() is not None
                 and s._remote_stage().source == "remote")
        t = s.telemetry()
        assert t["hq_active"] == "remote", t["hq_reason"]
        assert t["pipeline"].startswith("remote(") and t["remote_rtt_ms"] < 150
        s.close()


def test_health_monitor_reports_green_rtt_then_down():
    server = RealRemote()
    with server:
        monitor = remote.HealthMonitor(server.url, every_s=0.05, timeout_s=0.5).start()
        assert monitor.wait_checked(3)
        snap = monitor.snapshot()
        assert snap["ok"] is True and 0 < snap["rtt_ms"] < 150 and snap["error"] is None
    deadline = time.time() + 3
    while monitor.snapshot()["ok"]:
        assert time.time() < deadline, "monitor never noticed the server going away"
        time.sleep(0.05)
    snap = monitor.snapshot()
    assert snap["rtt_ms"] is None and snap["error"]
    monitor.close()


def test_ab_control_is_echoed_into_telemetry():
    s = session_for(None)
    assert s.telemetry()["ab"] == "processed"
    assert not [r for r in control(s, ab="raw") if r["type"] == "error"] and s.telemetry()["ab"] == "raw"
    bad = control(s, ab="both")
    assert bad[0]["type"] == "error" and bad[0]["stage"] == "protocol" and s.telemetry()["ab"] == "raw"
    control(s, ab="processed")
    assert s.telemetry()["ab"] == "processed"


def test_hq_reason_says_when_remote_replies_miss_the_playout_buffer(monkeypatch):
    """T-075: a connected remote whose RTT exceeds the buffer plays local fill, not the GPU."""
    s = session_for("ws://127.0.0.1:9/ws/session", "remote")

    class Late:
        source, remote_latency_ms = "delayed", 200.0

        def rtt_ms(self):
            return 4510.0

    monkeypatch.setattr(s, "_remote_stage", lambda playing=True: Late())
    status = s.hq_status()
    assert status["active"] == "remote" and status["reason"].startswith("remote late (RTT p50 4510 ms > 200 ms")
    s.close()


# ------------------------------------------------------------------ demo.sh --judge (T-109)


class SlowHealthRemote:
    """A GPU box whose ``/health`` takes ``delay_s`` to answer (iLab under load); echoes audio."""

    def __init__(self, delay_s: float):
        import asyncio

        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        app = FastAPI()

        @app.get("/health")
        async def health():
            await asyncio.sleep(delay_s)
            return {"ok": True, "device": "cuda"}

        @app.websocket("/ws/session")
        async def session(ws: WebSocket):
            await ws.accept()
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("bytes"):
                        f = unpack_frame(msg["bytes"])
                        await ws.send_bytes(pack_frame(KIND_AUDIO_OUT, f.seq, f.capture_ts, f.payload))
            except WebSocketDisconnect:
                return

        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                                    log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    url = RealRemote.url
    __enter__ = RealRemote.__enter__
    __exit__ = RealRemote.__exit__


@pytest.mark.slow  # ~5 s: the 4 s /health
def test_judge_probe_says_gpu_when_health_answers_slowly():
    with SlowHealthRemote(4.0) as server:
        t0 = time.perf_counter()
        ok, text = remote.judge_probe(server.url)
        assert ok, text
        assert text.startswith("RTT p50") and time.perf_counter() - t0 < 6.0  # one attempt, in budget


def test_judge_probe_reports_why_the_gpu_is_down():
    ok, text = remote.judge_probe(f"ws://127.0.0.1:{_free_port()}/ws/session", budget_s=3.0)
    assert not ok and "/health not ok" in text


@pytest.mark.slow  # ~13 s: the remote comes up 10 s in
def test_auto_switches_to_a_remote_that_comes_up_after_the_session_started():
    server = RealRemote()
    s = session_for(server.url, "auto")  # down now: no restart may be needed to pick it up later
    control(s, mode="focus_hq")
    feed = Feeder(s)
    feed.run(1.0)
    assert s.telemetry()["hq_active"] == "local"
    up = threading.Timer(9.0, server.__enter__)
    up.start()
    try:
        feed.run(9.0)
        up.join()
        t_up = time.perf_counter()
        feed.run(15.0, until=lambda: s.hq_status()["active"] == "remote")
        t = s.telemetry()
        assert t["hq_active"] == "remote", t["hq_reason"]
        assert time.perf_counter() - t_up < 15.0
        print(f"switched to remote {time.perf_counter() - t_up:.1f} s after it came up")
    finally:
        s.close()
        server.__exit__()
