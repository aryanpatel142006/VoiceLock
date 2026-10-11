"""T-156: a session whose Focus HQ plays on the GPU reports the GPU's state in its telemetry: the
remote ladder's tier, ``remote`` stats, the real output latency and the remote's target lock, and
the local labels again once the remote drops."""

import json
import threading
import time

import numpy as np
from websockets.sync.server import serve

from hearing import registry
from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    pack_frame,
    unpack_frame,
)
from hearing.server.session import ServerConfig, Session, load_stage_modules

FRAME = 320  # 20 ms at 16 kHz, like the judge page
TICK_FRAMES = 5  # one telemetry tick per 100 ms of audio (ServerConfig.telemetry_hz = 10)
REMOTE_ALGORITHMIC_MS = 625.0
DELAY_MS = 200.0


class _LocalGate:
    """The local fallback in these tests: passes audio through and reports the ``gate`` tier."""

    name = "t156_gate"
    sample_rate = 16000
    latency_ms = 0.0

    def process(self, chunk, ctx):
        ctx.telemetry.tier = "gate"
        return chunk

    def reset(self):
        pass


if "t156_gate" not in registry.names():
    registry.register("t156_gate")(_LocalGate)


class LadderRemote:
    """A fake GPU session server running a tier ladder: echoes ``audio in`` (until ``mute``) and
    sends the remote session's ``telemetry`` (tiers.active, target, acquisition, tse, latency)
    ahead of the first reply and then every ``TICK_FRAMES`` frames."""

    def __init__(self, active: str = "enrolled"):
        self.active = active
        self.mute = threading.Event()
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws/session"

    def telemetry(self) -> dict:
        return {"type": "telemetry", "tier": self.active, "device": "cuda",
                "latency_ms": {"algorithmic": REMOTE_ALGORITHMIC_MS, "total": 640.0},
                "tiers": {"active": self.active, "delay_ms": REMOTE_ALGORITHMIC_MS, "tiers": []},
                "target": {"track_id": None, "confidence": 0.87, "locked": True},
                "acquisition": {"state": "locked", "since_s": 1.0, "detail": "known voice: aryan"},
                "tse": {"confidence": 0.93},
                "enrollment": {"state": "calibrated", "source": "profile", "profile": "aryan"},
                "faces": []}

    def _handle(self, ws):
        n = 0
        for msg in ws:
            if isinstance(msg, str):
                continue
            f = unpack_frame(msg)
            if f.kind != KIND_AUDIO_IN or self.mute.is_set():
                continue
            if n % TICK_FRAMES == 0:
                ws.send(json.dumps(self.telemetry()))
            n += 1
            ws.send(pack_frame(KIND_AUDIO_OUT, f.seq, f.capture_ts, f.payload))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)


class FakeMonitor:
    def snapshot(self):
        return {"ok": True, "rtt_ms": 9.0, "error": None}

    def close(self):
        pass


def gpu_session(url: str) -> Session:
    load_stage_modules()
    return Session(ServerConfig(pipelines={"focus_hq": "t156_gate"}, remote_url=url, hq_backend="remote",
                                faces="none", vision_threaded=False), "cpu")


def noise(seq: int) -> np.ndarray:
    x = np.random.default_rng(seq).standard_normal(FRAME).astype(np.float32) * 0.2
    return pcm16_to_float(float_to_pcm16(x))


def run(session: Session, seconds: float, start_seq: int = 0, stop=None) -> tuple[list[dict], int]:
    """Feed 20 ms frames in real time; one telemetry tick per ``TICK_FRAMES`` frames. Each tick is
    tagged with ``_replies``: audio frames the remote had answered when the tick was taken."""
    ticks, seq = [], start_seq
    for _ in range(round(seconds / 0.02)):
        t0 = time.perf_counter()
        session.handle_binary(pack_frame(KIND_AUDIO_IN, seq, seq * 0.02, float_to_pcm16(noise(seq))))
        seq += 1
        if seq % TICK_FRAMES == 0:
            stage = session._remote_stage()
            ticks.append({**session.telemetry(), "_replies": stage.remote_hops if stage else 0})
            if stop is not None and stop(ticks[-1]):
                break
        time.sleep(max(0.0, 0.02 - (time.perf_counter() - t0)))
    return ticks, seq


def test_gpu_session_reports_the_remote_tier_stats_latency_and_target_then_local_on_drop():
    with LadderRemote("enrolled") as fake:
        s = gpu_session(fake.url)
        s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
        ticks, seq = run(s, 2.0)
        first = next(i for i, t in enumerate(ticks) if t["_replies"] > 0)
        settled = ticks[first + 1:]  # within 2 ticks of the first remote reply: the 1st and every later one
        assert settled, "the remote never answered"
        for t in settled:
            assert t["tier"] == "enrolled", t["tier"]
            assert t["hq_active"] == "remote" and t["hq_reason"] == "remote GPU playing"
            assert t["remote"]["source"] == "remote" and t["remote"]["connected"] is True
            assert t["remote"]["rtt_ms"]["p50"] is not None
            assert t["remote"]["remote_hops"] > 0 and "misses" in t["remote"]
            assert t["remote"]["remote_device"] == "cuda" and t["remote"]["remote_tier"] == "enrolled"
            assert t["latency_ms"]["total"] >= DELAY_MS + REMOTE_ALGORITHMIC_MS
            assert t["latency_ms"]["remote_stage"] == DELAY_MS + REMOTE_ALGORITHMIC_MS
            assert t["latency_ms"]["remote_rtt"] == t["remote"]["rtt_ms"]["p50"]
            assert t["target"]["locked"] is True and t["target"]["confidence"] == 0.87
            assert t["target"]["source"] == "remote"
            assert t["acquisition"]["state"] == "locked" and t["acquisition"]["source"] == "remote"
            assert t["tse"]["confidence"] == 0.93
            assert t["tiers"]["active"] == "enrolled" and t["tiers"]["remote"] is True

        fake.mute.set()  # connected but silent: the stage hands over to the local fallback
        dropped, _ = run(s, 2.0, seq, stop=lambda t: t["remote"]["source"] == "local")
        t = dropped[-1]
        assert t["remote"]["source"] == "local", "the remote never dropped"
        assert t["tier"] == "gate" and t["hq_active"] == "local"
        assert t["latency_ms"]["remote_stage"] == 0.0 and t["latency_ms"]["total"] < DELAY_MS
        assert t["target"] == {"track_id": None, "confidence": 0.0, "locked": False}
        assert t["acquisition"].get("source") is None
        s.close()


def test_local_backend_has_no_remote_block():
    s = Session(ServerConfig(pipelines={"focus_hq": "passthrough"}, faces="none", vision_threaded=False), "cpu")
    t = s.telemetry()
    assert t["remote"] is None and "remote_stage" not in t["latency_ms"]
    s.close()


def test_hq_reason_says_when_no_audio_has_reached_the_gpu_session():
    """The 2026-10-09 judge sessions: remote pinned, but the page never streamed the microphone."""
    s = gpu_session("ws://127.0.0.1:9/ws/session")
    s._hq_monitor = FakeMonitor()
    t = s.telemetry()
    assert t["seq"] is None and t["hq_active"] == "remote"
    assert t["hq_reason"] == "remote selected; no microphone audio from the client yet"
    s.close()


def test_session_report_prints_the_gpu_tier_timeline(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "session_report", Path(__file__).resolve().parents[1] / "scripts" / "session_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)

    def tick(t, source, tier):
        return {"t": t, "dir": "out", "event": {
            "type": "telemetry", "seq": int(t * 50), "tier": tier, "mode": "focus_hq",
            "latency_ms": {"algorithmic": 825.0, "total": 845.0, "remote_stage": 825.0},
            "target": {"track_id": None, "confidence": 0.87, "locked": True, "source": "remote"},
            "remote": {"source": source, "remote_tier": "enrolled" if source == "remote" else None,
                       "rtt_ms": {"p50": 9.0, "p95": 14.0}, "remote_hops": 100, "misses": 2,
                       "remote_device": "cuda", "latency_ms": 825.0}}}

    rows = [tick(0.0, "remote", "enrolled"), tick(2.5, "remote", "enrolled"), tick(5.0, "local", "gate")]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report.main(["session_report", str(tmp_path)])
    out = capsys.readouterr().out
    assert "GPU: ticks with remote stats 3/3" in out and "GPU tier {'enrolled': 2}" in out
    assert '"gpu": "remote/enrolled"' in out and '"gpu": "local/None"' in out
    assert "no audio frame ever reached" not in out
