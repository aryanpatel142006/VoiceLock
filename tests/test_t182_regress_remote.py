"""T-182: demo_regress --latency-preset / --strict-remote / --pre-agc, /health sessions_active."""
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hearing.server.app import create_app
from hearing.server.session import ServerConfig
from scripts import demo_regress as dr
from scripts import session_score as ss
from tests.test_session_score import _scene, _ticks

LSOF = """COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
ssh     41000 me    5u  IPv4 0x1      0t0  TCP 127.0.0.1:18765 (LISTEN)
Google  52000 me   30u  IPv4 0x2      0t0  TCP 127.0.0.1:50000->127.0.0.1:18765 (ESTABLISHED)
python  52111 me   12u  IPv4 0x3      0t0  TCP 127.0.0.1:50001->127.0.0.1:18765 (ESTABLISHED)
python  52111 me   13u  IPv4 0x4      0t0  TCP 127.0.0.1:50002->127.0.0.1:18765 (ESTABLISHED)
"""


def test_lsof_pids_drops_own_and_duplicates():
    assert dr.lsof_pids(LSOF, own_pid=41000) == [52000, 52111]
    assert dr.lsof_pids("") == []


def test_warning_names_the_pids():
    w = dr.tunnel_clients_warning([52000, 52111], 18765)
    assert "52000, 52111" in w and "18765" in w
    assert dr.tunnel_clients_warning([], 18765) is None


def test_tunnel_port_from_url():
    assert dr.tunnel_port("ws://127.0.0.1:18765/ws/session") == 18765
    assert dr.tunnel_port("wss://gpu.example/ws/session") == dr.TUNNEL_PORT


def test_flags_parse(monkeypatch):
    with pytest.raises(SystemExit):
        dr.run(["--latency-preset", "turbo"])
    monkeypatch.setattr(dr, "other_tunnel_clients", lambda port: [52000])
    monkeypatch.setenv("HEARING_REMOTE_URL", "ws://127.0.0.1:18765/ws/session")
    monkeypatch.setattr(dr.ss, "main_fallback", lambda p: p)
    monkeypatch.setattr(dr.ss, "resolve_stems", lambda *a: (_ for _ in ()).throw(FileNotFoundError("x")))
    # a missing clip stops before the warning; the abort path needs the clip, so test the strict flag via parsing only
    assert dr.run(["--strict-remote", "--pre-agc", "--latency-preset", "speed", "--clip", "/nonexistent.mp4"]) == 2


def test_health_sessions_active(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_FACES", "none")
    with TestClient(create_app(ServerConfig(console_dir=tmp_path))) as client:
        assert client.get("/health").json()["sessions_active"] == 0
        with client.websocket_connect("/ws/session"):
            assert client.get("/health").json()["sessions_active"] == 1
        assert client.get("/health").json()["sessions_active"] == 0


def _wav(path, y):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


def test_pre_agc_scores_every_metric_on_the_pre_agc_file(tmp_path):
    import json

    st, x, y, *_ = _scene(gain_o=0.1)
    bad = y + 0.5 * _scene(gain_o=1.0)[3]  # a leakier "leveled" out.wav
    _wav(tmp_path / "in.wav", x)
    _wav(tmp_path / "out.wav", bad / 4)
    _wav(tmp_path / "out_pre_agc.wav", y)
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(t) for t in _ticks(13.0)) + "\n")
    normal = ss.score_session(tmp_path, st, dnsmos="none")
    pre = ss.score_session(tmp_path, st, dnsmos="none", pre_agc=True)
    assert pre.leak_db == pytest.approx(-20.0, abs=1.0)
    assert pre.si_sdri_db > normal.si_sdri_db + 3
