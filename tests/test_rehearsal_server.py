"""T-285: rehearsal API: live matrix, take recorder."""

from __future__ import annotations

import json
import os
import wave

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hearing.server import rehearsal
from hearing.tse.enrolled import GATED_PLAY_ENV


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv(GATED_PLAY_ENV, raising=False)
    app = FastAPI()
    rehearsal.mount(app, tmp_path)
    yield TestClient(app)
    os.environ.pop(GATED_PLAY_ENV, None)


def test_options_list_matrix_and_scripts(client):
    opts = client.get("/api/rehearsal/options").json()
    assert set(opts["policies"]) == {"stream", "duck", "passthrough"}
    assert {s["kind"] for s in opts["scripts"]} == {"alternating", "overlap", "interruption", "freeform"}


def test_policy_switch_is_live_and_maps_controls(client):
    r = client.post("/api/rehearsal/config", json={"policy": "stream", "ladder": "ultra", "suppression": 0.5}).json()
    assert r["gated_play"] == "stream" and os.environ[GATED_PLAY_ENV] == "stream"
    assert r["controls"] == {"latency_preset": "speed", "ab": "processed", "suppression": 50}
    r = client.post("/api/rehearsal/config", json={"policy": "passthrough"}).json()
    assert GATED_PLAY_ENV not in os.environ and r["controls"]["ab"] == "raw"
    assert r["matrix"]["ladder"] == "ultra"  # earlier choices persist
    assert client.get("/api/rehearsal/config").json()["matrix"]["policy"] == "passthrough"


@pytest.mark.parametrize("body", [{"policy": "nope"}, {"suppression": 1.5}, {"suppression": True}, [1]])
def test_bad_config_is_400(client, body):
    assert client.post("/api/rehearsal/config", json=body).status_code == 400
    assert client.post("/api/rehearsal/config", content=b"{").status_code == 400


def test_take_records_audio_events_and_wraps_wav(client, tmp_path):
    take = client.post("/api/rehearsal/take", json={"mode": "script", "scenario": "../evil x"}).json()
    assert take["id"].endswith("_script_evil-x")
    pcm = (np.arange(1600) % 100).astype("<i2").tobytes()
    base = f"/api/rehearsal/take/{take['id']}"
    assert client.post(f"{base}/audio?stream=mic", content=pcm).json() == {"samples": 1600}
    assert client.post(f"{base}/audio?stream=out&rate=48000", content=pcm + b"x").json() == {"samples": 1600}
    assert client.post(f"{base}/events", json=[{"kind": "telemetry", "t_ms": 1}, {"kind": "marker"}]).json() == {"events": 2}
    summary = client.post(f"{base}/stop").json()
    folder = tmp_path / take["id"]
    assert folder.parent == tmp_path and folder.is_dir()
    with wave.open(str(folder / "mic.wav")) as w:
        assert (w.getframerate(), w.getnframes(), w.getnchannels()) == (16000, 1600, 1)
    with wave.open(str(folder / "out.wav")) as w:
        assert w.getframerate() == 48000
    assert not list(folder.glob("*.pcm"))
    lines = [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in lines] == ["telemetry", "marker"]
    assert json.loads((folder / "take.json").read_text())["samples"] == summary["samples"]
    assert summary["matrix"]["policy"] == "stream"
    assert client.post(f"{base}/stop").status_code == 404  # closed takes are forgotten


def test_take_errors(client):
    assert client.post("/api/rehearsal/take/nope/audio", content=b"ab").status_code == 404
    take = client.post("/api/rehearsal/take", json={}).json()
    base = f"/api/rehearsal/take/{take['id']}"
    assert client.post(f"{base}/events", json={"a": 1}).status_code == 400
    assert client.post(f"{base}/audio?rate=5", content=b"abcd").status_code == 409


def test_two_takes_in_one_second_get_distinct_folders(tmp_path):
    a = rehearsal.Take(tmp_path, "m", "s", {}, clock=lambda: 1000.0)
    b = rehearsal.Take(tmp_path, "m", "s", {}, clock=lambda: 1000.0)
    assert a.dir != b.dir


def test_default_matrix_is_applied_before_first_take(client):
    config = client.get("/api/rehearsal/config").json()
    assert config["controls"] == {"latency_preset": "balanced", "ab": "processed", "suppression": 80}
    assert config["gated_play"] == "stream"
    assert os.environ[GATED_PLAY_ENV] == "stream"


def test_create_app_mounts_rehearsal_options(monkeypatch):
    from hearing.server.app import create_app

    monkeypatch.delenv(GATED_PLAY_ENV, raising=False)
    with TestClient(create_app()) as client:
        assert GATED_PLAY_ENV not in os.environ  # ordinary server startup is not a rehearsal
        response = client.get("/api/rehearsal/options")
        assert response.status_code == 200
        assert response.json()["default"]["policy"] == "stream"


def test_interrupted_take_keeps_metadata_and_rates(tmp_path):
    take = rehearsal.Take(tmp_path, "freeform", "freeform", dict(rehearsal.DEFAULT_MATRIX))
    metadata = json.loads((take.dir / "take.json").read_text())
    assert metadata["stopped"] is None
    take.add_audio("mic", b"\x00\x00" * 480, rate=48000)
    metadata = json.loads((take.dir / "take.json").read_text())
    assert metadata["sample_rates"] == {"mic": 48000}
    assert metadata["samples"] == {"mic": 480}
    take.stop()


def test_phone_session_outputs_audio_at_declared_rate(monkeypatch):
    from hearing.server.app import create_app
    from hearing.server.protocol import (
        KIND_AUDIO_IN,
        KIND_AUDIO_OUT,
        pack_frame,
        unpack_frame,
    )
    from hearing.server.session import ServerConfig

    monkeypatch.delenv(GATED_PLAY_ENV, raising=False)
    config = ServerConfig(pipelines={"natural": "passthrough"})
    pcm = np.zeros(960, dtype="<i2").tobytes()  # 20 ms at the phone's 48 kHz
    with TestClient(create_app(config)) as client, client.websocket_connect("/ws/session") as ws:
        ws.send_json({"type": "hello", "sample_rate": 48000, "frame_ms": 20, "video": False,
                      "client": "rehearsal"})
        ws.receive_json()
        ws.send_bytes(pack_frame(KIND_AUDIO_IN, 1, 0.0, pcm))
        output = unpack_frame(ws.receive_bytes())
        assert output.kind == KIND_AUDIO_OUT
        assert len(output.payload) == 960 * 2  # output retains the declared 48 kHz, not 16 kHz
