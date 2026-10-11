"""Port ownership and immutable server provenance regressions (no media devices)."""
import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hearing import __version__
from hearing.server.app import create_app
from hearing.server.session import ServerConfig
from scripts.session_report import main

ROOT = Path(__file__).resolve().parents[1]


def test_health_and_telemetry_share_startup_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_FACES", "none")
    before = datetime.now(UTC)
    with TestClient(create_app(ServerConfig(console_dir=tmp_path))) as client:
        health = client.get("/health").json()
        assert health["git_rev"] == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        assert before <= datetime.fromisoformat(health["started_at"]) <= datetime.now(UTC)
        assert health["pid"] == os.getpid()
        assert health["version"] == __version__
        assert client.get("/health").json()["started_at"] == health["started_at"]
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({"type": "hello", "sample_rate": 16000, "frame_ms": 20, "video": False})
            event = ws.receive_json()
            while event["type"] != "telemetry":
                event = ws.receive_json()
            for key in ("git_rev", "started_at", "version", "pid"):
                assert event[key] == health[key]


@pytest.mark.parametrize("kind", ["hello", "telemetry", "server"])
def test_report_header_identifies_recording_producer(tmp_path, capsys, kind):
    (tmp_path / "events.jsonl").write_text(json.dumps({
        "t": 0.0, "dir": "out", "event": {"type": kind, "git_rev": "abc123",
        "started_at": "2026-10-06T10:00:00+00:00", "version": "0.1.0"}
    }) + "\n")
    main(["session_report.py", str(tmp_path)])
    header = capsys.readouterr().out.split("## client")[0]
    assert "abc123" in header
    assert "2026-10-06T10:00:00+00:00" in header
    assert "0.1.0" in header


def run_demo(port, *args):
    return subprocess.run(["bash", "scripts/demo.sh", *args, "--no-open"], cwd=ROOT,
                          env={**os.environ, "PORT": str(port)}, text=True,
                          capture_output=True, timeout=30, check=False)


@pytest.mark.parametrize("action", ["--restart", "--stop"])
def test_demo_refuses_unrelated_listener(action):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = run_demo(port, action)
        assert result.returncode != 0
        assert "conflict" in result.stderr.lower()
        assert str(os.getpid()) in result.stderr
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass  # unrelated listener remains alive


@pytest.mark.slow
def test_demo_restart_replaces_pid_and_stop_releases_port(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_FACES", "none")
    monkeypatch.setenv("HEARING_ASR", "mock")
    monkeypatch.setenv("HEARING_TTS", "mock")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # Codex's macOS sandbox denies ps. Substitute only argv inspection, and only
    # for the real hearing PID returned by this test's ephemeral health endpoint.
    try:
        ps_blocked = subprocess.run(["ps", "-p", str(os.getpid()), "-o", "command="],
                                    capture_output=True, check=False).returncode != 0
    except PermissionError:
        ps_blocked = True
    if ps_blocked:
        shim = tmp_path / "ps"
        shim.write_text(f"#!{ROOT / '.venv/bin/python'}\n" + """
import json, os, sys, urllib.request
with urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health') as r:
    health = json.load(r)
if str(health['pid']) != sys.argv[2]:
    sys.exit(1)
print('Python -m hearing serve' if 'command=' in sys.argv else health['started_at'])
""")
        shim.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    args = ("--restart", "--", "--no-warm", "--pipelines", "natural=passthrough")
    # --no-open must precede --, which forwards all subsequent options to serve.
    def launch():
        return subprocess.run(["bash", "scripts/demo.sh", "--no-open", *args], cwd=ROOT,
                              env={**os.environ, "PORT": str(port)}, text=True,
                              capture_output=True, timeout=30, check=False)
    try:
        first = launch()
        assert first.returncode == 0, first.stdout + first.stderr
        second = launch()
        assert second.returncode == 0, second.stdout + second.stderr
        first_identity = next(l for l in first.stdout.splitlines() if l.startswith("server PID"))
        second_identity = next(l for l in second.stdout.splitlines() if l.startswith("server PID"))
        assert first_identity.split()[2] != second_identity.split()[2]
        assert first_identity.split("started_at ")[1] != second_identity.split("started_at ")[1]
        assert "git_rev" in second_identity and "version" in second_identity
    finally:
        stopped = run_demo(port, "--stop")
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
