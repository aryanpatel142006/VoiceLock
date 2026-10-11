"""T-170: the GPU's ``git_rev`` reaches ``telemetry.remote``, the laptop's ``/health``, the team pins and
the JUDGE line, with "A100 rev <x> is behind local <y>: rerun gpu_up" when the GPU runs older code."""

import json
import subprocess
import threading

from websockets.sync.server import serve

from hearing.server import app as app_module
from hearing.server import remote
from hearing.server import team as team_preset
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    pack_frame,
    unpack_frame,
)
from tests.test_remote_telemetry import gpu_session, run


def _repo(tmp_path) -> tuple[str, str]:
    """A two-commit git repo at ``tmp_path``: (older, newer) revs."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                       cwd=tmp_path, text=True).strip()
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "old")
    old = git("rev-parse", "HEAD")
    git("commit", "-q", "--allow-empty", "-m", "new")
    return old, git("rev-parse", "HEAD")


def test_rev_warning_only_when_the_remote_lacks_the_local_commit(tmp_path):
    old, new = _repo(tmp_path)
    warn = team_preset.rev_warning(new, old, tmp_path)
    assert warn == f"A100 rev {old[:7]} is behind local {new[:7]}: rerun gpu_up"
    assert team_preset.rev_warning(new, f"{old}-dirty", tmp_path) == warn  # REVISION of a dirty sync
    assert team_preset.rev_warning(new, new, tmp_path) is None
    assert team_preset.rev_warning(old, new, tmp_path) is None  # the GPU is ahead: fine
    assert team_preset.rev_warning(new, None, tmp_path) is None
    assert team_preset.rev_warning(new, "unknown", tmp_path) is None
    assert team_preset.rev_warning(new, "f" * 40, tmp_path) is None  # a commit this checkout lacks
    assert team_preset.rev_warning(new, old, tmp_path / "missing") is None  # not a git repo


class RevRemote:
    """A fake GPU session server: ``/health`` answers with ``git_rev``; the socket echoes audio."""

    def __init__(self, rev: str):
        self.rev = rev
        self.server = serve(self._handle, "127.0.0.1", 0, process_request=self._health)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws/session"

    def _health(self, connection, request):
        if request.path == "/health":
            return connection.respond(200, json.dumps({"ok": True, "git_rev": self.rev}))
        return None

    def _handle(self, ws):
        for msg in ws:
            if isinstance(msg, bytes):
                f = unpack_frame(msg)
                if f.kind == KIND_AUDIO_IN:
                    ws.send(pack_frame(KIND_AUDIO_OUT, f.seq, f.capture_ts, f.payload))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)


def test_telemetry_remote_carries_the_gpu_git_rev_and_the_stale_warning(tmp_path, monkeypatch):
    old, new = _repo(tmp_path)
    monkeypatch.setattr(team_preset, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(team_preset, "local_rev", lambda: new)
    with RevRemote(old) as fake:
        s = gpu_session(fake.url)
        try:
            s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
            ticks, _ = run(s, 1.5, stop=lambda t: (t["remote"] or {}).get("git_rev"))
            t = ticks[-1]
            assert t["remote"]["git_rev"] == old
            assert t["remote"]["rev_warning"] == f"A100 rev {old[:7]} is behind local {new[:7]}: rerun gpu_up"
            # the laptop's /health says the same
            ident = app_module.remote_identity(fake.url)
            assert ident["remote_git_rev"] == old and "rerun gpu_up" in ident["remote_rev_warning"]
            monkeypatch.setattr(team_preset, "local_rev", lambda: old)  # same code on both: silent
            assert s.telemetry()["remote"]["rev_warning"] is None
        finally:
            s.close()


def _fake_probe(rev: str):
    def probe(url, seconds, connect_timeout=5.0, health_timeout=None):
        return {"ok": True, "health": {"ok": True, "git_rev": rev}, "rtt_ms": {"p50": 12.0, "p95": 15.0}}
    return probe


def test_judge_line_warns_when_the_gpu_rev_is_behind_and_not_when_equal(tmp_path, monkeypatch):
    old, new = _repo(tmp_path)
    monkeypatch.setattr(team_preset, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(remote, "local_rev", lambda: new)
    monkeypatch.setattr(remote, "probe", _fake_probe(old))
    ok, text = remote.judge_probe("ws://gpu.invalid:1/ws/session")
    assert ok and text == f"RTT p50 12 ms, p95 15 ms · A100 rev {old[:7]} is behind local {new[:7]}: rerun gpu_up"
    monkeypatch.setattr(remote, "probe", _fake_probe(new))
    ok, text = remote.judge_probe("ws://gpu.invalid:1/ws/session")
    assert ok and text == "RTT p50 12 ms, p95 15 ms"
    monkeypatch.setattr(remote, "probe", lambda *a, **k: {"ok": False, "health": {"ok": True, "git_rev": old}})
    ok, text = remote.judge_probe("ws://gpu.invalid:1/ws/session", attempts=1)
    assert not ok and text.startswith("no audio back") and text.endswith("rerun gpu_up")


def test_team_pins_list_the_gpu_rev_and_warning():
    pins = team_preset.pins({}, ("remote", "ok"), True, "abc1234", "A100 rev abc1234 is behind local def5678: rerun gpu_up")
    assert pins["remote_git_rev"] == "abc1234" and pins["rev_warning"].endswith("rerun gpu_up")
    assert team_preset.pins({}, ("local", "x"), True)["rev_warning"] is None
