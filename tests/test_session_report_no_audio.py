"""T-158: session_report warns when a judge page sat connected without ever streaming its mic."""

import importlib.util
import json
from pathlib import Path


def _report():
    spec = importlib.util.spec_from_file_location(
        "session_report", Path(__file__).resolve().parents[1] / "scripts" / "session_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _session(tmp_path, seconds, audio_frames=0):
    rows = [{"t": 0.01, "dir": "in", "event": {"type": "hello", "client": "judge", "devices_missing": []}}]
    rows += [{"t": i * 0.02, "dir": "in", "event": {"type": "binary", "kind": 1, "seq": i}} for i in range(audio_frames)]
    rows += [{"t": float(t), "dir": "out", "event": {"type": "telemetry", "seq": None, "tier": "gate"}}
             for t in range(seconds + 1)]
    rows.sort(key=lambda r: r["t"])
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_warns_on_a_15_s_judge_session_with_no_audio_frames(tmp_path, capsys):
    _report().main(["session_report", str(_session(tmp_path, 15))])
    out = capsys.readouterr().out
    assert "audio in frames 0" in out
    assert "WARNING: 0 audio frames in" in out and "client judge" in out


def test_no_warning_when_audio_streamed_or_session_short(tmp_path, capsys):
    report = _report()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    report.main(["session_report", str(_session(tmp_path / "a", 15, audio_frames=50))])
    report.main(["session_report", str(_session(tmp_path / "b", 5))])
    assert "WARNING: 0 audio frames in" not in capsys.readouterr().out
