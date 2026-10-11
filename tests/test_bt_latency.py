"""bt_latency (T-144): device refusal, dry run, JSON schema and the 300/350 ms bars, all on fake devices."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hearing.audio.devices import DeviceInfo
from scripts import bt_latency as bt

MAC_MIC = DeviceInfo(0, "MacBook Pro Microphone", 1, 0, 48000.0, "Core Audio")
MAC_SPK = DeviceInfo(1, "MacBook Pro Speakers", 0, 2, 48000.0, "Core Audio")
AIRPODS = DeviceInfo(2, "Aryan's AirPods Pro", 1, 2, 48000.0, "Core Audio")
DEVICES = [MAC_MIC, MAC_SPK, AIRPODS]


def _never_run(*a, **k):
    raise AssertionError("dry run must not start latency-test")


def test_dry_run_resolves_devices_and_prints_procedure(capsys, tmp_path):
    rc = bt.main(["--dry-run"], devices=lambda: DEVICES, run=_never_run, out_dir=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert "input  = MacBook Pro Microphone (#0)" in out and "output = Aryan's AirPods Pro (#2)" in out
    assert "ear detection OFF" in out and "2 cm" in out and "ANC OFF for this measurement only" in out
    assert "latency-test --json --in MacBook Pro Microphone --out Aryan's AirPods Pro --clicks 5" in out
    assert not list(tmp_path.iterdir())


def test_refuses_non_airpods_output(capsys):
    rc = bt.main(["--dry-run", "--out", "MacBook Pro Speakers"], devices=lambda: DEVICES, run=_never_run)
    out = capsys.readouterr().out
    assert rc == 2 and "REFUSED" in out and "not the AirPods" in out


def test_refuses_airpods_mic_and_missing_airpods():
    with pytest.raises(ValueError, match="HFP"):
        bt.check_devices("AirPods", "AirPods", DEVICES)
    with pytest.raises(ValueError, match="No output device"):
        bt.check_devices(bt.DEFAULT_IN, bt.DEFAULT_OUT, [MAC_MIC, MAC_SPK])


def _fake_run(per_run: list[list[float]]):
    calls = []

    def run(cmd, **kw):
        delays = per_run[len(calls)]
        calls.append(cmd)
        d = {"round_trip_ms": round(sorted(delays)[len(delays) // 2], 2) if delays else None,
             "delays_ms": delays, "clicks_sent": 5, "clicks_found": len(delays), "sample_rate": 16000}
        return SimpleNamespace(stdout="devices: log line\n" + json.dumps(d, indent=1), stderr="", returncode=0)

    run.calls = calls
    return run


def test_full_run_writes_json_schema_and_pass_row(capsys, tmp_path):
    run = _fake_run([[200, 210, 220, 230, 240]] * 3)
    rc = bt.main([], devices=lambda: DEVICES, run=run, out_dir=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0 and len(run.calls) == 3
    assert run.calls[0][-6:] == ["--out", "Aryan's AirPods Pro", "--clicks", "5", "--interval", "1"]
    (path,) = tmp_path.glob("bt_latency_*.json")
    data = json.loads(path.read_text())
    assert set(data) >= {"devices", "summary", "runs", "runs_requested", "clicks_per_run", "stamp"}
    s = data["summary"]
    assert set(s) >= {"median_ms", "range_ms", "p95_ms", "clicks_found", "clicks_sent", "pass", "bars"}
    assert s["median_ms"] == 220.0 and s["range_ms"] == [200.0, 240.0] and s["p95_ms"] == 240.0
    assert (s["clicks_found"], s["clicks_sent"], s["pass"]) == (15, 15, True)
    assert data["devices"]["output"] == "Aryan's AirPods Pro" and len(data["runs"]) == 3
    assert "PASS  median 220.0 ms (bar <= 300 ms)" in out and "PASS  p95 240.0 ms (bar <= 350 ms)" in out
    assert "| MacBook Pro Microphone | Aryan's AirPods Pro (A2DP, ANC off) | 15/15 | 220 | 200-240 | 240 | PASS |" in out


def test_bars_fail_on_slow_tail():
    s = bt.summarize([{"delays_ms": [280, 290, 295, 300, 420], "clicks_sent": 5, "clicks_found": 5}])
    assert s["median_pass"] and not s["p95_pass"] and not s["pass"]
    assert bt.verdict_lines(s)[1].startswith("FAIL  p95")
    s = bt.summarize([{"delays_ms": [310, 320, 330], "clicks_sent": 5, "clicks_found": 3}])
    assert not s["median_pass"] and bt.verdict_lines(s)[0].startswith("FAIL  median 320.0")


def test_no_clicks_prints_hint_and_fails(capsys, tmp_path):
    rc = bt.main(["--runs", "2"], devices=lambda: DEVICES, run=_fake_run([[], []]), out_dir=tmp_path)
    out = capsys.readouterr().out
    assert rc == 1 and "hint: no click detected" in out and "2 cm" in out
    assert "FAIL  median n/a" in out
    s = json.loads(next(tmp_path.glob("bt_latency_*.json")).read_text())["summary"]
    assert s["median_ms"] is None and s["clicks_found"] == 0 and s["pass"] is False
