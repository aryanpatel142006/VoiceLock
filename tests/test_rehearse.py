"""rehearse (T-147, T-157): --dry-run --no-hw on a fake session dir writes the sheet; empty cells print NOT MEASURED;
the human bars (over-talk, Space, Name Call, profile swap, mic report-only) and review findings F1-F5, Q-425, Q-430."""
from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from scripts import rehearse as rh
from scripts.judge_check import TEXT_WER_BAR

STAMP = "20261010T150000Z-1"
T0 = rh.session_start(rh.Path(STAMP))
SR = 16000
REAL_QUIET_ON_JUDGE = rh.quiet_on_judge


@pytest.fixture(autouse=True)
def _quiet_on_judge(monkeypatch):
    """T-149/T-150 merged: Quiet is on /judge (tests that need it off say so)."""
    monkeypatch.setattr(rh, "quiet_on_judge", lambda: True)
    monkeypatch.setenv("HEARING_PROFILES_DIR", "")  # recorded, so profiles_env()'s export is undone after the test
    monkeypatch.delenv("HEARING_PROFILES_DIR")


def _never(*a, **k):
    raise AssertionError("a dry run must not wait for Enter")


def _write_wav(path, x, sr=SR):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _tick(t, state, *, enroll="collecting", ab="processed", det=None, hq="local", delay=1500, tier="enrolled",
          profile=None, seq=1, remote=None):
    ev = {"type": "telemetry", "seq": seq, "acquisition": {"state": state},
          "enrollment": {"state": enroll, "profile": profile}, "ab": ab, "hq_active": hq,
          "tiers": {"active": tier, "delay_ms": delay}}
    if det is not None:
        ev["name_call"] = {"detections": det}
    if remote is not None:
        ev["remote"] = remote
    return {"t": t, "dir": "out", "event": ev}


def _ctl(t, **fields):
    return {"t": t, "dir": "in", "event": {"type": "control", **fields}}


def _sine(f, s, a=0.1):
    return a * np.sin(2 * np.pi * f * np.arange(int(SR * s)) / SR)


def fake_session(root, *, audio=True, captions=True, seq=1, output="Aryan's AirPods Pro", dets=(1, 2)):
    d = root / STAMP
    d.mkdir(parents=True)
    rows = [{"t": 0.0, "dir": "in", "event": {"type": "hello", "client": "judge", "input_device": rh.MIC,
                                               "output_device": output}}]
    rows += [_tick(0.1 * k, "searching", seq=seq) for k in range(10)]
    rows += [_tick(1.0 + 0.1 * k, "candidate", enroll="ready", seq=seq) for k in range(20)]
    rows += [_tick(3.0 + 0.1 * k, "locked", enroll="calibrated", seq=seq) for k in range(10)]
    # key 1 at 4.0: the gate plays for a moment (delay 0), then the Enrolled tier with aryan at 4.6 (0.6 s)
    rows += [_ctl(4.0, enroll={"action": "profile", "name": "aryan"}),
             _tick(4.2, "locked", tier="gate", delay=0, profile="aryan", seq=seq),
             _tick(4.6, "locked", profile="aryan", seq=seq)]
    rows += [_tick(5.0 + k, "locked", ab=("raw" if k % 2 else "processed"), seq=seq) for k in range(4)]
    rows += [_ctl(5.95, ab="raw"), _ctl(6.9, ab="processed"), _ctl(7.98, ab="raw")]
    rows += [_tick(9.0 + k, "locked", det=n, seq=seq) for k, n in enumerate(dets)]
    if captions:
        rows += [{"t": 11.0, "dir": "out", "event": {"type": "caption", "text": "see you at noon", "final": True}},
                 {"t": 11.5, "dir": "out", "event": {"type": "caption", "text": "see", "final": False}}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in sorted(rows, key=lambda r: r["t"])) + "\n")
    if audio:
        _write_wav(d / "in.wav", _sine(220, 1.0))
    return d


def _lines(sheet):
    return {line.split(" | ")[0].lstrip("| "): line for line in sheet.splitlines() if line.startswith("| ")}


def _row(rows, prefix):
    return next(r for r in rows if r[0].startswith(prefix))


def test_dry_run_no_hw_writes_the_sheet(tmp_path, capsys):
    d = fake_session(tmp_path / "sessions")
    ref = tmp_path / "ref.txt"
    ref.write_text("see you at noon")
    lat = tmp_path / "bt_latency_x.json"
    lat.write_text(json.dumps({"summary": {"median_ms": 240.0, "p95_ms": 380.0}}))
    out = tmp_path / "out"
    rc = rh.main(["--dry-run", "--no-hw", "--session", str(d), "--caption-ref", str(ref), "--latency-json", str(lat),
                  "--out", str(out)], ask=_never)
    sheet = next(out.glob("rehearsal_*.md")).read_text()
    printed = capsys.readouterr().out
    assert "Step 1/12: Devices and preflight" in printed and "Step 12/12: Sign the sheet" in printed
    assert rc == 1  # p95 380 > 350
    lines = _lines(sheet)
    assert "| 240 ms | <= 300 ms | PASS |" in lines["Round trip median (MacBook mic -> AirPods)"]
    assert "| 380 ms | <= 350 ms | FAIL |" in lines["Round trip p95"]
    assert "| 3.00 s | <= 8 s (Mac) | PASS |" in lines["Seconds to Isolated (Joseph alone)"]
    assert "| 1.00 s |" in lines["Enrollment ready"]
    assert "searching > candidate > locked" in lines["Pill changes"]
    assert "| 4 " in lines["Raw/Selective flips (Space x3)"] and lines["Raw/Selective flips (Space x3)"].endswith("PASS |")
    assert "| 50, 100, 20 ms | report-only | INFO |" in lines["A/B switch latency per flip"]
    assert "| 2/4 | >= 2/4 | PASS |" in lines["Name Call hits / attempts (4 clean calls)"]
    assert "| 0 | 0 expected | PASS |" in lines["Name Call false triggers (detections beyond the calls)"]
    swap = lines["Re-learn: profile swap (1/2) -> Enrolled with the new profile"]
    assert "worst 0.60 s (per swap s: 0.6)" in swap and "<= 1.5 s each (T-155)" in swap and swap.endswith("PASS |")
    assert "| 0.000 |" in lines["Caption WER vs the script"] and lines["Caption WER vs the script"].endswith("PASS |")
    # mic peak / rms are report-only: the number is printed, never PASS/FAIL against -48..-2
    assert "| -20.0 dBFS |" in lines["Mic peak (in.wav)"] and lines["Mic peak (in.wav)"].endswith("INFO |")
    assert lines["Mic RMS (in.wav)"].endswith("INFO |")
    assert "| local |" in lines["HQ backend playing"]
    # never measured in a dry run: printed as such, never a default
    for item in ("Worklet underruns", "Bud out and back: audio kept playing", "judge_check device rows",
                 "Isolated while Aryan talks over Joseph", "GPU RTT p50",
                 "Wrong speaker during the over-talk (ECAPA vs profiles, out.wav)"):
        assert lines[item].endswith(f"| {rh.NOT_MEASURED} |"), lines[item]
    assert "Overall: **FAIL**" in sheet


def test_empty_session_prints_not_measured_everywhere(tmp_path):
    d = tmp_path / STAMP
    d.mkdir()
    (d / "events.jsonl").write_text("")
    out = tmp_path / "out"
    rc = rh.main(["--dry-run", "--no-hw", "--session", str(d), "--out", str(out)], ask=_never)
    sheet = next(out.glob("rehearsal_*.md")).read_text()
    rows = [line for line in sheet.splitlines() if line.startswith("| ") and not line.startswith("| item")]
    assert rows
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[1] == rh.NOT_MEASURED and cells[3] == rh.NOT_MEASURED, line
    assert rc == 0 and "Overall: **INCOMPLETE**" in sheet


def test_step_windows_scope_the_numbers(tmp_path):
    d = fake_session(tmp_path, audio=False, captions=False)
    marks = {"isolate": T0 + 1.0, "relearn": T0 + 3.9, "overtalk": T0 + 4.4, "space": T0 + 5.0,
             "namecall": T0 + 8.5, "captions": T0 + 10.5}
    m = rh.measure([d], marks, "see you")
    assert m["isolated_s"] == 2.0  # from the "Joseph alone" Enter, not from Start
    assert m["overtalk_locked"] == 1.0
    assert m["name_calls"] == 1  # detections 1 -> 2 inside the step
    assert m["swaps"] == [0.6]
    assert m["caption_words"] is None and m["caption_wer"] is None  # nothing recorded in the captions step
    m = rh.measure([d], {"captions": T0 + 9.5}, "see you")
    assert m["caption_words"] == 0 and m["caption_wer"] == 1.0  # recorded, but no caption: measured, FAIL
    assert m["rms_db"] is None and m["peak_db"] is None


def test_gpu_bar_and_live_steps_wait_for_enter(tmp_path, monkeypatch):
    d = fake_session(tmp_path / "s")
    rows = rh.sheet_rows(rh.measure([d], {}, None), None, None, {}, gpu=True)
    iso = _row(rows, "Seconds to Isolated")
    assert iso[2] == "<= 6 s (GPU)" and iso[3] == "PASS"
    calls = []
    monkeypatch.setattr(rh, "auto_devices", lambda no_hw: calls.append(("devices", no_hw)))
    monkeypatch.setattr(rh, "auto_server", lambda port, no_hw, gpu=False: calls.append(("server", port)))
    monkeypatch.setattr(rh, "auto_judge", lambda port: [("PASS", "Demo mic in use (MacBook Pro Microphone)", "ok")])
    monkeypatch.setattr(rh.subprocess, "run", lambda *a, **k: calls.append(("run", a[0])))
    monkeypatch.setenv("HEARING_SESSION_LOG", str(tmp_path / "none"))
    answers = iter(["", "", "", "", "", "", "", "", "", "", "", "y", ""])
    rc = rh.main(["--no-hw", "--port", "0", "--out", str(tmp_path / "o")], ask=lambda _p: next(answers))
    sheet = next((tmp_path / "o").glob("rehearsal_*.md")).read_text()
    assert ("devices", True) in calls and ("server", 0) in calls and not any(c[0] == "run" for c in calls)
    assert "| Bud out and back: audio kept playing | yes | yes | PASS |" in sheet
    assert "| judge_check: Demo mic in use (MacBook Pro Microphone) | ok |  | PASS |" in sheet
    assert rc == 0


def test_f1_server_restart_sets_the_profiles_dir(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.delenv("HEARING_HQ_BACKEND", raising=False)
    monkeypatch.setattr(rh.subprocess, "run", lambda cmd, **k: seen.update(cmd=cmd, env=k["env"]))
    monkeypatch.setattr("scripts.judge_check.wait_warm", lambda port: None)
    rh.auto_server(0, no_hw=True)
    assert seen["env"]["HEARING_PROFILES_DIR"] == str(rh.ROOT / "eval" / "fixtures" / "generated" / "profiles")
    monkeypatch.setenv("HEARING_PROFILES_DIR", str(tmp_path))
    rh.auto_server(0, no_hw=True)
    assert seen["env"]["HEARING_PROFILES_DIR"] == str(tmp_path) and "--team" in seen["cmd"]
    assert "HEARING_HQ_BACKEND" not in seen["env"]  # T-181: the Mac unless --gpu
    rh.auto_server(0, no_hw=True, gpu=True)
    assert seen["env"]["HEARING_HQ_BACKEND"] == "remote"


def test_f2_hq_delay_only_while_the_hq_tier_plays(tmp_path):
    d = tmp_path / STAMP
    d.mkdir()
    rows = [_tick(0.1 * k, "searching", tier="gate", delay=0.0) for k in range(30)]
    rows += [_tick(3.0 + 0.1 * k, "locked", delay=1500) for k in range(10)]
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = rh.measure([d], {}, None)
    assert m["hq_delay_ms"] == 1500.0  # the gate's 0.0 (30 ticks of 40) no longer wins the median
    row = _row(rh.sheet_rows(m, None, None, {}, None), "HQ delay")
    assert row[1] == "1500 ms (per tier: gate 0 ms, enrolled 1500 ms)"
    # while the GPU plays, its tier and delay count (T-156 telemetry.remote)
    gpu = {"source": "remote", "remote_tier": "enrolled", "delay_ms": 900, "rtt_ms": {"p50": 42.0},
           "remote_device": "cuda"}
    rows = [_tick(0.1 * k, "locked", tier="gate", delay=0, remote=gpu) for k in range(5)]
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = rh.measure([d], {}, None)
    assert m["hq_delay_ms"] == 900 and m["gpu"] and m["gpu_rtt_ms"] == 42.0
    rows = rh.sheet_rows(m, None, None, {}, None)
    assert _row(rows, "GPU tier / device")[1] == "enrolled / cuda"
    assert _row(rows, "GPU played")[1] == "100 %" and _row(rows, "Seconds to Isolated")[2] == "<= 6 s (GPU)"


def test_f3_already_locked_at_the_mark_is_not_a_pass(tmp_path):
    d = fake_session(tmp_path)
    m = rh.measure([d], {"isolate": T0 + 3.5}, None)
    assert m["isolated_already"] and m["isolated_s"] is None
    iso = _row(rh.sheet_rows(m, None, None, {}, None), "Seconds to Isolated")
    assert iso[1] == "already locked at mark" and iso[3] == rh.NOT_MEASURED
    m = rh.measure([d], {"isolate": T0 + 3.5}, None)
    assert m["enroll_ready_s"] is not None and m["enroll_ready_s"] >= 0  # ready at the first tick after the mark
    m = rh.measure([d], {}, None)  # no mark: from Start (hello at 0.0), enrollment ready 1.0 s
    assert m["enroll_ready_s"] == 1.0
    late_hello = tmp_path / "late" / STAMP
    late_hello.parent.mkdir()
    late_hello.mkdir()
    rows = [_tick(0.5, "candidate", enroll="ready"),
            {"t": 1.0, "dir": "in", "event": {"type": "hello", "input_device": rh.MIC}}]
    (late_hello / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert rh.measure([late_hello], {}, None)["enroll_ready_s"] == -0.5  # not clamped to 0


@pytest.mark.parametrize("label, verdict", [("Default - AirPods Pro", "FAIL"), ("System default", "FAIL"),
                                            ("Aryan's AirPods Pro", "PASS"), ("MacBook Pro Speakers", "FAIL")])
def test_f4_default_output_fails(tmp_path, label, verdict):
    d = fake_session(tmp_path, output=label)
    row = _row(rh.sheet_rows(rh.measure([d], {}, None), None, None, {}, None), "Output in use")
    assert row[1] == label and row[3] == verdict


def test_f5_step_text(monkeypatch):
    by_key = {s.key: s for s in rh.STEPS}
    assert "Shift+N" in by_key["namecall"].say and "4 times" in by_key["namecall"].say
    assert "press 1" in by_key["relearn"].say and "press 2" in by_key["relearn"].say
    assert "Shift+C" not in by_key["relearn"].say


def test_f5_captions_not_measured_without_quiet_on_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "quiet_on_judge", lambda: False)
    d = fake_session(tmp_path)
    m = rh.measure([d], {}, "see you at noon")
    assert m["caption_words"] is None and m["caption_wer"] is None
    rows = rh.sheet_rows(m, None, None, {}, None)
    assert _row(rows, "Caption WER")[3] == rh.NOT_MEASURED


def test_quiet_on_judge_reads_judge_js(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "ROOT", tmp_path)
    js = tmp_path / "web" / "console" / "judge.js"
    js.parent.mkdir(parents=True)
    js.write_text('if (key === " ") {}')
    assert not REAL_QUIET_ON_JUDGE()
    js.write_text('} else if (key === "q") { setQuiet(); }')
    assert REAL_QUIET_ON_JUDGE()


def test_q425_no_audio_reached_the_engine(tmp_path, capsys):
    d = fake_session(tmp_path / "s", seq=None)
    out = tmp_path / "out"
    rc = rh.main(["--dry-run", "--no-hw", "--session", str(d), "--out", str(out)], ask=_never)
    sheet = next(out.glob("rehearsal_*.md")).read_text()
    assert rh.NO_AUDIO in capsys.readouterr().out and rh.NO_AUDIO in sheet
    assert "| FAIL |" not in sheet and rc == 0
    lines = _lines(sheet)
    for item in ("Seconds to Isolated (Joseph alone)", "Raw/Selective flips (Space x3)", "Mic peak (in.wav)",
                 "Name Call hits / attempts (4 clean calls)", "Isolated while Aryan talks over Joseph"):
        assert lines[item].endswith(f"| {rh.NOT_MEASURED} |"), lines[item]
    assert lines["Mic in use (page hello)"].endswith("| PASS |")  # the page's hello is not audio


def test_q430_captions_use_judge_checks_bar():
    assert TEXT_WER_BAR == 0.90 and not hasattr(rh, "CAPTION_WER_BAR")
    row = _row(rh.sheet_rows({"caption_wer": 0.7}, None, None, {}, None), "Caption WER")
    assert row[2].startswith("<= 0.9") and row[3] == "PASS"  # judge_check passes 0.7, so must the sheet


def test_bars_overtalk_space_namecall_swap():
    rows = rh.sheet_rows({"overtalk_locked": 0.89, "space_toggles": 2, "name_calls": 6, "swaps": [0.4, float("inf")]},
                         None, None, {}, None)
    assert _row(rows, "Isolated while Aryan")[1:] == ("89 %", ">= 90 % of ticks", "FAIL")
    assert _row(rows, "Raw/Selective flips")[1:] == ("2", ">= 3", "FAIL")
    assert _row(rows, "Name Call hits")[1] == "4/4" and _row(rows, "Name Call hits")[3] == "PASS"
    assert _row(rows, "Name Call false")[1:] == ("2", "0 expected", "FAIL")
    swap = _row(rows, "Re-learn")
    assert swap[1] == "worst never (per swap s: 0.4, never)" and swap[3] == "FAIL"
    rows = rh.sheet_rows({"name_calls": 1}, None, None, {}, None)
    assert _row(rows, "Name Call hits")[1:] == ("1/4", ">= 2/4", "FAIL")


def test_swap_needs_the_new_profile_in_the_enrolled_tier():
    ticks = [dict(_tick(t, "locked", tier=tier, profile=p)["event"], w=t)
             for t, tier, p in ((0.5, "enrolled", "joseph"), (1.0, "gate", "aryan"), (2.0, "enrolled", "aryan"))]
    controls = [{"w": 0.2, "enroll": {"action": "profile", "name": "aryan"}}]
    assert rh.swap_latencies(controls, ticks) == [1.8]  # over the 1.5 s bar
    assert rh.swap_latencies([{"w": 0.0, "ab": "raw"}], ticks) is None


def test_wrong_speaker_share_from_out_wav(tmp_path):
    prof = tmp_path / "profiles"
    for name, f in (("joseph", 220), ("aryan", 440)):
        (prof / name).mkdir(parents=True)
        _write_wav(prof / name / "a.wav", _sine(f, 2.0))
    d = tmp_path / STAMP
    d.mkdir()
    heard = np.concatenate([_sine(220, 3.0), _sine(440, 1.0), np.zeros(SR)])  # 3 s Joseph, 1 s Aryan, 1 s silence
    _write_wav(d / "out.wav", heard)
    rows = []
    for k in range(50):  # 100 ms frames, tapped as they were sent
        rows.append({"t": 0.1 * k, "dir": "out", "event": {"type": "binary", "kind": 3, "bytes": 3200}})
        rows.append(_tick(0.1 * k + 0.05, "locked", profile="joseph"))
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def embed(x):
        spec = np.abs(np.fft.rfft(x, n=SR))
        return np.array([spec[220], spec[440]])

    m = rh.measure([d], {"overtalk": T0}, None, embed=embed, profiles_root=prof)
    assert m["wrong_speaker"] == pytest.approx(0.25)  # silence is skipped
    row = _row(rh.sheet_rows(m, None, None, {}, None), "Wrong speaker")
    assert row[1:] == ("25.0 %", "report-only (Mac measured 6-25 %, best 8.6-9.1 %)", "INFO")
    assert rh.measure([d], {"overtalk": T0}, None, embed=embed, profiles_root=tmp_path / "none")["wrong_speaker"] is None
