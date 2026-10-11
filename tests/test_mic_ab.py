"""T-171: iPhone-over-USB mic A/B (scripts/mic_ab.py), the preset's input_device override and judge_check's label row."""

from __future__ import annotations

import datetime as dt
import json
import wave

import numpy as np

from hearing.server import team
from scripts import judge_check as jc
from scripts import mic_ab
from scripts import rehearse as rh

IPHONE = "Aryan patel's iPhone Microphone"
FAKE = f"{IPHONE},MacBook Pro Microphone,AirPods Pro"
SR = 16000


# ---------------------------------------------------------------- label override


def test_env_overrides_the_preset_mic_label(monkeypatch):
    monkeypatch.delenv(team.INPUT_ENV, raising=False)
    assert team.input_device({"input_device": "MacBook Pro Microphone"}) == ("MacBook Pro Microphone", "config/demo-team.yaml")
    assert team.input_device({}) == (team.DEMO_INPUT, "default")
    monkeypatch.setenv(team.INPUT_ENV, f"  {IPHONE} ")
    assert team.input_device({"input_device": "MacBook Pro Microphone"}) == (IPHONE, team.INPUT_ENV)
    assert team.demo_devices({"input_device": "MacBook Pro Microphone"})["input"] == IPHONE
    assert team.pins({"input_device": "MacBook Pro Microphone"}, ("local", "x"), True)["input_device"] == IPHONE


def test_judge_check_passes_the_iphone_label_without_an_hfp_warning(monkeypatch):
    monkeypatch.delenv("HEARING_EXPECT_MIC", raising=False)
    monkeypatch.setenv(team.INPUT_ENV, IPHONE)
    devs = {"client": "judge", "input": IPHONE, "output": "AirPods Pro", "missing": []}
    rows = {c: (v, d) for v, c, d in jc.check_devices("http://x", fetch=lambda url: devs)}
    assert rows[f"Demo mic in use ({IPHONE})"][0] == "PASS"
    assert rows["Mic is not a Bluetooth (HFP) mic"][0] == "PASS"
    assert rows[jc.LABEL_CRIT] == ("PASS", f"{IPHONE!r} (from {team.INPUT_ENV})")
    assert rows[jc.IPHONE_CRIT][0] == "SKIP" and "screen off" in rows[jc.IPHONE_CRIT][1]
    assert not any(v == "FAIL" for v, _ in rows.values())


def test_label_row_fails_a_bluetooth_label_and_shows_the_default(monkeypatch):
    monkeypatch.delenv("HEARING_EXPECT_MIC", raising=False)
    monkeypatch.setenv(team.INPUT_ENV, "AirPods Pro")
    assert jc.label_rows()[0][0] == "FAIL"
    monkeypatch.delenv(team.INPUT_ENV)
    rows = jc.label_rows()
    assert rows == [("PASS", jc.LABEL_CRIT, "'MacBook Pro Microphone' (from config/demo-team.yaml)")]


# ---------------------------------------------------------------- gate


def _rows(delays=(250.0,) * 5, peak=-20.0, wrong=0.05):
    return [{"delays": list(delays), "peak_db": peak, "wrong": wrong} for _ in range(3)]


def test_gate_passes_only_when_every_bar_holds():
    ok, line = mic_ab.gate({"iphone": _rows(), "macbook": _rows(wrong=0.2)})
    assert ok and line.startswith("VERDICT: iPhone over USB PASSES")
    cases = [({"iphone": _rows(delays=(320.0,) * 5)}, "median 320 > 300"),
             ({"iphone": _rows(delays=(250, 250, 250, 250, 360))}, "p95 360 > 350"),
             ({"iphone": _rows(peak=-1.0)}, "peak outside"),
             ({"iphone": _rows(peak=-50.0)}, "peak outside"),
             ({"iphone": _rows(wrong=0.2)}, "is not lower than MacBook"),
             ({"iphone": _rows(delays=())}, "round trip not measured")]
    for over, why in cases:
        ok, line = mic_ab.gate({"macbook": _rows(wrong=0.2), **over})
        assert not ok and line.startswith("VERDICT: FALLBACK = MacBook Pro Microphone, target at 40 cm") and why in line
    assert not mic_ab.gate({"iphone": _rows()})[0]  # no MacBook share to compare: fallback


def test_refuses_a_bluetooth_mic_and_picks_the_iphone_by_name():
    devices = mic_ab.fake_devices(FAKE)
    assert mic_ab.pick_mics(devices, None, "MacBook Pro Microphone") == {"iphone": IPHONE, "macbook": "MacBook Pro Microphone"}
    assert mic_ab.pick_mics(devices, "nope", "MacBook Pro Microphone")["iphone"] is None
    assert mic_ab.refuse_bluetooth("AirPods Pro") and mic_ab.refuse_bluetooth(IPHONE) is None


# ---------------------------------------------------------------- dry run


def test_dry_run_no_hw_writes_the_sheet_and_the_listening_log(tmp_path):
    rc = mic_ab.main(["--dry-run", "--no-hw", "--fake-devices", FAKE, "--results", str(tmp_path)],
                     ask=lambda m: (_ for _ in ()).throw(AssertionError("dry run asked")),
                     sleep=lambda s: (_ for _ in ()).throw(AssertionError("dry run slept")))
    assert rc == 1  # nothing measured: the verdict is the fallback
    sheet = next(tmp_path.glob("mic_ab_*.md")).read_text()
    assert sum(ln.startswith("| iphone | ") and ln.split("|")[2].strip().isdigit() for ln in sheet.splitlines()) == 3
    assert sum(ln.startswith("| macbook | ") and ln.split("|")[2].strip().isdigit() for ln in sheet.splitlines()) == 3
    assert "## Target absent" in sheet and "dropout" in sheet and "VERDICT: FALLBACK" in sheet
    listen = next(p for p in tmp_path.glob("listening_*.md") if not p.name.endswith("_key.md")).read_text()
    assert "wrong person (y/n) | leak (y/n) | robotic (y/n) | delay (y/n)" in listen
    assert "| B | ANC on, no app | - |" in listen
    key = next(tmp_path.glob("listening_*_key.md")).read_text()
    assert all(w in key for w in ("Raw", "Natural", "Focus")) and all(f"| {c} |  |  |" in listen for c in "ABC")
    saved = json.loads(next(tmp_path.glob("mic_ab_*.json")).read_text())
    assert saved["marks"]["labels"]["iphone"] == IPHONE


def test_dry_run_prints_the_recorded_demo_line_with_the_profiles(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HEARING_SESSION_LOG", raising=False)
    monkeypatch.delenv("HEARING_PROFILES_DIR", raising=False)
    mic_ab.main(["--dry-run", "--no-hw", "--fake-devices", FAKE, "--results", str(tmp_path)])
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines() if "scripts/demo.sh" in ln]
    assert len(lines) == 2
    for ln, label in zip(lines, (IPHONE, "MacBook Pro Microphone"), strict=True):
        assert ln == ("HEARING_SESSION_LOG=results/sessions HEARING_PROFILES_DIR=$PWD/eval/fixtures/generated/profiles "
                      f'bash scripts/demo.sh --judge --team --mic "{label}" --restart')


def test_scoring_without_a_session_dir_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEARING_SESSION_LOG", str(tmp_path / "empty"))  # main() exports both: restore them after
    monkeypatch.setenv("HEARING_PROFILES_DIR", str(tmp_path / "profiles"))
    rc = mic_ab.main(["--no-hw", "--fake-devices", FAKE, "--results", str(tmp_path / "r")],
                     ask=lambda m: "", sleep=lambda s: None)
    saved = next((tmp_path / "r").glob("mic_ab_*.json"))
    assert not list((tmp_path / "r").glob("mic_ab_*.md"))
    rc2 = mic_ab.main(["--from", str(saved), "--results", str(tmp_path / "r2")])
    rc3 = mic_ab.main(["--from", str(saved), "--session", str(tmp_path / "gone"), "--results", str(tmp_path / "r2")])
    out = capsys.readouterr().out
    assert rc == rc2 == rc3 == 3
    assert out.count("REFUSED to score: no session dir under") == 2 and "HEARING_SESSION_LOG=" in out
    assert f"session dir missing or without events.jsonl: {tmp_path / 'gone'}" in out
    assert not (tmp_path / "r2").exists() or not list((tmp_path / "r2").glob("mic_ab_*.md"))


# ---------------------------------------------------------------- numbers from a recorded session


def _wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _session(root, x_in, x_out, gap_at=None, gap_s=0.0):
    """A tap session: one 10 ms in/out binary frame per hop, in.wav / out.wav; returns (dir, wall-clock start)."""
    start = dt.datetime(2026, 10, 9, 12, 0, 0, tzinfo=dt.UTC)
    d = root / f"{start.strftime('%Y%m%dT%H%M%SZ')}-1"
    d.mkdir()
    _wav(d / "in.wav", x_in)
    _wav(d / "out.wav", x_out)
    rows, t = [], 0.0
    for k in range(len(x_in) // 160):
        if gap_at is not None and k == gap_at:
            t += gap_s
        rows.append({"t": t, "dir": "in", "event": {"type": "binary", "kind": 1, "seq": k, "bytes": 320}})
        rows.append({"t": t, "dir": "out", "event": {"type": "binary", "kind": 3, "seq": k, "bytes": 320}})
        t += 0.01
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d, start.timestamp()


def test_take_numbers_read_level_from_the_solo_stretches_and_leak_from_the_overlap(tmp_path, monkeypatch):
    n = int(25 * SR)
    x = np.full(n, 0.01, np.float32)
    x[int(2 * SR)] = 0.5  # peak while the target talks alone (-6 dBFS)
    x[int(10 * SR)] = 0.9  # louder peak in the overlap: not the target's level
    d, t0 = _session(tmp_path, x, x)
    seen = {}

    def windows(parts, target, **kw):
        seen["s"] = sum(len(p) for p in parts) / SR
        return [("aryan", -20.0)] * 8 + [("joseph", -32.0)] * 2

    monkeypatch.setattr(rh, "speaker_windows", windows)
    r = mic_ab.take_numbers([d], t0, "aryan")
    assert abs(r["peak_db"] - 20 * np.log10(0.5)) < 0.1
    assert abs(r["wrong"] - 0.2) < 1e-9 and abs(r["leak_db"] + 12.0) < 1e-9
    assert 9.9 <= seen["s"] <= 10.1  # only the 10 s overlap is scored


def test_absent_numbers_and_dropouts(tmp_path, monkeypatch):
    n = int(25 * SR)
    x_in = np.full(n, 0.1, np.float32)
    x_out = np.full(n, 0.01, np.float32)
    x_in[int(3 * SR):int(3.3 * SR)] = 0.0  # 300 ms of digital silence (a Continuity pause)
    d, t0 = _session(tmp_path, x_in, x_out, gap_at=2000, gap_s=0.25)
    monkeypatch.setattr(rh, "speaker_windows", lambda parts, target, **kw: [("joseph", -40.0)] * 3)
    a = mic_ab.absent_numbers([d], t0, "aryan")
    assert abs(a["out_rel_in_db"] + 20.0) < 0.2 and abs(a["wrong"] - 0.3) < 1e-9
    drop = mic_ab.dropout_numbers([d], t0, t0 + 30)
    assert 250 <= drop["gap_ms"] <= 270 and 295 <= drop["silence_ms"] <= 305 and drop["fps_median"] == 100
    m = {"labels": {}, "rows": {}, "dropout": drop}
    assert "| FAIL |" in mic_ab.render(m, "x")
