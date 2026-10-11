"""T-114: verify_full's demo_regress exit-2 handling, judge --check --no-hw, demo_regress --no-baseline."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import demo_regress as dr
from scripts import judge_check as jc

ROOT = Path(__file__).resolve().parents[1]


def fixture_missing(log_text: str, tmp_path: Path) -> bool:
    """Run verify_full.sh's own ``fixture_missing`` shell function on a demo_regress log."""
    script = (ROOT / "scripts/verify_full.sh").read_text()
    fn = script[script.index("fixture_missing() {"):]
    fn = fn[: fn.index("\n}\n") + 3]
    log = tmp_path / "demo.log"
    log.write_text(log_text)
    return subprocess.run(["bash", "-c", f'{fn}\nfixture_missing "$1"', "_", str(log)], check=False).returncode == 0


@pytest.mark.parametrize(("log", "skipped"), [
    ("no demo clip at /x/clip.mp4 (nor in the main checkout /m): run .venv/bin/python -m eval.make_fixtures --demo\n", True),
    ("!! no demo manifest at /x/manifest.json (nor in /m); run eval.make_fixtures --demo (clip /x/clip.mp4)\n", True),
    ("Traceback (most recent call last):\n  ...\n!! demo_regress crashed: RuntimeError: boom\n", False),
    ("server on :5000\n!! focus_hq: no session recorded\n!! 1 mode(s) recorded no session\n", False),
    ("", False),
])
def test_verify_full_skips_exit_2_only_for_a_missing_clip_or_manifest(tmp_path, log, skipped):
    assert fixture_missing(log, tmp_path) is skipped


def test_verify_full_fails_other_exit_2():
    s = (ROOT / "scripts/verify_full.sh").read_text()
    assert '[ "$drc" -eq 2 ] && fixture_missing "$dlog"' in s


def test_no_baseline_runs_never_become_the_baseline(tmp_path):
    def write(stamp, sdri, **extra):
        (tmp_path / f"demo_regress_{stamp}.json").write_text(json.dumps(
            {"variant": "fast:natural:12:lock", "clip": "/d/demo-mf-easy/clip.mp4",
             "scores": [{"mode": "natural", "si_sdri_db": sdri, "true_dropout_s": 0.0}], **extra}))

    write("20261001-000000", 5.0)
    write("20261002-000000", -9.0, no_baseline=True)  # a doc / dry-run replay
    runs = dr.previous(tmp_path, "fast:natural:12:lock", "now", clip="demo-mf-easy")
    assert len(runs) == 1
    assert dr.baseline(runs)["scores"][0]["si_sdri_db"] == 5.0


@pytest.mark.parametrize(("value", "on"), [("1", True), ("true", True), ("", False), ("0", False)])
def test_no_baseline_env(monkeypatch, value, on):
    monkeypatch.setenv("HEARING_REGRESS_NO_BASELINE", value)
    assert dr.no_baseline_env() is on


def test_judge_check_no_hw_skips_camera_latency_and_gpu(tmp_path, monkeypatch, capsys):
    calls = {}

    def no_hw(*_a, **_k):
        raise AssertionError("--no-hw must not touch hardware or the GPU")

    def regress(backend, no_baseline=False):
        calls.update(backend=backend, no_baseline=no_baseline)
        return [("PASS", "Natural on the clip", "ok"), ("PASS", "Focus HQ on the clip", "ok")]

    monkeypatch.setattr(jc, "probe_gpu", no_hw)
    monkeypatch.setattr(jc, "check_latency", no_hw)
    monkeypatch.setattr(jc, "check_camera", no_hw)
    monkeypatch.setattr(jc, "check_flow", lambda *a: ("PASS", "lock->Isolated 5.0 s", []))
    real_devices = jc.check_devices  # not whatever demo server is up on :8000
    monkeypatch.setattr(jc, "check_devices", lambda url, expect=None, **kw: real_devices(url, expect, fetch=lambda u: None, **kw))
    monkeypatch.setattr(jc, "check_regress", regress)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    assert jc.main(["--no-hw", "--clip", str(clip)]) == 0
    out = capsys.readouterr().out
    assert calls == {"backend": "mac", "no_baseline": True}
    hw = [line for line in out.splitlines() if "skipped (no hardware)" in line]
    assert len(hw) == 3 and all(line.startswith("SKIP") for line in hw)
    assert "GPU probe" in out and "FAIL" not in out


def test_demo_sh_forwards_no_hw_to_the_judge_check():
    s = (ROOT / "scripts/demo.sh").read_text()
    assert "--no-hw) NO_HW=1 ;;" in s
    assert '[ "$NO_HW" = 1 ] && CHECK_ARGS+=(--no-hw)' in s  # T-130: forwarded with --team too
    assert '[ "$TEAM" = 1 ] && CHECK_ARGS+=(--team)' in s
    assert 'exec "$PY" scripts/judge_check.py ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}' in s
