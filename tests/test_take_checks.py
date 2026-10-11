"""T-172: target-absent row and the listening log."""

from __future__ import annotations

import numpy as np

from scripts import take_checks as tc

SR = 16000


def talker(seconds=10.0, level=0.1):
    t = np.arange(round(seconds * SR)) / SR
    return (level * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def test_suppressed_output_reads_low_and_no_wrong_speaker():
    x = talker()
    r = tc.target_absent(x, np.concatenate([np.zeros(SR, np.float32), 0.001 * x]), delay_ms=1000)
    assert r["blocks"] == 20 and r["out_re_in_db"] < -50 and r["wrong_share"] == 0.0


def test_leaking_output_is_wrong_speaker():
    x = talker()
    r = tc.target_absent(x, 0.5 * x, delay_ms=0)
    assert abs(r["out_re_in_db"] + 6.0) < 0.1 and r["wrong_share"] == 1.0
    assert "wrong-speaker share 100 %" in tc.absent_row(r)


def test_silent_stretch_is_not_measured():
    z = np.zeros(SR * 5, np.float32)
    assert "NOT MEASURED" in tc.absent_row(tc.target_absent(z, z))


def test_listening_log(tmp_path):
    answers = iter(["y", "n", "maybe", "n", "y"] + ["n"] * 4)
    path = tc.listening_log(["key2", "absent"], lambda _: next(answers), tmp_path, stamp="x")
    lines = path.read_text().splitlines()
    assert path.name == "listening_x.md"
    assert lines[-2] == "| key2 | yes | no | no | yes |" and lines[-1] == "| absent | no | no | no | no |"
