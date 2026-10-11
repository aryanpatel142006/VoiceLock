"""T-135 (Q-350): separation SI-SDRi is scored on the output before the judge profile's AGC."""

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np

from hearing.server.tap import SessionTap

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tap_writes_the_pre_agc_output(tmp_path):
    tap = SessionTap(tmp_path)
    tap.pre_agc(np.arange(320, dtype="<i2").tobytes())
    tap.close()
    with wave.open(str(tap.dir / "out_pre_agc.wav")) as w:
        assert w.getnframes() == 320 and w.getframerate() == 16000


def test_regression_is_judged_on_separation_before_leveling():
    dr = _load("demo_regress")
    assert dr.separation({"si_sdri_db": 4.0, "si_sdri_pre_agc_db": 8.9}) == 8.9
    assert dr.separation({"si_sdri_db": 4.0}) == 4.0
    base = dr.baseline([{"scores": [{"mode": "focus_hq", "si_sdri_db": 4.0, "si_sdri_pre_agc_db": 9.0,
                                     "true_dropout_s": 0.0}]}])
    assert base["scores"][0]["si_sdri_db"] == 9.0
    now = [{"mode": "focus_hq", "si_sdri_db": 4.0, "si_sdri_pre_agc_db": 5.0, "true_dropout_s": 0.0}]
    msg = dr.regressions(now, base)
    assert msg and "separation SI-SDRi, before output leveling" in msg[0]


def test_pre_agc_sdri_needs_a_matching_file(tmp_path):
    ss = _load("session_score")
    assert ss.pre_agc_sdri(tmp_path, np.zeros(16000), 16000, None, []) is None
    with wave.open(str(tmp_path / "out_pre_agc.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(8000, dtype="<i2").tobytes())
    assert ss.pre_agc_sdri(tmp_path, np.zeros(16000), 16000, None, []) is None  # does not cover out.wav


def test_pause_lift_compares_out_with_pre_agc_where_the_target_is_silent():
    """T-179: the room in target-silent frames, out.wav over out_pre_agc.wav, in dB."""
    ss = _load("session_score")
    e_pre = np.array([1.0, 1e-4, 1e-4, 1.0])
    e_out = np.array([8.0, 2e-4, 2e-4, 8.0])  # speech +9 dB, pauses +3 dB
    silent = np.array([False, True, True, False])
    assert abs(ss.pause_lift(e_out, e_pre, silent) - 10 * np.log10(2)) < 1e-6
    assert ss.pause_lift(e_out, e_pre, np.zeros(4, bool)) is None
    assert ss.pause_lift(e_out, np.zeros(4), silent) is None  # digital silence before the AGC
