import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "demo_noise", Path(__file__).resolve().parents[1] / "scripts" / "demo_noise.py")
dn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dn)

DEVS = [
    {"name": "MacBook Pro Microphone", "max_output_channels": 0},
    {"name": "MacBook Pro Speakers", "max_output_channels": 2},
    {"name": "External Headphones", "max_output_channels": 2},
    {"name": "JBL Flip 6", "max_output_channels": 2},
]


def test_resolve_by_name_and_index():
    assert dn.resolve_device("jbl", DEVS) == 3
    assert dn.resolve_device("2", DEVS) == 2
    with pytest.raises(SystemExit):
        dn.resolve_device("0", DEVS)  # input only
    with pytest.raises(SystemExit):
        dn.resolve_device("nothing", DEVS)


def test_ambiguous_name_rejected():
    with pytest.raises(SystemExit):
        dn.resolve_device("er", DEVS)


def test_refuses_engine_output():
    with pytest.raises(SystemExit, match="refusing"):
        dn.choose_devices("External", "External Headphones", DEVS, 1)
    with pytest.raises(SystemExit, match="refusing"):
        dn.choose_devices("Speakers", None, DEVS, 1)  # engine uses the default output
    assert dn.choose_devices("JBL", "External Headphones", DEVS, 1) == 3
    assert dn.choose_devices("JBL", None, DEVS, 1) == 3


def test_device_required():
    with pytest.raises(SystemExit):
        dn.choose_devices(None, None, DEVS, 1)


def test_args_defaults():
    a = dn.parse_args(["--device", "JBL", "--calibrate"])
    assert a.calibrate and a.source == "3" and a.level == -20.0


def test_player_keys_and_loop():
    src = {m: np.ones(100, np.float32) * 0.1 for m in "123"}
    p = dn.Player(src)
    assert p.key("1") and p.mode == "1"
    p.key("-")
    assert p.gain_db == -22
    p.key("+"), p.key("+"), p.key("+"), p.key("+"), p.key("+"), p.key("+"), p.key("+"), p.key("+")
    assert p.gain_db == -6 or p.gain_db <= -3
    p.key(" ")
    out = np.ones((160, 1), np.float32)
    p.callback(out, 160, None, None)
    assert not out.any()
    assert p.key("q") is False


def test_synthetic_sources_have_target_rms():
    b = dn.build_babble([], seconds=1)
    assert abs(float(np.sqrt(np.mean(b**2))) - 0.1) < 0.01
