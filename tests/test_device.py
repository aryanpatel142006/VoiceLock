import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_import_does_not_load_optional_ml_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import hearing.device; "
                "assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("override", ["cpu", "cuda", "mps"])
def test_explicit_override_needs_no_torch(monkeypatch, override):
    from hearing import device

    monkeypatch.setenv("HEARING_DEVICE", override)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert device.pick() == override


def test_bad_override_is_actionable(monkeypatch):
    from hearing import device

    monkeypatch.setenv("HEARING_DEVICE", "gpu")
    with pytest.raises(ValueError, match="HEARING_DEVICE"):
        device.pick()


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_automatic_device_priority(monkeypatch, cuda, mps, expected):
    from hearing import device

    monkeypatch.delenv("HEARING_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        ),
    )
    assert device.pick() == expected


def test_missing_optional_packages_report_cpu_and_null(monkeypatch):
    from hearing import device

    monkeypatch.delenv("HEARING_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert device.pick() == "cpu"
    info = device.describe()
    assert info["device"] == "cpu"
    assert info["torch"] is None
    assert info["onnxruntime_providers"] is None
    assert all(
        key in info
        for key in ["platform", "machine", "hostname", "python", "cpu_count"]
    )
    json.dumps(info)


def test_describe_reports_installed_optional_packages(monkeypatch):
    from hearing import device

    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(__version__="test-version")
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )
    info = device.describe()
    assert info["torch"] == "test-version"
    assert info["onnxruntime_providers"] == ["CPUExecutionProvider"]


def test_device_cli_prints_json():
    result = subprocess.run(
        [sys.executable, "-m", "hearing.device"],
        env={**os.environ, "HEARING_DEVICE": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["device"] == "cpu"
    assert info["machine"]


def test_real_pick_returns_supported_device_and_cpu_override_wins(monkeypatch):
    """Unmocked: whatever runtimes are installed, pick() names a supported device."""
    from hearing import device

    monkeypatch.delenv("HEARING_DEVICE", raising=False)
    auto = device.pick()
    assert auto in {"cuda", "mps", "cpu"}
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    assert device.pick() == "cpu"
    assert device.describe()["device"] == "cpu"
