"""demo.sh --port / --no-open flag parsing (T-378). Only --stop and argument errors: no server starts."""

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def demo(*args):
    return subprocess.run(
        ["bash", "scripts/demo.sh", *args],
        cwd=ROOT,
        env={k: v for k, v in os.environ.items() if k != "PORT"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_port_without_value_is_an_error():
    r = demo("--port")
    assert r.returncode == 2
    assert "--port needs a number" in r.stderr


def test_port_out_of_range_is_rejected():
    r = demo("--port", "70000", "--stop")
    assert r.returncode == 2
    assert "PORT must be 1-65535" in r.stderr


def test_port_zero_picks_a_free_port_and_prints_it():
    r = demo("--port", "0", "--stop", "--no-open")
    assert r.returncode == 0, r.stderr
    picked = re.search(r"^port (\d+) \(picked by --port 0\)$", r.stdout, re.MULTILINE)
    assert picked, r.stdout
    port = int(picked.group(1))
    assert 1 <= port <= 65535
    assert f"Stopped: :{port} is free" in r.stdout
