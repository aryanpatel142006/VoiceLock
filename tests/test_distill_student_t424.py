"""T-424: the distilled causal student is causal, streams exactly, and reports its latency."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "gpu" / "T-424" / "student.py"
_spec = importlib.util.spec_from_file_location("t424_student", _PATH)
student = importlib.util.module_from_spec(_spec)
sys.modules["t424_student"] = student
_spec.loader.exec_module(student)


def _net(variant):
    torch.manual_seed(0)
    return student.build(variant).eval()


@pytest.mark.parametrize("variant", sorted(student.VARIANTS))
def test_output_never_looks_further_ahead_than_one_window(variant):
    net = _net(variant)
    x = torch.randn(1, 8000) * 0.1
    with torch.no_grad():
        full = net(x)
        n = 5000
        prefix = net(x[:, :n])
    keep = n - net.win
    assert full.shape == (1, 2, 8000)
    assert net(x, residual=True).shape == (1, 3, 8000)
    assert torch.allclose(full[..., :keep], prefix[..., :keep], atol=1e-6)
    assert net.latency_ms == 1000.0 * net.win / student.SR <= 50.0


def test_streamer_matches_whole_pass_for_any_chunking():
    net = _net("tcn8s")
    x = torch.randn(9000) * 0.1
    with torch.no_grad():
        full = net(x[None], residual=True)[0]
    st = student.Streamer(net)
    sizes = [160, 37, 256, 1000, 1, 160]
    outs, i, k = [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        outs.append(st.process(x[i : i + n]))
        i, k = i + n, k + 1
    y = torch.cat(outs, dim=1)
    assert len(x) - y.shape[1] < net.win  # at most one window still pending
    assert torch.allclose(y, full[:, : y.shape[1]], atol=1e-5)


def test_variants_fit_the_size_budget():
    for v in student.VARIANTS:
        assert 0.5e6 < student.count_params(student.build(v)) < 3.0e6, v
