import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_P = Path(__file__).resolve().parents[1] / "experiments/theory/T-251/prep.py"
spec = importlib.util.spec_from_file_location("t251_prep", _P)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.mark.parametrize("mode", prep.CONDITIONING)
def test_shapes_and_causality(mode):
    torch.manual_seed(1)
    m = prep.build_model(mode)
    y = m(torch.randn(2, 3200), torch.randn(2, 192), torch.randn(2, 10, 192))
    assert y.shape == (2, 3200)
    assert prep.causality_leak(m, mode, seed=1) < 1e-6


def test_noncausal_control_is_detected():
    class Leaky(torch.nn.Module):
        def forward(self, x, emb=None, seq=None):
            return x - x.mean(dim=-1, keepdim=True)  # global mean sees the future

    assert prep.causality_leak(Leaky(), "none", seed=1) > 1e-3


def test_conditioning_changes_output_and_cost_ordering():
    torch.manual_seed(2)
    x = torch.randn(1, 1600)
    for mode in ("project", "film", "xattn"):
        m = prep.build_model(mode).eval()
        a = m(x, torch.randn(1, 192), torch.randn(1, 5, 192))
        b = m(x, torch.randn(1, 192), torch.randn(1, 5, 192))
        assert (a - b).abs().max() > 0
    p = {k: prep.n_params(prep.build_model(k)) for k in prep.CONDITIONING}
    assert p["none"] < p["project"] < p["film"] < p["xattn"]


def test_manifest_deterministic_and_counts(tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir()
    for i in range(3):
        (fx / f"{i}.wav").write_bytes(b"x")
    a = prep.manifest(tmp_path / "a", fx, 2, 251)
    b = prep.manifest(tmp_path / "b", fx, 2, 251)
    assert a["config_sha256"] == b["config_sha256"]
    assert a["inputs"] == {"fixtures_dir_set": True, "available_wavs": 2, "capped": True}
    assert prep.manifest(tmp_path / "c", None, 5, 251)["inputs"]["available_wavs"] == 0
    assert json.loads((tmp_path / "a" / "manifest.json").read_text())["status"] == "complete"
