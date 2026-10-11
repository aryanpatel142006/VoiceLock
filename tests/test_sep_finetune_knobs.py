"""T-374: HEARING_SEP_CKPT / HEARING_PICK knobs and the fine-tune loss (no weights)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hearing import models
from hearing.tse import blind

TRAIN = Path(__file__).resolve().parents[1] / "experiments/gpu/T-374/train_sep.py"


def test_knobs_default_off(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARING_SEP_CKPT", raising=False)
    monkeypatch.delenv("HEARING_PICK", raising=False)
    monkeypatch.setattr(models, "cache_dir", lambda: tmp_path)
    assert not blind.pick_fixed()
    with pytest.raises(RuntimeError, match="weights absent"):
        blind._checkpoint()  # base checkpoint path, no override
    root = tmp_path / blind.MODEL
    root.mkdir()
    (root / "last_best_checkpoint.pt").write_bytes(b"x")
    (root / "last_best_checkpoint").write_text("last_best_checkpoint.pt\n")
    assert blind._checkpoint() == (root / "last_best_checkpoint.pt").resolve()


def test_sep_ckpt_override_and_pick(monkeypatch, tmp_path):
    ckpt = tmp_path / "best.pt"
    monkeypatch.setenv("HEARING_SEP_CKPT", str(ckpt))
    with pytest.raises(RuntimeError, match="not a file"):
        blind._checkpoint()
    ckpt.write_bytes(b"x")
    assert blind._checkpoint() == ckpt
    monkeypatch.setenv("HEARING_PICK", " Fixed ")
    assert blind.pick_fixed()


def test_fixed_order_loss_penalises_swapped_slots():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("t374_train_sep", TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    refs = torch.from_numpy(rng.standard_normal((1, 2, 8000)).astype(np.float32))
    refs[0, 1, :4000] = 0
    mix = refs.sum(dim=1)
    right = mod.slot_losses(refs * 3.0, refs, mix)  # any shared scale is undone by the gain
    swapped = mod.slot_losses(refs.flip(1), refs, mix)
    assert mod.objective(right, "fixed") < -25  # ~ -30 dB, the tau floor
    assert mod.objective(swapped, "fixed") > 0
    assert mod.objective(swapped, "pit") == pytest.approx(float(mod.objective(right, "pit")), abs=1e-3)
    silent = torch.zeros_like(refs)
    silent[0, 0] = refs[0, 0]  # slot 1 silent: L0 term rewards an empty stream
    quiet = mod.slot_losses(silent, silent, silent.sum(dim=1))
    assert quiet[0, 0, 1] < -25


def test_fixed_slot_only_with_knob(monkeypatch):
    monkeypatch.delenv("HEARING_PICK", raising=False)
    assert blind.fixed_slot("aryan") is None
    monkeypatch.setenv("HEARING_PICK", "fixed")
    assert blind.fixed_slot("Aryan") == 0
    assert blind.fixed_slot("joseph") == 1
    assert blind.fixed_slot("grid-s1") is None
    assert blind.fixed_slot(None) is None
