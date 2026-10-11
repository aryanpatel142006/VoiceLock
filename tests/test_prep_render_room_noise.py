from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from eval.prep_render_room_noise import render, render_variant


def test_render_variant_is_additive_and_reproducible() -> None:
    clean = np.zeros(32, dtype=np.float32)
    clean[0] = 1.0
    noise = np.ones(8, dtype=np.float32)
    ir = np.array([1.0, 0.5], dtype=np.float32)
    first = render_variant(clean, noise, ir, 5)
    second = render_variant(clean, noise, ir, 5)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    mix, target, scaled_noise = first
    np.testing.assert_allclose(mix, target + scaled_noise, atol=1e-7)


def test_render_resamples_inputs_and_records_synthetic_ir(tmp_path) -> None:
    clean_dir, noise_dir = tmp_path / "clean", tmp_path / "noise"
    clean_dir.mkdir()
    noise_dir.mkdir()
    sf.write(clean_dir / "a.wav", np.full(80, 0.1, dtype=np.float32), 8000)
    sf.write(noise_dir / "n.wav", np.full(160, 0.1, dtype=np.float32), 16000)
    rows = render(clean_dir, noise_dir, tmp_path / "out", limit=1)
    row = rows[0]
    assert row["sample_rate"] == 16000
    assert row["ir"]["kind"] == "SYNTHETIC_acoustic_IR"
    assert row["input_sample_rates"] == {"clean": 8000, "noise": 16000}
    assert row["simulated_distance"] is False
    manifest = json.loads((tmp_path / "out" / row["id"] / "manifest.json").read_text())
    mix = np.asarray(sf.read(tmp_path / "out" / row["id"] / "mix.wav", dtype="float32")[0])
    target = np.asarray(sf.read(tmp_path / "out" / row["id"] / "target.wav", dtype="float32")[0])
    noise = np.asarray(sf.read(tmp_path / "out" / row["id"] / "noise.wav", dtype="float32")[0])
    assert manifest["additivity"].startswith("mix = target + noise")
    np.testing.assert_allclose(mix, target + noise, atol=2e-4)
