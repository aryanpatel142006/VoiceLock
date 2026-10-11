from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.cache_spectrograms import build_cache, extract_features


def _fixture(root: Path) -> Path:
    fixture = root / "tiny" / "000_demo"
    fixture.mkdir(parents=True)
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    sf.write(fixture / "mix.wav", (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    (fixture / "manifest.json").write_text(
        json.dumps({"id": "000_demo", "set": "tiny", "sample_rate": sr})
    )
    return fixture


def test_extract_features_has_stable_shapes_and_finite_values() -> None:
    audio = np.sin(2 * np.pi * 440 * np.arange(3200) / 16_000).astype(np.float32)

    features = extract_features(audio, 16_000)

    assert features["stft"].shape == (21, 257)
    assert features["filterbank"].shape == (21, 40)
    assert features["log_mel"].shape == (21, 40)
    assert all(np.isfinite(value).all() for value in features.values())


def test_build_cache_writes_hashed_manifest_and_reuses_cache(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    _fixture(fixture_root)
    cache_root = tmp_path / "cache"

    first = build_cache(fixture_root, cache_root)
    cached_file = cache_root / "tiny" / "000_demo" / "mix.npz"
    before = cached_file.read_bytes()
    second = build_cache(fixture_root, cache_root)

    assert first == second
    assert cached_file.read_bytes() == before
    manifest = json.loads((cache_root / "manifest.json").read_text())
    assert manifest["schema"] == 1
    assert manifest["entries"][0]["source"] == "tiny/000_demo/mix.wav"
    assert len(manifest["entries"][0]["source_sha256"]) == 64
    assert len(manifest["entries"][0]["cache_sha256"]) == 64
