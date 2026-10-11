from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.prep_acoustic_plan import (
    build_plan,
    inspect_rir,
    recipe_hash,
)


def test_inspect_rir_rejects_zero_impulse(tmp_path: Path) -> None:
    path = tmp_path / "zero.wav"
    sf.write(path, np.zeros(64, dtype=np.float32), 16_000)

    result = inspect_rir(path)

    assert result.valid is False
    assert "zero" in result.reason


def test_plan_has_deterministic_synthetic_alignment_and_provenance(tmp_path: Path) -> None:
    noise = tmp_path / "noise.wav"
    sf.write(noise, np.ones(160, dtype=np.float32), 16_000)
    metadata = tmp_path / "noise.json"
    metadata.write_text(
        json.dumps(
            {
                "kind": "noise",
                "source": "cached-noise",
                "licence": "fixture-only",
                "sample_rate": 16000,
            }
        )
    )

    first = build_plan(tmp_path, seed=7, max_recipes=4)
    second = build_plan(tmp_path, seed=7, max_recipes=4)

    assert first["available"]["noise"] == 1
    assert first["available"]["rir"] == 0
    assert first["planned"]["recipes"] == 4
    assert first["recipes"] == second["recipes"]
    assert all(r["rir"]["synthetic"] for r in first["recipes"])
    assert all(r["rir"]["alignment"]["peak_sample"] == 0 for r in first["recipes"])
    assert all(r["provenance"]["noise"]["source"] == "cached-noise" for r in first["recipes"])
    assert all(recipe_hash(r) == r["config_hash"] for r in first["recipes"])


def test_incompatible_rir_rate_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "room_rir.wav"
    impulse = np.zeros(32, dtype=np.float32)
    impulse[0] = 1
    sf.write(path, impulse, 48_000)

    result = inspect_rir(path, required_rate=16_000)

    assert result.valid is False
    assert "rate" in result.reason
