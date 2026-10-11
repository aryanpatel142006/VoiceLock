import hashlib
import json

import numpy as np
import soundfile as sf

from eval.render_gain_pause import Recipe, render_recipe, write_shards


def test_render_recipe_applies_negative_sir_and_target_pause(tmp_path):
    target = np.full(1600, 0.2, dtype=np.float32)
    interferer = np.full(1600, 0.2, dtype=np.float32)
    recipe = Recipe("joseph", "aryan", -15.0, "target_pause", 0.05, 0)

    mix, metadata = render_recipe(target, interferer, recipe, 16_000)

    expected_interferer = 0.2 * 10 ** (15 / 20)
    assert np.allclose(mix[:800], 0.2, rtol=1e-5)
    assert np.allclose(mix[800:1600], expected_interferer, rtol=1e-5)
    assert metadata["sir_db"] == -15.0
    assert metadata["target_silent_intervals"] == [[0.05, 0.1]]


def test_write_shards_is_deterministic_and_hashes_waveforms(tmp_path):
    sources = {
        "aryan": np.linspace(-0.25, 0.25, 1600, dtype=np.float32),
        "joseph": np.linspace(0.25, -0.25, 1600, dtype=np.float32),
    }
    recipes = [Recipe("aryan", "joseph", 0.0, "target_on", 0.0, 7)]

    first = write_shards(sources, recipes, tmp_path / "one", 16_000)
    second = write_shards(sources, recipes, tmp_path / "two", 16_000)

    assert first["count"] == second["count"] == 1
    assert first["files"] == second["files"]
    shard = tmp_path / "one" / first["files"][0]
    manifest = json.loads((shard.parent / "manifest.json").read_text())
    audio, _ = sf.read(shard, dtype="float32")
    assert manifest["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert np.array_equal(audio, sf.read(tmp_path / "two" / second["files"][0], dtype="float32")[0])
