import json

import numpy as np

from eval.render_room_noise import render_example, render_shards


def test_render_example_convolves_room_and_hits_requested_snr():
    speech = np.zeros(1600, dtype=np.float32)
    speech[200] = 1.0
    noise = np.ones(1600, dtype=np.float32)
    result = render_example(
        speech,
        noise,
        snr_db=6.0,
        rir_name="small_room",
        seed=4,
        example_id="x",
    )

    assert result["target"].shape == speech.shape
    assert result["room_ir"]["name"] == "small_room"
    assert result["mix"].dtype == np.float32
    assert np.max(np.abs(result["mix"])) <= 0.9 + 1e-6
    measured = 20 * np.log10(
        np.sqrt(np.mean(result["target"] ** 2))
        / np.sqrt(np.mean(result["noise"] ** 2))
    )
    np.testing.assert_allclose(measured, 6.0, atol=1e-3)


def test_render_shards_resume_uses_atomic_checkpoint(tmp_path):
    out = tmp_path / "room_noise"
    first = render_shards(out, count=3, shards=2, seed=7)
    assert first["completed_shards"] == [0, 1]
    checkpoint = json.loads((out / "checkpoint.json").read_text())
    assert checkpoint["completed_shards"] == [0, 1]
    assert all((out / f"shard-{i:04d}" / "manifest.json").exists() for i in range(2))

    second = render_shards(out, count=3, shards=2, seed=7, resume=True)
    assert second == first
    assert not list(out.glob("*.tmp"))
