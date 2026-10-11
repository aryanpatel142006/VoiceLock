import json

import numpy as np
import soundfile as sf

from eval.extract_ecapa_matrix import extract_matrix, load_segments


def _write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(values, dtype=np.float32), 16_000)


def test_load_segments_labels_clean_and_corrupted(tmp_path):
    _write(tmp_path / "clean" / "alice.wav", np.ones(16_000))
    _write(tmp_path / "corrupted" / "alice_noise.wav", np.ones(16_000) * 0.5)

    segments = load_segments(tmp_path)

    assert [(s.condition, s.speaker, s.path.name) for s in segments] == [
        ("clean", "alice", "alice.wav"),
        ("corrupted", "alice_noise", "alice_noise.wav"),
    ]


def test_extract_matrix_crops_and_writes_pairwise_cosine_distances(tmp_path):
    _write(tmp_path / "clean" / "a.wav", np.ones(32_000))
    _write(tmp_path / "clean" / "b.wav", np.r_[np.ones(16_000), -np.ones(16_000)])
    _write(tmp_path / "corrupted" / "a.wav", np.ones(32_000) * 0.5)

    def embed(audio):
        return np.array([np.mean(audio), np.std(audio), 1.0])

    output = tmp_path / "matrix.npz"
    result = extract_matrix(
        tmp_path,
        embedder=embed,
        crops=(1.0, 2.0),
        hop_s=1.0,
        output=output,
    )

    assert result["sample_rate"] == 16_000
    assert result["crops"] == [1.0, 2.0]
    assert output.exists()
    saved = np.load(output, allow_pickle=False)
    assert saved["matrix_1.0"].shape == (6, 6)
    np.testing.assert_allclose(np.diag(saved["matrix_2.0"]), 0.0)
    np.testing.assert_allclose(saved["matrix_1.0"], saved["matrix_1.0"].T)
    assert json.loads((output.with_suffix(".json")).read_text())["n_embeddings"] == {"1.0": 6, "2.0": 3}


def test_drift_under_noise_is_zero_for_identical_and_grows_with_corruption(tmp_path):
    rng = np.random.default_rng(0)
    tone = np.sin(2 * np.pi * 200 * np.arange(32_000) / 16_000)
    _write(tmp_path / "clean" / "a.wav", tone)
    _write(tmp_path / "clean" / "b.wav", tone)
    _write(tmp_path / "corrupted" / "a.wav", tone)  # untouched
    _write(tmp_path / "corrupted" / "b.wav", tone + rng.normal(0, 2.0, tone.size))

    def embed(audio):  # energy-sensitive toy embedding
        spectrum = np.abs(np.fft.rfft(audio))[:400]
        return np.r_[spectrum[::40], np.std(audio)]

    result = extract_matrix(tmp_path, embedder=embed, crops=(1.0, 2.0))
    for crop in (1.0, 2.0):
        names, matrix = result["labels"][crop], result["matrices"][crop]
        idx = {n.split("/", 1)[0] + n.split("/")[-1]: i for i, n in enumerate(names)}
        same = matrix[idx["cleana.wav@0.000s"], idx["corrupteda.wav@0.000s"]]
        noisy = matrix[idx["cleanb.wav@0.000s"], idx["corruptedb.wav@0.000s"]]
        assert same < 1e-9
        assert noisy > same + 1e-3
    drift = result["drift"]["1.0"]
    assert drift["n_pairs"] == 4
    assert drift["max"] > drift["mean"] > 0
