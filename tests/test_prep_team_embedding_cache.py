import json

import numpy as np
import soundfile as sf

from eval.prep_team_embedding_cache import build_cache, margin_report


class FakeEmbedder:
    def embed(self, audio):
        x = np.asarray(audio)
        return np.array([x.mean(), np.mean(x * x), 1.0], dtype=np.float32)


def test_build_cache_contains_normalized_crops_and_margin_percentiles(tmp_path):
    root = tmp_path / "team"
    for name, frequency in (("aryan", 220), ("joseph", 330)):
        path = root / name / "take.wav"
        path.parent.mkdir(parents=True)
        t = np.arange(16000 * 2, dtype=np.float32) / 16000
        sf.write(path, (0.4 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), 16000)

    cache = build_cache(root, embedder=FakeEmbedder(), sample_rate=16000)
    assert cache["embeddings"].shape[:3] == (2, 4, 8)
    norms = np.linalg.norm(cache["embeddings"], axis=-1)
    np.testing.assert_allclose(norms[np.isfinite(norms)], 1.0)

    report = margin_report(cache)
    assert set(report) == {"sample_rate", "speakers", "crops"}
    assert [entry["crop_s"] for entry in report["crops"]] == [0.25, 0.5, 1.0, 2.0]
    for entry in report["crops"]:
        assert set(entry["similarity"]) == {"within", "between", "margin"}
        assert set(entry["margin"]) == {"p05", "p25", "p50", "p75", "p95"}


def test_report_round_trips_as_json(tmp_path):
    root = tmp_path / "team"
    for name in ("aryan", "joseph"):
        path = root / name / "take.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.ones(32000, dtype=np.float32) * (0.1 if name == "aryan" else 0.2), 16000)
    report = margin_report(build_cache(root, embedder=FakeEmbedder()))
    encoded = json.dumps(report)
    assert json.loads(encoded)["crops"][0]["crop_s"] == 0.25
