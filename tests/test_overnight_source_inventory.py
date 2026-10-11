import json
import wave

import numpy as np

from eval.overnight_source_inventory import build_inventory


def _write_wav(path, samples, sample_rate=16000):
    samples = np.asarray(samples, dtype=np.float32)
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def test_inventory_records_audio_metrics_ids_and_mix_consistency(tmp_path):
    root = tmp_path / "tiny"
    root.mkdir()
    target = np.array([0.25, -0.25, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.1, 0.1, 0.0, 0.0], dtype=np.float32)
    mix = target + other
    for name, samples in (("target.wav", target), ("other.wav", other), ("mix.wav", mix)):
        _write_wav(root / name, samples)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "id": "tiny-scene",
                "sample_rate": 16000,
                "target_id": "speaker-a-utt-1",
                "interferer_id": "speaker-b-utt-2",
                "transcript": "declared only",
                "target_intervals": [[0.0, 0.000125]],
                "paths": {"target": "target.wav", "other": "other.wav", "mix": "mix.wav"},
            }
        )
    )

    rows, report = build_inventory(root, tmp_path / "catalog.jsonl")

    assert {row["role"] for row in rows} == {"target", "other", "mix"}
    target_row = next(row for row in rows if row["role"] == "target")
    assert target_row["sample_rate"] == 16000
    assert target_row["channels"] == 1
    assert target_row["speaker_id"] == "speaker-a"
    assert target_row["utterance_id"] == "speaker-a-utt-1"
    assert target_row["source_intervals_s"] == [[0.0, 0.000125]]
    assert report["mix_checks"][0]["matches_sum"] is True
    assert target_row["transcript_reference"] == "manifest.json:transcript"
    assert json.loads((tmp_path / "catalog.jsonl").read_text().splitlines()[0])["sha256"]


def test_inventory_separates_missing_and_corrupt_inputs(tmp_path):
    root = tmp_path / "set"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"paths": {"target": "missing.wav", "mix": "broken.wav"}})
    )
    (root / "broken.wav").write_bytes(b"not a wav")

    rows, report = build_inventory(root, tmp_path / "catalog.jsonl")

    assert rows == []
    assert report["missing"] == ["missing.wav"]
    assert report["corrupt"] == ["broken.wav"]
