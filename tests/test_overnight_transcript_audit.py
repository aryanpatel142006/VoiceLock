import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.overnight_transcript_audit import audit_manifest, audit_roots


def _manifest(tmp_path: Path, **updates):
    root = tmp_path / "fixture"
    root.mkdir(parents=True)
    sf.write(root / "target.wav", np.zeros(16_000, dtype=np.float32), 16_000)
    data = {
        "dataset": "test-public",
        "license": "CC BY 4.0",
        "sample_rate": 16_000,
        "duration_s": 1.0,
        "speech": [
            {
                "file": "target.wav",
                "id": "spk-0001",
                "speaker_id": "spk",
                "text": "ONE TWO",
                "source_duration_s": 1.0,
            }
        ],
    }
    data.update(updates)
    path = root / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_exact_reference_has_complete_safe_interval_and_alignment(tmp_path):
    path = _manifest(
        tmp_path,
        speech=[
            {
                "file": "target.wav",
                "id": "spk-0001",
                "speaker_id": "spk",
                "text": "ONE TWO",
                "start_s": 0.0,
                "end_s": 1.0,
                "words": [
                    {"word": "ONE", "start_s": 0.0, "end_s": 0.4},
                    {"word": "TWO", "start_s": 0.5, "end_s": 0.9},
                ],
            }
        ],
    )

    record = audit_manifest(path)[0]

    assert record["transcript_status"] == "exact_reference"
    assert record["word_boundaries"] == "aligned"
    assert record["safe_interval_s"] == [0.0, 1.0]
    assert record["source_file_status"] == "ok"


def test_missing_reference_is_automatic_unverified(tmp_path):
    path = _manifest(tmp_path, speech=[{"file": "target.wav", "id": "spk-0001"}])

    record = audit_manifest(path)[0]

    assert record["transcript_status"] == "automatic_unverified"
    assert record["word_boundaries"] == "missing"
    assert record["safe_interval_s"] is None


def test_repeated_interferer_is_not_claimed_full(tmp_path):
    path = _manifest(
        tmp_path,
        speech=[
            {
                "file": "target.wav",
                "id": "spk-0001",
                "text": "ONE TWO",
                "source_duration_s": 1.0,
            },
            {
                "file": "target.wav",
                "id": "spk-0002",
                "text": "THREE FOUR",
                "source_duration_s": 1.0,
            },
        ],
        noise={"file": "noise.wav", "tiled_period_s": 1.0},
    )

    records = audit_manifest(path)

    assert records[1]["utterance_status"] == "repeated_or_looped"


def test_source_id_mismatch_is_reported(tmp_path):
    path = _manifest(
        tmp_path,
        speech=[
            {
                "file": "target.wav",
                "id": "wrong-id",
                "speaker_id": "spk",
                "text": "ONE TWO",
                "source": {"id": "source-id"},
            }
        ],
    )

    record = audit_manifest(path)[0]

    assert record["source_id_status"] == "mismatch"
    assert any("source-id" in issue for issue in record["issues"])


def test_audit_roots_is_bounded_and_writes_ignored_jsonl(tmp_path):
    root = tmp_path / "fixtures"
    root.mkdir()
    for index in range(3):
        path = _manifest(root / str(index))
        assert path.exists()
    output = tmp_path / "transcript_validity.jsonl"

    records = audit_roots([root], limit=2, output=output)

    assert len(records) == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
