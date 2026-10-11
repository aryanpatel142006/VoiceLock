import hashlib
import json
from pathlib import Path

from eval.overnight_split_audit import (
    assign_partitions,
    audit_rows,
    intervals_overlap,
    load_manifest_rows,
)


def row(**overrides):
    value = {
        "asset_id": "a",
        "speaker_id": "spk",
        "source_id": "utt",
        "start_s": 0.0,
        "end_s": 1.0,
        "provenance": {"dataset": "fixture", "license": "CC BY 4.0"},
    }
    value.update(overrides)
    return value


def test_intervals_touching_at_boundary_do_not_overlap():
    assert not intervals_overlap((0, 1), (1, 2))
    assert intervals_overlap((0, 1.01), (1, 2))


def test_same_source_is_kept_in_one_partition():
    rows = [row(asset_id="a", source_id="utt", start_s=0), row(asset_id="b", source_id="utt", start_s=1)]
    assigned = assign_partitions(rows, seed=7)
    assert len({item["partition"] for item in assigned}) == 1


def test_audit_rejects_interval_overlap_and_transformed_duplicate():
    digest = hashlib.sha256(b"same").hexdigest()
    rows = [
        row(asset_id="a", source_id="utt-a", start_s=0, end_s=1, sha256=digest),
        row(
            asset_id="a-crop",
            source_id="utt-a",
            start_s=0.5,
            end_s=1.5,
            sha256=digest,
            transform="gain",
        ),
    ]
    result = audit_rows(rows)
    reasons = {reason for item in result["excluded"] for reason in item["reasons"]}
    assert "temporal_source_overlap" in reasons
    assert "duplicate_sha256" in reasons


def test_unknown_provenance_is_ineligible():
    result = audit_rows([row(provenance={})])
    assert result["eligible"] == []
    assert "unknown_provenance" in result["excluded"][0]["reasons"]


def test_load_manifest_rows_reads_librispeech_manifest(tmp_path: Path):
    manifest = {
        "dataset": "openslr/librispeech_asr",
        "split": "test-clean",
        "license": "CC BY 4.0",
        "speech": [
            {
                "file": "speech.wav",
                "id": "1-2-3",
                "speaker_id": "1",
                "text": "hello",
                "source_duration_s": 1.0,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    rows = load_manifest_rows(path)
    assert rows[0]["source_id"] == "1-2-3"
    assert rows[0]["speaker_id"] == "1"
    assert rows[0]["transcript_valid"] is True
