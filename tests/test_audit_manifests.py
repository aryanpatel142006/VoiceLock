import json

import numpy as np
import soundfile as sf

from eval.audit_manifests import audit_shards


def _shard(root, name="shard-001", *, sample_rate=16000, channels=1, duration=1.0):
    shard = root / name
    shard.mkdir()
    audio = np.zeros((round(sample_rate * duration), channels), dtype=np.float32)
    if channels == 1:
        audio = audio[:, 0]
    audio_path = shard / "render.wav"
    sf.write(audio_path, audio, sample_rate, subtype="FLOAT")
    manifest = {
        "schema_version": 1,
        "shard_id": name,
        "audio": "render.wav",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_s": duration,
    }
    (shard / "manifest.json").write_text(json.dumps(manifest))
    return shard


def test_audit_accepts_compliant_float32_shard(tmp_path):
    _shard(tmp_path)

    report = audit_shards(tmp_path)

    assert report["status"] == "ok"
    assert report["summary"]["shards"] == 1
    assert report["summary"]["audio_files"] == 1
    assert report["issues"] == []
    assert len(report["shards"][0]["sha256"]) == 64


def test_audit_reports_format_missing_file_and_duration_issues(tmp_path):
    _shard(tmp_path, "bad-format", sample_rate=8000, channels=2, duration=0.1)
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shard_id": "missing",
                "audio": "not-rendered.wav",
                "sample_rate": 16000,
                "channels": 1,
                "duration_s": 2.0,
            }
        )
    )

    report = audit_shards(tmp_path, min_duration_s=0.5, max_duration_s=1.5)

    codes = {issue["code"] for issue in report["issues"]}
    assert {"sample_rate", "channels", "duration", "missing_audio"} <= codes
    assert report["status"] == "error"


def test_audit_detects_manifest_hash_mismatch_and_duration_mismatch(tmp_path):
    shard = _shard(tmp_path, duration=1.0)
    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["duration_s"] = 2.0
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    report = audit_shards(tmp_path)

    codes = {issue["code"] for issue in report["issues"]}
    assert {"duration_mismatch", "hash_mismatch"} <= codes


def test_audit_rejects_invalid_json_manifest(tmp_path):
    shard = tmp_path / "broken"
    shard.mkdir()
    (shard / "manifest.json").write_text("{")

    report = audit_shards(tmp_path)

    assert report["summary"]["manifests"] == 1
    assert report["issues"][0]["code"] == "invalid_manifest"


def test_audit_checks_declared_audio_metadata_and_empty_roots(tmp_path):
    _shard(tmp_path)
    manifest_path = tmp_path / "shard-001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sample_rate"] = 8000
    manifest_path.write_text(json.dumps(manifest))

    report = audit_shards(tmp_path)

    assert any(issue["code"] == "metadata_mismatch" for issue in report["issues"])
    assert audit_shards(tmp_path / "empty")["status"] == "error"
