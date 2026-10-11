import json
from pathlib import Path

import pytest

from eval.prep_transfer_manifest import (
    ManifestError,
    build_manifest,
    sha256_file,
)


def _catalog(tmp_path: Path, *, file_name: str = "clip.wav") -> tuple[Path, Path]:
    root = tmp_path / "catalog"
    root.mkdir()
    source = root / file_name
    source.write_bytes(b"public fixture bytes")
    catalog = root / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "dataset": "public-demo",
                "revision": "rev-7",
                "license": {"spdx": "CC-BY-4.0", "url": "https://example.test/license"},
                "source_group": "speaker-a",
                "files": [{"path": file_name, "source_id": "utterance-1"}],
            }
        ),
        encoding="utf-8",
    )
    return catalog, source


def test_manifest_is_content_addressed_and_has_argument_arrays(tmp_path):
    catalog, source = _catalog(tmp_path)

    result = build_manifest([catalog], max_bytes=1024)

    assert result["available"]["files"] == 1
    assert result["planned"]["files"] == 1
    item = result["files"][0]
    assert item["sha256"] == sha256_file(source)
    assert "local_path" not in item
    assert item["source_id"] == "utterance-1"
    assert item["license"]["spdx"] == "CC-BY-4.0"
    assert all(isinstance(arg, list) for arg in result["transfer"]["commands"])
    assert all("source_id" not in " ".join(args) for args in result["transfer"]["commands"])


def test_manifest_rejects_symlink_escape(tmp_path):
    catalog, _ = _catalog(tmp_path)
    outside = tmp_path / "private.wav"
    outside.write_bytes(b"private")
    (catalog.parent / "escape.wav").symlink_to(outside)
    data = json.loads(catalog.read_text())
    data["files"][0]["path"] = "escape.wav"
    catalog.write_text(json.dumps(data))

    with pytest.raises(ManifestError, match="escapes catalog root"):
        build_manifest([catalog])


def test_manifest_detects_catalog_hash_mismatch(tmp_path):
    catalog, _ = _catalog(tmp_path)
    data = json.loads(catalog.read_text())
    data["files"][0]["sha256"] = "0" * 64
    catalog.write_text(json.dumps(data))

    with pytest.raises(ManifestError, match="sha256 mismatch"):
        build_manifest([catalog])


def test_resume_and_byte_cap_are_deterministic(tmp_path):
    catalog, source = _catalog(tmp_path)
    first = build_manifest([catalog], max_bytes=1)
    assert first["planned"]["files"] == 0
    assert first["excluded"][0]["reason"] == "byte_cap"

    resumed = build_manifest([catalog], max_bytes=1024, completed_hashes=set())
    assert resumed["files"][0]["sha256"] == sha256_file(source)
    resumed_again = build_manifest(
        [catalog], max_bytes=1024, completed_hashes={resumed["files"][0]["sha256"]}
    )
    assert resumed_again["planned"]["files"] == 0
    assert resumed_again["excluded"][0]["reason"] == "resume_complete"
