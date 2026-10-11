from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.overnight_data_ready import assemble_ready, validate_case


def _write_case(root: Path, *, license_text: str = "CC BY 4.0") -> Path:
    case = root / "case-a"
    case.mkdir(parents=True)
    paths = {}
    for name, value in {
        "mix": np.full(160, 0.1, np.float32),
        "target": np.full(160, 0.05, np.float32),
        "interferer": np.full(160, 0.02, np.float32),
        "noise": np.full(160, 0.03, np.float32),
    }.items():
        path = case / f"{name}.wav"
        sf.write(path, value, 16000, subtype="PCM_16")
        paths[name] = path.name
    manifest = {
        "id": "case-a",
        "set": "overnight",
        "sample_rate": 16000,
        "duration_s": 0.01,
        "target_id": "target-1",
        "interferer_id": "other-1",
        "transcript": "a valid transcript",
        "paths": paths,
        "source": {
            "dataset": "example",
            "split": "train",
            "license": license_text,
            "target": {"row_id": "target-1"},
            "interferer": {"row_id": "other-1"},
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))
    return case


def test_validate_case_reports_hashes_audio_and_stem_sum(tmp_path: Path):
    case = _write_case(tmp_path)

    result = validate_case(case / "manifest.json")

    assert result["status"] == "ready"
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert result["transcript"]["status"] == "complete"
    assert result["license"] == "CC BY 4.0"
    assert result["stems_sum_max_abs"] < 1e-4
    assert result["files"]["mix"]["sha256"] == hashlib.sha256((case / "mix.wav").read_bytes()).hexdigest()


def test_validate_case_rejects_path_escape(tmp_path: Path):
    case = _write_case(tmp_path)
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["paths"]["mix"] = "../outside.wav"
    (case / "manifest.json").write_text(json.dumps(manifest))

    result = validate_case(case / "manifest.json")

    assert result["status"] == "excluded"
    assert result["reason"] == "path_escape"


@pytest.mark.parametrize("failure", ["missing", "hash"])
def test_validate_case_rejects_missing_or_hash_mismatch(tmp_path: Path, failure: str):
    case = _write_case(tmp_path)
    if failure == "missing":
        (case / "noise.wav").unlink()
    else:
        manifest = json.loads((case / "manifest.json").read_text())
        manifest["hashes"] = {"mix": "0" * 64}
        (case / "manifest.json").write_text(json.dumps(manifest))

    result = validate_case(case / "manifest.json")

    assert result["status"] == "excluded"
    assert result["reason"] == ("missing_file" if failure == "missing" else "hash_mismatch")


def test_validate_case_marks_target_absent_and_excludes_it_from_sisdr(tmp_path: Path):
    case = _write_case(tmp_path)
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["target_id"] = ""
    manifest["transcript"] = ""
    (case / "manifest.json").write_text(json.dumps(manifest))

    result = validate_case(case / "manifest.json")

    assert result["status"] == "ready"
    assert result["target_absent"] is True
    assert result["sisdr_eligible"] is False
    assert result["transcript"]["status"] == "unavailable"


def test_assemble_ready_writes_jsonl_and_deduplicates_original_sources(tmp_path: Path):
    case = _write_case(tmp_path / "fixtures")
    index = case.parent / "index.json"
    entry = json.loads((case / "manifest.json").read_text())
    entry["paths"] = {key: f"case-a/{value}" for key, value in entry["paths"].items()}
    index.write_text(json.dumps({"fixtures": [entry, entry]}))
    output = tmp_path / "results" / "ready.jsonl"

    summary = assemble_ready([tmp_path / "fixtures"], output, limit=12)

    assert summary["ready"] == 1
    assert summary["excluded"] == 0
    assert len(output.read_text().splitlines()) == 1
