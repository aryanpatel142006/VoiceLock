import json
from pathlib import Path

from eval.prep_checkpoint_audit import audit, write_snapshot


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_distinguishes_zero_null_and_absent_sisdr(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    results = tmp_path / "results.json"
    _write(manifest, {"config_hash": "c", "cases": [
        {"case_id": "zero", "source_hash": "s"},
        {"case_id": "null", "source_hash": "s"},
        {"case_id": "absent", "source_hash": "s", "target_present": False},
    ]})
    _write(results, {"cases": [
        {"case_id": "zero", "status": "complete", "source_hash": "s",
         "metrics": {"baseline": {"si_sdri": 0.0}}},
        {"case_id": "null", "status": "complete", "source_hash": "s",
         "metrics": {"baseline": {"si_sdri": None}}},
        {"case_id": "absent", "status": "complete", "source_hash": "s",
         "metrics": {"baseline": {"si_sdri": 3.0}}},
    ]})

    report = audit(manifest, [results])
    by_id = {row["case_id"]: row for row in report["cases"]}
    assert by_id["zero"]["status"] == "complete"
    assert by_id["zero"]["metrics"]["baseline"]["si_sdri"] == 0.0
    assert by_id["null"]["status"] == "complete"
    assert by_id["null"]["metrics"]["baseline"]["si_sdri"] is None
    assert by_id["absent"]["status"] == "invalid"
    assert "si_sdri_not_applicable" in by_id["absent"]["issues"]


def test_audit_rejects_duplicates_stale_hash_and_partial_checkpoint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    result = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.json"
    _write(manifest, {"config_hash": "new", "model_hash": "m", "cases": [
        {"case_id": "a", "source_hash": "s"},
        {"case_id": "b", "source_hash": "s"},
    ]})
    _write(result, {"cases": [
        {"case_id": "a", "status": "complete", "source_hash": "old"},
        {"case_id": "a", "status": "complete", "source_hash": "s"},
    ]})
    _write(checkpoint, {"case_id": "b", "status": "partial", "source_hash": "s",
                        "config_hash": "new", "model_hash": "m",
                        "stages": {"baseline": "complete"}})

    report = audit(manifest, [result, checkpoint])
    by_id = {row["case_id"]: row for row in report["cases"]}
    assert by_id["a"]["status"] == "invalid"
    assert "duplicate_case_id" in by_id["a"]["issues"]
    assert "source_hash_mismatch" in by_id["a"]["issues"]
    assert by_id["b"]["status"] == "partial"
    assert report["resume"]["case_ids"] == ["a", "b"]


def test_write_snapshot_is_atomic_and_caps_resume_files(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "audit.json"
    report = {"cases": [], "resume": {"files": [str(i) for i in range(700)]}}
    write_snapshot(destination, report)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert len(saved["resume"]["files"]) == 500
    assert not list(destination.parent.glob("*.tmp"))


def test_interrupted_json_is_reported_without_aborting_audit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    broken = tmp_path / "checkpoint.json"
    _write(manifest, {"cases": [{"case_id": "a"}]})
    broken.write_text('{"case_id":"a","status":"partial"', encoding="utf-8")

    report = audit(manifest, [broken])
    assert report["invalid_files"][0]["file"] == str(broken.resolve())
    assert report["cases"][0]["status"] == "missing"
