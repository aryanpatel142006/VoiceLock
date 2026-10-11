"""Audit offline overnight result/checkpoint JSON without changing its inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

MAX_FILES = 500
METRIC_GROUPS = ("baseline", "selected", "oracle", "mixer", "agc", "delivered")
SI_KEYS = {"si_sdr", "si_sdri", "sisdr", "sisdr_i"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable_json:{type(exc).__name__}"
    return (value, None) if isinstance(value, dict) else (None, "top_level_not_object")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_groups(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        group: value[group]
        for group in METRIC_GROUPS
        if isinstance(value.get(group), dict)
    }


def _bad_metrics(value: Any, prefix: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            bad.extend(_bad_metrics(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            bad.extend(_bad_metrics(item, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        bad.append(prefix)
    return bad


def _entries(document: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    values = document.get("cases", document.get("results"))
    if isinstance(values, list):
        entries = [item.copy() for item in values if isinstance(item, dict)]
        inherited = {
            key: document[key]
            for key in ("config_hash", "model_hash", "profile_hash", "source_hash")
            if key in document
        }
        for entry in entries:
            for key, value in inherited.items():
                entry.setdefault(key, value)
        return entries
    if document.get("case_id") is not None:
        return [document]
    return []


def _expected_hashes(manifest: dict[str, Any]) -> dict[str, Any]:
    hashes = manifest.get("hashes")
    hashes = hashes if isinstance(hashes, dict) else {}
    return {
        name: manifest.get(f"{name}_hash", hashes.get(name))
        for name in ("config", "model", "profile", "source")
    }


def audit(manifest_path: str | Path, inputs: list[str | Path]) -> dict[str, Any]:
    """Return a deterministic integrity report for a manifest and JSON inputs."""
    manifest_file = Path(manifest_path).resolve()
    manifest, manifest_error = _load(manifest_file)
    if manifest is None:
        raise ValueError(f"invalid manifest: {manifest_error}")
    expected = _expected_hashes(manifest)
    planned = [
        item for item in manifest.get("cases", [])
        if isinstance(item, dict) and item.get("case_id") is not None
    ]
    planned_by_id = {str(item["case_id"]): item for item in planned}
    records: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    invalid_files: list[dict[str, Any]] = []
    paths = sorted({Path(item).resolve() for item in inputs})
    for path in paths[:MAX_FILES]:
        document, error = _load(path)
        if error:
            invalid_files.append({"file": str(path), "issue": error})
            continue
        assert document is not None
        for entry in _entries(document, path):
            if entry.get("case_id") is not None:
                case_id = str(entry["case_id"])
                records.setdefault(case_id, []).append((entry, path))

    cases: list[dict[str, Any]] = []
    for case_id in sorted(set(planned_by_id) | set(records)):
        expected_case = planned_by_id.get(case_id, {})
        found = records.get(case_id, [])
        issues: list[str] = []
        if not found:
            status = "missing"
            entry: dict[str, Any] = {}
        else:
            entry = found[0][0]
            if len(found) > 1:
                issues.append("duplicate_case_id")
            for name, expected_hash in expected.items():
                actual = entry.get(f"{name}_hash", expected_hash)
                if expected_hash is not None and actual != expected_hash:
                    issues.append(f"{name}_hash_mismatch")
            if expected_case.get("source_hash") is not None and entry.get("source_hash") != expected_case["source_hash"]:
                issues.append("source_hash_mismatch")
            metric_value = entry.get("metrics", {})
            bad = _bad_metrics(metric_value)
            if bad:
                issues.append("non_finite_metric")
            target_present = expected_case.get("target_present", entry.get("target_present", True))
            groups = _metric_groups(metric_value)
            if not target_present:
                for values in groups.values():
                    if any(key in values and values[key] is not None for key in SI_KEYS):
                        issues.append("si_sdri_not_applicable")
                        break
            status_value = str(entry.get("status", "complete" if not entry.get("partial") else "partial"))
            status = "partial" if status_value in {"partial", "running", "in_progress"} else "complete"
            if issues:
                status = "invalid"
            if status == "complete" and not groups and "metrics" in entry:
                issues.append("metrics_missing_groups")
                status = "invalid"
        cases.append({
            "case_id": case_id,
            "status": status,
            "issues": sorted(set(issues)),
            "source_hash": entry.get("source_hash", expected_case.get("source_hash")),
            "config_hash": entry.get("config_hash", expected.get("config")),
            "model_hash": entry.get("model_hash", expected.get("model")),
            "profile_hash": entry.get("profile_hash", expected.get("profile")),
            "metrics": entry.get("metrics", {}),
            "files": sorted({str(path) for _, path in found}),
        })

    resume_cases = [row["case_id"] for row in cases if row["status"] != "complete"]
    resume_files = sorted({
        file for row in cases if row["status"] != "complete" for file in row["files"]
    })[:MAX_FILES]
    available = sum(row["status"] in {"complete", "partial", "invalid"} for row in cases)
    report = {
        "schema": "overnight_prep_checkpoint_audit.v1",
        "manifest": str(manifest_file),
        "manifest_sha256": _hash_file(manifest_file),
        "planned_cases": len(planned_by_id),
        "available_cases": available,
        "complete_cases": sum(row["status"] == "complete" for row in cases),
        "partial_cases": sum(row["status"] == "partial" for row in cases),
        "invalid_cases": sum(row["status"] == "invalid" for row in cases),
        "missing_cases": sum(row["status"] == "missing" for row in cases),
        "invalid_files": invalid_files,
        "cases": cases,
        "resume": {
            "case_ids": resume_cases[:MAX_FILES],
            "files": resume_files,
            "max_files": MAX_FILES,
            "source_changes_invalidate_reuse": True,
        },
    }
    return report


def write_snapshot(destination: str | Path, report: dict[str, Any]) -> None:
    """Write a complete snapshot atomically; never append to an existing report."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    resume = report.get("resume")
    if isinstance(resume, dict) and isinstance(resume.get("files"), list):
        resume["files"] = sorted(map(str, resume["files"]))[:MAX_FILES]
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.inputs)
    write_snapshot(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "planned_cases": report["planned_cases"],
        "available_cases": report["available_cases"],
        "resume_cases": len(report["resume"]["case_ids"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
