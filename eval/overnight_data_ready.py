"""Validate existing overnight fixture manifests and write a metadata-only handoff.

This module never downloads or rewrites fixtures.  It only reads manifests and WAV headers,
then writes one JSON object per usable case to an ignored results path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

STEMS = ("mix", "target", "interferer", "noise")
DEFAULT_FIXTURES = Path(os.environ.get("HEARING_FIXTURES_DIR", Path(__file__).parent / "fixtures" / "generated"))
DEFAULT_OUTPUT = Path(os.environ.get("HEARING_RESULTS_DIR", Path(__file__).parent / "results")) / "overnight_grunt" / "ready.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _license(manifest: dict[str, Any]) -> str:
    source = manifest.get("source", {})
    if isinstance(source, dict) and isinstance(source.get("license"), str):
        return source["license"].strip()
    for role in ("target", "interferer"):
        value = source.get(role) if isinstance(source, dict) else None
        if isinstance(value, dict) and isinstance(value.get("license"), str):
            return value["license"].strip()
    return ""


def _record_failure(manifest_path: Path, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "excluded", "manifest": str(manifest_path), "reason": reason, **extra}


def validate_case(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate one existing fixture manifest without changing any source files."""
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return _record_failure(manifest_path, "malformed_manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("paths"), dict):
        return _record_failure(manifest_path, "malformed_manifest")

    case_root = manifest_path.parent
    paths: dict[str, Path] = {}
    for stem in STEMS:
        raw = manifest["paths"].get(stem)
        if not isinstance(raw, str) or not raw:
            return _record_failure(manifest_path, "invalid_reference")
        path = (case_root / raw).resolve()
        if case_root not in path.parents:
            return _record_failure(manifest_path, "path_escape")
        paths[stem] = path

    target_absent = not bool(manifest.get("target_id")) or bool(manifest.get("target_absent"))
    transcript = str(manifest.get("transcript") or "").strip()
    transcript_info = {"status": "complete" if transcript else "unavailable"}
    license_text = _license(manifest)
    if not license_text:
        return _record_failure(
            manifest_path, "missing_license", target_absent=target_absent, transcript=transcript_info
        )
    if not manifest.get("interferer_id"):
        return _record_failure(
            manifest_path, "invalid_reference", target_absent=target_absent, transcript=transcript_info
        )

    files: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    header: tuple[int, int, int] | None = None
    expected_hashes = manifest.get("hashes", {})
    if not isinstance(expected_hashes, dict):
        return _record_failure(manifest_path, "malformed_manifest")
    for stem, path in paths.items():
        if not path.is_file():
            return _record_failure(manifest_path, "missing_file", missing=stem)
        try:
            info = sf.info(str(path))
            arrays[stem], sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        except (OSError, RuntimeError, ValueError):
            return _record_failure(manifest_path, "invalid_audio", stem=stem)
        if info.channels != 1 or sample_rate != 16000:
            return _record_failure(manifest_path, "invalid_audio_format", stem=stem)
        if arrays[stem].ndim != 1 or arrays[stem].size == 0:
            return _record_failure(manifest_path, "invalid_audio", stem=stem)
        current_hash = _sha256(path)
        expected = expected_hashes.get(stem)
        if expected is not None and expected != current_hash:
            return _record_failure(manifest_path, "hash_mismatch", stem=stem)
        files[stem] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": current_hash,
            "frames": int(info.frames),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_s": float(info.duration),
        }
        current_header = (info.frames, info.samplerate, info.channels)
        if header is None:
            header = current_header
        elif current_header != header:
            return _record_failure(manifest_path, "inconsistent_audio")

    total = arrays["target"] + arrays["interferer"] + arrays["noise"]
    stem_error = float(np.max(np.abs(total - arrays["mix"])))
    if stem_error > 1e-3:
        return _record_failure(manifest_path, "stems_do_not_add_up", stems_sum_max_abs=stem_error)

    source = manifest.get("source", {})
    dataset = source.get("dataset", "") if isinstance(source, dict) else ""
    split = source.get("split", "") if isinstance(source, dict) else ""
    target_partition = manifest.get("target_id") or "target-absent"
    partition = f"{dataset}|{split}|{target_partition}|{manifest['interferer_id']}"
    return {
        "status": "ready",
        "manifest": str(manifest_path),
        "id": str(manifest.get("id", manifest_path.parent.name)),
        "set": str(manifest.get("set", "")),
        "sample_rate": 16000,
        "channels": 1,
        "duration_s": float(manifest.get("duration_s", files["mix"]["duration_s"])),
        "files": files,
        "license": license_text,
        "source": source,
        "source_partition": partition,
        "target_id": manifest.get("target_id", ""),
        "interferer_id": manifest["interferer_id"],
        "target_absent": target_absent,
        "sisdr_eligible": not target_absent,
        "transcript": transcript_info,
        "stems_sum_max_abs": stem_error,
    }


def _manifest_paths(root: Path) -> list[Path]:
    indexes = sorted(root.rglob("index.json"))
    found: list[Path] = []
    for index in indexes:
        try:
            data = json.loads(index.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for entry in data.get("fixtures", []) if isinstance(data, dict) else []:
            paths = entry.get("paths", {}) if isinstance(entry, dict) else {}
            first = paths.get("mix") if isinstance(paths, dict) else None
            if isinstance(first, str):
                found.append((index.parent / first).parent / "manifest.json")
    if found:
        return sorted(set(found))
    return sorted(root.rglob("manifest.json"))


def assemble_ready(
    roots: list[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    limit: int = 12,
) -> dict[str, Any]:
    """Collect distinct usable cases, writing only ready records to ``output``."""
    candidates = sorted({path for root in roots for path in _manifest_paths(Path(root))})
    records: list[dict[str, Any]] = []
    excluded = Counter()
    seen_partitions: set[str] = set()
    for manifest_path in candidates:
        result = validate_case(manifest_path)
        if result["status"] != "ready":
            excluded[result.get("reason", "unknown")] += 1
            continue
        partition = result["source_partition"]
        if partition in seen_partitions:
            excluded["duplicate_source_partition"] += 1
            continue
        seen_partitions.add(partition)
        records.append(result)
        if len(records) >= limit:
            break
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    return {
        "ready": len(records),
        "excluded": sum(excluded.values()),
        "excluded_reasons": dict(excluded),
        "pending": max(0, limit - len(records)),
        "candidates": len(candidates),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-root", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    roots = args.fixtures_root or [DEFAULT_FIXTURES]
    summary = assemble_ready(roots, args.output, limit=args.limit)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
