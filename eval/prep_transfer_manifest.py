"""Build safe, replayable transfer manifests for prepared public eval inputs.

The output is portable metadata: local paths are intentionally never emitted.  Transfer
recipes use symbolic argv tokens and must be materialised by the GPU runner explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "selective-hearing.prep-transfer/v1"
MAX_TRANSFER_BYTES = 2 * 1024**3


class ManifestError(ValueError):
    """Raised when a catalog is unsafe or cannot be verified."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _catalog_entries(path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read catalog {path}: {exc}") from exc
    if not isinstance(catalog, dict) or not isinstance(catalog.get("files"), list):
        raise ManifestError(f"catalog {path} must contain a files array")
    license_info = catalog.get("license")
    if not license_info:
        raise ManifestError(f"catalog {path} lacks licence evidence")
    return path.parent.resolve(), catalog, catalog["files"]


def _license(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"spdx": value}
    if isinstance(value, dict) and (value.get("spdx") or value.get("name")):
        return dict(value)
    raise ManifestError("licence evidence must include spdx or name")


def _relative_path(root: Path, raw: Any) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ManifestError("file path must be a non-empty relative path")
    lexical = (root / raw).resolve(strict=False)
    if not _inside(root, lexical):
        raise ManifestError(f"path escapes catalog root: {raw}")
    if not lexical.is_file():
        raise ManifestError(f"file is missing or not regular: {raw}")
    return lexical, lexical.relative_to(root).as_posix()


def _resume_hashes(resume: Path | None) -> set[str]:
    if resume is None:
        return set()
    data = json.loads(resume.read_text(encoding="utf-8"))
    hashes = set(data.get("checkpoint", {}).get("complete_sha256", []))
    hashes.update(item["sha256"] for item in data.get("files", []) if item.get("sha256"))
    return hashes


def build_manifest(
    catalogs: list[str | os.PathLike[str]],
    *,
    max_bytes: int = MAX_TRANSFER_BYTES,
    shard_index: int = 0,
    shard_count: int = 1,
    completed_hashes: set[str] | None = None,
) -> dict[str, Any]:
    """Verify catalogs and return a deterministic, content-addressed transfer plan."""
    if not 0 <= shard_index < shard_count or shard_count < 1:
        raise ValueError("shard_index must be within shard_count")
    if not 0 <= max_bytes <= MAX_TRANSFER_BYTES:
        raise ValueError(f"max_bytes must be between 0 and {MAX_TRANSFER_BYTES}")
    completed = completed_hashes or set()
    candidates: dict[str, dict[str, Any]] = {}
    available = 0
    for catalog_path in sorted((Path(item).resolve() for item in catalogs), key=str):
        root, catalog, entries = _catalog_entries(catalog_path)
        licence = _license(catalog["license"])
        dataset = str(catalog.get("dataset", ""))
        revision = str(catalog.get("revision", ""))
        source_group = str(catalog.get("source_group", ""))
        if not dataset or not revision or not source_group:
            raise ManifestError(f"catalog {catalog_path} lacks dataset, revision, or source_group")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ManifestError(f"catalog {catalog_path} has a non-object file entry")
            actual, relative = _relative_path(root, entry.get("path"))
            digest = sha256_file(actual)
            declared = entry.get("sha256")
            if declared is not None and declared != digest:
                raise ManifestError(f"sha256 mismatch for {relative}")
            available += 1
            reference = {
                "source_id": str(entry.get("source_id") or f"{dataset}:{relative}"),
                "dataset": dataset,
                "revision": revision,
                "source_group": source_group,
                "relative_path": relative,
                "sha256": digest,
                "bytes": actual.stat().st_size,
                "license": licence,
            }
            if "speaker_id" in entry:
                reference["speaker_id"] = str(entry["speaker_id"])
            if "utterance_id" in entry:
                reference["utterance_id"] = str(entry["utterance_id"])
            if "interval_s" in entry:
                reference["interval_s"] = entry["interval_s"]
            item = candidates.setdefault(
                digest,
                {
                    "sha256": digest,
                    "bytes": reference["bytes"],
                    "references": [],
                    "license": licence,
                },
            )
            item["references"].append(reference)
    unique = sorted(candidates.values(), key=lambda item: item["sha256"])
    files: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    used = 0
    for item in unique:
        digest = item["sha256"]
        if int(digest[:8], 16) % shard_count != shard_index:
            excluded.append({"sha256": digest, "reason": "other_shard"})
        elif digest in completed:
            excluded.append({"sha256": digest, "reason": "resume_complete"})
        elif used + item["bytes"] > max_bytes:
            excluded.append({"sha256": digest, "reason": "byte_cap"})
        else:
            used += item["bytes"]
            files.append(
                {
                    "sha256": digest,
                    "bytes": item["bytes"],
                    "license": item["license"],
                    "source_id": item["references"][0]["source_id"],
                    "references": sorted(item["references"], key=lambda ref: ref["source_id"]),
                }
            )
    commands = [
        [
            "rsync",
            "--archive",
            "--protect-args",
            "--",
            f"CATALOG_ROOT/./{item['references'][0]['relative_path']}",
            f"GPU_DEST/{item['sha256']}",
        ]
        for item in files
    ]
    return {
        "schema": SCHEMA,
        "portable": True,
        "shard": {"index": shard_index, "count": shard_count},
        "limits": {"max_bytes": max_bytes},
        "available": {"files": available, "unique_bytes": sum(item["bytes"] for item in unique)},
        "planned": {"files": len(files), "bytes": used},
        "files": files,
        "excluded": excluded,
        "checkpoint": {"complete_sha256": [item["sha256"] for item in files]},
        "transfer": {
            "commands": commands,
            "substitutions": {
                "CATALOG_ROOT": "absolute local catalog parent, supplied by runner",
                "GPU_DEST": "unique destination shard directory, supplied by runner",
            },
            "no_shell": True,
        },
    }


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=MAX_TRANSFER_BYTES)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    result = build_manifest(
        args.catalog,
        max_bytes=args.max_bytes,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        completed_hashes=_resume_hashes(args.resume),
    )
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "available_files": result["available"]["files"],
                "planned_files": result["planned"]["files"],
                "planned_bytes": result["planned"]["bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
