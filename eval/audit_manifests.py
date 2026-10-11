"""Validate rendered shard manifests and their audio headers.

The audit is metadata-first: it reads audio headers and hashes files without
decoding samples, making it suitable for an overnight render directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import soundfile as sf

REQUIRED_FIELDS = ("schema_version", "shard_id", "audio", "sample_rate", "channels", "duration_s")
DEFAULT_MIN_DURATION_S = 0.01
DEFAULT_MAX_DURATION_S = 3600.0


def _issue(manifest: Path, code: str, message: str) -> dict[str, str]:
    return {"manifest": str(manifest), "code": code, "message": message}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_name(manifest: dict[str, Any]) -> str | None:
    value = manifest.get("audio")
    if value is None:
        value = manifest.get("audio_path")
    return value if isinstance(value, str) and value else None


def audit_shards(
    root: str | Path,
    *,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> dict[str, Any]:
    """Audit every ``manifest.json`` below *root* and return a JSON report."""
    root = Path(root).resolve()
    issues: list[dict[str, str]] = []
    manifests = sorted(root.rglob("manifest.json")) if root.exists() else []
    seen_ids: dict[str, Path] = {}
    audio_files = 0
    records: list[dict[str, Any]] = []

    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            issues.append(_issue(manifest_path, "invalid_manifest", str(exc)))
            continue
        if not isinstance(payload, dict):
            issues.append(_issue(manifest_path, "invalid_manifest", "manifest must be a JSON object"))
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        for field in missing:
            issues.append(_issue(manifest_path, "missing_field", f"missing required field: {field}"))

        shard_id = payload.get("shard_id")
        if isinstance(shard_id, str) and shard_id:
            previous = seen_ids.get(shard_id)
            if previous is not None:
                issues.append(
                    _issue(manifest_path, "duplicate_shard_id", f"{shard_id!r} already declared by {previous}")
                )
            else:
                seen_ids[shard_id] = manifest_path
        else:
            issues.append(_issue(manifest_path, "invalid_field", "shard_id must be a non-empty string"))

        audio_name = _audio_name(payload)
        if audio_name is None:
            continue
        audio_path = (manifest_path.parent / audio_name).resolve()
        try:
            audio_path.relative_to(root)
        except ValueError:
            issues.append(_issue(manifest_path, "audio_outside_root", audio_name))
            continue
        if not audio_path.is_file():
            issues.append(_issue(manifest_path, "missing_audio", audio_name))
            continue

        audio_files += 1
        try:
            info = sf.info(audio_path)
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(_issue(manifest_path, "invalid_audio", str(exc)))
            continue
        audio_hash = _sha256(audio_path)
        records.append(
            {
                "manifest": str(manifest_path),
                "shard_id": payload.get("shard_id"),
                "audio": str(audio_path.relative_to(root)),
                "sha256": audio_hash,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "subtype": info.subtype,
                "duration_s": info.frames / info.samplerate,
            }
        )

        if info.samplerate != 16000:
            issues.append(_issue(manifest_path, "sample_rate", f"expected 16000, got {info.samplerate}"))
        if info.channels != 1:
            issues.append(_issue(manifest_path, "channels", f"expected mono, got {info.channels}"))
        if info.subtype != "FLOAT":
            issues.append(_issue(manifest_path, "sample_format", f"expected float32 WAV, got {info.subtype}"))
        if payload.get("sample_rate") != info.samplerate or payload.get("channels") != info.channels:
            issues.append(
                _issue(
                    manifest_path,
                    "metadata_mismatch",
                    f"declared {payload.get('sample_rate')}Hz/{payload.get('channels')}ch, "
                    f"measured {info.samplerate}Hz/{info.channels}ch",
                )
            )

        actual_duration = info.frames / info.samplerate
        declared_duration = payload.get("duration_s")
        if not isinstance(declared_duration, (int, float)) or isinstance(declared_duration, bool):
            issues.append(_issue(manifest_path, "invalid_field", "duration_s must be numeric"))
        else:
            if not min_duration_s <= actual_duration <= max_duration_s:
                issues.append(
                    _issue(
                        manifest_path,
                        "duration",
                        f"expected {min_duration_s:g}-{max_duration_s:g}s, got {actual_duration:.6f}s",
                    )
                )
            if abs(float(declared_duration) - actual_duration) > 0.01:
                issues.append(
                    _issue(
                        manifest_path,
                        "duration_mismatch",
                        f"declared {float(declared_duration):.6f}s, measured {actual_duration:.6f}s",
                    )
                )

        declared_hash = payload.get("sha256") or payload.get("audio_sha256")
        if declared_hash is not None:
            if not isinstance(declared_hash, str) or len(declared_hash) != 64:
                issues.append(_issue(manifest_path, "invalid_hash", "sha256 must be a 64-character hex digest"))
            elif declared_hash.lower() != audio_hash:
                issues.append(_issue(manifest_path, "hash_mismatch", audio_name))

    if not manifests:
        issues.append(_issue(root, "no_manifests", "no manifest.json files found"))
    summary = {
        "manifests": len(manifests),
        "shards": len(seen_ids),
        "audio_files": audio_files,
        "issues": len(issues),
    }
    return {
        "status": "ok" if not issues else "error",
        "root": str(root),
        "summary": summary,
        "shards": records,
        "issues": issues,
        "duration_bounds_s": {"min": min_duration_s, "max": max_duration_s},
    }


audit_manifests = audit_shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", help="Rendered shard directory")
    parser.add_argument("--root", dest="root_option", type=Path, help="Rendered shard directory")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    root = args.root_option or args.root
    if root is None:
        parser.error("a rendered shard directory is required (pass ROOT or --root ROOT)")
    report = audit_shards(
        root, min_duration_s=args.min_duration, max_duration_s=args.max_duration
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
