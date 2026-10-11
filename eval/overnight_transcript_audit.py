"""Audit transcript and alignment claims in overnight fixture manifests.

This module deliberately audits metadata only.  It does not run ASR, forced
alignment, or any identity inference, and therefore labels absent references
as automatic/unverified rather than guessing a transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_metadata(path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    info = sf.info(path)
    return {
        "sample_rate": info.samplerate,
        "frames": info.frames,
        "duration_s": round(info.frames / info.samplerate, 6),
        "channels": info.channels,
    }


def _alignment_status(item: dict[str, Any]) -> str:
    words = item.get("words") or item.get("word_boundaries") or item.get("alignment")
    if not words:
        return "missing"
    if isinstance(words, list) and all(
        isinstance(word, dict)
        and ("start_s" in word or "start" in word)
        and ("end_s" in word or "end" in word)
        for word in words
    ):
        return "aligned"
    return "present_unusable"


def _safe_interval(item: dict[str, Any]) -> list[float] | None:
    start = item.get("start_s", item.get("start"))
    end = item.get("end_s", item.get("end"))
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and 0 <= start < end:
        return [float(start), float(end)]
    return None


def _source_id_status(item: dict[str, Any]) -> tuple[str, str | None]:
    declared = item.get("id") or item.get("utterance_id")
    source = item.get("source")
    source_id = source.get("id") if isinstance(source, dict) else None
    if source_id is not None and declared is not None and source_id != declared:
        return "mismatch", f"source id {source_id!r} differs from utterance id {declared!r}"
    if source_id is not None or declared is not None:
        return "ok", None
    return "missing", None


def _utterance_status(
    item: dict[str, Any], manifest: dict[str, Any], seen_ids: set[str]
) -> str:
    if (
        manifest.get("tiled_period_s") is not None
        or isinstance(manifest.get("noise"), dict)
        and manifest["noise"].get("tiled_period_s") is not None
        or manifest.get("looped") is True
        or item.get("looped") is True
        or item.get("repeat") is True
    ):
        return "repeated_or_looped"
    utterance_id = item.get("id") or item.get("utterance_id")
    if utterance_id and utterance_id in seen_ids:
        return "repeated_or_looped"
    clip_duration = manifest.get("duration_s")
    source_duration = item.get("source_duration_s")
    if (
        isinstance(clip_duration, (int, float))
        and isinstance(source_duration, (int, float))
        and clip_duration + 0.05 < source_duration
    ):
        return "truncated"
    return "full_reference_candidate"


def _entries(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(manifest.get("speech", [])):
        if isinstance(item, dict):
            entries.append(("target" if index == 0 else "interferer", item))
    for item in manifest.get("sentences", []):
        if isinstance(item, dict):
            who = str(item.get("who", "unknown"))
            entries.append((who, item))
    return entries


def audit_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Return one public-safe validity record per transcript-bearing entry."""
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for role, item in _entries(manifest):
        issues: list[str] = []
        text = item.get("text")
        transcript_status = "exact_reference" if isinstance(text, str) and text.strip() else "automatic_unverified"
        source_file = item.get("file")
        file_path = root / source_file if isinstance(source_file, str) else None
        file_status = "ok" if file_path and file_path.is_file() else "missing"
        if file_status == "missing":
            issues.append("source file missing")
        source_id_status, source_issue = _source_id_status(item)
        if source_issue:
            issues.append("source_id_mismatch")
            issues.append(source_issue)
        utterance_status = _utterance_status(item, manifest, seen_ids)
        if utterance_status in {"truncated", "repeated_or_looped"}:
            issues.append(utterance_status)
        utterance_id = item.get("id") or item.get("utterance_id")
        if utterance_id:
            seen_ids.add(str(utterance_id))
        safe_interval = _safe_interval(item)
        if safe_interval is None:
            issues.append("sentence boundaries missing")
        alignment = _alignment_status(item)
        if alignment != "aligned":
            issues.append(f"word boundaries {alignment}")
        records.append(
            {
                "manifest": str(manifest_path),
                "fixture_id": manifest_path.parent.name,
                "role": role,
                "dataset": manifest.get("dataset")
                or manifest.get("source", {}).get("dataset"),
                "license": manifest.get("license")
                or manifest.get("source", {}).get("license"),
                "license_status": "declared"
                if manifest.get("license") or manifest.get("source", {}).get("license")
                else "missing",
                "utterance_id": utterance_id,
                "speaker_id": item.get("speaker_id"),
                "source_file": source_file,
                "source_file_sha256": _sha256(file_path) if file_path else None,
                "source_audio": _audio_metadata(file_path) if file_path else {},
                "source_file_status": file_status,
                "source_id_status": source_id_status,
                "transcript_status": transcript_status,
                "utterance_status": utterance_status,
                "word_boundaries": alignment,
                "safe_interval_s": safe_interval,
                "issues": issues,
            }
        )
    return records


def audit_roots(
    roots: list[str | Path], *, limit: int = 300, output: str | Path | None = None
) -> list[dict[str, Any]]:
    """Audit at most ``limit`` manifests, in deterministic path order."""
    if limit < 1:
        raise ValueError("limit must be positive")
    manifests = sorted({Path(root).resolve() for root in roots})
    paths: list[Path] = []
    for root in manifests:
        if root.is_file() and root.name == "manifest.json":
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.rglob("manifest.json")))
    records: list[dict[str, Any]] = []
    for path in paths[:limit]:
        records.extend(audit_manifest(path))
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="fixture roots or manifest paths")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("transcript_validity.jsonl"))
    args = parser.parse_args()
    records = audit_roots(args.roots, limit=args.limit, output=args.output)
    print(json.dumps({"manifests": len({record["manifest"] for record in records}), "records": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
