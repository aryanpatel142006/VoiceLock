"""Inventory local speech/stem fixtures without acquiring or transcribing media.

The command is deliberately metadata-first: headers and manifests are read before
decoding, and only a bounded amount of audio is decoded for signal metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

MAX_FILES = 1000
MAX_DECODE_SECONDS = 60 * 60
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_for(path: Path, root: Path) -> tuple[Path | None, dict[str, Any]]:
    for parent in (path.parent, *path.parents):
        if root not in parent.parents and parent != root:
            continue
        candidate = parent / "manifest.json"
        if candidate.is_file():
            try:
                return candidate, json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                return candidate, {}
        if parent == root:
            break
    return None, {}


def _path_role(
    path: Path, manifest: dict[str, Any], manifest_path: Path | None
) -> tuple[str, str | None]:
    paths = manifest.get("paths") or manifest.get("stems") or {}
    for role, value in paths.items():
        if isinstance(value, str) and (
            Path(value).as_posix() == path.name
            or (manifest_path and Path(value) == path.relative_to(manifest_path.parent))
        ):
            return str(role), str(role)
    stem = path.stem.lower()
    for role in ("target", "interferer", "other", "noise", "mix"):
        if role in stem:
            return role, role
    return "unknown", None


def _metadata(path: Path, manifest_path: Path | None, manifest: dict[str, Any], role: str):
    source = manifest.get("source")
    source = source if isinstance(source, dict) else {}
    utterance = manifest.get(f"{role}_id") or manifest.get("utterance_id")
    speaker = manifest.get(f"{role}_speaker_id")
    if isinstance(utterance, str) and speaker is None:
        speaker = utterance.split("-", 1)[0]
        if not speaker.isdigit() and "-utt-" in utterance:
            speaker = utterance.split("-utt-", 1)[0]
    for entry in manifest.get("speech", []):
        if entry.get("file") == path.name:
            utterance = entry.get("id", utterance)
            speaker = str(entry.get("speaker_id", speaker)) if entry.get("speaker_id") else speaker
    face_intervals = None
    for face in manifest.get("faces", []):
        if face.get("role") == ("interferer" if role == "other" else role):
            utterance = utterance or face.get("actor")
            speaker = speaker or face.get("actor")
            if face.get("speaking"):
                face_intervals = face["speaking"]
                break
    intervals = manifest.get(f"{role}_intervals")
    if intervals is None and role == "target":
        intervals = manifest.get("target_intervals")
    if intervals is None and role in {"other", "interferer"}:
        intervals = manifest.get("interferer_intervals")
    if intervals is None:
        intervals = face_intervals
    if intervals is None:
        sentences = [
            [item.get("start_s"), item.get("end_s")]
            for item in manifest.get("sentences", [])
            if item.get("who") in {role, "interferer" if role == "other" else role}
        ]
        intervals = sentences or None
    transcript_ref = None
    if manifest_path and ("transcript" in manifest or "text" in manifest):
        transcript_ref = f"{manifest_path.name}:"
        transcript_ref += "transcript" if "transcript" in manifest else "text"
    return {
        "dataset": source.get("dataset") or manifest.get("dataset"),
        "speaker_id": speaker,
        "utterance_id": utterance,
        "source_intervals_s": intervals,
        "transcript_reference": transcript_ref,
    }


def _signal_metrics(path: Path, info: Any, remaining_samples: int) -> tuple[dict[str, float] | None, int]:
    if remaining_samples <= 0:
        return None, 0
    chunks = []
    used = 0
    with sf.SoundFile(path) as audio:
        while used < remaining_samples:
            chunk = audio.read(
                min(262144, remaining_samples - used), dtype="float32", always_2d=True
            )
            if len(chunk) == 0:
                break
            if used + len(chunk) > remaining_samples:
                chunk = chunk[: remaining_samples - used]
            chunks.append(chunk)
            used += len(chunk)
    if not chunks:
        values = np.zeros(0, dtype=np.float32)
    else:
        values = np.concatenate(chunks, axis=0)
    flat = values.mean(axis=1) if values.ndim == 2 else values
    peak = float(np.max(np.abs(flat))) if len(flat) else 0.0
    rms = float(np.sqrt(np.mean(np.square(flat)))) if len(flat) else 0.0
    return {
        "peak": peak,
        "rms": rms,
        "silence_fraction": float(np.mean(np.abs(flat) < 1e-4)) if len(flat) else 1.0,
        "decoded_duration_s": len(flat) / float(info.samplerate),
    }, used


def _mix_check(files: dict[str, Path]) -> dict[str, Any]:
    mix = files.get("mix")
    stems = [files.get(name) for name in ("target", "interferer", "other", "noise")]
    stems = [path for path in stems if path is not None]
    if mix is None or not stems:
        return {"mix": str(mix) if mix else None, "stems": [str(p) for p in stems], "matches_sum": None}
    try:
        mix_pcm, _ = sf.read(mix, dtype="int16", always_2d=False)
        stem_arrays = [sf.read(path, dtype="int16", always_2d=False)[0] for path in stems]
        if any(len(arr) != len(mix_pcm) for arr in stem_arrays):
            return {"mix": str(mix), "stems": [str(p) for p in stems], "matches_sum": False}
        summed = np.sum(np.stack(stem_arrays), axis=0, dtype=np.int64)
        return {
            "mix": str(mix),
            "stems": [str(p) for p in stems],
            "max_pcm_error": int(np.max(np.abs(summed - mix_pcm))) if len(mix_pcm) else 0,
            "matches_sum": bool(np.all(np.abs(summed - mix_pcm) <= 2)),
        }
    except (OSError, RuntimeError, ValueError):
        return {"mix": str(mix), "stems": [str(p) for p in stems], "matches_sum": False}


def build_inventory(root: str | os.PathLike, output: str | os.PathLike):
    root = Path(root).resolve()
    output = Path(output)
    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    report: dict[str, Any] = {
        "root": str(root),
        "files_seen": min(len(candidates), MAX_FILES),
        "missing": [],
        "corrupt": [],
        "duplicates": {},
        "mix_checks": [],
        "decoded_seconds": 0.0,
        "decode_limit_seconds": MAX_DECODE_SECONDS,
    }
    rows: list[dict[str, Any]] = []
    hashes: dict[str, list[str]] = {}
    header_records: list[tuple[Path, Any, Path | None, dict[str, Any]]] = []
    for path in candidates[:MAX_FILES]:
        manifest_path, manifest = _manifest_for(path, root)
        try:
            info = sf.info(path)
            digest = _sha256(path)
        except (OSError, RuntimeError, ValueError):
            report["corrupt"].append(str(path.relative_to(root)))
            continue
        hashes.setdefault(digest, []).append(str(path.relative_to(root)))
        header_records.append((path, info, manifest_path, manifest))
    report["duplicates"] = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for relative in (manifest.get("paths") or manifest.get("stems") or {}).values():
            if not isinstance(relative, str):
                continue
            expected = manifest_path.parent / relative
            if not expected.is_file():
                report["missing"].append(str(expected.relative_to(root)))
    report["missing"] = sorted(set(report["missing"]))
    remaining = MAX_DECODE_SECONDS * 16000
    grouped: dict[str, dict[str, Path]] = {}
    for path, info, manifest_path, manifest in header_records:
        role, _ = _path_role(path, manifest, manifest_path)
        metrics, used = _signal_metrics(path, info, remaining)
        remaining -= used
        report["decoded_seconds"] += used / float(info.samplerate)
        meta = _metadata(path, manifest_path, manifest, role)
        row = {
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "sha256": hashes[next(key for key, value in hashes.items() if str(path.relative_to(root)) in value)],
            "bytes": path.stat().st_size,
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_s": float(info.frames / info.samplerate),
            "role": role,
            **meta,
            "metrics": metrics,
        }
        rows.append(row)
        paths = manifest.get("paths") or manifest.get("stems") or {}
        for manifest_role, relative in paths.items():
            expected = path.parent / relative if isinstance(relative, str) else None
            if expected == path:
                grouped.setdefault(str(manifest_path), {})[str(manifest_role)] = path
    for manifest_path, files in grouped.items():
        report["mix_checks"].append(_mix_check(files))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures"))
    parser.add_argument("--output", required=True, help="Ignored JSONL destination for source_catalog")
    parser.add_argument("--summary", help="Optional JSON summary destination")
    args = parser.parse_args()
    _, report = build_inventory(args.root, args.output)
    if args.summary:
        Path(args.summary).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
