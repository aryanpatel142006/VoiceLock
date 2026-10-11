"""Build a conservative, deterministic leakage audit for overnight data shards.

The input is metadata only. Audio is never downloaded or opened by this module. Rows
with missing provenance are explicitly excluded instead of being treated as safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PARTITIONS = ("enrollment", "development", "holdout")
REQUIRED_PROVENANCE = ("dataset", "license")


def intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    """Return whether two half-open time intervals overlap by a positive duration."""
    return max(left[0], right[0]) < min(left[1], right[1])


def _source_key(item: dict[str, Any]) -> str:
    return str(item.get("source_id") or item.get("utterance_id") or item.get("asset_id"))


def _speaker_key(item: dict[str, Any]) -> str:
    return str(item.get("speaker_id") or item.get("speaker") or "unknown")


def _stable_bucket(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_partitions(rows: list[dict[str, Any]], seed: int = 229) -> list[dict[str, Any]]:
    """Assign source groups deterministically; reserve holdout speakers."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        groups[(_speaker_key(item), _source_key(item))].append(item)
    speakers = sorted({_speaker_key(item) for item in rows})
    holdout_speakers = {
        speaker
        for speaker in speakers
        if _stable_bucket(f"speaker:{speaker}", seed) % 5 == 0
    }
    if len(holdout_speakers) == len(speakers) and speakers:
        holdout_speakers.remove(min(holdout_speakers))
    output: list[dict[str, Any]] = []
    for (speaker, source), members in sorted(groups.items()):
        if speaker in holdout_speakers:
            partition = "holdout"
        else:
            partition = "development" if _stable_bucket(source, seed) % 4 == 0 else "enrollment"
        for item in members:
            copy = dict(item)
            copy["partition"] = partition
            output.append(copy)
    return output


def _provenance_known(item: dict[str, Any]) -> bool:
    provenance = item.get("provenance")
    return isinstance(provenance, dict) and all(provenance.get(key) for key in REQUIRED_PROVENANCE)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_rows(rows: list[dict[str, Any]], seed: int = 229) -> dict[str, list[dict[str, Any]]]:
    """Assign and audit rows, returning public-safe eligible/excluded records."""
    assigned = assign_partitions(rows, seed)
    excluded: list[dict[str, Any]] = []
    reasons_by_asset: dict[str, set[str]] = defaultdict(set)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in assigned:
        asset = str(item.get("asset_id", ""))
        if not _provenance_known(item):
            reasons_by_asset[asset].add("unknown_provenance")
        source = _source_key(item)
        by_source[source].append(item)
        digest = item.get("sha256")
        if digest:
            by_hash[str(digest)].append(item)
    for members in by_source.values():
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if intervals_overlap(
                    (float(left.get("start_s", 0)), float(left.get("end_s", 0))),
                    (float(right.get("start_s", 0)), float(right.get("end_s", 0))),
                ):
                    reasons_by_asset[str(left.get("asset_id", ""))].add("temporal_source_overlap")
                    reasons_by_asset[str(right.get("asset_id", ""))].add("temporal_source_overlap")
    for members in by_hash.values():
        if len(members) > 1:
            for item in members:
                reasons_by_asset[str(item.get("asset_id", ""))].add("duplicate_sha256")
    holdout_speakers = {_speaker_key(item) for item in assigned if item.get("partition") == "holdout"}
    for item in assigned:
        if item.get("partition") != "holdout" and _speaker_key(item) in holdout_speakers:
            reasons_by_asset[str(item.get("asset_id", ""))].add("speaker_overlap_with_holdout")
    eligible = [item for item in assigned if not reasons_by_asset.get(str(item.get("asset_id", "")))]
    for item in assigned:
        reasons = sorted(reasons_by_asset.get(str(item.get("asset_id", "")), set()))
        if reasons:
            copy = dict(item)
            copy["reasons"] = reasons
            excluded.append(copy)
    return {"eligible": eligible, "excluded": excluded}


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    """Extract speech rows from a fixture manifest without reading media."""
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text())
    base_provenance = {
        "dataset": data.get("dataset"),
        "split": data.get("split"),
        "license": data.get("license"),
    }
    rows: list[dict[str, Any]] = []
    for entry in data.get("speech", []):
        text = str(entry.get("text", "")).strip()
        media_path = manifest_path.parent / entry.get("file", "")
        rows.append(
            {
                "asset_id": str(entry.get("id") or entry.get("file")),
                "speaker_id": str(entry.get("speaker_id", "")),
                "source_id": str(entry.get("id") or entry.get("file")),
                "start_s": 0.0,
                "end_s": float(entry.get("source_duration_s", data.get("duration_s", 0.0))),
                "transcript": text,
                "transcript_valid": bool(text),
                "provenance": {key: value for key, value in base_provenance.items() if value},
                "path": str(media_path.resolve()),
                "sha256": _file_sha256(media_path),
            }
        )
    return rows


def discover_manifests(roots: list[Path]) -> list[Path]:
    return sorted(path for root in roots for path in root.rglob("manifest.json") if path.is_file())


def build_ledger(roots: list[Path], seed: int = 229) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for manifest in discover_manifests(roots):
        rows.extend(load_manifest_rows(manifest))
    result = audit_rows(rows, seed)
    public_result = {
        key: [{field: value for field, value in item.items() if field != "path"} for item in values]
        for key, values in result.items()
    }
    return {
        "schema": 1,
        "partition_seed": seed,
        "source_manifests": [str(path) for path in discover_manifests(roots)],
        "counts": {
            "input": len(rows),
            "eligible": len(result["eligible"]),
            "excluded": len(result["excluded"]),
        },
        **public_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("eval/fixtures")])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=229)
    args = parser.parse_args()
    ledger = build_ledger(args.roots, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
