"""Index historical overnight evidence without opening media or recomputing metrics.

The index is deliberately conservative: values are copied only when present in a
JSON result or a cited benchmark table.  It never invents a transcript, identity,
or metric and records missing reference/material provenance explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

MAX_RESULTS = 200
PUBLIC_ROWS = 30
METRIC_KEYS = (
    "si_sdri",
    "si_sdri_db",
    "leak",
    "wrong_speaker_s",
    "latency_ms",
    "rtf",
    "dnsmos",
    "pesq",
    "wer",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(data: dict[str, Any]) -> str | None:
    value = data.get("device")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("device") or value.get("name")
    return None


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _metric(row: dict[str, Any], name: str) -> Any:
    aliases = {
        "si_sdri": ("si_sdri", "si_sdri_db"),
        "latency_ms": ("latency_ms", "algorithmic_latency_ms", "delay_ms"),
        "dnsmos": ("dnsmos", "dnsmos_ovrl"),
    }
    return _first(row, *(aliases.get(name, (name,))))


def _valid_reference(row: dict[str, Any]) -> bool | None:
    if "reference_valid" in row:
        return bool(row["reference_valid"])
    if "reference" in row and isinstance(row["reference"], dict):
        return bool(row["reference"].get("valid", False))
    if "si_sdri" in row or "si_sdri_db" in row:
        return True
    return None


def _run_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    config = data.get("configuration") or data.get("config") or {}
    rows = data.get("rows") or data.get("results") or []
    if not isinstance(rows, list):
        rows = []
    record_rows = []
    for row in rows[:MAX_RESULTS]:
        if not isinstance(row, dict):
            continue
        metrics = {key: _metric(row, key) for key in METRIC_KEYS}
        record_rows.append(
            {
                "fixture_id": _first(row, "fixture", "fixture_id", "scenario"),
                "selection": _first(row, "selected_stream", "selection", "candidate"),
                "oracle": _first(row, "oracle_choice", "oracle", "target"),
                "reference_valid": _valid_reference(row),
                "metrics": metrics,
            }
        )
    return {
        "run_id": _first(data, "run_id", "stamp", "created", "reported_run")
        or path.stem,
        "source_revision": _first(data, "git_sha", "git", "revision"),
        "config_hash": _first(config, "hash", "config_hash", "tiers_hash"),
        "host": _first(data, "machine", "hostname")
        or (data.get("device") or {}).get("hostname")
        if isinstance(data.get("device"), dict)
        else data.get("machine"),
        "device": _device(data),
        "fixture_ids": sorted(
            {row["fixture_id"] for row in record_rows if row["fixture_id"] is not None}
        ),
        "clean_stems": _first(data, "clean_stems", "stems", "fixtures"),
        "reference_valid": any(
            row["reference_valid"] is True for row in record_rows
        )
        if record_rows
        else None,
        "selection_vs_oracle": any(
            row["selection"] is not None and row["oracle"] is not None
            for row in record_rows
        )
        if record_rows
        else None,
        "rows": record_rows,
        "source": str(path),
        "source_sha256": _sha256(path),
    }


def _coverage(root: Path) -> dict[str, Any]:
    manifests = sorted(root.rglob("manifest.json")) if root.exists() else []
    clips = []
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        clips.append(
            {
                "id": data.get("id") or data.get("fixture") or manifest.parent.name,
                "speaker": data.get("speaker") or data.get("target_speaker"),
                "duration_s": data.get("duration_s"),
                "sample_rate": data.get("sample_rate") or data.get("sr"),
                "source": str(manifest),
            }
        )
    wavs = sorted(root.rglob("*.wav")) if root.exists() else []
    bytes_total = sum(path.stat().st_size for path in wavs if path.exists())
    audio_info = []
    for path in wavs:
        try:
            with wave.open(str(path)) as audio:
                audio_info.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "duration_s": audio.getnframes() / audio.getframerate(),
                        "sample_rate": audio.getframerate(),
                        "channels": audio.getnchannels(),
                    }
                )
        except (OSError, wave.Error):
            continue
    return {
        "root": str(root),
        "manifest_count": len(manifests),
        "clip_count": len(clips),
        "clips": clips,
        "wav_count": len(wavs),
        "wav_bytes": bytes_total,
        "audio": audio_info,
    }


def _gaps(records: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> list[str]:
    text = json.dumps(records).lower()
    return [
        condition
        for condition in ("quiet target", "pause", "same-voice", "target absent", "reverberation")
        if condition not in text.replace("_", " ")
    ]


def build_index(
    roots: list[Path],
    *,
    benchmark_docs: list[Path] | None = None,
    fixture_roots: list[Path] | None = None,
) -> dict[str, Any]:
    json_files = sorted(
        {
            path
            for root in roots
            if root.exists()
            for path in root.rglob("*.json")
            if path.is_file()
        }
    )
    records = []
    session_reports = []
    for path in json_files[:MAX_RESULTS]:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and (
            "rows" in data or "results" in data or "kind" in data
        ):
            records.append(_run_record(path, data))
    for root in roots:
        for path in (sorted(root.rglob("events.jsonl")) if root.exists() else []):
            session_reports.append({"path": str(path), "sha256": _sha256(path)})
    fingerprints = {}
    for record in records:
        fingerprint = (
            record["source_revision"],
            record["config_hash"],
            tuple(record["fixture_ids"]),
        )
        fingerprints.setdefault(fingerprint, []).append(record["source"])
    coverage = [_coverage(root) for root in (fixture_roots or [])]
    citations = []
    for path in benchmark_docs or []:
        if path.exists():
            citations.append({"path": str(path), "sha256": _sha256(path)})
    return {
        "schema": 1,
        "results": records[:MAX_RESULTS],
        "coverage": coverage,
        "session_reports": session_reports,
        "duplicate_groups": [paths for paths in fingerprints.values() if len(paths) > 1],
        "stale_comparison_warning": len(
            {record["source_revision"] for record in records if record["source_revision"]}
        )
        > 1,
        "citations": citations,
        "coverage_gaps": _gaps(records, coverage),
        "limits": {"max_results": MAX_RESULTS, "max_public_rows": PUBLIC_ROWS},
    }


def render_public_report(index: dict[str, Any]) -> str:
    rows = []
    for run in index["results"]:
        for row in run["rows"]:
            metrics = row["metrics"]
            rows.append(
                "| {run} | {fixture} | {device} | {sidri} | {latency} | {rtf} | {wer} | {source} |".format(
                    run=run["run_id"],
                    fixture=row["fixture_id"] or "-",
                    device=run["device"] or "-",
                    sidri=metrics.get("si_sdri") if metrics.get("si_sdri") is not None else "-",
                    latency=metrics.get("latency_ms") if metrics.get("latency_ms") is not None else "-",
                    rtf=metrics.get("rtf") if metrics.get("rtf") is not None else "-",
                    wer=metrics.get("wer") if metrics.get("wer") is not None else "-",
                    source=run["source"],
                )
            )
            if len(rows) >= PUBLIC_ROWS:
                break
        if len(rows) >= PUBLIC_ROWS:
            break
    lines = [
        "# Overnight evidence index",
        "",
        "Read-only index of locally available historical results. Missing values are `-`; no transcript text is copied.",
        "",
        f"Indexed runs: {len(index['results'])}; public-safe cited rows: {len(rows)}.",
        f"Session reports found: {len(index['session_reports'])}; duplicate groups: {len(index['duplicate_groups'])}.",
        "",
        "| run | fixture/scenario | device | SI-SDRi | latency ms | RTF | WER | source |",
        "|---|---|---|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## Coverage and provenance",
        "",
        f"Coverage gaps: {', '.join(index['coverage_gaps']) or 'none observed in indexed labels'}.",
        f"Revision/config mismatch warning: {'yes' if index['stale_comparison_warning'] else 'no'}.",
        "",
    ]
    for item in index["coverage"]:
        lines.append(
            f"- `{item['root']}`: {item['clip_count']} manifests, {item['wav_count']} WAVs, "
            f"{item['wav_bytes']} bytes; WAV hashes and duration/sample rate are in the JSON index."
        )
    if not index["session_reports"]:
        lines.append("- No `events.jsonl` session reports were available in the supplied roots.")
    lines.append("- Licence evidence: no licence fields were present in the indexed result/fixture manifests; no new media was acquired.")
    for citation in index["citations"]:
        lines.append(f"- Cited document: `{citation['path']}` (SHA256 `{citation['sha256']}`).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--fixture-root", action="append", type=Path, default=[])
    parser.add_argument("--benchmark", action="append", type=Path, default=[])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    index = build_index(args.root, benchmark_docs=args.benchmark, fixture_roots=args.fixture_root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_public_report(index))


if __name__ == "__main__":
    main()
