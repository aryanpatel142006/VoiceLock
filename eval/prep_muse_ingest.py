"""Offline ingestion and coverage stratification for Muse measurement exports.

This module intentionally does not run models.  It normalizes reports from prior
tasks, retaining missing values as missing (rather than converting them to zero)
and writes a deterministic manifest for bounded overnight ablations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

MAX_INPUTS = 500
MAX_BYTES = 1_000_000_000
MAX_CASES = 300
METRICS = ("si_sdri", "leak", "wrong_speaker", "dropout", "latency_ms", "target_loss")
STAGES = ("oracle", "picked", "mixer")
ABLATIONS = (
    {"id": "role-x-sir", "cases": 48, "factors": "quiet role x SIR", "hypothesis": "quiet targets increase picked-selection loss"},
    {"id": "condition", "cases": 32, "factors": "condition x stage", "hypothesis": "occlusion changes oracle-to-picked loss"},
    {"id": "stage", "cases": 32, "factors": "oracle/picked/mixer", "hypothesis": "mixer reintroduction dominates played leakage"},
    {"id": "telemetry", "cases": 24, "factors": "telemetry present/missing", "hypothesis": "missing telemetry hides deadline failures"},
    {"id": "revision", "cases": 24, "factors": "revision/config", "hypothesis": "inconsistent revisions explain metric shifts"},
)


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _records_from_object(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        for key in ("records", "rows", "results", "measurements", "evidence"):
            if isinstance(obj.get(key), list):
                return [item for item in obj[key] if isinstance(item, dict)]
        return [obj]
    return []


def load_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load JSON, JSONL, or CSV rows without network access."""
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                records.extend(dict(row) for row in csv.DictReader(handle))
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.extend(_records_from_object(json.loads(line)))
        elif suffix == ".json":
            with path.open(encoding="utf-8") as handle:
                records.extend(_records_from_object(json.load(handle)))
        else:
            raise ValueError(f"unsupported report format: {path}")
    return records


def normalize_record(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    quiet = _number(_first(row, "quiet_target_db", "quiet_target_sir", "target_level_db"))
    stage = str(_first(row, "output_stage", "stage", "path") or "unknown").lower()
    aliases = {"oracle-selected": "oracle", "oracle_selection": "oracle", "select": "picked", "selected": "picked", "played": "mixer"}
    canonical_stage = aliases.get(stage, stage)
    sir = _number(_first(row, "sir_db", "sir", "signal_interference_ratio_db"))
    condition = str(_first(row, "condition", "scenario", "scene") or "unknown")
    role = "quiet" if quiet is not None and quiet < 0 else "nominal" if quiet is not None else "unknown"
    metrics: dict[str, float | None] = {}
    statuses: dict[str, str] = {}
    nested = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    for metric in METRICS:
        value = _first(row, metric, *({"latency_ms": ("latency", "mouth_to_ear_ms"), "wrong_speaker": ("wrong_voice", "wrong_speaker_seconds"), "dropout": ("dropouts",)}.get(metric, ())))
        if value is None:
            value = nested.get(metric)
        metrics[metric] = _number(value)
        statuses[metric] = "missing" if metrics[metric] is None else "zero" if metrics[metric] == 0 else "present"
    provenance = {
        "source": source,
        "task": _first(row, "task", "task_id", "run_task"),
        "speaker": _first(row, "speaker", "speaker_id", "original_speaker"),
        "utterance": _first(row, "utterance", "utterance_id"),
        "interval": _first(row, "interval", "interval_s", "start_end"),
        "hash": _first(row, "hash", "source_hash", "input_hash"),
        "licence": _first(row, "licence", "license", "licence_evidence"),
    }
    telemetry = row.get("telemetry") if isinstance(row.get("telemetry"), dict) else {}
    return {
        "provenance": provenance,
        "revision": _first(row, "revision", "git_sha", "commit"),
        "config_hash": _first(row, "config_hash", "configuration_hash"),
        "strata": {"quiet_target_role": role, "sir_db": sir, "condition": condition, "output_stage": canonical_stage},
        "metrics": metrics,
        "metric_status": statuses,
        "telemetry_present": bool(telemetry) or any(key in row for key in ("telemetry", "p50_ms", "p95_ms", "rtf")),
        "telemetry": telemetry,
        "stage_label_issue": canonical_stage not in STAGES or (
            "output_stage" in row and "stage" in row and str(row["output_stage"]).lower() != str(row["stage"]).lower()
        ),
    }


def build_report(rows: list[dict[str, Any]], *, seed: int = 0, planned_cases: int = 192) -> dict[str, Any]:
    """Build JSON-serializable coverage, missingness, and loss summaries."""
    if rows and "provenance" not in rows[0]:
        rows = [normalize_record(row, source=str(row.get("source", "unknown"))) for row in rows]
    by_task: dict[str, set[str]] = {}
    for row in rows:
        task = row["provenance"]["task"] or "unknown"
        if row["revision"] is not None:
            by_task.setdefault(str(task), set()).add(str(row["revision"]))
    conflicts = [{"task": task, "revisions": sorted(revisions)} for task, revisions in sorted(by_task.items()) if len(revisions) > 1]
    strata: dict[str, int] = {}
    for row in rows:
        key = "|".join(str(row["strata"][field]) for field in ("quiet_target_role", "sir_db", "condition", "output_stage"))
        strata[key] = strata.get(key, 0) + 1
    metric_summary: dict[str, dict[str, int]] = {}
    for metric in METRICS:
        metric_summary[metric] = {status: sum(row["metric_status"][metric] == status for row in rows) for status in ("present", "zero", "missing")}
    losses = {}
    for stage in STAGES:
        values = [row["metrics"]["target_loss"] for row in rows if row["strata"]["output_stage"] == stage and row["metrics"]["target_loss"] is not None]
        losses[stage] = {"count": len(values), "mean": sum(values) / len(values) if values else None, "missing": sum(row["strata"]["output_stage"] == stage and row["metrics"]["target_loss"] is None for row in rows)}
    return {
        "schema": "muse-ingest/v1",
        "seed": seed,
        "counts": {"available_records": len(rows), "planned_ablation_cases": planned_cases, "planned_cap": MAX_CASES},
        "coverage": {"strata": dict(sorted(strata.items())), "stage_label_issues": sum(row["stage_label_issue"] for row in rows)},
        "metrics": metric_summary,
        "losses": losses,
        "telemetry": {"present_records": sum(row["telemetry_present"] for row in rows), "missing_records": sum(not row["telemetry_present"] for row in rows)},
        "revision_conflicts": conflicts,
        "provenance_requirements": ["speaker", "utterance", "interval", "hash", "licence"],
        "ablation_proposals": list(ABLATIONS),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=192)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if len(args.inputs) > MAX_INPUTS or args.max_cases < 0 or args.max_cases > MAX_CASES:
        parser.error(f"inputs <= {MAX_INPUTS} and max-cases <= {MAX_CASES} are required")
    total_bytes = sum(path.stat().st_size for path in args.inputs)
    if total_bytes > MAX_BYTES:
        parser.error(f"input bytes exceed {MAX_BYTES}")
    input_hash = hashlib.sha256()
    for path in args.inputs:
        input_hash.update(path.read_bytes())
    checkpoint = args.checkpoint_dir / "summary.json" if args.checkpoint_dir else None
    if args.resume and checkpoint and checkpoint.exists():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        if cached.get("input_hash") == input_hash.hexdigest() and cached.get("complete") is True:
            _atomic_write(args.out, cached)
            print(json.dumps({"records": cached["counts"]["available_records"], "out": str(args.out), "resumed": True}, sort_keys=True))
            return 0
    started = time.monotonic()
    rows = [normalize_record(row, source=path.name) for path in args.inputs for row in load_records([path])]
    report = build_report(rows, seed=args.seed, planned_cases=args.max_cases)
    report["input"] = {"files": [path.name for path in args.inputs], "bytes": total_bytes, "elapsed_s": round(time.monotonic() - started, 6)}
    report["input"]["sha256"] = input_hash.hexdigest()
    if checkpoint:
        report["complete"] = True
        _atomic_write(checkpoint, report)
        report["checkpoint"] = str(checkpoint)
    _atomic_write(args.out, report)
    print(json.dumps({"records": len(rows), "out": str(args.out), "planned_cases": args.max_cases}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
