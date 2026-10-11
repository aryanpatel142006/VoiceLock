"""Build a public-safe morning evidence table from overnight result JSON.

The input may be a result object (``rows``/``results``), a JSON array, or
JSONL.  No score is imputed: missing metrics remain ``unknown``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from statistics import quantiles
from typing import Any

PESQ_PASS = 2.8
PESQ_REJECT = 2.5
SIG_PASS = 3.4
SIG_REJECT = 3.2


def _read(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        rows = value.get("rows", value.get("results", value.get("records")))
        return [item for item in rows or [] if isinstance(item, dict)]
    raise ValueError("input must contain an object, array, or JSONL objects")


def _metric(row: dict[str, Any], *names: str) -> float | None:
    metrics = row.get("metrics")
    for name in names:
        value = row.get(name)
        if value is None and isinstance(metrics, dict):
            value = metrics.get(name)
        if isinstance(value, dict):
            value = value.get("value", value.get("mean"))
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _value(row: dict[str, Any], *names: str, default: str = "unknown") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return str(value)
    return default


def _hash(row: dict[str, Any], *names: str) -> str | None:
    value = _value(row, *names, default="")
    return value or None


def _gate(pesq: float | None, sig: float | None) -> str:
    if pesq is None or sig is None:
        return "unknown"
    if pesq < PESQ_REJECT or sig < SIG_REJECT:
        return "reject"
    if pesq >= PESQ_PASS and sig >= SIG_PASS:
        return "pass"
    return "review"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[int(percentile) - 1]


def build_table(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows by stage, speaker, SIR, and condition."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    source_rows = list(rows)
    for row in source_rows:
        key = (
            _value(row, "stage", "pipeline", "model"),
            _value(row, "speaker", "speaker_id", "target_speaker"),
            _value(row, "sir_db", "sir", "SIR"),
            _value(row, "condition", "scenario", "case"),
        )
        groups.setdefault(key, []).append(row)

    entries: list[dict[str, Any]] = []
    for (stage, speaker, sir, condition), members in sorted(groups.items()):
        pesq = [_metric(r, "pesq_wb", "pesq") for r in members]
        sig = [_metric(r, "dnsmos_sig", "dnsmos_sig", "dnsmos_signal", "DNSMOS_SIG") for r in members]
        latency = [_metric(r, "latency_ms", "latency") for r in members]
        pesq_values = [x for x in pesq if x is not None]
        sig_values = [x for x in sig if x is not None]
        lat_values = [x for x in latency if x is not None]
        target_absent = any(
            str(r.get("target_present", r.get("target", ""))).lower() in {"false", "absent", "none"}
            for r in members
        )
        input_hashes = {_hash(r, "input_hash", "source_hash", "input_sha256") for r in members}
        model_hashes = {_hash(r, "model_revision", "model_hash", "model_sha256") for r in members}
        config_hashes = {_hash(r, "config_revision", "config_hash", "config_sha256") for r in members}
        eligible = (
            not target_absent
            and None not in input_hashes
            and None not in model_hashes
            and None not in config_hashes
            and len(input_hashes) == len(model_hashes) == len(config_hashes) == 1
        )
        entries.append(
            {
                "stage": stage,
                "speaker": speaker,
                "sir_db": sir,
                "condition": condition,
                "samples": len(members),
                "available": {"pesq_wb": len(pesq_values), "dnsmos_sig": len(sig_values), "latency_ms": len(lat_values)},
                "missing": {
                    "pesq_wb": len(members) - len(pesq_values),
                    "dnsmos_sig": len(members) - len(sig_values),
                    "latency_ms": len(members) - len(lat_values),
                },
                "p50_ms": _percentile(lat_values, 50),
                "p95_ms": _percentile(lat_values, 95),
                "pesq_wb": sum(pesq_values) / len(pesq_values) if pesq_values else None,
                "dnsmos_sig": sum(sig_values) / len(sig_values) if sig_values else None,
                "quality_gate": _gate(
                    sum(pesq_values) / len(pesq_values) if pesq_values else None,
                    sum(sig_values) / len(sig_values) if sig_values else None,
                ),
                "comparison_eligible": eligible,
                "revisions": {
                    "input": sorted(x for x in input_hashes if x is not None),
                    "model": sorted(x for x in model_hashes if x is not None),
                    "config": sorted(x for x in config_hashes if x is not None),
                },
            }
        )
    return {"schema": "morning-evidence-v1", "samples_input": len(source_rows), "groups": entries}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true", help="reuse a verified complete checkpoint")
    parser.add_argument("--limit", type=int, default=500, help="maximum result rows to consume")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.resume and args.checkpoint and args.output.is_file() and args.checkpoint.is_file():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("complete") is True and checkpoint.get("input_sha256") == input_hash:
            print(json.dumps({"resumed": True, "output": str(args.output)}))
            return 0
    rows = _read(args.input)
    if len(rows) > args.limit:
        raise ValueError(f"input has {len(rows)} rows; limit is {args.limit}")
    table = build_table(rows)
    table["input_sha256"] = input_hash
    table["input_path"] = args.input.name
    _atomic_json(args.output, table)
    if args.checkpoint:
        _atomic_json(args.checkpoint, {"complete": True, "input_sha256": table["input_sha256"], "output": args.output.name})
    print(json.dumps({"samples_input": table["samples_input"], "groups": len(table["groups"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
