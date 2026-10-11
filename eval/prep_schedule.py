"""Create a deterministic, balanced, metadata-only overnight experiment schedule.

This prepares cases for other workers; it never renders or augments audio.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

SIRS = (0, -6, -10, -15)
CONDITIONS = ("absent", "pause", "onset")
NOISE = ("clean", "noise")
ROOMS = ("anechoic", "room")
ENROLLMENTS = (1, 2, 3)
WINDOWS = (1, 2)
HOPS = (0.5, 1.0)
PARTITIONS = ("development", "final")
MAX_CASES = 300


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _existing_cases(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    hashes: list[str] = []
    for path in paths:
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        data = _read_json(path)
        rows = data.get("cases", [])
        if not isinstance(rows, list):
            raise TypeError(f"cases must be a list: {path}")
        cases.extend(row for row in rows if isinstance(row, dict))
    return cases, hashes


def _pair_rows(value: Any) -> list[dict[str, Any]]:
    rows = value.get("pairs", []) if isinstance(value, dict) else value
    return [row for row in rows if isinstance(row, dict) and row.get("pair_id")]


def _split_map(value: Any) -> dict[str, str]:
    rows = value.get("eligible", []) if isinstance(value, dict) else value
    result = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("source_id") and row.get("partition") in PARTITIONS:
            result[str(row["source_id"])] = str(row["partition"])
    return result


def _partition(pair: dict[str, Any], splits: dict[str, str]) -> str | None:
    ids = [
        str(pair.get(role, {}).get("source_id", ""))
        for role in ("target", "interferer")
    ]
    values = {splits.get(source_id) for source_id in ids if source_id}
    if not splits:
        return PARTITIONS[int(_digest(pair.get("pair_id", ""))[:2], 16) % 2]
    if len(values) != 1:
        return None
    return values.pop()


def _available(pair: dict[str, Any]) -> bool:
    return all(
        isinstance(pair.get(role), dict)
        and pair[role].get("source_id")
        and pair[role].get("stats", {}).get("sha256")
        for role in ("target", "interferer")
    ) and pair.get("target", {}).get("speaker_id") != pair.get("interferer", {}).get("speaker_id")


def _case_id(fields: dict[str, Any]) -> str:
    return "prep-" + _digest(fields)[:16]


def build_schedule(
    pairs: list[dict[str, Any]],
    existing_paths: list[Path],
    *,
    seed: int = 234,
    max_cases: int = MAX_CASES,
    splits: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a balanced schedule and coverage report without reading media."""
    if not 1 <= max_cases <= MAX_CASES:
        raise ValueError(f"max_cases must be in [1, {MAX_CASES}]")
    splits = splits or {}
    existing, existing_hashes = _existing_cases(existing_paths)
    completed = {
        str(row["case_id"])
        for row in existing
        if row.get("case_id") and row.get("status", "complete") == "complete"
    }
    usable = [
        pair for pair in pairs if _available(pair) and _partition(pair, splits) in PARTITIONS
    ]
    candidates = []
    for pair, role, sir, condition, noise, room, enrollment, window, hop in itertools.product(
        usable,
        ("target", "interferer"),
        SIRS,
        CONDITIONS,
        NOISE,
        ROOMS,
        ENROLLMENTS,
        WINDOWS,
        HOPS,
    ):
        partition = _partition(pair, splits)
        fields = {
            "pair_id": pair["pair_id"],
            "target_role": role,
            "sir_db": sir,
            "condition": condition,
            "noise": noise,
            "room": room,
            "enrollment_s": enrollment,
            "window_s": window,
            "hop_s": hop,
            "partition": partition,
        }
        case_id = _case_id(fields)
        if case_id not in completed:
            candidates.append((int(_digest({"seed": seed, **fields})[:16], 16), fields | {"case_id": case_id}))
    candidates.sort(key=lambda item: item[0])
    available_count = len(candidates)
    budget = max(0, max_cases - len(existing))
    selected: list[dict[str, Any]] = []
    # Round-robin across the primary cells makes the smoke prefix balanced.
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for _, row in candidates:
        key = (row["target_role"], row["sir_db"], row["condition"])
        cells.setdefault(key, []).append(row)
    for rows in cells.values():
        rows.sort(key=lambda row: int(_digest({"seed": seed, **row})[:16], 16))
    keys = sorted(cells, key=lambda key: _digest({"seed": seed, "cell": key}))
    while len(selected) < budget and any(cells.values()):
        for key in keys:
            if cells[key] and len(selected) < budget:
                selected.append(cells[key].pop(0))
    smoke: list[dict[str, Any]] = []
    remaining = list(selected)
    smoke_counts = {name: {} for name in ("target_role", "sir_db", "condition")}
    while remaining and len(smoke) < min(12, len(selected)):
        def score(row: dict[str, Any]) -> tuple[Any, ...]:
            counts = tuple(smoke_counts[name].get(row[name], 0) for name in smoke_counts)
            return (*counts, _digest({"seed": seed, "smoke": row["case_id"]}))

        choice = min(remaining, key=score)
        remaining.remove(choice)
        smoke.append(choice)
        for name, counts in smoke_counts.items():
            counts[choice[name]] = counts.get(choice[name], 0) + 1
    selected = smoke + remaining
    covered = {(row["target_role"], row["sir_db"], row["condition"]) for row in selected}
    missing = [
        {"target_role": role, "sir_db": sir, "condition": condition}
        for role, sir, condition in itertools.product(("target", "interferer"), SIRS, CONDITIONS)
        if (role, sir, condition) not in covered
    ]
    factor_values = {
        "target_role": ("target", "interferer"),
        "sir_db": SIRS,
        "condition": CONDITIONS,
        "noise": NOISE,
        "room": ROOMS,
        "enrollment_s": ENROLLMENTS,
        "window_s": WINDOWS,
        "hop_s": HOPS,
    }
    factor_coverage = {
        name: {str(value): sum(row[name] == value for row in selected) for value in values}
        for name, values in factor_values.items()
    }
    pairwise_coverage: dict[str, dict[str, int]] = {}
    factor_names = tuple(factor_values)
    for index, left_name in enumerate(factor_names):
        for right_name in factor_names[index + 1 :]:
            name = f"{left_name}__{right_name}"
            pairwise_coverage[name] = {}
            for row in selected:
                key = f"{row[left_name]}={row[right_name]}"
                pairwise_coverage[name][key] = pairwise_coverage[name].get(key, 0) + 1
    config = {
        "seed": seed,
        "max_cases": max_cases,
        "factors": {"sir_db": SIRS, "condition": CONDITIONS, "noise": NOISE, "room": ROOMS},
    }
    return {
        "schema": "overnight-prep-schedule/v1",
        "config": config,
        "config_sha256": _digest(config),
        "input_sha256": _digest(
            [{"pair_id": pair.get("pair_id"), "target": pair.get("target", {}).get("stats", {}).get("sha256"),
              "interferer": pair.get("interferer", {}).get("stats", {}).get("sha256")} for pair in pairs]
        ),
        "accounting": {
            "global_cap": max_cases,
            "existing_cases": len(existing),
            "planned_cases": len(selected),
            "available_input_cases": available_count,
            "available_pairs": len(usable),
        },
        "cases": selected,
        "smoke_cases": smoke,
        "coverage": {
            "factor": factor_coverage,
            "pairwise": pairwise_coverage,
            "primary_cells": sorted(
                [{"target_role": role, "sir_db": sir, "condition": condition} for role, sir, condition in covered],
                key=lambda row: (row["target_role"], row["sir_db"], row["condition"]),
            ),
            "missing_cells": missing,
        },
        "missing_cells": missing,
        "checkpoint": {
            "existing_manifest_sha256": existing_hashes,
            "resume_skips_verified": True,
            "completed_case_ids": len(completed),
            "stop_reason": "global_case_cap_or_input_exhausted",
        },
        "consumer": {
            "command": "python -m eval.prep_schedule --pairs <pairs.json> --splits <splits.json> --output <schedule.json>",
            "case_mapping": "cases[].case_id -> cases[]; smoke_cases[] is the first 12 balanced cases",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--existing", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--max-cases", type=int, default=MAX_CASES)
    args = parser.parse_args()
    pair_data = _read_json(args.pairs)
    split_data = _read_json(args.splits) if args.splits else {}
    result = build_schedule(
        _pair_rows(pair_data),
        args.existing,
        seed=args.seed,
        max_cases=args.max_cases,
        splits=_split_map(split_data),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["accounting"], sort_keys=True))


if __name__ == "__main__":
    main()
