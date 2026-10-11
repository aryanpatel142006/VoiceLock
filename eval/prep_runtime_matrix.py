"""Prepare bounded, hardware-free runtime/deadline experiment manifests.

This module never imports a model or starts a server.  It describes the serial
window wrappers that T-215 can run, and supplies deterministic telemetry
fixtures for scheduler edge cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TASK_ID = "T-242"
DEFAULT_WINDOWS = (0.5, 1.0, 2.0)
DEFAULT_REPETITIONS = 3
DEFAULT_CASE_CAP = 24
DEFAULT_TIME_CAP_S = 1800
DEFAULT_BYTE_CAP = 1_000_000_000
EVENT_CASES = ("stale-server", "never-enrolled", "missing-tier", "late-window")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _supported_pairs(path: Path) -> set[tuple[float, float]]:
    """Extract explicitly configured window/hop pairs without parsing YAML."""
    text = path.read_text(encoding="utf-8")
    pairs = set()
    for window, hop in re.findall(
        r"window_s\s*[:=]\s*([0-9.]+).*?hop_s\s*[:=]\s*([0-9.]+)", text, re.DOTALL
    ):
        pairs.add((float(window), float(hop)))
    return pairs


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    candidate: str
    window_s: float
    hop_s: float
    phase: str
    repetition: int
    seed: int
    deadline_s: float
    fallback: str
    required_telemetry: tuple[str, ...]


def available_inputs(repo: Path) -> dict[str, Any]:
    configs = sorted(repo.glob("config/tiers*.yaml"))
    return {
        "config_files": [
            {"path": str(path.relative_to(repo)), "sha256": sha256(path)}
            for path in configs
        ],
        "configured_pairs": {
            path.name: [list(pair) for pair in sorted(_supported_pairs(path))]
            for path in configs
        },
    }


def build_matrix(
    repo: Path,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = 242,
    case_cap: int = DEFAULT_CASE_CAP,
) -> list[MatrixCase]:
    if repetitions < 1 or case_cap < 1:
        raise ValueError("repetitions and case_cap must be positive")
    inputs = available_inputs(repo)
    local = {
        tuple(pair)
        for pairs in inputs["configured_pairs"].values()
        for pair in pairs
    }
    requested = [(window, hop) for window in DEFAULT_WINDOWS for hop in (window / 2, window / 4)]
    pairs = [(window, hop) for window, hop in requested if (window, hop) in local]
    if not pairs:
        raise ValueError("no requested window/hop pair is present in tier configs")
    cases: list[MatrixCase] = []
    candidate_pairs = {
        # The checked-in configs use short windows only for enrolled and the
        # 2-second AV wrapper; do not invent unsupported combinations.
        "enrolled": [(window, hop) for window, hop in pairs if window <= 1.0],
        "av_mossformer2": [(window, hop) for window, hop in pairs if window == 2.0],
    }
    for candidate, supported in candidate_pairs.items():
        for window, hop in supported:
            for phase in ("warmup", "steady"):
                for repetition in range(repetitions):
                    cases.append(
                        MatrixCase(
                            case_id=f"{candidate}-{window:g}-{hop:g}-{phase}-{repetition}",
                            candidate=candidate,
                            window_s=window,
                            hop_s=hop,
                            phase=phase,
                            repetition=repetition,
                            seed=seed + repetition,
                            deadline_s=hop,
                            fallback="raw_mix",
                            required_telemetry=(
                                "capture_ts",
                                "deadline_ts",
                                "emitted_ts",
                                "status",
                                "tier",
                                "device",
                            ),
                        )
                    )
    if len(cases) > case_cap:
        raise ValueError(f"planned {len(cases)} cases exceeds case_cap={case_cap}")
    return cases


def dry_run(
    repo: Path,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = 242,
    case_cap: int = DEFAULT_CASE_CAP,
    time_cap_s: int = DEFAULT_TIME_CAP_S,
    byte_cap: int = DEFAULT_BYTE_CAP,
) -> dict[str, Any]:
    if time_cap_s < 1 or byte_cap < 1:
        raise ValueError("time_cap_s and byte_cap must be positive")
    inputs = available_inputs(repo)
    cases = build_matrix(repo, repetitions=repetitions, seed=seed, case_cap=case_cap)
    estimate_bytes = len(cases) * 512
    if estimate_bytes > byte_cap:
        raise ValueError("estimated manifest exceeds byte_cap")
    return {
        "task_id": TASK_ID,
        "dry_run": True,
        "inference": False,
        "server_started": False,
        "mic_used": False,
        "gpu_used": False,
        "seed": seed,
        "caps": {"cases": case_cap, "time_s": time_cap_s, "bytes": byte_cap},
        "planned_cases": len(cases),
        "available_inputs": inputs,
        "config_hashes": {entry["path"]: entry["sha256"] for entry in inputs["config_files"]},
        "cases": [asdict(case) for case in cases],
    }


def make_event_fixtures(destination: Path, *, seed: int = 242) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {
        "stale-server": ("stale_server", "health_revision_mismatch"),
        "never-enrolled": ("fallback", "enrollment_missing"),
        "missing-tier": ("fallback", "tier_unavailable"),
        "late-window": ("deadline_missed", "deadline_exceeded"),
    }
    manifest = {"task_id": TASK_ID, "seed": seed, "cases": []}
    for index, case_id in enumerate(EVENT_CASES):
        status, reason = expected[case_id]
        event = {
            "seq": index,
            "capture_ts": index * 0.5,
            "deadline_ts": index * 0.5 + 0.25,
            "emitted_ts": index * 0.5 + (0.3 if case_id == "late-window" else 0.1),
            "status": status,
            "reason": reason,
            "tier": "enrolled" if case_id == "stale-server" else "audio-only",
            "device": None,
            "tier_evidence_required": True,
            "config_device_claim": "cuda" if case_id == "stale-server" else None,
        }
        path = destination / f"{case_id}.json"
        path.write_text(json.dumps(event, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest["cases"].append({"case_id": case_id, "path": path.name, "expected_status": status})
    (destination / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def account_timeline(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for event in events:
        required = ("capture_ts", "deadline_ts", "emitted_ts", "status", "tier")
        missing = [key for key in required if key not in event]
        if missing:
            raise ValueError(f"event {event.get('seq', '?')} missing {missing}")
        late = float(event["emitted_ts"]) > float(event["deadline_ts"])
        status = str(event["status"])
        rows.append(
            {
                "seq": event.get("seq"),
                "late": late,
                "fallback": status in {"fallback", "deadline_missed"},
                "status": status,
                "tier": event["tier"],
                "tier_evidence": bool(event.get("tier_evidence_required"))
                and bool(event.get("tier")),
                "device_evidence": event.get("device") is not None,
            }
        )
    return {
        "events": rows,
        "n_events": len(rows),
        "n_late": sum(row["late"] for row in rows),
        "n_fallback": sum(row["fallback"] for row in rows),
        "gpu_evidence_events": sum(
            row["device_evidence"] and row["tier"] != "audio-only" for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--case-cap", type=int, default=DEFAULT_CASE_CAP)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = dry_run(args.repo, repetitions=args.repetitions, case_cap=args.case_cap, seed=args.seed)
    if args.fixtures:
        report["event_fixtures"] = make_event_fixtures(args.fixtures, seed=args.seed)
    text = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
