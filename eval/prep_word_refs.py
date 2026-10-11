"""Prepare conservative word-retention references from existing alignments.

This module deliberately does not transcribe or infer timings.  It turns
boundary-backed source records into scoring plans and marks everything else
unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _words(text: str | None) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(text or "")]


def _alignment(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("alignment")
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        word = entry.get("word", entry.get("text"))
        start, end = entry.get("start"), entry.get("end")
        if (
            isinstance(word, str)
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and 0 <= float(start) < float(end)
        ):
            result.append({"text": word.lower(), "start": float(start), "end": float(end)})
    return result


def _speaker_plan(
    record: dict[str, Any] | None, crop: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    if not record:
        return {"present": False, "words": []}, "speaker absent"
    transcript = record.get("transcript")
    aligned = _alignment(record)
    if not aligned:
        return {
            "present": bool(transcript),
            "transcript_unverified": bool(transcript),
            "words": [],
        }, "no word boundaries; cropped or transcript-only reference is unavailable"
    start = float(crop["start"]) if crop and isinstance(crop.get("start"), (int, float)) else None
    end = float(crop["end"]) if crop and isinstance(crop.get("end"), (int, float)) else None
    words = []
    occurrence: dict[str, int] = {}
    for token in aligned:
        if start is not None and (token["start"] < start or token["end"] > end):
            continue
        occurrence[token["text"]] = occurrence.get(token["text"], 0) + 1
        words.append({**token, "occurrence": occurrence[token["text"]]})
    if not words:
        return {"present": True, "transcript_unverified": False, "words": []}, "crop contains no complete aligned words"
    return {
        "present": True,
        "transcript_unverified": False,
        "words": words,
    }, None


def build_plan(cases: list[dict[str, Any]], *, seed: int | None = None) -> dict[str, Any]:
    planned = []
    for case in cases:
        crop = case.get("crop") if isinstance(case.get("crop"), dict) else None
        target, target_reason = _speaker_plan(case.get("target"), crop)
        interferer, interferer_reason = _speaker_plan(case.get("interferer"), crop)
        shared = sorted(
            {word["text"] for word in target["words"]}
            & {word["text"] for word in interferer["words"]}
        )
        reasons = [reason for reason in (target_reason, interferer_reason) if reason]
        status = "available" if target["words"] else "unavailable"
        planned.append(
            {
                "id": str(case.get("id", "")),
                "status": status,
                "unavailable_reason": "; ".join(reasons) if reasons else None,
                "target": target,
                "interferer": interferer,
                "ambiguous_shared_words": shared,
                "recipe": {
                    "target_deletions": "score target words as references; count missing words as deletions",
                    "competitor_insertions": "score competitor-only words as insertions; do not assign shared words",
                    "shared_word_policy": "flag ambiguous shared words and exclude them from attribution",
                },
                "provenance": {
                    "source": case.get("source", {}),
                    "source_hash": _digest(case.get("source", {})),
                    "case_hash": _digest(case),
                    "crop": crop,
                },
            }
        )
    return {
        "kind": "word_reference_plan",
        "schema": 1,
        "seed": seed,
        "config": {"seed": seed, "max_cases": len(cases), "word_boundary_policy": "actual-boundaries-only"},
        "config_hash": _digest({"schema": 1, "seed": seed}),
        "cases": planned,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _public_summary(plan: dict[str, Any], input_path: Path) -> dict[str, Any]:
    cases = plan["cases"]
    return {
        "kind": "word_reference_summary",
        "schema": plan["schema"],
        "input_hash": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "config_hash": plan["config_hash"],
        "seed": plan["seed"],
        "planned_cases": len(cases),
        "available_cases": sum(item["status"] == "available" for item in cases),
        "unavailable_cases": sum(item["status"] == "unavailable" for item in cases),
        "available_target_words": sum(len(item["target"]["words"]) for item in cases),
        "ambiguous_shared_word_cases": sum(bool(item["ambiguous_shared_words"]) for item in cases),
        "policy": "No ASR inference; no invented alignment; transcript-only and crop-incomplete words unavailable.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON source cases or {cases: [...]}")
    parser.add_argument("--output", required=True, type=Path, help="private reference plan")
    parser.add_argument("--summary", required=True, type=Path, help="public-safe summary")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="optional ignored directory for atomic per-case resume checkpoints",
    )
    args = parser.parse_args(argv)
    if args.max_cases < 1 or args.max_cases > 500:
        parser.error("--max-cases must be between 1 and 500")
    payload = json.loads(args.input.read_text())
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        parser.error("input must be a list or object containing cases")
    if len(cases) > args.max_cases:
        parser.error(f"input has {len(cases)} cases; cap is {args.max_cases}")
    seed = args.seed if args.seed is not None else payload.get("seed") if isinstance(payload, dict) else None
    if args.checkpoint_dir is None:
        plan = build_plan(cases, seed=seed)
    else:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        planned = []
        for index, case in enumerate(cases):
            case_hash = _digest(case)
            checkpoint = args.checkpoint_dir / f"{index:05d}-{case_hash[:16]}.json"
            item = None
            if checkpoint.exists():
                try:
                    cached = json.loads(checkpoint.read_text())
                    if cached.get("case_hash") == case_hash and isinstance(cached.get("item"), dict):
                        item = cached["item"]
                except (OSError, json.JSONDecodeError):
                    item = None
            if item is None:
                item = build_plan([case], seed=seed)["cases"][0]
                _atomic_write(checkpoint, {"case_hash": case_hash, "item": item})
            planned.append(item)
        plan = build_plan([], seed=seed)
        plan["cases"] = planned
        plan["config"]["max_cases"] = len(cases)
    plan["input_hash"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    _atomic_write(args.output, plan)
    _atomic_write(args.summary, _public_summary(plan, args.input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
