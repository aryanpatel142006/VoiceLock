"""Score target-word retention from offline ASR transcripts.

Input files contain one scene per JSONL line, or a JSON list (optionally under
``scenes``).  A scene needs a target transcript, a separated-output
transcript, and an interference ratio (normally SIR in dB).  Audio and ASR
are intentionally outside this module: overnight jobs can run ASR separately
and score the resulting transcripts without network access or model weights.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval.metrics import normalize_text, wer

_REFERENCE_KEYS = ("target_text", "reference", "clean_target_text", "transcript")
_HYPOTHESIS_KEYS = ("separated_transcript", "hypothesis", "output_transcript", "asr_text")
_RATIO_KEYS = ("interference_ratio", "sir_db", "interference_db", "ratio")


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _validated_record(record: dict[str, Any], source: Path | None = None) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(f"{source or 'record'}: expected an object")
    reference = _first(record, _REFERENCE_KEYS)
    hypothesis = _first(record, _HYPOTHESIS_KEYS)
    if (
        hypothesis is None
        and any(key in record for key in _REFERENCE_KEYS[:-1])
        and isinstance(record.get("transcript"), str)
    ):
        hypothesis = record["transcript"]
    ratio = _first(record, _RATIO_KEYS)
    missing = [
        name
        for name, value in (
            ("target_text", reference),
            ("separated_transcript", hypothesis),
            ("interference_ratio", ratio),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"{source or 'record'}: missing {', '.join(missing)}")
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError(f"{source or 'record'}: transcript fields must be strings")
    try:
        ratio = float(ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source or 'record'}: interference_ratio must be numeric") from exc
    if not math.isfinite(ratio):
        raise ValueError(f"{source or 'record'}: interference_ratio must be finite")
    return {
        **record,
        "scene_id": str(record.get("scene_id", record.get("id", ""))),
        "target_text": reference,
        "separated_transcript": hypothesis,
        "interference_ratio": ratio,
    }


def _read_file(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("scenes", payload) if isinstance(payload, dict) else payload
            if isinstance(values, dict):
                values = [values]
        else:
            return []
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(values, list):
        raise TypeError(f"{path}: expected a list of scene records")
    return [_validated_record(value, path) for value in values]


def load_records(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load and validate JSON/JSONL scene records in deterministic path order."""
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
            )
        else:
            paths.append(path)
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_read_file(path))
    if not records:
        raise ValueError("no scene records found")
    return records


def _score_one(record: dict[str, Any]) -> dict[str, Any]:
    reference = normalize_text(record["target_text"])
    hypothesis = normalize_text(record["separated_transcript"])
    reference_words = reference.split()
    if not reference_words:
        raise ValueError(f"scene {record['scene_id']!r}: target_text has no words")
    import jiwer

    alignment = jiwer.process_words(reference, hypothesis)
    return {
        "scene_id": record["scene_id"],
        "interference_ratio": record["interference_ratio"],
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis.split()),
        "deletions": alignment.deletions,
        "insertions": alignment.insertions,
        "substitutions": alignment.substitutions,
        "wer": float(wer(hypothesis, reference)),
        "word_loss_pct": float(100 * alignment.deletions / len(reference_words)),
    }


def score_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Score scenes and return per-scene plus interference-ratio summaries."""
    scenes = [_score_one(_validated_record(record)) for record in records]
    groups: dict[float, list[dict[str, Any]]] = {}
    for scene in scenes:
        groups.setdefault(scene["interference_ratio"], []).append(scene)
    by_ratio: dict[str, dict[str, Any]] = {}
    for ratio in sorted(groups):
        group = groups[ratio]
        by_ratio[_ratio_key(ratio)] = {
            "interference_ratio": ratio,
            "scenes": len(group),
            "wer": sum(row["wer"] for row in group) / len(group),
            "word_loss_pct": sum(row["word_loss_pct"] for row in group) / len(group),
            "reference_words": sum(row["reference_words"] for row in group),
            "deletions": sum(row["deletions"] for row in group),
        }
    return {"scenes": scenes, "by_interference_ratio": by_ratio}


def _ratio_key(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def write_report(report: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="scene JSON/JSONL files or directories")
    parser.add_argument("-o", "--output", type=Path, help="write the JSON report to this path")
    args = parser.parse_args()
    report = score_records(load_records(args.inputs))
    if args.output:
        write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
