"""Canonical result record shared by every agent (eval runner, scheduler, distillation jobs, Arena).

One record = one candidate evaluated at one rung on a named, hashed set of fixed mixtures.
Records are appended to ``results.jsonl``; ``validate`` is the single gate every producer passes.
There is deliberately no weighted aggregate: the four Pareto dimensions stay separate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "hearing.arena.result/1"
ACTIONS = ("CONTINUE", "STOP", "HUMAN_REVIEW")

# metric -> (dimension, direction): +1 higher is better, -1 lower is better.
METRICS: dict[str, tuple[str, int]] = {
    "estoi": ("intelligibility", +1),
    "stoi": ("intelligibility", +1),
    "wer": ("intelligibility", -1),
    "dnsmos_sig": ("naturalness", +1),
    "dnsmos_bak": ("naturalness", +1),
    "dnsmos_ovrl": ("naturalness", +1),
    "si_sdri": ("interference", +1),
    "wrong_speaker_frac": ("interference", -1),
    "switches_per_min": ("interference", -1),
    "absent_leak_db": ("interference", -1),
    "present_miss_frac": ("interference", -1),
    "rtf": ("latency", -1),
    "algo_latency_ms": ("latency", -1),
    "compute_p95_ms": ("latency", -1),
    "overrun_frac": ("latency", -1),
    "target_dropout_s": ("interference", -1),
}
DIMENSIONS = ("intelligibility", "naturalness", "interference", "latency")
# Metrics a STOP may rest on; naturalness (DNSMOS) is auxiliary but can veto a STOP.
QUALITY = ("estoi", "stoi", "wer", "si_sdri", "wrong_speaker_frac")
PERCEPTUAL = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")

REQUIRED = ("schema", "run_id", "experiment_id", "candidate", "baseline_id", "rung", "inputs",
            "per_mixture", "dimensions", "metrics", "compute", "decision", "human_verdict", "created_at")
CANDIDATE_REQUIRED = ("id", "kind", "model", "checkpoint", "spec", "preprocessing", "seed")


def dimensions(means: dict) -> dict[str, dict]:
    """Group flat metric means into the four Pareto dimensions (missing metrics are omitted)."""
    out: dict[str, dict] = {d: {} for d in DIMENSIONS}
    for key, value in means.items():
        if key in METRICS and value is not None:
            out[METRICS[key][0]][key] = value
    return out


def validate(record: dict) -> dict:
    """Raise ``ValueError`` unless ``record`` is a well-formed canonical result; return it."""
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"result record missing {missing}")
    if record["schema"] != SCHEMA:
        raise ValueError(f"unknown schema {record['schema']!r}")
    cand = record["candidate"]
    missing = [k for k in CANDIDATE_REQUIRED if k not in cand]
    if missing:
        raise ValueError(f"candidate missing {missing}")
    if record["decision"].get("action") not in ACTIONS:
        raise ValueError(f"decision.action must be one of {ACTIONS}")
    if set(record["dimensions"]) != set(DIMENSIONS):
        raise ValueError("dimensions must be exactly intelligibility/naturalness/interference/latency")
    ids = record["inputs"].get("mixture_ids", [])
    if [r["mixture_id"] for r in record["per_mixture"]] != ids:
        raise ValueError("per_mixture rows must follow inputs.mixture_ids")
    if any(k in record for k in ("score", "aggregate", "winner")):
        raise ValueError("no weighted aggregate or winner field: a winner needs a human Arena verdict")
    for key in ("consumed_s", "audio_s"):
        if not math.isfinite(float(record["compute"].get(key, math.nan))):
            raise ValueError(f"compute.{key} must be a finite number")
    json.dumps(record, allow_nan=False)  # strict JSON: NaN is not portable across agents
    return record


def append(path: Path, record: dict) -> None:
    validate(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
