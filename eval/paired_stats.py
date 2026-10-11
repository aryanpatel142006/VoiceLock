"""Seed variability and paired window comparisons (ported from T-307).

Usage from the repository root::

    .venv/bin/python -m eval.paired_stats seeds head_results.json
    .venv/bin/python -m eval.paired_stats mcnemar audit_a.json audit_b.json

Seeds use T-307's ``seed* -> eval_set -> head -> numeric_metric`` mapping;
``train`` and n_queries/n_speakers/hard_n counts are excluded. Every metric
must exist and be finite in all requested seeds. SD is the sample SD (ddof=1),
undefined for one seed, and describes run variability, not a confidence interval.

Audits use T-323's top-level ``scenes`` list. Each scene has string ``scene``,
``target``, ``other`` and a ``rows`` list with integer ``k``, numeric ``t`` and
boolean ``wrong`` (any other-dominant played-output time owned by that window).
Pairs match on (scene, target, other, k), with identical t, independent of order.
Paths must match literally; duplicates, missing pairs and empty inputs are errors.
Embedded adaptive/fixed-gate arms are not read: supply two top-level audit files.

Correctness is NOT wrong. The table has A incorrect/correct rows and B
incorrect/correct columns; b = A incorrect/B correct, c = A correct/B incorrect.
``p`` is the two-sided exact binomial McNemar p-value; ``p_asymptotic`` uses
continuity-corrected chi-square(1). No discordance gives chi2=0 and both p=1.
T-323 windows overlap and can be temporally correlated, violating independent
pair assumptions: these p-values are exploratory, not evidence of independent
replication. Use independent scenes/seeds or a cluster-aware analysis for claims.
Binary window counts do not measure wrong-speaker duration or effect uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import binomtest

_COUNTS = {"n_queries", "n_speakers", "hard_n"}


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def load_results(path: Path) -> tuple[dict, list[str]]:
    """Load a T-307 result and discover all seed entries."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("seed results must be an object")
    seeds = sorted(k for k in data if k.startswith("seed"))
    if not seeds:
        raise ValueError(f"no seed* entries in {path}")
    return data, seeds


def seed_stats(data: dict, seeds: list[str]) -> dict:
    """Return (mean, sample SD) for complete seed metrics; singleton SD is None."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide nonempty unique seeds")
    flattened = []
    for seed in seeds:
        if seed not in data or not isinstance(data[seed], dict):
            raise ValueError(f"missing or invalid seed {seed}")
        metrics = {}
        for set_name, heads in data[seed].items():
            if set_name == "train" or not isinstance(heads, dict):
                continue
            for head, values in heads.items():
                if not isinstance(values, dict):
                    raise TypeError(f"invalid metrics for {seed}/{set_name}/{head}")
                for metric, value in values.items():
                    if metric in _COUNTS:
                        continue
                    if not _finite_number(value):
                        raise ValueError(f"nonfinite or nonnumeric metric {seed}/{set_name}/{head}/{metric}")
                    metrics[set_name, head, metric] = float(value)
        flattened.append(metrics)
    if not flattened[0] or any(d.keys() != flattened[0].keys() for d in flattened):
        raise ValueError("metrics must be nonempty and present in every seed")
    out = {}
    for (set_name, head, metric) in flattened[0]:
        values = [d[set_name, head, metric] for d in flattened]
        out.setdefault(set_name, {}).setdefault(head, {})[metric] = (
            mean(values), stdev(values) if len(values) > 1 else None)
    return out


def mcnemar_from_outcomes(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Compare nonempty, already aligned boolean correctness vectors."""
    if not correct_a or len(correct_a) != len(correct_b):
        raise ValueError("paired outcomes must have equal nonzero length")
    if any(type(v) is not bool for v in [*correct_a, *correct_b]):
        raise ValueError("outcomes must be booleans")
    table = [[0, 0], [0, 0]]
    for a, b_value in zip(correct_a, correct_b, strict=True):
        table[int(a)][int(b_value)] += 1
    b, c = table[0][1], table[1][0]
    discordant = b + c
    chi2 = max(abs(b - c) - 1, 0) ** 2 / discordant if discordant else 0.0
    p = float(binomtest(b, discordant, 0.5).pvalue) if discordant else 1.0
    return {"n_pairs": len(correct_a), "table": table, "b": b, "c": c,
            "discordant": discordant, "chi2": chi2, "p": p,
            "p_asymptotic": math.erfc(math.sqrt(chi2 / 2)),
            "significant_0.05": p < 0.05}


def _audit_rows(audit: dict) -> dict:
    if not isinstance(audit, dict) or not isinstance(audit.get("scenes"), list):
        raise TypeError("audit must contain a scenes list")
    out = {}
    scene_keys = set()
    for scene in audit["scenes"]:
        if not isinstance(scene, dict):
            raise TypeError("invalid scene")
        identity = tuple(scene.get(name) for name in ("scene", "target", "other"))
        if any(not isinstance(v, str) or not v for v in identity):
            raise ValueError("scene/target/other must be nonempty strings")
        if identity in scene_keys:
            raise ValueError("duplicate scene identity")
        scene_keys.add(identity)
        if not isinstance(scene.get("rows"), list) or not scene["rows"]:
            raise ValueError("scene must contain nonempty rows")
        for row in scene["rows"]:
            if not isinstance(row, dict) or type(row.get("k")) is not int or row["k"] < 0:
                raise ValueError("window k must be a nonnegative integer")
            if not _finite_number(row.get("t")) or row["t"] < 0 or type(row.get("wrong")) is not bool:
                raise ValueError("window needs finite nonnegative t and boolean wrong")
            key = (*identity, row["k"])
            if key in out:
                raise ValueError("duplicate window identity")
            out[key] = (row["t"], not row["wrong"])
    if not out:
        raise ValueError("audit has no windows")
    return out


def compare_audits(audit_a: dict, audit_b: dict) -> dict:
    """Strictly match two T-323 audit objects and compute paired statistics."""
    a, b = _audit_rows(audit_a), _audit_rows(audit_b)
    if a.keys() != b.keys():
        raise ValueError(f"missing pairs: A-only={len(a.keys() - b.keys())}, B-only={len(b.keys() - a.keys())}")
    keys = sorted(a)
    if any(a[k][0] != b[k][0] for k in keys):
        raise ValueError("paired windows have different timestamps")
    return mcnemar_from_outcomes([a[k][1] for k in keys], [b[k][1] for k in keys])


def print_table(stats: dict, seeds: list[str]) -> None:
    """Print every evaluated head, including heads beyond T-307 raw/ap/ce."""
    print(f"seeds: {len(seeds)} ({', '.join(seeds)}); sample SD, not confidence intervals")
    for set_name, heads in stats.items():
        print(f"== {set_name} ==")
        for head, metrics in heads.items():
            for metric, (avg, sd) in metrics.items():
                value = f"{avg:.4f}+/-{sd:.4f}" if sd is not None else f"{avg:.4f}; SD undefined (one seed)"
                print(f"{head} {metric}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    seeds = commands.add_parser("seeds", help="mean +/- sample SD over T-307 seeds")
    seeds.add_argument("result", type=Path)
    paired = commands.add_parser("mcnemar", help="strict paired T-323 audit comparison")
    paired.add_argument("audit_a", type=Path)
    paired.add_argument("audit_b", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "seeds":
            data, names = load_results(args.result)
            print_table(seed_stats(data, names), names)
        else:
            result = compare_audits(json.loads(args.audit_a.read_text()), json.loads(args.audit_b.read_text()))
            result["limitation"] = "Overlapping/correlated windows: p-values exploratory; independent-pair assumption required."
            print(json.dumps(result, indent=2))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
