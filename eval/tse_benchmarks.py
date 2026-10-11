"""Validate and print the cited TSE research catalog; no models or network required.

Run: .venv/bin/python -m eval.tse_benchmarks > docs/research/TSE_BENCHMARKS.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "docs/research/tse_benchmarks.json"
STREAMING = {
    "offline", "verified-subsecond", "causal-unverified", "chunked-over-limit",
    "protocol-subsecond", "not-established", "mixed-online-offline",
}


def validate(data: dict) -> None:
    """Reject broken provenance, missing comparisons, and unsupported latency labels."""
    sources = data["sources"]
    for source in sources.values():
        if not source["url"].startswith("https://") or not source["title"]:
            raise ValueError("sources require an HTTPS URL and title")
    benchmark_ids = {row["id"] for row in data["benchmarks"]}
    for section in ("benchmarks", "methods"):
        rows = data[section]
        if len(rows) < 8:
            raise ValueError(f"{section} requires at least eight entries")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {section} IDs")
        for row in rows:
            if row["streaming"] not in STREAMING:
                raise ValueError("unknown streaming status")
            if row["streaming"] in {"verified-subsecond", "protocol-subsecond"}:
                latency = row["latency"]
                if not latency or latency["unit"] != "ms" or not 0 <= latency["value"] < 1000:
                    raise ValueError("subsecond status requires cited latency below one second")
            if not row["results"] and not row.get("result_gap"):
                raise ValueError("missing results require an explicit evidence gap")
            for result in row["results"]:
                if result["dataset"] not in benchmark_ids:
                    raise ValueError("unknown result dataset")

    def walk(value):
        if isinstance(value, dict):
            if "value" in value:
                if not isinstance(value["value"], (float, int)) or isinstance(value["value"], bool):
                    raise ValueError("measurement value must be numeric")
                if not math.isfinite(value["value"]):
                    raise ValueError("measurement must be finite")
                for key in ("metric", "unit", "source", "locator", "year", "conditions"):
                    if not value[key]:
                        raise ValueError(f"measurement missing {key}")
            for key, item in value.items():
                if key == "source" and item not in sources:
                    raise ValueError(f"unknown source: {item}")
                if (
                    key == "sources" and isinstance(item, list)
                    and (not item or any(ref not in sources for ref in item))
                ):
                    raise ValueError("invalid source list")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    for candidate in data.get("research_lane", {}).get("candidates", []):
        if candidate["ready"] and not candidate.get("weights_revision"):
            raise ValueError("runnable candidates require pinned weights")


def render(data: dict) -> str:
    """Render every catalog section, preserving metric names and source locators."""
    validate(data)
    sources = data["sources"]
    datasets = {row["id"]: row["name"] for row in data["benchmarks"]}

    def cite(ref):
        return f"[{ref}]({sources[ref]['url']})"

    def refs(row):
        return " ".join(cite(ref) for ref in row["sources"])

    def measurement(claim):
        if claim is None:
            return "Not reported / not verified"
        return (
            f"{claim['value']:g} {claim['unit']} {claim['metric']} "
            f"({claim['year']}; {claim['conditions']}; {claim['locator']}) {cite(claim['source'])}"
        )

    def results(row):
        if not row["results"]:
            return row["result_gap"]

        def method_label(result):
            mode = result.get("inference_mode")
            return result["method"] + (f" [{mode}]" if mode else "")

        return "<br>".join(
            f"{method_label(result)} on {datasets[result['dataset']]} — {measurement(result)}"
            for result in row["results"]
        )

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    def table(headers, rows):
        return [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
        ]

    lines = [
        "# Streaming target-speaker extraction benchmarks",
        "", f"Research snapshot: {data['as_of']}. {data['scope']}", "",
        ("Generated from [tse_benchmarks.json](tse_benchmarks.json). "
        "Regenerate with `.venv/bin/python -m eval.tse_benchmarks > docs/research/TSE_BENCHMARKS.md`. "
        "The JSON is authoritative; the renderer works offline."), "",
        *data["interpretation"], "", "## Benchmark table", "",
    ]
    lines += table(
        ["Benchmark", "Task / data / protocol", "Reported best / reference result and year",
         "Streaming status of listed result / latency", "Limits"],
        [
            [row["name"], f"{row['task']} {refs(row)}", results(row),
             f"{row['streaming']}; {measurement(row['latency'])}",
             f"{row['best_scope']}; {row['caveat']}"]
            for row in data["benchmarks"]
        ],
    )
    lines += ["", "## Method table", ""]
    lines += table(
        ["Method", "Architecture / target cue", "Published metric on named set",
         "Streaming / latency", "Runtime evidence", "Code / licence / limits"],
        [
            [row["name"], f"{row['architecture']}; {row['enrollment']} {refs(row)}",
             results(row), f"{row['streaming']}; {measurement(row['latency'])}",
             "<br>".join(measurement(claim) for claim in row["runtime"]) or "Not reported",
             f"{row['code']}; {row['license']}; {row['caveat']}"]
            for row in data["methods"]
        ],
    )
    lines += ["", "## Focused hunts", ""]
    for hunt in data["hunts"]:
        lines += [f"### {hunt['name']}", "", f"{hunt['finding']} {refs(hunt)}", ""]
        lines += [f"- {measurement(claim)}" for claim in hunt["evidence"]]
        lines += ["", f"Remaining gap: {hunt['gap']}", ""]
    lines += ["## Steal list", ""]
    lines += table(
        ["Idea", "Evidence", "Proposed next experiment / adoption gate"],
        [[row["idea"], f"{row['evidence']} {refs(row)}", row["action"]]
         for row in data["steal_list"]],
    )
    if lane := data.get("research_lane"):
        lines += ["", "## Research lane handoff", "", lane["status"], ""]
        lines += table(
            ["Candidate / release pin", "Weights / licence", "Chunk / state / enrollment", "Adapter gate"],
            [[(f"{row['name']}; code {row['code_revision']}; ready={row['ready']}. "
               f"{row['release']} {refs(row)}"),
              f"weights revision: {row['weights_revision'] or 'not verified'}; {row['license']}",
              f"{row['chunk_state']} {row['enrollment']}", row["adapter"]]
             for row in lane["candidates"]],
        )
        lines += ["", "### Resident input inspection", ""]
        lines += [f"- `{row['path']}`: {row['status']}" for row in lane["resident_inputs"]]
        lines += ["", "### Alternative short-utterance identity methods", ""]
        lines += [f"- {row['idea']}: {row['gate']} {refs(row)}"
                  for row in lane["identity_alternatives"]]
        lines += ["", "### Matched falsification protocol", ""]
        lines += [f"- {key}: {value}" for key, value in lane["falsification_protocol"].items()]
        lines += ["", lane["coordination"], ""]
    lines += ["", "## Sources", ""]
    for ref, source in sources.items():
        lines += [f"- {cite(ref)} — {source['title']}. {source['note']}"]
    lines += ["", "## Evidence gaps", ""]
    lines += [f"- {gap}" for gap in data["gaps"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    try:
        print(render(json.loads(args.data.read_text())), end="")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Invalid catalog: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
