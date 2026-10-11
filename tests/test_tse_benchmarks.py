"""Catalog regressions: provenance, honest latency labels, and reproducible output."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/research/tse_benchmarks.json"


def run_catalog(*args):
    return subprocess.run(
        [sys.executable, "-m", "eval.tse_benchmarks", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_catalog_cli_prints_both_tables_and_hunts():
    result = run_catalog()
    assert result.returncode == 0, result.stderr
    assert "| Benchmark |" in result.stdout
    assert "| Method |" in result.stdout
    assert "Phone recordings" in result.stdout
    assert "Steal list" in result.stdout


def test_checked_in_document_is_reproducible():
    result = run_catalog()
    assert result.returncode == 0, result.stderr
    assert result.stdout == (ROOT / "docs/research/TSE_BENCHMARKS.md").read_text()


@pytest.mark.parametrize("mutation", ["source", "metric", "latency", "coverage"])
def test_invalid_catalog_fails_before_printing_tables(tmp_path, mutation):
    data = copy.deepcopy(json.loads(CATALOG.read_text()))
    if mutation == "source":
        data["methods"][0]["results"][0]["source"] = "missing-source"
    elif mutation == "metric":
        del data["methods"][0]["results"][0]["metric"]
    elif mutation == "latency":
        data["methods"][0]["streaming"] = "verified-subsecond"
        data["methods"][0]["latency"] = None
    else:
        data["benchmarks"] = data["benchmarks"][:1]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    result = run_catalog("--data", path)
    assert result.returncode != 0
    assert "Invalid catalog:" in result.stderr
    assert not result.stdout


def test_method_results_name_the_scoring_dataset():
    result = run_catalog()
    assert result.returncode == 0, result.stderr
    methods = result.stdout.split("## Method table", 1)[1].split("## Focused hunts", 1)[0]
    assert "MossFormer2 on WSJ0-2mix" in methods
    assert "USEF-TFGridNet on LibriMix max-mode TSE" in methods


def test_research_lane_exposes_release_gaps_and_matched_test():
    result = run_catalog()
    assert result.returncode == 0, result.stderr
    assert "## Research lane handoff" in result.stdout
    assert "dc11cb18ee8636064a7851f38b9348143c343cdc" in result.stdout
    assert "Matched falsification protocol" in result.stdout


def test_runnable_claim_requires_pinned_weights(tmp_path):
    data = json.loads(CATALOG.read_text())
    data["research_lane"] = {"candidates": [{"ready": True, "weights_revision": None}]}
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(data))
    result = run_catalog("--data", path)
    assert result.returncode != 0
    assert "pinned weights" in result.stderr


def test_mixed_track_results_keep_their_inference_mode(tmp_path):
    """Dropping track labels falsely assigns the online rule to offline scores."""
    data = json.loads(CATALOG.read_text())
    row = next(row for row in data["benchmarks"] if row["id"] == "real-tse")
    row["results"][0]["inference_mode"] = "online"
    row["results"][1]["inference_mode"] = "offline"
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(data))
    result = run_catalog("--data", path)
    assert result.returncode == 0, result.stderr
    assert "CARTSE [online] on REAL-TSE" in result.stdout
    assert "MERL [offline] on REAL-TSE" in result.stdout
