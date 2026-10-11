from pathlib import Path

import pytest

from eval.prep_runtime_matrix import (
    EVENT_CASES,
    account_timeline,
    build_matrix,
    dry_run,
    make_event_fixtures,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_is_bounded_and_hashes_available_configs():
    report = dry_run(ROOT, repetitions=1, case_cap=24)
    assert report["inference"] is False
    assert report["gpu_used"] is False
    assert report["planned_cases"] == 8
    assert report["config_hashes"]
    assert all(case["required_telemetry"] for case in report["cases"])


def test_matrix_only_uses_configured_pairs():
    cases = build_matrix(ROOT, repetitions=1, case_cap=24)
    configured = {
        tuple(pair)
        for pairs in dry_run(ROOT, repetitions=1)["available_inputs"]["configured_pairs"].values()
        for pair in pairs
    }
    assert {(case.window_s, case.hop_s) for case in cases} <= configured


def test_event_fixtures_have_expected_statuses(tmp_path):
    manifest = make_event_fixtures(tmp_path)
    assert [item["case_id"] for item in manifest["cases"]] == list(EVENT_CASES)
    assert (tmp_path / "manifest.json").is_file()
    assert all(item["expected_status"] for item in manifest["cases"])


def test_timeline_uses_observed_device_not_config_claim():
    summary = account_timeline(
        [
            {
                "seq": 1,
                "capture_ts": 0.0,
                "deadline_ts": 0.25,
                "emitted_ts": 0.3,
                "status": "deadline_missed",
                "tier": "enrolled",
                "device": None,
                "config_device_claim": "cuda",
            }
        ]
    )
    assert summary["n_late"] == 1
    assert summary["gpu_evidence_events"] == 0


def test_timeline_rejects_incomplete_events():
    with pytest.raises(ValueError, match="missing"):
        account_timeline([{"seq": 1, "status": "ok"}])
