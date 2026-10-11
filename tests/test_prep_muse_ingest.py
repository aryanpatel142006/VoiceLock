import json

from eval.prep_muse_ingest import build_report, load_records, normalize_record


def test_normalize_preserves_provenance_and_strata():
    record = normalize_record(
        {
            "task_id": "T-224",
            "speaker": "target-a",
            "utterance": "u7",
            "interval": [1.0, 2.0],
            "source_hash": "abc",
            "quiet_target_db": -10,
            "sir": 5,
            "condition": "occluded",
            "stage": "mixer",
            "si_sdri": 0,
        },
        source="report.json",
    )
    assert record["strata"] == {
        "quiet_target_role": "quiet",
        "sir_db": 5.0,
        "condition": "occluded",
        "output_stage": "mixer",
    }
    assert record["provenance"]["speaker"] == "target-a"
    assert record["metrics"]["si_sdri"] == 0.0
    assert record["metric_status"]["si_sdri"] == "zero"


def test_missing_metric_is_not_zero():
    record = normalize_record({"task": "T-225", "stage": "oracle"}, source="x.json")
    assert record["metrics"]["si_sdri"] is None
    assert record["metric_status"]["si_sdri"] == "missing"


def test_report_separates_missing_zero_and_loss_roles():
    rows = [
        normalize_record(
            {
                "task": "T-224",
                "speaker": "a",
                "utterance": "1",
                "interval": [0, 1],
                "hash": "h1",
                "quiet_target_db": 0,
                "sir_db": 0,
                "condition": "clean",
                "stage": "oracle",
                "si_sdri": 1,
                "target_loss": 0,
                "telemetry": {"p95_ms": 4},
            },
            source="a.json",
        ),
        normalize_record(
            {
                "task": "T-224",
                "speaker": "a",
                "utterance": "2",
                "interval": [1, 2],
                "hash": "h2",
                "quiet_target_db": 0,
                "sir_db": 0,
                "condition": "clean",
                "stage": "picked",
                "target_loss": 1,
            },
            source="a.json",
        ),
    ]
    report = build_report(rows, seed=17)
    assert report["counts"]["available_records"] == 2
    assert report["metrics"]["si_sdri"]["zero"] == 0
    assert report["metrics"]["target_loss"]["zero"] == 1
    assert report["metrics"]["si_sdri"]["missing"] == 1
    assert report["losses"]["picked"]["count"] == 1
    assert report["telemetry"]["missing_records"] == 1


def test_loader_accepts_jsonl_and_detects_revision_conflict(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"task": "T-226", "revision": "a", "stage": "oracle"}),
                json.dumps({"task": "T-226", "revision": "b", "stage": "oracle"}),
            ]
        )
    )
    rows = load_records([path])
    report = build_report(rows, seed=1)
    assert report["counts"]["available_records"] == 2
    assert report["revision_conflicts"] == [{"task": "T-226", "revisions": ["a", "b"]}]
