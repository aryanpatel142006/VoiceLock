import json

from eval.prep_schedule import build_schedule


def _pair(pair_id, target, interferer):
    return {
        "pair_id": pair_id,
        "target": {"source_id": target, "speaker_id": target, "stats": {"sha256": target}},
        "interferer": {
            "source_id": interferer,
            "speaker_id": interferer,
            "stats": {"sha256": interferer},
        },
        "speaker_group": "same_documented_sex",
    }


def test_schedule_is_reproducible_and_stays_under_global_cap():
    pairs = [_pair("p1", "t1", "i1"), _pair("p2", "t2", "i2")]
    left = build_schedule(pairs, [], seed=7, max_cases=24)
    right = build_schedule(pairs, [], seed=7, max_cases=24)

    assert left == right
    assert len(left["cases"]) <= 24
    assert len(left["smoke_cases"]) == 12
    assert all(case["case_id"] in {item["case_id"] for item in left["cases"]} for case in left["smoke_cases"])


def test_schedule_keeps_roles_separate_and_reports_missing_cells():
    pairs = [_pair("p1", "t1", "i1")]
    result = build_schedule(pairs, [], seed=1, max_cases=12)

    assert {case["target_role"] for case in result["cases"]} == {"target", "interferer"}
    assert {case["partition"] for case in result["cases"]} <= {"development", "final"}
    assert result["missing_cells"]
    assert all("sir_db" in cell and "condition" in cell for cell in result["missing_cells"])
    assert "sir_db__condition" in result["coverage"]["pairwise"]


def test_existing_gpu_cases_count_against_cap_and_checkpoints_are_resumable(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text(json.dumps({"cases": [{"case_id": f"old-{n}"} for n in range(299)]}))
    pairs = [_pair("p1", "t1", "i1")]

    result = build_schedule(pairs, [existing], seed=2, max_cases=300)

    assert result["accounting"]["existing_cases"] == 299
    assert len(result["cases"]) == 1
    checkpoint = result["checkpoint"]
    assert checkpoint["resume_skips_verified"] is True
