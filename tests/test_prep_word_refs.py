import json
from pathlib import Path

from eval.prep_word_refs import build_plan, main


def _case(**updates):
    case = {
        "id": "case-1",
        "sample_rate": 16000,
        "target": {
            "speaker": "target-a",
            "transcript": "red blue red",
            "alignment": [
                {"word": "red", "start": 0.0, "end": 0.2},
                {"word": "blue", "start": 0.3, "end": 0.5},
                {"word": "red", "start": 0.6, "end": 0.8},
            ],
        },
        "interferer": {
            "speaker": "other-b",
            "transcript": "blue green",
            "alignment": [
                {"word": "blue", "start": 0.35, "end": 0.55},
                {"word": "green", "start": 0.7, "end": 0.9},
            ],
        },
        "source": {"dataset": "fixture", "utterance": "u1", "hash": "abc"},
        "crop": {"start": 0.0, "end": 1.0},
    }
    case.update(updates)
    return case


def test_build_plan_flags_shared_words_and_preserves_repeated_tokens():
    plan = build_plan([_case()])
    item = plan["cases"][0]

    assert item["status"] == "available"
    assert [word["text"] for word in item["target"]["words"]] == ["red", "blue", "red"]
    assert item["target"]["words"][0]["occurrence"] == 1
    assert item["target"]["words"][2]["occurrence"] == 2
    assert item["ambiguous_shared_words"] == ["blue"]
    assert item["recipe"]["target_deletions"] == "score target words as references; count missing words as deletions"


def test_crop_without_alignment_is_unavailable_and_not_full_word_scored():
    case = _case(
        target={
            "speaker": "target-a",
            "transcript": "red blue",
            "alignment": None,
        },
        crop={"start": 0.2, "end": 0.6},
    )
    item = build_plan([case])["cases"][0]

    assert item["status"] == "unavailable"
    assert item["target"]["words"] == []
    assert "no word boundaries" in item["unavailable_reason"]


def test_cli_writes_public_summary_and_private_refs(tmp_path: Path):
    source = tmp_path / "input.json"
    output = tmp_path / "refs.json"
    summary = tmp_path / "summary.json"
    checkpoints = tmp_path / "checkpoints"
    source.write_text(json.dumps({"seed": 17, "cases": [_case()]}))

    assert main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--max-cases",
            "1",
            "--checkpoint-dir",
            str(checkpoints),
        ]
    ) == 0
    result = json.loads(output.read_text())
    public = json.loads(summary.read_text())
    assert result["config"]["seed"] == 17
    assert public["available_cases"] == 1
    assert public["planned_cases"] == 1
    assert "target-a" not in summary.read_text()
    assert len(list(checkpoints.glob("*.json"))) == 1
