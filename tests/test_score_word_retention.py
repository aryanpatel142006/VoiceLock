import json

import pytest

from eval.score_word_retention import (
    load_records,
    score_records,
    write_report,
)


def test_scores_wer_and_deletion_word_loss():
    rows = score_records(
        [
            {
                "scene_id": "s1",
                "interference_ratio": -5,
                "target_text": "the quick brown fox",
                "separated_transcript": "the brown fox",
            },
            {
                "scene_id": "s2",
                "interference_ratio": 0,
                "target_text": "hello world",
                "separated_transcript": "hello word",
            },
        ]
    )

    assert rows["scenes"][0]["wer"] == pytest.approx(0.25)
    assert rows["scenes"][0]["word_loss_pct"] == pytest.approx(25.0)
    assert rows["scenes"][1]["wer"] == pytest.approx(0.5)
    assert rows["scenes"][1]["word_loss_pct"] == pytest.approx(0.0)
    assert rows["by_interference_ratio"]["-5"]["word_loss_pct"] == pytest.approx(25.0)
    assert rows["by_interference_ratio"]["0"]["wer"] == pytest.approx(0.5)


def test_load_records_accepts_jsonl_and_rejects_missing_fields(tmp_path):
    source = tmp_path / "scenes.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "x",
                "sir_db": 10,
                "reference": "one two",
                "transcript": "one two",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_records([source])[0]["scene_id"] == "x"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"scene_id": "bad"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target_text"):
        load_records([bad])


def test_write_report_is_stable_json(tmp_path):
    report = score_records(
        [
            {
                "scene_id": "x",
                "interference_ratio": 3.5,
                "target_text": "one",
                "separated_transcript": "",
            }
        ]
    )
    destination = tmp_path / "report.json"
    write_report(report, destination)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["scenes"][0]["scene_id"] == "x"
    assert loaded["scenes"][0]["word_loss_pct"] == 100.0
