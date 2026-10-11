import json
import wave

from eval.overnight_evidence_index import build_index, render_public_report


def _wav(path):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 160)


def test_indexes_results_without_inventing_missing_metrics(tmp_path):
    result = tmp_path / "run.json"
    result.write_text(
        json.dumps(
            {
                "stamp": "run-1",
                "git_sha": "abc",
                "device": {"device": "cpu", "hostname": "host"},
                "configuration": {"config_hash": "cfg"},
                "rows": [
                    {
                        "fixture": "quiet-target",
                        "selected_stream": 1,
                        "oracle_choice": 1,
                        "si_sdri": 2.5,
                        "latency_ms": 13,
                    }
                ],
            }
        )
    )
    fixture = tmp_path / "fixtures" / "one"
    fixture.mkdir(parents=True)
    (fixture / "manifest.json").write_text(
        json.dumps({"id": "one", "duration_s": 0.01, "sample_rate": 16000})
    )
    _wav(fixture / "target.wav")

    index = build_index([tmp_path], fixture_roots=[tmp_path / "fixtures"])
    record = index["results"][0]
    row = record["rows"][0]
    assert record["run_id"] == "run-1"
    assert record["config_hash"] == "cfg"
    assert row["metrics"]["si_sdri"] == 2.5
    assert row["metrics"]["wer"] is None
    assert index["coverage"][0]["wav_bytes"] == 364
    assert len(record["source_sha256"]) == 64


def test_public_report_is_capped_and_does_not_copy_transcripts(tmp_path):
    result = tmp_path / "run.json"
    result.write_text(
        json.dumps(
            {
                "rows": [
                    {"fixture": str(i), "transcript": "private words", "wer": 0.2}
                    for i in range(40)
                ]
            }
        )
    )
    report = render_public_report(build_index([tmp_path]))
    assert sum(line.startswith("| run | ") and "fixture/scenario" not in line for line in report.splitlines()) == 30
    assert "private words" not in report
