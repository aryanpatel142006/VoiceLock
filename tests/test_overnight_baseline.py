"""Offline evidence gates; no models, network or audio hardware."""

import hashlib

import pytest

from eval.overnight_baseline import collect, quality_status


def test_fake_and_empty_results_never_become_measured():
    assert (
        quality_status({"fake": True, "mode": "real", "results": [1]})
        == "rejected_mock"
    )
    assert quality_status({"fake": False, "mode": "real", "results": []}) == "missing"
    assert quality_status({"fake": False, "mode": "real", "results": [1]}) == "measured"


def test_freeze_hashes_actual_bytes_and_marks_missing_evidence(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "source.py").write_bytes(b"abc")
    report = collect(tmp_path, "frozen-rev")
    assert report["files"]["repo/source.py"] == hashlib.sha256(b"abc").hexdigest()
    assert report["source_revision"] == "frozen-rev"
    assert report["team_status"] == "missing"
    assert report["asr_status"] == "missing"
    assert report["warmup_status"] == "missing"
    (tmp_path / "repo" / "source.py").write_bytes(b"changed")
    assert collect(tmp_path, "frozen-rev")["files"] != report["files"]


def test_missing_source_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect(tmp_path, "revision")


def test_cli_preserves_existing_checkpoint(tmp_path):
    from eval.overnight_baseline import main

    (tmp_path / "repo").mkdir()
    out = tmp_path / "checkpoint.json"
    args = ["--root", str(tmp_path), "--source-revision", "rev", "--out", str(out)]
    assert main(args) == 0
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert out.read_bytes() == original


def test_unverified_reference_counts_insertions_without_claiming_verification():
    from eval.overnight_baseline import reference_wer

    row = reference_wer("Hello world", "hello brave world again", False)
    assert row["wer"] == 1.0
    assert row["reference_status"] == "reference unverified (auto transcript)"
    assert reference_wer("Hello, WORLD!", "hello world", True)["wer"] == 0.0
    assert reference_wer("one two three", "one four")["word_errors"] == 2
    with pytest.raises(ValueError, match="empty"):
        reference_wer("", "speech", False)


def test_collect_reads_real_asr_and_rejects_mock_checkpoint(tmp_path):
    import json

    (tmp_path / "repo").mkdir()
    results = tmp_path / "results" / "t210-a4500-20261010T0315"
    results.mkdir(parents=True)
    asr = {
        "model": "base.en",
        "mock": False,
        "load_s": 3.4,
        "warmup_s": 0.5,
        "reference": "hello world",
        "results": [{"input": "target", "text": "Hello world", "seconds": 0.2}],
    }
    (results / "asr.json").write_text(json.dumps(asr))
    (results / "public.json").write_text(
        json.dumps({"mode": "real", "fake": False, "results": [{"role": "A"}]})
    )
    (results / "readiness.json").write_text(
        json.dumps({"cuda": True, "load_s": 10, "first_window_warmup_s": 7})
    )
    report = collect(tmp_path, "rev")
    assert report["asr_status"] == "measured"
    assert report["public_status"] == "measured"
    assert report["warmup_status"] == "measured"
    assert report["asr"]["scores"][0]["wer"] == 0
    assert report["asr"]["scores"][0]["reference_status"].startswith(
        "reference unverified"
    )
    assert (
        report["artifact_sha256"]["asr.json"]
        == hashlib.sha256((results / "asr.json").read_bytes()).hexdigest()
    )
    asr["mock"] = True
    (results / "asr.json").write_text(json.dumps(asr))
    assert collect(tmp_path, "rev")["asr_status"] == "rejected_mock"


def test_collect_reports_unverified_unless_explicit_audit(tmp_path):
    import json

    (tmp_path / "repo").mkdir()
    results = tmp_path / "results" / "t210-a4500-20261010T0315"
    results.mkdir(parents=True)
    
    # Write an asr.json that falsely claims verified reference
    asr = {
        "model": "base.en",
        "mock": False,
        "load_s": 3.4,
        "warmup_s": 0.5,
        "reference": "hello world",
        "reference_status": "verified reference",
        "results": [{"input": "target", "text": "Hello world", "seconds": 0.2}],
    }
    (results / "asr.json").write_text(json.dumps(asr))
    
    # Without explicit audit evidence, collect should downgrade it
    report = collect(tmp_path, "rev")
    assert report["asr"]["reference_status"] == "reference unverified (auto transcript)"
    assert report["asr"]["scores"][0]["reference_status"] == "reference unverified (auto transcript)"
    assert report["transcript_reference_status"] == "reference unverified (auto transcript)"
    
    # With explicit audit evidence, it stays verified
    report_verified = collect(tmp_path, "rev", explicit_verified=True)
    assert report_verified["asr"]["reference_status"] == "verified reference"
    assert report_verified["asr"]["scores"][0]["reference_status"] == "verified reference"
    assert report_verified["transcript_reference_status"] == "verified reference"
