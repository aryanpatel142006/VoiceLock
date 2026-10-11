import json

from eval.prep_morning_table import build_table, main


def _row(**kwargs):
    return {
        "stage": "dfn",
        "speaker": "s1",
        "sir_db": 0,
        "condition": "target_present",
        "input_hash": "input-a",
        "model_revision": "model-a",
        "config_revision": "config-a",
        **kwargs,
    }


def test_aggregates_counts_percentiles_and_gate():
    result = build_table([_row(pesq_wb=2.8, dnsmos_sig=3.4, latency_ms=10), _row(pesq_wb=3.0, dnsmos_sig=3.6, latency_ms=20)])
    group = result["groups"][0]
    assert group["samples"] == 2
    assert group["available"]["pesq_wb"] == 2
    assert group["p50_ms"] == 15
    assert group["p95_ms"] == 19.5
    assert group["quality_gate"] == "pass"
    assert group["comparison_eligible"] is True


def test_all_missing_is_unknown():
    group = build_table([_row()])["groups"][0]
    assert group["quality_gate"] == "unknown"
    assert group["available"]["pesq_wb"] == 0
    assert group["missing"]["dnsmos_sig"] == 1


def test_invalid_comparison_hashes_are_ineligible():
    group = build_table([_row(pesq=2.9, dnsmos_sig=3.5), _row(input_hash="input-b", pesq=2.9, dnsmos_sig=3.5)])["groups"][0]
    assert group["comparison_eligible"] is False
    assert group["revisions"]["input"] == ["input-a", "input-b"]


def test_target_absent_is_not_eligible():
    group = build_table([_row(target_present=False, pesq=3, dnsmos_sig=4)])["groups"][0]
    assert group["comparison_eligible"] is False


def test_cli_checkpoint_and_atomic_output(tmp_path, capsys):
    source = tmp_path / "results.json"
    output = tmp_path / "table.json"
    checkpoint = tmp_path / "checkpoint.json"
    source.write_text(json.dumps({"rows": [_row(pesq=2.9, dnsmos_sig=3.5)]}))
    assert main([str(source), "--output", str(output), "--checkpoint", str(checkpoint)]) == 0
    assert json.loads(output.read_text())["samples_input"] == 1
    assert json.loads(checkpoint.read_text())["complete"] is True
    assert "groups\": 1" in capsys.readouterr().out
    assert main(
        [str(source), "--output", str(output), "--checkpoint", str(checkpoint), "--resume"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["resumed"] is True
