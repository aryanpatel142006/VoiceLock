import json
from pathlib import Path

import pytest

from eval import bench
from hearing import registry
from hearing.audio.io import PIPELINE_SR

LIMIT = 3  # fixtures per pipeline; keeps the module well under verify_fast's budget


@pytest.fixture(scope="module")
def fixtures_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixtures")
    bench.fixture_dirs(root, "tiny")  # generates the tiny set offline
    return root


@pytest.fixture(scope="module")
def report(fixtures_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    specs = ["passthrough", "gain(db=-6)", "delay(ms=20)"]
    return bench.run(specs, "tiny", "off", fixtures_root, out, limit=LIMIT)


def _rows(report, spec):
    rows = [r for r in report["rows"] if r["stages"] == spec]
    assert len(rows) == LIMIT
    return rows


def test_passthrough_matches_the_mix(report):
    for r in _rows(report, "passthrough"):
        assert abs(r["si_sdri"]) < 0.01
        assert r["si_sdr"] == pytest.approx(r["si_sdr_mix"], abs=1e-6)
        assert r["pesq"] == pytest.approx(r["pesq_mix"], abs=1e-6)
        assert r["stoi"] == pytest.approx(r["stoi_mix"], abs=1e-6)
        assert r["latency_ms"] == 0.0
        assert r["chunks"] == r["duration_s"] * 100


def test_gain_is_scale_invariant(report):
    for p, g in zip(_rows(report, "passthrough"), _rows(report, "gain(db=-6)")):
        assert p["fixture"] == g["fixture"]
        assert g["si_sdr"] == pytest.approx(p["si_sdr"], abs=1e-3)


def test_delay_is_latency_compensated(report):
    for r in _rows(report, "delay(ms=20)"):
        assert abs(r["si_sdri"]) < 0.01
        assert r["latency_ms"] == 20.0
        assert r["chunks"] == r["duration_s"] * 100 + 2  # two 10 ms flush chunks


def test_report_schema_and_json(report):
    assert report["failures"] == []
    assert {"device", "git_sha", "machine", "python", "fixture_set", "rows", "aggregates"} <= set(report)
    assert report["device"]["hostname"]
    saved = json.loads(Path(report["path"]).read_text())  # strict JSON: nan written as null
    assert saved["device"] == report["device"]
    assert len(saved["rows"]) == 3 * LIMIT
    agg = {a["stages"]: a for a in saved["aggregates"]}
    assert set(agg) == {"passthrough", "gain(db=-6)", "delay(ms=20)"}
    assert agg["delay(ms=20)"]["metrics"]["latency_ms"]["mean"] == 20.0
    assert agg["passthrough"]["compute_ms"]["chunks"] == sum(r["chunks"] for r in _rows(report, "passthrough"))
    assert saved["rows"][0]["wer"] is None


def test_table_has_one_row_per_pipeline(report):
    table = bench.format_table(report["aggregates"])
    lines = table.splitlines()
    assert lines[0].startswith("| stages | n | latency_ms | p50_ms | p95_ms | rtf |")
    assert len(lines) == 2 + 3
    assert lines[4].startswith("| delay(ms=20) | 3 | 20.0 |")


def test_cli_prints_table_and_writes_json(fixtures_root, tmp_path, capsys):
    rc = bench.main(
        ["--stages", "passthrough", "--stages", "gain(db=-6)>delay(ms=20)", "--fixtures", str(fixtures_root),
         "--out", str(tmp_path), "--limit", "1", "--wer", "mock"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "| passthrough | 1 | 0.0 |" in out
    assert "| gain(db=-6)>delay(ms=20) | 1 | 20.0 |" in out
    (path,) = tmp_path.glob("*.json")
    rows = json.loads(path.read_text())["rows"]
    assert [r["wer"] for r in rows] == [0.0, 0.0]  # the mock echoes the reference transcript


def test_fast_generates_missing_tiny_set(tmp_path, capsys):
    fixtures, out = tmp_path / "fx", tmp_path / "out"
    assert bench.main(["--fast", "--fixtures", str(fixtures), "--out", str(out), "--limit", "1"]) == 0
    assert (fixtures / "tiny" / "index.json").exists()
    assert "| passthrough | 1 |" in capsys.readouterr().out


class _Boom:
    name = "boom"
    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def process(self, chunk, ctx):
        raise RuntimeError("kaboom")

    def reset(self):
        pass


def test_failure_exits_nonzero_naming_stage_and_fixture(fixtures_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(registry._factories, "boom", _Boom)
    rc = bench.main(["--stages", "boom", "--fixtures", str(fixtures_root), "--out", str(tmp_path), "--limit", "1"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "stages='boom' fixture=000_snr-5_sir0_babble" in err
    assert "kaboom" in err


def test_missing_full_set_is_an_error(tmp_path, capsys):
    assert bench.main(["--set", "full", "--fixtures", str(tmp_path), "--out", str(tmp_path)]) == 2
    assert "make_fixtures --full" in capsys.readouterr().err
