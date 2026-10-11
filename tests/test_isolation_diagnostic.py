import json

import numpy as np
import pytest

from eval import isolation_diagnostic as d


def _run(case):
    t, o = d.fake_stems(8.0)
    mix = (t + o).astype(np.float32)
    runner = d.FakeRunner(case, t, o, 1.0)
    rep = d.replay(runner, mix, t, o)
    return d.summarize(rep, mix, t, o, "all"), rep


@pytest.mark.parametrize(
    ("case", "expected"),
    [("separator_missing_target", "separation"), ("wrong_selector", "selector"),
     ("mix_back_leak", "mix_back"), ("missed_deadline", "deadline"), ("ok", "none")],
)
def test_dry_run_classifies_constructed_failures(case, expected):
    summary, _ = _run(case)
    assert summary["dominant_failure"] == expected
    if case == "ok":
        assert summary["categories"]["ok"] > 0


def test_deadline_counts_raw_fallback_and_hop():
    summary, _ = _run("missed_deadline")
    assert summary["n_deadline_missed"] == summary["n_raw_fallback"] > 0
    assert summary["hop_s"] == 1.0 and summary["window_s"] == 2.0


def test_oracle_vs_deployable_labelled_with_denominators():
    summary, _ = _run("wrong_selector")
    rows = {r["estimator"]: r for r in summary["rows"]}
    assert rows["oracle_pick"]["oracle"] and not rows["oracle_pick"]["deployable"]
    assert rows["actual_choice"]["deployable"]
    assert rows["oracle_pick"]["sample_rate_hz"] == 16000 and rows["oracle_pick"]["n_windows"] > 0
    assert summary["ceiling_gap_db"] > 10


def test_target_absent_uses_leak_not_sisdr():
    t, o = d.fake_stems(8.0)
    t = t.copy()
    t[3 * d.SR:5 * d.SR] = 0
    mix = (t + o).astype(np.float32)
    rep = d.replay(d.FakeRunner("ok", t, o, 1.0), mix, t, o)
    absent = [r for r in rep.windows if not r["target_active"]]
    assert absent and all(r["final_sisdr"] is None and "leak_db" in r for r in absent)
    row = next(r for r in d.summarize(rep, mix, t, o)["rows"] if r["estimator"].startswith("target_absent"))
    assert row["sisdr_mean_db"] is None and row["leak_mean_db"] is not None


def test_scale_target_ratio():
    t, o = d.fake_stems(2.0)
    ts, os_ = d.scale_target(t, o, -10.0)
    assert d._db(d._rms(ts)) - d._db(d._rms(os_)) == pytest.approx(-10.0, abs=0.01)


def test_probes_negative_gate_and_margins():
    t, o = d.fake_stems(8.0)
    p = d.embedding_probes(d.FakeEmbedder(), t[:48000], o[:48000], t, o)
    assert set(p["crop_margin"]) == {"0.5s", "1.0s", "2.0s", "4.0s"}
    assert isinstance(p["negative_profile_dropped_at_0.7"], bool)
    assert p["crop_margin"]["2.0s"] > 0


def test_cli_dry_run_writes_manifest(tmp_path, capsys):
    out = tmp_path / "m.json"
    assert d.main(["--dry-run", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["fake"] is True and "NOT measured" in data["label"]
    text = capsys.readouterr().out
    assert "ORACLE (not deployable)" in text and "FAKE" in text


def test_real_run_missing_inputs_is_blocker(tmp_path, capsys):
    assert d.main(["--scene", str(tmp_path), "--profiles", str(tmp_path)]) == 2
    assert d.main([]) == 2
