"""``eval.bench --full`` plumbing on the tiny set: mock WER, mock DNSMOS, cheap stages only."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from eval import bench
from eval.bench import BankEntry

LIMIT = 2


@pytest.fixture(scope="module")
def fixtures_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixtures")
    bench.fixture_dirs(root, "tiny")
    return root


BANK = [
    BankEntry("baseline", "none", "passthrough"),
    BankEntry("natural", "audio-only", "gain(db=-6)", 1.0),
    BankEntry("natural", "gate", "gate", 1.0, "oracle"),
    BankEntry("natural", "audio-only", "delay(ms=20)", 0.5),
    BankEntry("focus_hq", "upper-bound", "tse:oracle@1", 1.0),
    BankEntry("focus_hq", "focus_hq", "tse:identity@1", 0.5),
    BankEntry("quiet", "none", "silence", 1.0, score_audio=False, wer=False, dnsmos=False),
    BankEntry("baseline", "none", "passthrough", condition="noise_only"),
    BankEntry("baseline", "none", "passthrough", fset="av"),
]
SLIDER = [BankEntry("natural", "gate", "gate", s, "oracle", wer=False) for s in (0.0, 1.0)]


@pytest.fixture(scope="module")
def report(fixtures_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return bench.run_full("tiny", "mock", "mock", fixtures_root, out, limit=LIMIT, real_dir=None, av_set=None,
                          bank=BANK, slider=SLIDER, tts_engines=[])


def _agg(report, spec, s, target="none", condition="mix"):
    a = bench._find(report, spec, s, target, condition)
    assert a is not None, spec
    return a


def test_every_entry_runs_and_json_is_strict(report):
    assert report["failures"] == []
    saved = json.loads(Path(report["path"]).read_text())
    assert saved["kind"] == "bench-full"
    assert {a["label"] for a in saved["aggregates"] if a["group"] == "bank"} == {e.label for e in BANK[:-1]}
    assert saved["skipped"] == [{"label": BANK[-1].label, "reason": saved["skipped"][0]["reason"]}]  # no AV set
    assert Path(report["path"]).with_suffix(".md").exists()
    assert saved["dnsmos"] == "mock" and saved["wer"] == "mock"


def test_slider_reuses_main_table_runs(report):
    slider = bench._bank(report, "slider")
    assert len(slider) == 2
    reused = _agg(report, "gate", 1.0, "oracle")
    assert sum(1 for r in report["rows"] if r["label"] == reused["label"]) == LIMIT  # run once, listed twice


def test_oracle_gate_ducks_only_when_target_silent(report):
    full = _agg(report, "gate", 1.0, "oracle")
    off = next(a for a in report["aggregates"] if a["spec"] == "gate" and a["suppression"] == 0.0)
    assert bench._m(off, "si_sdri") == pytest.approx(0.0, abs=0.05)  # s = 0: room untouched
    assert bench._m(full, "si_sdri") > 0.1  # ducking the target's pauses removes interferer energy


def test_honest_latency_has_no_lag_error(report):
    for spec, s in (("passthrough", 0.5), ("delay(ms=20)", 0.5)):
        assert bench._m(_agg(report, spec, s), "lag_error_ms") == pytest.approx(0.0, abs=0.1)


def test_tse_tiers_through_the_window_extractor(report):
    oracle = _agg(report, "tse:oracle@1", 1.0)
    assert bench._m(oracle, "si_sdri") > 20
    assert bench._m(oracle, "latency_ms") == 1000.0
    assert oracle["compute_unit"] == "window"
    identity = _agg(report, "tse:identity@1", 0.5)  # mix scaled onto the mix + (1 - s) residual = the mix
    assert bench._m(identity, "si_sdri") == pytest.approx(0.0, abs=0.05)


def test_quiet_mode_skips_audio_scores(report):
    q = _agg(report, "silence", 1.0)
    assert q["n"] == LIMIT
    assert math.isnan(bench._m(q, "si_sdri"))


def test_markdown_and_claims(report, tmp_path):
    md = bench.render_markdown(report)
    for heading in ("### Every mode and tier", "### Chart 1", "### Chart 2", "### Chart 3",
                    "### Real phone clips", "### Slide-safe one-line claims"):
        assert heading in md
    assert "| quiet | none | `silence` |" in md
    assert isinstance(report["claims"], list)
    doc = tmp_path / "BENCHMARKS.md"
    doc.write_text("# Benchmarks\n\nintro\n\n## Per-task rows\nkeep me\n")
    bench.update_doc(doc, "first")
    bench.update_doc(doc, "second")
    text = doc.read_text()
    assert text.count(bench.DOC_START) == 1 and "second" in text and "first" not in text
    assert "keep me" in text and text.index("keep me") < text.index(bench.DOC_START)


def test_from_json_rerenders_the_same_doc(report, tmp_path):
    path = bench.write_results(tmp_path, report)
    again = bench.load_report(path)
    assert again["claims"] == report["claims"]
    assert bench.render_markdown(again).split("\n")[2:] == bench.render_markdown(report).split("\n")[2:]
    doc = tmp_path / "B.md"
    rc = bench.main(["--full", "--from-json", str(path), "--doc", str(doc)])
    assert rc == (1 if report["failures"] else 0)
    assert bench.DOC_START in doc.read_text() and path.with_suffix(".md").exists()


def test_claims_name_the_cpu():
    nat = {"spec": bench.NATURAL, "suppression": 1.0, "target": "none", "condition": "noise_only", "fset": "audio",
           "n": 1, "mode": "natural", "tier": "audio-only", "compute_ms": {"p95": 0.5},
           "metrics": {"latency_ms": {"mean": 13.0}, "rtf": {"mean": 0.05}, "si_sdri": {"mean": 0.0}}}
    report = {"device": {"machine": "arm64", "hostname": "h"}, "cpu": "Apple M4 Pro",
              "fixture_set": {"name": "tiny", "count": 1, "tse_count": 1}, "wer": "mock", "dnsmos": "mock",
              "aggregates": [nat], "synth": {}, "real_clips": {}}
    assert "on an Apple M4 Pro laptop (h)" in bench.claims(report)[0]


def test_text_claim_names_the_better_caption_input():
    def text(tier, wer):
        return {"spec": "x", "suppression": 1.0, "target": "none", "condition": "mix", "fset": "audio", "n": 2,
                "mode": "text", "tier": tier, "group": "bank", "compute_ms": {"p95": 0.0},
                "metrics": {"wer": {"mean": wer}}}
    report = {"device": {"machine": "arm64", "hostname": "h"}, "fixture_set": {"name": "tiny", "count": 2,
              "tse_count": 2}, "wer": "whisper", "dnsmos": "mock", "synth": {}, "real_clips": {},
              "aggregates": [text("audio-only", 0.73), text("none", 0.59)]}
    (claim,) = bench.claims(report)
    assert "WER 0.59 from the raw mix vs 0.73" in claim


def test_residual_lag_detects_a_wrong_latency():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000 * 2).astype(np.float32)
    late = np.concatenate([np.zeros(480, np.float32), x[:-480]])  # 30 ms behind
    assert bench.residual_lag_ms(late, x, 16000) == pytest.approx(30.0)
    assert bench.residual_lag_ms(x, late, 16000) == pytest.approx(-30.0)
    assert math.isnan(bench.residual_lag_ms(np.zeros(100), x, 16000))


def test_claims_skip_missing_or_negative_numbers():
    report = {
        "device": {"machine": "arm64", "hostname": "h"},
        "fixture_set": {"name": "tiny", "count": 2, "tse_count": 2},
        "wer": "mock", "dnsmos": "mock",
        "aggregates": [], "synth": {}, "real_clips": {},
    }
    assert bench.claims(report) == []


def test_cli_full_writes_doc(fixtures_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "full_bank", lambda windows: BANK[:2])
    monkeypatch.setattr(bench, "slider_bank", list)
    monkeypatch.setattr(bench, "synth_latency", lambda engines: [])
    doc = tmp_path / "B.md"
    rc = bench.main(["--full", "--set", "tiny", "--wer", "mock", "--dnsmos", "mock", "--limit", "1",
                     "--fixtures", str(fixtures_root), "--out", str(tmp_path / "res"), "--doc", str(doc),
                     "--real", str(tmp_path / "none")])
    assert rc == 0
    assert bench.DOC_START in doc.read_text()
    assert "updated" in capsys.readouterr().out


def test_render_writes_an_aligned_clip(tmp_path):
    from hearing.audio import io as aio

    src = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "tiny" / "speech_a.wav"
    out = tmp_path / "out.wav"
    info = bench.render(src, "delay(ms=20)", out, 0.5)
    x, _ = aio.load(src)
    y, _ = aio.load(out)
    assert info["latency_ms"] == 20.0 and len(y) == len(x)
    assert np.allclose(x, y, atol=1e-3)
    assert bench.main(["--render", str(src), "--stages", "gain(db=-6)", "--render-out", str(tmp_path / "g.wav")]) == 0
    assert (tmp_path / "g.wav").exists()


def test_noise_only_condition_drops_the_talker(report):
    noisy = _agg(report, "passthrough", 0.5, condition="noise_only")
    mixed = _agg(report, "passthrough", 0.5)
    assert noisy["n"] == mixed["n"] == LIMIT
    assert bench._m(noisy, "si_sdr_mix") > bench._m(mixed, "si_sdr_mix")  # no competing talker: an easier input
    assert bench._m(noisy, "si_sdri") == pytest.approx(0.0, abs=0.01)


def test_load_any_fixture_accepts_av_manifests(fixtures_root, tmp_path):
    import shutil

    src = bench.fixture_dirs(fixtures_root, "tiny")[0]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    m = json.loads((dst / "manifest.json").read_text())
    m.update(layout="two_face", target_box=[0.1, 0.1, 0.3, 0.3])
    m["paths"]["video"] = "video.mp4"
    (dst / "manifest.json").write_text(json.dumps(m))
    manifest, stems = bench.load_any_fixture(dst)
    assert manifest.id == m["id"] and len(stems.mix) == len(stems.target)
    noise_only = bench.apply_condition(stems, "noise_only")
    assert np.allclose(noise_only.mix, stems.target + stems.noise)
    with pytest.raises(ValueError):
        bench.apply_condition(stems, "bogus")
