"""Offline tests for the T-248 failure-localizing benchmark prototype (synthetic arrays only)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[1] / "experiments/theory/T-248/bench_design.py"
_spec = importlib.util.spec_from_file_location("bench_design_t248", _P)
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)


def _case(scenario="overlap", **kw):
    c = {
        "case_id": "x",
        "group": "g00",
        "role": "A",
        "target_db": 0,
        "room": "matched",
        "scenario": scenario,
        "voice_bucket": "same",
        "split": "dev",
        "case_seed": 7,
    }
    return c | kw


def test_si_sdr_scale_invariant_and_noise_sensitive():
    r = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    assert bd.si_sdr(3 * r, r) > 60
    assert (
        8 < bd.si_sdr(r + 0.3 * np.random.default_rng(1).standard_normal(8000), r) < 12
    )


def test_matrix_is_deterministic_and_splits_by_group():
    a, b = bd.build_matrix(248), bd.build_matrix(248)
    assert a == b
    dev = {c["group"] for c in a if c["split"] == "dev"}
    held = {c["group"] for c in a if c["split"] == "heldout"}
    assert dev and held and not dev & held  # source-group disjoint
    assert {c["role"] for c in a} == {"A", "B"} and {c["target_db"] for c in a} >= {
        -10,
        -15,
    }
    assert {c["room"] for c in a} == {"matched", "mismatched"}


def test_oracle_selector_has_no_false_speech_and_full_target():
    sig = bd.synth_case(_case("target_absent"))
    nf = sig["n"] // bd.HOP
    y = bd.render(sig, bd.oracle_choice(sig, nf))
    assert bd.target_absent_false_speech(y, sig)["false_speech_s"] == 0.0
    sig = bd.synth_case(_case("overlap"))
    orc = bd.oracle_choice(sig, nf)
    m = bd.selector_metrics(orc, orc, sig["tw"], sig["ow"], nf)
    assert (
        m["target_miss_rate"] == 0
        and m["wrong_speaker_rate"] == 0
        and m["pause_swap_count"] == 0
    )


def test_selector_error_localized_and_measured():
    sig = bd.synth_case(_case("pause_swap"))
    nf = sig["n"] // bd.HOP
    rng = np.random.default_rng(0)
    sel = bd.selector_model(sig, nf, None, rng, onset_delay=6, hold=0, p_false=1.0)
    m = bd.selector_metrics(sel, bd.oracle_choice(sig, nf), sig["tw"], sig["ow"], nf)
    assert m["pause_swap_rate"] == 1.0
    assert m["reacq_latency_ms"][0] == 60.0
    wb = bd.word_boundary_metrics(sig, sel, bd.render(sig, sel))
    assert all(w["onset_clip_ms"] >= 60 for w in wb)


def test_hold_reduces_tail_clip_but_costs_false_speech():
    sig = bd.synth_case(_case("overlap"))
    nf = sig["n"] // bd.HOP
    rng = np.random.default_rng(0)
    s0 = bd.selector_model(sig, nf, None, rng, onset_delay=0, hold=0)
    s5 = bd.selector_model(sig, nf, None, rng, onset_delay=0, hold=5)
    t0 = np.mean(
        [
            w["tail_clip_ms"]
            for w in bd.word_boundary_metrics(sig, s0, bd.render(sig, s0))
        ]
    )
    t5 = np.mean(
        [
            w["tail_clip_ms"]
            for w in bd.word_boundary_metrics(sig, s5, bd.render(sig, s5))
        ]
    )
    assert t5 <= t0
    f0 = bd.target_absent_false_speech(bd.render(sig, s0), sig)["false_speech_s"]
    f5 = bd.target_absent_false_speech(bd.render(sig, s5), sig)["false_speech_s"]
    assert f5 > f0


def test_streaming_fallback_holds_choice():
    c = np.array([0, 1, 1, 2, 2, 0], np.int8)
    ms = np.array([1, 1, 1, 20, 1, 1.0])
    fb, miss = bd.apply_fallback(c, ms)
    assert fb[3] == 1 and miss.sum() == 1
    s = bd.streaming_metrics(np.array([1, 11, 12, 1, 1, 13.0]))
    assert s["n_miss"] == 3 and s["max_miss_burst_frames"] == 2


def test_cluster_bootstrap_resamples_groups_and_is_seeded():
    by = {"a": [1.0, 1.0], "b": [3.0, 3.0], "c": [5.0]}
    r1, r2 = bd.cluster_bootstrap(by, 200, 1), bd.cluster_bootstrap(by, 200, 1)
    assert r1 == r2 and r1[1] <= r1[0] <= r1[2] and r1[1] < r1[2]


def test_ablation_oracle_separator_beats_full_and_preservation_reported():
    r = bd.eval_case(_case("overlap"), bd.default_knobs())
    a = r["ablation_si_sdr_db"]
    assert a["oracle_separator"] > a["full"]
    assert np.isfinite(r["preservation_si_sdr_db"])


def test_run_resume_cap_and_heldout_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEARING_RESULTS_DIR", str(tmp_path))
    assert bd.main(["run", "--split", "heldout", "--max-cases", "1"]) == 2
    assert bd.main(["run", "--max-cases", "3", "--seed", "5"]) == 0
    od = tmp_path / "overnight_prep/T-248/dev-seed5"
    man = json.loads((od / "manifest.json").read_text())
    assert man["computed"] == 3 and man["real_inputs_used"] == 0
    assert bd.main(["run", "--max-cases", "3", "--seed", "5"]) == 0
    man = json.loads((od / "manifest.json").read_text())
    assert man["computed"] == 0 and man["resumed_skipped"] == 3
    # torn checkpoint is recomputed, not trusted
    victim = next((od / "cases").glob("*.json"))
    victim.write_text("{")
    assert bd.main(["run", "--max-cases", "3", "--seed", "5"]) == 0
    assert json.loads((od / "manifest.json").read_text())["computed"] == 1
    # time cap stops deterministically
    assert bd.main(["run", "--max-cases", "5", "--seed", "6", "--time-cap-s", "0"]) == 0
    assert (
        json.loads(
            (tmp_path / "overnight_prep/T-248/dev-seed6/manifest.json").read_text()
        )["status"]
        == "stopped_cap"
    )
