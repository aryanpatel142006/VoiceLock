"""scripts/session_score.py on synthetic stems with known answers (no server, no weights)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import session_score as ss

SR = ss.SR


def _speechy(rng, n, on_s):
    """Noise bursts gated on/off like speech: (signal, active mask)."""
    x = rng.standard_normal(n).astype(np.float32) * 0.1
    env = np.zeros(n, np.float32)
    for a, b in on_s:
        env[int(a * SR):int(b * SR)] = 1.0
    return x * env, env > 0


def _stems(seed=0):
    rng = np.random.default_rng(seed)
    t, _ = _speechy(rng, 3 * SR, [(0.1, 1.4), (1.8, 2.8)])
    o, _ = _speechy(rng, 3 * SR, [(0.0, 0.9), (1.2, 2.2), (2.5, 3.0)])
    nz = (rng.standard_normal(3 * SR) * 0.02).astype(np.float32)
    return ss.Stems(mix=t + o + nz, target=t, other=o, noise=nz, fixture="synthetic")


def _loop(a, starts, n):
    out = np.zeros(n, np.float32)
    for s in starts:
        m = min(len(a), n - s)
        out[s:s + m] = a[:m]
    return out


def _ticks(delay_ms, n_s=12.0):
    return [{"t": round(i * 0.1, 2), "dir": "out",
             "event": {"type": "telemetry", "latency_ms": {"algorithmic": delay_ms, "resample": 0, "total": delay_ms + 25},
                       "tier": "gate"}} for i in range(int(n_s * 10))] + [
        {"t": 0.0, "dir": "in", "event": {"type": "control", "mode": "natural"}}]


# AAC-style drifting loop starts (the real demo MP4 drifts by up to ~50 ms per loop)
STARTS = [0, 3 * SR + 128, 6 * SR + 2304, 9 * SR + 2432]
N = 12 * SR


def _scene(gain_t=1.0, gain_o=0.1, delay=208):
    st = _stems()
    x = _loop(st.mix, STARTS, N)
    t, o = _loop(st.target, STARTS, N), _loop(st.other, STARTS, N)
    clean = gain_t * t + gain_o * o
    y = np.zeros(N, np.float32)
    y[delay:] = clean[:N - delay]
    return st, x, y, t, o, clean


def test_place_stems_finds_drifting_loops():
    st, x, *_ = _scene()
    placed, loops = ss.place_stems(x, st)
    assert [round(lp["start_s"] * SR) for lp in loops] == STARTS
    assert all(lp["fit_db"] > 60 for lp in loops)
    assert np.allclose(placed["target"], _loop(st.target, STARTS, N), atol=1e-5)


def test_known_si_sdr_leak_and_delay():
    st, x, y, t, o, _ = _scene(gain_o=0.1, delay=208)
    s = ss.score_arrays(x, y, st, _ticks(13.0), session="synthetic", dnsmos="none")
    assert s.delay_ms == pytest.approx(13.0)
    assert s.reported_total_ms == pytest.approx(38.0)
    # output = target + 0.1 * other: SI-SDR = 10 log10(|t|^2 / |0.1 o|^2) over the scored region
    expected = 10 * np.log10(np.sum(t**2) / np.sum((0.1 * o) ** 2))
    assert s.si_sdr_db == pytest.approx(expected, abs=0.3)
    assert s.si_sdri_db == pytest.approx(s.si_sdr_db - s.si_sdr_in_db, abs=0.02)
    assert s.leak_db == pytest.approx(-20.0, abs=0.5)
    assert s.true_dropout_s == 0.0
    assert s.other_dominant_s == 0.0
    assert s.mode == "natural"


def test_prior_is_refined_by_xcorr():
    st, x, y, *_ = _scene(delay=208 + 48)  # engine says 13 ms, the stream is 16 ms late
    s = ss.score_arrays(x, y, st, _ticks(13.0), dnsmos="none")
    assert s.delay_ms == pytest.approx(16.0)
    assert s.leak_db == pytest.approx(-20.0, abs=0.5)


def test_true_dropouts_vs_intended_ducking():
    st, x, y, t, _, _ = _scene(gain_o=0.0, delay=0)
    # mute 4.0-5.0 s of output: the target talks there (loop 2 starts at ~3.008 s -> 0.99-1.99 s of the source)
    y[4 * SR:5 * SR] = 0.0
    s = ss.score_arrays(x, y, st, _ticks(0.0), dnsmos="none")
    t_act = ss.vad(t)[4 * SR // ss.FRAME:5 * SR // ss.FRAME]
    assert s.true_dropout_s == pytest.approx(t_act.sum() * ss.FRAME / SR, abs=0.1)
    assert s.true_dropout_s > 0.5
    # the other talker alone is fully ducked: counted as ducking, never as a dropout
    assert s.ducking_s > 1.0
    assert s.leak_db < -60


def test_silent_output_has_no_si_sdr():
    st, x, *_ = _scene()
    s = ss.score_arrays(x, np.zeros(N, np.float32), st, _ticks(0.0), dnsmos="none")
    assert np.isnan(s.si_sdr_db) and s.as_dict()["si_sdr_db"] is None
    assert "SI-SDRi -" in s.line()


def test_regression_flags():
    from scripts.demo_regress import regressions

    before = {"scores": [{"mode": "natural", "si_sdri_db": 3.0, "true_dropout_s": 0.0}]}
    assert regressions([{"mode": "natural", "si_sdri_db": 2.5, "true_dropout_s": 0.4}], before) == []
    flagged = regressions([{"mode": "natural", "si_sdri_db": 1.5, "true_dropout_s": 0.8}], before)
    assert len(flagged) == 2


def test_baseline_is_the_median_of_previous_runs(tmp_path):
    import json

    from scripts.demo_regress import baseline, previous, regressions

    for i, drop in enumerate([1.2, 2.8, 2.7, 1.3, 2.4, 0.1]):  # the oldest (0.1) falls outside the 5-run window
        (tmp_path / f"demo_regress_2026100{9 - i}-000000.json").write_text(json.dumps(
            {"variant": "fast:focus_hq:12", "scores": [{"mode": "focus_hq", "si_sdri_db": 0.0, "true_dropout_s": drop}]}))
    (tmp_path / "demo_regress_20261010-000000.json").write_text(json.dumps({"variant": "full:x:24", "scores": []}))
    runs = previous(tmp_path, "fast:focus_hq:12", "20261011-000000")
    assert len(runs) == 5
    base = baseline(runs)
    b = base["scores"][0]
    assert (b["true_dropout_s"], b["true_dropout_s_spread"]) == (2.4, 1.6)
    # 2.78 s against one lucky 1.2 s run flagged before; within the median + measured spread it does not
    assert regressions([{"mode": "focus_hq", "si_sdri_db": 0.5, "true_dropout_s": 2.78}], base) == []
    assert len(regressions([{"mode": "focus_hq", "si_sdri_db": 0.5, "true_dropout_s": 4.1}], base)) == 1
    assert baseline([]) is None


def test_a_regression_must_reproduce_on_the_second_replay():
    from scripts.demo_regress import best_of, regressions

    base = {"scores": [{"mode": "focus_hq", "si_sdri_db": 0.5, "true_dropout_s": 1.0}]}
    first = {"mode": "focus_hq", "si_sdri_db": -1.0, "true_dropout_s": 3.0}
    assert len(regressions([first], base)) == 2
    assert regressions([best_of(first, {"mode": "focus_hq", "si_sdri_db": 0.2, "true_dropout_s": 1.1})], base) == []
    both_bad = best_of(first, {"mode": "focus_hq", "si_sdri_db": -0.8, "true_dropout_s": 2.5})
    assert len(regressions([both_bad], base)) == 2


def test_alignment_is_piecewise_across_tier_switches():
    """T-095: a gate (13 ms) -> enrolled (1.5 s) -> gate run is aligned per tier segment, so the
    score of a flapping ladder is the audio's, not an alignment artefact."""
    st, x, _, t, o, clean = _scene(gain_o=0.1)
    plan = [(0.0, 4.0, 13.0, "gate"), (4.0, 8.0, 1500.0, "enrolled"), (8.0, 12.0, 13.0, "gate")]
    y = np.zeros(N, np.float32)
    for a, b, d_ms, _tier in plan:  # output [a, b) plays input [a - d, b - d): replays, then skips 1.5 s
        a, b, d = int(a * SR), int(b * SR), int(d_ms * SR / 1000)
        y[a:b] = np.concatenate([np.zeros(2 * SR, np.float32), clean])[2 * SR + a - d:2 * SR + b - d]
    ticks = [{"t": round(i * 0.1, 2), "dir": "out", "event": {
        "type": "telemetry", "tier": tier, "tiers": {"active": tier, "delay_ms": d_ms}}}
        for i in range(120) for a, b, d_ms, tier in plan if a <= i * 0.1 < b]
    s = ss.score_arrays(x, y, st, ticks, dnsmos="none")
    assert [(d["tier"], d["measured_ms"]) for d in s.delays] == [
        ("gate", 13.0), ("enrolled", 1500.0), ("gate", 13.0)]
    assert all(d["corr"] > 2 * ss.MIN_CORR for d in s.delays)  # measured by xcorr, not the prior kept
    expected = 10 * np.log10(np.sum(t**2) / np.sum((0.1 * o) ** 2))
    assert s.si_sdr_db == pytest.approx(expected, abs=0.5)
    assert s.true_dropout_s == 0.0
    assert (s.first_isolated_s, s.tier_switches, s.tier_switches_after_isolated) == (4.0, 2, 1)  # T-130: enrolled -> gate
    # one global delay (the old failure mode this guards against) would score far worse
    one = ss.score_arrays(x, y, st, _ticks(13.0), dnsmos="none")
    assert one.si_sdr_db < s.si_sdr_db - 10


LEFT, RIGHT = [0.07, 0.24, 0.25, 0.63], [0.55, 0.24, 0.25, 0.63]  # demo-mf-easy's two faces


def _target_ticks(tracks):
    """Telemetry ticks with both faces; ``tracks`` = the session's target track per tick (0 = left)."""
    faces = [{"track_id": 0, "box": LEFT}, {"track_id": 2, "box": RIGHT}]
    return [{"t": i * 0.1, "faces": faces, "target": {"track_id": tid}} for i, tid in enumerate(tracks)]


def test_target_agreement_by_box_side():
    target = {"side": "left", "box": LEFT}
    assert ss.target_agreement(_target_ticks([0] * 10), target) == 1.0
    assert ss.target_agreement(_target_ticks([2] * 10), target) == 0.0
    # ticks without a target (searching) do not count; 3 of 4 targeted ticks are the left face
    assert ss.target_agreement(_target_ticks([None, None, 0, 0, 0, 2]), target) == 0.75
    assert ss.target_agreement(_target_ticks([None, None]), target) is None
    assert ss.target_agreement(_target_ticks([0]), {}) is None  # no manifest box: unknown, not wrong


def test_wrong_target_is_flagged_in_the_score_line_and_table():
    st, x, *_ = _scene()
    st.target_face = {"side": "left", "box": LEFT}
    rows = [{"t": tk["t"], "dir": "out", "event": dict(tk, type="telemetry")} for tk in _target_ticks([2] * 100)]
    s = ss.score_arrays(x, x, st, rows, dnsmos="none")
    assert s.target_agreement == 0.0
    assert "target agreement 0% WRONG TARGET" in s.line()
    assert "WRONG TARGET" in ss.table([s])
    s.target_agreement = 0.8
    assert "WRONG TARGET" not in s.line() and "target agreement 80%" in s.line()
    s.target_agreement = None
    assert "WRONG TARGET" not in s.line()


def test_regress_oracle_locks_the_manifest_side():
    from scripts.replay import OracleLock

    oracle = OracleLock({"target": {"side": "left", "box": LEFT}})
    msg = {"type": "telemetry", "faces": [{"track_id": 2, "box": RIGHT}, {"track_id": 5, "box": LEFT}]}
    assert oracle.on_message(msg) == {"type": "control", "target": {"lock_track_id": 5}}
    assert oracle.on_message(msg) is None  # still on screen: no re-lock


def test_enrolled_phase_scores_only_enrolled_ticks_with_shared_crop():
    st, x, y, *_ = _scene(gain_o=0.1, delay=208)
    rows = _ticks(13.0)
    for r in rows:
        if r["dir"] == "out":
            r["event"]["tier"] = "enrolled" if r["t"] >= 6.0 else "gate"
    full = ss.score_arrays(x, y, st, rows, session="s", dnsmos="none")
    enr = ss.score_arrays(x, y, st, rows, session="s", dnsmos="none", enrolled_phase=True)
    assert enr.seconds < full.seconds
    assert 4.0 < enr.seconds <= 6.0
    assert enr.delay_ms == pytest.approx(13.0)
    assert enr.si_sdri_db == pytest.approx(enr.si_sdr_db - enr.si_sdr_in_db, abs=0.02)


def test_wrong_reported_delay_falls_back_to_the_measured_lag_and_says_so():
    # T-108: the remote path reported the local fallback's 1500 ms; the stream was 1700 ms late
    st, x, y, *_ = _scene(delay=round(1.7 * SR))
    s = ss.score_arrays(x, y, st, _ticks(1500.0), dnsmos="none")
    assert s.delay_ms == pytest.approx(1700.0)
    assert s.alignment.startswith("measured:") and "reported 1500 ms" in s.alignment
    assert "ALIGNMENT measured" in s.line()
    assert s.leak_db == pytest.approx(-20.0, abs=0.5)


def test_correct_reported_delay_keeps_reported_alignment():
    st, x, y, *_ = _scene(delay=round(1.7 * SR))
    s = ss.score_arrays(x, y, st, _ticks(1700.0), dnsmos="none")
    assert s.delay_ms == pytest.approx(1700.0) and s.alignment == "reported"
    assert "ALIGNMENT" not in s.line()


def test_regress_folder_scores_only_subfolders_with_events(tmp_path):
    reg = tmp_path / "regress-20261006-120000"
    for name in ("s1", "s2", "empty"):
        (reg / name).mkdir(parents=True)
    (reg / "s1" / "events.jsonl").write_text("{}\n")
    (reg / "s2" / "events.jsonl").write_text("{}\n")
    (reg / "server.log").write_text("log")
    single = tmp_path / "solo"
    single.mkdir()
    (single / "events.jsonl").write_text("{}\n")
    assert ss.session_dirs([reg, single]) == [reg / "s1", reg / "s2", single]
    assert ss.main([str(reg / "empty")]) == 2


def test_missing_worktree_files_fall_back_to_the_main_checkout(tmp_path, monkeypatch):
    wt, main = tmp_path / "wt", tmp_path / "main"
    (main / "eval/fixtures/demo/demo-mf-easy").mkdir(parents=True)
    (main / "eval/fixtures/demo/demo-mf-easy/clip.mp4").write_bytes(b"")
    (wt / "eval").mkdir(parents=True)
    monkeypatch.setattr(ss, "ROOT", wt)
    monkeypatch.setattr(ss, "MAIN", main)
    rel = "eval/fixtures/demo/demo-mf-easy/clip.mp4"
    assert ss.main_fallback(wt / rel) == main / rel
    assert ss.main_fallback(wt / "eval/nope.wav") == wt / "eval/nope.wav"
    (wt / rel).parent.mkdir(parents=True)
    (wt / rel).write_bytes(b"")
    assert ss.main_fallback(wt / rel) == wt / rel  # the worktree's own copy wins
    with pytest.raises(FileNotFoundError, match="--clip"):
        ss.resolve_stems(wt / "eval/fixtures/demo/other/manifest.json")


def test_demo_regress_exit_codes(tmp_path, monkeypatch, capsys):
    from scripts import demo_regress as dr

    assert dr.main(["--bogus"]) == 2
    assert dr.main(["--clip", str(tmp_path / "missing.mp4")]) == 2
    assert "--clip" in capsys.readouterr().err
    clip = tmp_path / "c" / "clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"")

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(dr.ss, "resolve_stems", boom)
    assert dr.main(["--clip", str(clip.parent)]) == 2  # a crash is not a regression (1)
    assert "kaboom" in capsys.readouterr().err


def test_pre_agc_rides_out_wav_alignment_on_the_same_crop():
    """T-179 (Q-485): a y_pre that is y / 2 (a flat -6 dB AGC) scores the same SI-SDRi on out.wav's crop,
    its speech level sits 6 dB under out.wav's, and the AGC 'lifted' the pauses by +6 dB."""
    st, x, y, *_ = _scene(gain_o=0.1, delay=208)
    s = ss.score_arrays(x, y, st, _ticks(13.0), dnsmos="none", y_pre=y / 2)
    assert s.si_sdri_pre_agc_same_crop_db == pytest.approx(s.si_sdri_db, abs=0.01)
    assert s.speech_level_db - s.speech_level_pre_agc_db == pytest.approx(20 * np.log10(2), abs=0.01)
    assert s.pause_lift_db == pytest.approx(20 * np.log10(2), abs=0.01)
    assert s.speech_level_p90_db >= s.speech_level_db - 3.0
    assert ss.score_arrays(x, y, st, _ticks(13.0), dnsmos="none").si_sdri_pre_agc_same_crop_db is None
