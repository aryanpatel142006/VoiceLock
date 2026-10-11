"""judge_check (T-100): per-backend Focus HQ bars on the locked regress, latency-test invocation, SKIP."""
from __future__ import annotations

import json
from types import SimpleNamespace

from hearing.cli import build_parser
from scripts import judge_check as jc


def _data(focus: dict, *, locked: bool = True, regressions: list | None = None) -> dict:
    base = {"si_sdri_db": 0.1, "leak_db": -8.0, "true_dropout_s": 0.0, "first_isolated_s": None,
            "delay_ms": 13.0, "target_agreement": 1.0}
    return {"target_locked": locked, "regressions": regressions or [], "free_selection": None,
            "scores": [{**base, "mode": "natural"}, {**base, "mode": "focus_hq", **focus}]}


def _focus(lines):
    return next(line for line in lines if line[1].startswith("Focus HQ"))


MAC_PASS = {"si_sdri_db": 3.71, "leak_db": -19.7, "first_isolated_s": 7.3, "target_agreement": 1.0}


def test_mac_bar_passes_the_locked_baseline_and_says_which_bar():
    verdict, _, detail = _focus(jc.regress_rows(_data(MAC_PASS), "mac", "r.json"))
    assert verdict == "PASS"
    assert "Mac ladder bar: SI-SDRi >= +3 dB, leak <= -12 dB, Isolated <= 8 s" in detail
    assert "locked to the manifest target" in detail


def test_gpu_bar_is_plan_8_db():
    verdict, _, detail = _focus(jc.regress_rows(_data(MAC_PASS), "gpu", "r.json"))
    assert verdict == "FAIL" and "GPU backend bar: SI-SDRi >= +8 dB" in detail
    assert _focus(jc.regress_rows(_data({**MAC_PASS, "si_sdri_db": 9.0}), "gpu", "r.json"))[0] == "PASS"


def test_mac_bar_fails_on_leak_late_isolation_and_wrong_target():
    for bad, word in (({"leak_db": -4.4}, "leak"), ({"first_isolated_s": 9.5}, "Isolated"),
                      ({"first_isolated_s": None}, "never"), ({"target_agreement": 0.38}, "WRONG TARGET")):
        verdict, _, detail = _focus(jc.regress_rows(_data({**MAC_PASS, **bad}), "mac", "r.json"))
        assert verdict == "FAIL" and word in detail, bad


def test_regression_fails_both_rows():
    lines = jc.regress_rows(_data(MAC_PASS, regressions=["focus_hq: SI-SDRi +3.7 -> +0.2 dB"]), "mac", "r.json")
    assert [v for v, _, _ in lines] == ["FAIL", "FAIL"]


def test_latency_command_parses_with_the_real_cli():
    args = build_parser().parse_args(jc.LATENCY_CMD[2:])
    assert args.command == "latency-test" and args.json


def test_is_wired():
    assert jc.is_wired("External Headphones")
    assert jc.is_wired("USB Audio DAC")
    for name in ("MacBook Pro Speakers", "Aryan's AirPods Pro", "LG Display Audio", "", None):
        assert not jc.is_wired(name), name


DEMO = {"input": "MacBook Pro Microphone", "output": "AirPods", "bt_round_trip_max_ms": 300.0}
NAMES = ["MacBook Pro Microphone", "MacBook Pro Speakers", "Aryan's AirPods Pro 3", "External Headphones"]


def _latency_run(delays, seen=None):
    def run(cmd, **k):
        if seen is not None:
            seen["cmd"] = cmd
        return SimpleNamespace(stdout=json.dumps({"round_trip_ms": sorted(delays)[len(delays) // 2], "delays_ms": delays}),
                               stderr="", returncode=0)
    return run


def test_demo_devices_come_from_the_team_preset():
    from hearing.server import team

    assert team.demo_devices() == DEMO
    assert team.demo_devices({"input_device": "X", "output_device": "Y", "bt_round_trip_max_ms": 250}) == {
        "input": "X", "output": "Y", "bt_round_trip_max_ms": 250.0}


def test_latency_runs_on_the_preset_devices_on_the_bluetooth_path():
    """T-143: AirPods no longer SKIP: latency-test runs with --in/--out the preset devices, 300/350 ms bars."""
    seen = {}
    verdict, detail = jc.check_latency(names=lambda: NAMES, demo=DEMO, run=_latency_run([240, 250, 260, 255, 270], seen))
    assert verdict == "PASS" and "Bluetooth path" in detail and "Aryan's AirPods Pro 3" in detail
    cmd = seen["cmd"][1:]
    assert cmd[:len(jc.LATENCY_CMD)] == jc.LATENCY_CMD and "--mode" not in cmd
    assert cmd[cmd.index("--in") + 1] == "MacBook Pro Microphone" and cmd[cmd.index("--out") + 1] == "Aryan's AirPods Pro 3"
    args = build_parser().parse_args(cmd[2:])
    assert args.command == "latency-test" and args.input == "MacBook Pro Microphone"


def test_latency_bluetooth_fails_on_the_median_or_the_p95():
    verdict, detail = jc.check_latency(names=lambda: NAMES, demo=DEMO, run=_latency_run([310, 320, 330, 305, 315]))
    assert verdict == "FAIL" and "Bluetooth path" in detail and "median" in detail
    verdict, detail = jc.check_latency(names=lambda: NAMES, demo=DEMO,
                                       run=_latency_run([250, 250, 250, 250, 250, 250, 250, 250, 250, 420]))
    assert verdict == "FAIL" and "p95" in detail


def test_latency_wired_override_keeps_the_100_ms_bar():
    verdict, detail = jc.check_latency(names=lambda: NAMES, demo={**DEMO, "output": "External Headphones"},
                                       run=_latency_run([40, 42, 44, 41, 43]))
    assert verdict == "PASS" and "wired path" in detail


def test_latency_fails_without_the_demo_devices_and_never_runs_the_test():
    def run(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("latency-test ran without the demo devices")

    verdict, detail = jc.check_latency(names=lambda: NAMES[:2], demo=DEMO, run=run)
    assert verdict == "FAIL" and "AirPods" in detail and "not connected" in detail
    verdict, detail = jc.check_latency(names=lambda: NAMES[1:], demo=DEMO, run=run)
    assert verdict == "FAIL" and "MacBook Pro Microphone" in detail
    verdict, detail = jc.check_latency(names=lambda: NAMES, demo={**DEMO, "output": "Speakers"}, run=run)
    assert verdict == "FAIL" and "speakers" in detail


def _devs(**kw):
    d = {"client": "judge", "input": "MacBook Pro Microphone", "output": "Aryan's AirPods Pro 3", "missing": []}
    return lambda url: {**d, **kw}


def _rows(**kw):
    return {crit: (verdict, detail) for verdict, crit, detail in jc.check_devices("http://x", fetch=_devs(**kw), demo=DEMO)}


MIC = "Demo mic in use (MacBook Pro Microphone)"
HFP = "Mic is not a Bluetooth (HFP) mic"
SPK = "Output is not the laptop speakers"
OUT = "Output is the demo output (AirPods)"


def test_devices_pass_on_the_builtin_mic_and_airpods():  # fake-label case 1
    rows = _rows()
    assert [rows[c][0] for c in (MIC, HFP, SPK, OUT)] == ["PASS"] * 4
    assert "MacBook Pro Microphone" in rows[MIC][1] and "AirPods" in rows[OUT][1]
    assert rows["macOS Mic Mode is Standard (not Voice Isolation)"][0] == "SKIP"
    assert rows["AirPods settings (ANC on, Transparency/Conversation Awareness off)"][0] == "SKIP"


def test_devices_fail_on_the_airpods_mic_hfp():  # case 2
    rows = _rows(input="Aryan's AirPods Pro 3")
    assert rows[MIC][0] == "FAIL" and rows[HFP][0] == "FAIL" and "HFP" in rows[HFP][1]


def test_devices_fail_on_a_bluetooth_headset_mic():  # case 3
    rows = _rows(input="Bluetooth Headset")
    assert rows[HFP][0] == "FAIL"


def test_devices_fail_on_an_external_mic():  # case 4
    rows = _rows(input="RØDE VideoMic GO II")
    assert rows[MIC][0] == "FAIL" and "RØDE" in rows[MIC][1] and rows[HFP][0] == "PASS"


def test_devices_fail_on_the_builtin_speakers():  # case 5
    rows = _rows(output="Default - MacBook Pro Speakers")
    assert rows[SPK][0] == "FAIL" and "speakers" in rows[SPK][1]
    assert rows[OUT][0] == "FAIL"


def test_devices_fail_on_wired_headphones_as_output():  # case 6
    rows = _rows(output="External Headphones")
    assert rows[SPK][0] == "PASS" and rows[OUT][0] == "FAIL" and "External Headphones" in rows[OUT][1]


def test_devices_fail_on_a_missing_saved_mic_and_output():  # case 7
    rows = _rows(missing=["mic", "output"])
    assert rows[MIC][0] == "FAIL" and "NOT FOUND" in rows[MIC][1]
    assert rows[OUT][0] == "FAIL" and "NOT FOUND" in rows[OUT][1]


def test_devices_pass_on_a_default_prefixed_airpods_label_and_env_override(monkeypatch):  # case 8
    rows = _rows(input="Default - MacBook Pro Microphone (Built-in)", output="Default - AirPods Pro")
    assert [rows[c][0] for c in (MIC, HFP, SPK, OUT)] == ["PASS"] * 4
    monkeypatch.setenv("HEARING_EXPECT_MIC", "RODE")
    verdicts = {c: v for v, c, _ in jc.check_devices("http://x", fetch=_devs(input="RØDE VideoMic GO II"))}
    assert verdicts["Demo mic in use (RODE)"] == "PASS"


def test_devices_without_a_page_fail_only_when_required():
    def down(url):
        raise OSError("refused")
    assert jc.check_devices("http://x", fetch=down, demo=DEMO, required=True)[0][0] == "FAIL"
    assert jc.check_devices("http://x", fetch=down, demo=DEMO)[0][0] == "SKIP"
    assert jc.check_devices("http://x", fetch=lambda url: None, demo=DEMO)[0][0] == "SKIP"


def test_pill_changes_mirror_the_page_hysteresis():
    """T-135: judge-pill.js PillHysteresis in Python: isolated instant, other states after 0.5 s."""
    from scripts.judge_check import pill_changes, pill_state

    assert pill_state({}) == "idle"
    assert pill_state({"acquisition": {"state": "locked", "detail": "known voice: aryan"}}) == "isolated"
    assert pill_state({"acquisition": {"state": "acquiring", "detail": "calibrating 1.0/2.0 s"}}) == "calibrating"
    # calibrated, the ladder not yet on an isolating tier: still "calibrating" (locking on), not "speak"
    done = {"acquisition": {"state": "locked-visual", "detail": "face selected; waiting for isolation"},
            "auto_calibrate": {"state": "done"}}
    assert pill_state(done) == "calibrating"
    seq = [(0.0, "look"), (0.1, "speak"), (0.3, "look"), (0.5, "speak"), (1.1, "speak"), (1.2, "calibrating"),
           (1.4, "speak"), (2.0, "isolated"), (2.1, "look")]
    changes, history = pill_changes(seq)
    assert history == ["look", "speak", "isolated"] and changes == 2


def test_jitter_underruns_simulates_the_worklet_buffer():
    from scripts.judge_check import jitter_underruns

    steady = [(i * 0.02, 0.02) for i in range(200)]
    assert jitter_underruns(steady, 120) == 0
    gap = steady[:100] + [(t + 0.2, d) for t, d in steady[100:]]  # one 200 ms stall
    assert jitter_underruns(gap, 120) == 1
    burst = steady[:100] + [(t + 0.08, d) for t, d in steady[100:]]  # 80 ms late: the 120 ms buffer absorbs it
    assert jitter_underruns(burst, 120) == 0
    assert jitter_underruns(burst, 40) == 1


def test_regress_row_takes_the_team_clip_and_profile(monkeypatch, tmp_path):
    """T-130: ``--team`` scores Natural/Focus HQ on demo-team-easy with the preset on; without it the
    command is unchanged (demo_regress's default clip, no profile)."""
    monkeypatch.setenv("HEARING_RESULTS_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(jc.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=2, stderr=""))
    jc.check_regress("mac", no_baseline=True)
    jc.check_regress("mac", no_baseline=True, clip=tmp_path / "demo-team-easy/clip.mp4", profile="team")
    assert "--clip" not in calls[0] and "--profile" not in calls[0]
    assert calls[1][-4:] == ["--clip", str(tmp_path / "demo-team-easy/clip.mp4"), "--profile", "team"]
