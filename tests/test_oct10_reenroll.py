"""T-368: the held-out scene list, SIR remix, batch rows, log parsing and level matching (no weights, no audio files)."""

import json

import numpy as np

from eval import oct10_reenroll as ore


def test_scene_list_both_roles_every_sir_and_noise(tmp_path):
    scenes = ore.scene_list(tmp_path)
    assert len(scenes) == 8
    assert {(t, o) for t, o, *_ in scenes} == {("aryan", "joseph"), ("joseph", "aryan")}
    assert {d.name for *_, d in scenes} == {"holdout-sir0-none", "holdout-sir0-cafe", "holdout-sir-6-none", "holdout-sir-6-cafe"}
    assert all(d.parent.name == t for t, *_, d in scenes)
    assert all(d.name == ore.scene_name(s, n) and n in d.name for _, _, s, n, d in scenes)


def test_remix_puts_other_louder_by_sir_and_keeps_peak():
    rng = np.random.default_rng(0)
    st = {k: (0.1 * rng.standard_normal(16000)).astype(np.float32) for k in ("target", "other", "noise")}
    st["mix"] = st["target"] + st["other"] + st["noise"]
    out = ore.remix(st, -6.0)
    ratio = np.sqrt(np.mean(out["other"] ** 2) / np.mean(out["target"] ** 2)) / np.sqrt(np.mean(st["other"] ** 2) / np.mean(st["target"] ** 2))
    assert abs(20 * np.log10(ratio) - 6.0) < 1e-3
    assert np.allclose(out["mix"], out["target"] + out["other"] + out["noise"], atol=1e-6)
    assert np.max(np.abs(out["mix"])) <= ore.PEAK + 1e-6
    same = ore.remix({k: v * 0.1 for k, v in st.items()}, 0.0)
    assert np.allclose(same["other"], st["other"] * 0.1)


def test_batch_has_every_condition_and_scene_with_the_t284_command(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_FIXTURES_DIR", str(tmp_path / "fx"))
    monkeypatch.setenv("HEARING_RESULTS_DIR", str(tmp_path / "res"))
    text = ore.batch()
    rows = [line for line in text.splitlines() if "demo_regress.py" in line]
    assert len(rows) == len(ore.CONDITIONS) * 8
    assert all("--modes focus_hq --seconds 30 --no-free --no-baseline --profile team --isolation standard" in r for r in rows)
    assert sum("HEARING_GATED_FLOOR_DB=20" in r for r in rows) == 4 * 8
    head = [r for r in rows if "HEARING_TEAM_HEAD" in r]
    assert len(head) == 8 and all(f"{tmp_path / 'res' / ore.HEAD_FILE}" in r and "profiles-v2" in r for r in head)
    assert sum(f"HEARING_PROFILES_DIR={tmp_path / 'fx' / 'profiles'} " in r for r in rows) == 16
    assert sum(f"HEARING_PROFILES_DIR={tmp_path / 'fx' / ore.MIXED_DIR} " in r for r in rows) == 16


def test_batch_filters_labels_and_sirs_scene_major(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_FIXTURES_DIR", str(tmp_path / "fx"))
    text = ore.batch(labels=["mixed", "new+head+floor20"], sirs=[-6.0])
    rows = [line.split()[2:5] for line in text.splitlines() if line.startswith('echo "ROW ')]
    assert len(rows) == 2 * 4 and text.count("demo_regress.py") == 8
    assert {r[0] for r in rows} == {"mixed", "new+head+floor20"}
    assert all("sir-6" in r[2] for r in rows)
    assert [r[2] for r in rows[:4]] == ["holdout-sir-6-cafe"] * 4
    assert [r[0] for r in rows[:2]] == ["new+head+floor20", "mixed"] and rows[0][1] == "aryan:joseph"


def test_merge_logs_later_row_wins(tmp_path):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text("ROW new aryan:joseph s1 load 1 1 1\nwrote demo_regress_20261010-170000.json\nEXIT 0\n"
                 "ROW old aryan:joseph s1 load 1 1 1\nEXIT 1\n")
    b.write_text("ROW old aryan:joseph s1 load 2 2 2\nwrote demo_regress_20261010-190000.json\nEXIT 0\n")
    rows = {r["label"]: r for r in ore.merge_logs([a, b])}
    assert rows["new"]["run"] == "demo_regress_20261010-170000"
    assert rows["old"]["run"] == "demo_regress_20261010-190000" and rows["old"]["load"] == "2 / 2 / 2"


def test_parse_log_takes_label_load_run_and_exit():
    log = (
        "ROW new+floor20 joseph:aryan holdout-sir-6-cafe load 3.10 4.20 5.30\n"
        "noise\nwrote /x/eval/results/demo_regress_20261010-170000.json\nEXIT 0\n"
        "ROW old aryan:joseph holdout-sir0-none load 2.00 2.10 2.20\nEXIT 2\n"
    )
    rows = ore.parse_log(log)
    assert rows[0] == {"label": "new+floor20", "target": "joseph", "other": "aryan", "scene": "holdout-sir-6-cafe",
                       "load": "3.10 / 4.20 / 5.30", "run": "demo_regress_20261010-170000", "exit": 0}
    assert rows[1]["run"] is None and rows[1]["exit"] == 2


def test_telemetry_columns(tmp_path):
    ev = [{"dir": "out", "event": {"type": "telemetry", "tier": "audio-only"}},
          {"dir": "out", "event": {"type": "telemetry", "tier": "enrolled", "tse": {"confidence": 0},
                                   "tiers": {"tiers": [{"name": "audio-only", "health": []}, {"name": "enrolled", "health": [
                                       {"candidate": "enrolled", "windows": 10, "overruns": 0, "late": 1, "compute_p95_ms": 250.0}]}]}}},
          {"dir": "out", "event": {"type": "telemetry", "tier": "enrolled", "tse": {"confidence": 0.7}}},
          {"dir": "in", "event": {"type": "telemetry", "tier": "enrolled"}}]
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    c = ore.telemetry_columns(p)
    assert c == {"enrolled_share": 66.7, "gated": "1/2", "misses": "1 / 10", "p95": 250.0}


def test_level_match_hits_reference_and_limits_peak():
    x = np.zeros(16000, np.float32)
    x[:8000] = 0.01 * np.sin(np.arange(8000) / 5)
    y = ore.level_match(x, -26.0)
    a = y[np.abs(y) > 1e-3]
    assert abs(20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2))) + 26.0) < 0.5
    assert np.max(np.abs(ore.level_match(x, 0.0))) <= 0.89 + 1e-6


def test_totals_only_over_scenes_every_condition_measured():
    def row(label, target, scene, od, share=80.0):
        return {"label": label, "target": target, "scene": scene, "od_s": od, "enrolled_share": share, "drop": 0.0,
                "leak": -10.0, "sdri_pre": 5.0, "target_down_10dB_s": 1.0, "misses": "1 / 58"}
    labels = [c[0] for c in ore.CONDITIONS]
    rows = [row(lab, "aryan", "s1", 1.0) for lab in labels] + [row(lab, "joseph", "s1", 2.0) for lab in labels]
    rows += [row(lab, "aryan", "s2", 9.0, 40.0 if lab == "old" else 80.0) for lab in labels]  # s2: old NOT MEASURED
    tot = ore.totals(rows)
    assert set(tot) == set(labels) and tot["old"]["rows"] == 2 and tot["old"]["wrong_s"] == 3.0
    assert tot["new"]["scenes"] == ["aryan/s1", "joseph/s1"] and tot["new"]["misses"] == 2
    assert ore.totals(rows, "joseph")["mixed"]["wrong_s"] == 2.0
