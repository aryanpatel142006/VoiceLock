import json

import numpy as np
import pytest

from eval import isolation_diagnostic as d
from eval import isolation_live_path as lp


def _args(**kw):
    a = lp.parse(["--dry-run"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _rep(case):
    t, o = d.fake_stems(8.0)
    mix = (t + o).astype(np.float32)
    return lp.condition_report(lambda: d.FakeRunner(case, t, o, 1.0), t, o, mix, _args(split="all")), t, o, mix


def test_real_mixer_class_is_invoked_and_proxy_is_labelled():
    rep, *_ = _rep("ok")
    st = rep["stages"]
    assert st["real_mixer"]["real"] and not st["real_mixer"]["proxy"]
    assert st["real_mixer"]["class"] == "hearing.fusion.mixer.Mixer"
    assert st["proxy_mixer"]["proxy"] and not st["proxy_mixer"]["real"]
    assert not st["oracle_candidate"]["deployable"] and st["cf_floor24_mixer"]["counterfactual"]
    assert st["live_wrapper"]["status"] == "not_run"


def test_real_mixer_uses_confidence_floor():
    n = 16000
    est = np.zeros(n, np.float32)
    mix = np.full(n, 0.1, np.float32)  # residual = mix; target silent
    lo, info_lo = lp.run_real_mixer(est, mix, np.full(n, 0.0, np.float32))
    hi, info_hi = lp.run_real_mixer(est, mix, np.full(n, 1.0, np.float32))
    assert abs(info_lo["residual_floor_db_mean"] - 6.0) < 1e-3 and abs(info_hi["residual_floor_db_mean"] - 40.0) < 1e-3
    assert np.abs(hi).max() < np.abs(lo).max() / 10
    cf, _ = lp.run_real_mixer(est, mix, None, max_duck_db=24.0, use_conf=False)
    assert abs(float(np.abs(cf[-1])) - 0.1 * 10 ** (-24 / 20)) < 1e-4


@pytest.mark.parametrize(("case", "attr"), [("separator_missing_target", "model_target_loss"),
                                            ("wrong_selector", "wrong_selection"),
                                            ("missed_deadline", "scheduler_failure"), ("ok", "none")])
def test_failure_attribution(case, attr):
    rep, *_ = _rep(case)
    assert rep["failure_attribution"] == attr
    assert rep["counters"]["window_s"] == 2.0 and rep["counters"]["hop_s"] == 1.0


def test_labels_absolute_vs_improvement_and_target_absent_leak():
    rep, t, o, _ = _rep("ok")
    s = rep["stages"]["actual_selected"]
    assert s["sisdr_absolute_db"] is not None and s["sisdri_db"] is not None
    assert abs(s["sisdri_db"] - (s["sisdr_absolute_db"] - s["raw_sisdr_absolute_db"])) < 1e-6
    assert rep["stages"]["raw_mix"]["sisdri_db"] == pytest.approx(0.0)
    t2 = t.copy()
    t2[4 * 16000:] = 0  # target absent in the second half
    mix2 = (t2 + o).astype(np.float32)
    r2 = lp.condition_report(lambda: d.FakeRunner("ok", t2, o, 1.0), t2, o, mix2, _args(split="all"))
    assert r2["stages"]["real_mixer"]["n_target_absent_windows"] > 0
    assert r2["stages"]["real_mixer"]["target_absent_leak_db"] is not None


def test_dry_run_writes_manifest_with_gaps(tmp_path, capsys):
    out = tmp_path / "m.json"
    assert lp.main(["--dry-run", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["fake"] and len(data["results"]) == 5
    assert data["gaps"]["word_check"]["status"] == "missing_transcripts"
    assert "not_exercised" in data["gaps"]["strict_pause_gate_session_hooks"]
    assert "FAKE" in capsys.readouterr().out


def test_missing_inputs_blocks(capsys):
    assert lp.main(["--scene", "/nonexistent", "--profiles", "/nonexistent"]) == 2
