"""T-340 knob sweep: provenance filter, scene shape, scoring and Pareto/best selection (no weights)."""
from __future__ import annotations

import numpy as np

from eval import knob_sweep as ks


def _item(speaker, source, aug, sha):
    return {"speaker": speaker, "source_file": source, "augmentation": aug, "source_sha256": sha,
            "path": f"{aug}/{speaker}/{source}"}


def test_provenance_keeps_only_profile_cuts_with_matching_hash():
    sha = {"aryan": "a" * 64, "joseph": "j" * 64}
    man = {"files": [_item("aryan", "profile-01.wav", "clean", "a" * 64),
                     _item("aryan", "fixture-01.wav", "clean", "f" * 64),
                     _item("joseph", "profile-01.wav", "opus16", "j" * 64),
                     _item("joseph", "profile-01.wav", "pitch_pm2", "j" * 64),  # changes the voice: not used
                     _item("joseph", "profile-01.wav", "clean", "x" * 64)]}  # hash mismatch
    keep, check = ks.provenance(man, sha)
    assert [(f["speaker"], f["augmentation"]) for f in keep] == [("aryan", "clean"), ("joseph", "opus16")]
    assert check["refused"]["fixture-01.wav:clean"] == 1
    assert all(f["source_file"] == "profile-01.wav" for f in keep)


def test_compose_levels_and_sir():
    rng = np.random.default_rng(0)
    pool = [rng.standard_normal(ks.SR * 2).astype(np.float32) * 0.1 for _ in range(4)]
    st = ks.compose(pool, pool, -10.0, None)
    n = round(ks.DUR_S * ks.SR)
    assert all(len(x) == n for x in st.values())
    # target alone in the first section, other only from the alternate section on
    assert np.abs(st["other"][: 4 * ks.SR]).max() == 0
    assert np.abs(st["target"][: 4 * ks.SR]).max() > 0
    ratio = ks._active_rms(st["other"]) / ks._active_rms(st["target"])
    assert abs(20 * np.log10(ratio) - 10.0) < 0.5  # SIR -10 dB: the other is 10 dB louder
    np.testing.assert_allclose(st["mix"], st["target"] + st["other"] + st["noise"], atol=1e-6)


def test_speech_chunks_cut_on_silence():
    x = np.zeros(ks.SR * 6, np.float32)
    x[ks.SR: ks.SR * 2] = 0.1 * np.sin(np.arange(ks.SR) / 3.0)
    x[ks.SR * 3: ks.SR * 6] = 0.1 * np.sin(np.arange(ks.SR * 3) / 3.0)
    parts = ks.speech_chunks(x)
    assert [round(len(p) / ks.SR, 1) for p in parts] == [1.0, 2.0, 1.0]


def test_score_target_only_output():
    rng = np.random.default_rng(1)
    t = rng.standard_normal(ks.SR * 4) * 0.1
    o = rng.standard_normal(ks.SR * 4) * 0.1
    nz = np.zeros_like(t)
    sc = {"mix": t + o, "A": t, "B": o, "noise": nz}
    clean = ks.score(t.copy(), sc)
    assert clean["wrong_s"] == 0 and clean["atten_s"] == 0 and clean["leak_db"] < -40
    quiet = ks.score(0.05 * t + o, sc)  # target 26 dB down, other untouched
    assert quiet["wrong_s"] > 3 and quiet["atten_s"] > 3


def test_floor_env_reaches_composition(monkeypatch):
    mix = np.ones(ks.SR * 2, np.float32)
    sc = {"mix": mix}
    wins = [{"k": k, "i": i, "y": np.zeros(ks.SR, np.float32), "conf": 0.0, "gated": True}
            for k, i in enumerate(range(0, ks.SR + 1, ks.SR // 2))]
    for k in ks.ENVS:
        monkeypatch.delenv(k, raising=False)
    unset = ks.compose_output(sc, wins)
    monkeypatch.setenv("HEARING_GATED_FLOOR_DB", "20")
    floor20 = ks.compose_output(sc, wins)
    np.testing.assert_allclose(unset[ks.SR // 2], 10 ** (-6 / 20), rtol=1e-5)
    np.testing.assert_allclose(floor20[ks.SR // 2], 0.1, rtol=1e-5)


def test_pareto_and_pick_best():
    def m(w, a, leak=-5.0):
        return {"wrong_s": w, "leak_db": leak, "atten_s": a, "dropout_s": 0.0}

    base = ks.BASELINE
    res = {base: {"s1": m(5, 1), "s2": m(5, 1)},
           "good": {"s1": m(2, 1.5), "s2": m(2, 1.8)},
           "cheap": {"s1": m(4, 1), "s2": m(4, 1)},
           "costly": {"s1": m(0, 4), "s2": m(0, 1)},  # attenuation +3 s on s1: fails the bar
           "worse": {"s1": m(6, 2), "s2": m(6, 2)}}
    tot = ks.totals(res)
    front = ks.pareto(tot)
    assert "worse" not in front and base not in front and set(front) == {"costly", "good", "cheap"}
    assert ks.pick_best(res) == ["good", "cheap"]


def test_setting_name_and_grid():
    assert ks.setting_name(0.0, None, None, None) == ks.BASELINE == "c0_funset_uunset_zero"
    assert ks.setting_name(-0.3, 20.0, 10.0, "stream") == "c-0.3_f20_u10_stream"
    assert len(ks.settings()) == 4 * 3 * 3 * 2
