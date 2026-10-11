"""T-341: the forensic scorer bookkeeping on a synthetic cache (no weights, no audio)."""

import numpy as np

from eval import team_head_forensics as thf


def _cache(rng, sep=0.6):
    """Two speakers on orthogonal axes; each scene window holds one target and one other stream."""
    axis = {"aryan": np.eye(192)[0], "joseph": np.eye(192)[1]}

    def emb(name, n):
        x = axis[name] * sep + rng.normal(scale=0.05, size=(n, 192))
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    c = {}
    for t in thf.NAMES:
        o = "joseph" if t == "aryan" else "aryan"
        c[f"train/{t}"] = emb(t, 20)
        c[f"pilot_test/{t}"] = emb(t, 10)
        c[f"enroll/{t}"] = emb(t, 1)
        for scene in (*thf.TEST_SCENES, thf.DEV_SCENE):
            k = f"scene/{t}/{scene}"
            lab = rng.integers(0, 2, size=12)
            e = np.stack([emb(t, 12), emb(o, 12)], axis=1)
            e[lab == 1] = e[lab == 1][:, ::-1]
            c[k + "/present/emb"], c[k + "/present/target_stream"] = e, lab
            c[k + "/present/target_db"] = np.full(12, -20.0)
            c[k + "/present/other_db"] = np.full(12, -25.0)
            c[k + "/absent/emb"] = np.stack([emb(o, 12), emb(o, 12)], axis=1)
            c[k + "/unknown/emb"] = -np.stack([emb(t, 12), emb(t, 12)], axis=1)  # far from the target
    return c


def test_wilson_bounds():
    lo, hi = thf.wilson(97, 97)
    assert 0.95 < lo < 1.0 and hi == 1.0
    assert np.isnan(thf.wilson(0, 0)[0])


def test_evaluate_separable_speakers_rank_and_reject():
    c = _cache(np.random.default_rng(0))
    enroll = {t: c[f"enroll/{t}"][0] for t in thf.NAMES}
    res = thf.evaluate(c, {"raw": lambda e, t: e @ enroll[t]})["raw"]
    assert res["top1"]["k"] == res["top1"]["n"] == 48
    assert res["fa_absent"]["rate"] <= 0.2 and res["fa_unknown"]["rate"] == 0.0
    assert res["margin_mean"] > 0.3
    assert set(res["per_scene"]) == {f"{t}/{s}" for t in thf.NAMES for s in thf.TEST_SCENES}


def test_pilot_eval_counts_two_way_errors():
    c = _cache(np.random.default_rng(1))
    proto = {t: c[f"train/{t}"][:10].mean(0) for t in thf.NAMES}
    out = thf.pilot_eval(c, {"raw cos(proto10)": lambda e, t: e @ proto[t]})
    assert out["raw cos(proto10)"]["n"] == 20 and out["raw cos(proto10)"]["errors"] == 0
