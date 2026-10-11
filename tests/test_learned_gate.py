"""T-407: learned continuous-gain gate (hearing.tse.learned_gate) and its offline fit (eval.learned_gate_fit)."""

import json

import numpy as np
import pytest

from eval import learned_gate_fit as fit
from hearing.tse import learned_gate as lg


def test_gain_policy_is_continuous_and_cost_weighted():
    pol = {"leak_weight": 1.0, "lo": 0.2, "hi": 0.6, "hyst": 0.1, "leak_close": 0.6}
    assert lg.gain_db(0.9, 0.0, False, pol) == 0.0
    assert lg.gain_db(0.1, 0.0, False, pol) == lg.FLOOR_DB
    mid = lg.gain_db(0.4, 0.0, False, pol)
    assert lg.FLOOR_DB < mid < 0.0
    assert lg.gain_db(0.4, 0.0, True, pol) > mid  # hysteresis: an open gate stays more open
    assert lg.gain_db(0.99, 0.7, True, pol) == lg.FLOOR_DB  # p_leak closes it whatever p_target says
    assert lg.gain_db(0.9, 0.5, False, pol) < lg.gain_db(0.9, 0.0, False, pol)


def test_window_row_history():
    a = lg.window_row(-20, -3, -30, 0.5, 0.1, 0.0, 0.4, 0.3, 0.2)
    assert a.shape == (len(lg.FEATURES),) and a[-1] == 0.0
    b = lg.window_row(-20, -40, -2, 0.0, 0.1, 0.1, 0.5, 0.3, 0.2, prev=a)
    assert b[-1] == 1.0 and b[lg.FEATURES.index("prev_vad_t")] == 1.0
    assert b[lg.FEATURES.index("vad_t")] == 0.0 and b[lg.FEATURES.index("vad_p")] == 1.0


def test_head_roundtrip_and_streaming_gate(tmp_path):
    n = len(lg.FEATURES)
    w_t = np.zeros(n + 1)
    w_t[lg.FEATURES.index("t_db")] = 4.0  # louder target slot -> target active
    head = lg.LearnedHead(np.zeros(n), np.ones(n), w_t, np.full(n + 1, 0.0) - np.eye(n + 1)[-1] * 5)
    path = tmp_path / "head.json"
    path.write_text(json.dumps(head.to_dict()))
    loaded = lg.load_head(path)
    assert isinstance(loaded, lg.LearnedHead)
    gate = lg.LearnedGate(loaded)
    loud = gate.step(-20, -1, -30, 0.5, 0.1, 0.0, 0.4, 0.3, 0.2)
    gate.reset()
    quiet = gate.step(-20, -60, -1, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2)
    assert loud > quiet
    with pytest.raises(ValueError):
        lg.LearnedHead(np.zeros(3), np.ones(3), np.zeros(4), np.zeros(4))


def test_shipped_head_is_off():
    assert lg.HEAD is None  # not adopted (T-407 offline verdict): nothing to load without a JSON


def test_fit_logistic_separates():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 2))
    y = (x[:, 0] > 0).astype(float)
    beta = fit.fit_logistic(x, y, np.ones(len(y)))
    assert beta[0] > 1.0 and abs(beta[1]) < beta[0] / 3


def test_sequences_split_renders_and_scenes():
    split = np.array(["train"] * 6 + ["val"] * 3 + ["holdout"] * 4)
    kind = np.array(["a"] * 9 + ["x:s1", "x:s1", "y:s1", "y:s1"])
    seq = fit.sequences(split, kind)
    assert list(seq) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]


def _fake_npz(path, rng):
    """A small synthetic dump with the T-391 layout: slot 0 is loud when Aryan talks, slot 1 when Joseph does."""
    rows, labs, split, kind = [], [], [], []
    for sp, n in (("train", 120), ("val", 30)):
        for _ in range(n * 3):
            a, j = rng.random() < 0.6, rng.random() < 0.5
            s0 = -2 + rng.normal() if a else -40 + rng.normal()
            s1 = -2 + rng.normal() if j else -40 + rng.normal()
            rows.append([-20, s0, s1, 0.5 if a else 0.0, 0.1, 0.1, 0.5 if j else 0.0, 0.3, 0.3])
            labs.append([-3 if a else -100, -3 if j else -100, -100])
            split.append(sp)
            kind.append("aj")
    for kd in ("aryan:sir0", "joseph:sir0"):
        for _ in range(12):
            rows.append([-20, -3, -3, 0.5, 0.1, 0.1, 0.5, 0.3, 0.3])
            labs.append([-3, -3, -100])
            split.append("holdout")
            kind.append(kd)
    np.savez(path, features=np.asarray(rows, np.float32), labels=np.asarray(labs, np.float32),
             split=np.asarray(split), kind=np.asarray(kind), ckpt=np.asarray("fake.pt"))


def test_fit_main_end_to_end(tmp_path, monkeypatch):
    npz, out = tmp_path / "g.npz", tmp_path / "out.json"
    _fake_npz(npz, np.random.default_rng(1))
    monkeypatch.setattr(fit, "bootstrap", lambda *a, **k: {"muted_s": [0, 0], "wrong_pct": [0, 0], "leak_db": None})
    fit.main(["--features", str(npz), "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["fit"]["target_acc"] > 0.9
    assert {r["target"] for r in res["rows"]} == {"aryan", "joseph"}
    assert all(set(r) >= {"arm4", "arm5", "learned"} for r in res["rows"])
    assert lg.LearnedHead.from_dict(res["head"]).policy == res["val"]["policy"]
