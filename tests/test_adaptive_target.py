import numpy as np

from eval.adaptive_target import AdaptiveTargetCandidate, solo_activity


class _Emb:
    def embed(self, a):
        return np.asarray(a, np.float64)


def _cand():
    c = AdaptiveTargetCandidate(separator=type("S", (), {"device": "cpu"})(), embedder=_Emb())
    rng = np.random.default_rng(0)
    t = rng.normal(size=(6, 8)) * 0.1 + np.eye(8)[0]
    o = rng.normal(size=(6, 8)) * 0.1 + np.eye(8)[1]
    c._pieces = {"target": (1, t / np.linalg.norm(t, axis=1, keepdims=True), None),
                 "other": (2, o / np.linalg.norm(o, axis=1, keepdims=True), None)}
    return c


def _ready(c, margin=2.0):
    c.scoring, c.margin_score, c.selected_stream = "lda", margin, 0
    c.vetoed = c.held = c.gated = False


SOLO = {"any_s": 1.0, "both_s": 0.0, "overlap": False}


def test_solo_activity():
    assert solo_activity(SOLO)
    assert not solo_activity({"any_s": 0.0, "both_s": 0.0, "overlap": False})
    assert not solo_activity({"any_s": 1.0, "both_s": 0.1, "overlap": False})
    assert not solo_activity({"any_s": 1.0, "both_s": 0.5, "overlap": True})


def test_updates_only_when_confident_and_solo():
    c, x = _cand(), np.eye(8)[0] * 0.9 + np.eye(8)[2] * 0.1
    _ready(c)
    assert c.learn(x, SOLO) and c.updates == 1
    _ready(c, margin=0.1)  # below hold 0.4
    assert not c.learn(x, SOLO)
    _ready(c)
    assert not c.learn(x, {"any_s": 1.0, "both_s": 0.4, "overlap": True})
    for flag in ("vetoed", "held", "gated"):
        _ready(c)
        setattr(c, flag, True)
        assert not c.learn(x, SOLO)
    c.scoring = "cosine"
    c.vetoed = c.held = c.gated = False
    assert not c.learn(x, SOLO)
    assert c.updates == 1


def test_profile_pieces_untouched_and_lda_moves_and_reset():
    c, x = _cand(), np.eye(8)[0] * 0.5 + np.eye(8)[2] * 0.8
    before = c._pieces["target"][1].copy()
    w0 = c._closed_set()[0].copy()
    _ready(c)
    c.learn(x / np.linalg.norm(x), SOLO)
    assert np.array_equal(c._pieces["target"][1], before)
    assert not np.allclose(c._closed_set()[0], w0)
    c.reset()
    assert c.crops == [] and np.allclose(c._closed_set()[0], w0)


def test_crop_cap():
    c = _cand()
    for _ in range(20):
        _ready(c)
        c.learn(np.eye(8)[0], SOLO)
    assert len(c.crops) == c.max_crops
