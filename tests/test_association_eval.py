"""Scene parsing, rendering and scoring of the association eval (T-061), no weights."""

from __future__ import annotations

import numpy as np
import pytest

from eval import association_eval as ae


def _talker(seed: int) -> ae.Talker:
    rng = np.random.default_rng(seed)
    speech = rng.integers(0, 255, (10, 36, 48, 3), dtype=np.uint8)
    idle = rng.integers(0, 255, (5, 36, 48, 3), dtype=np.uint8)
    audio = (ae.LEVEL * np.sqrt(2) * np.sin(np.arange(10 * ae.SPF) * (0.05 + 0.01 * seed))).astype(np.float32)
    return ae.Talker(f"s{seed}", speech, audio, idle, (0.25, 0.25, 0.5, 0.5))


SCENE = {
    "name": "t",
    "duration_s": 0.4,
    "persons": {"A": [[0, 0.2, "speak"], [0.2, 0.4, "occluded"]], "B": [[0, 0.2, "mouth"], [0.2, 0.4, "absent"]]},
    "target": [[0, "A"], [0.2, "B"]],
}


def test_committed_scenes_parse():
    scenes = ae.load_scenes()
    names = {s.name for s in scenes}
    assert len(scenes) == 10
    assert set(ae.FAST_SCENES) <= names
    for s in scenes:
        assert s.n_frames == round(s.duration_s * ae.FPS)


@pytest.mark.parametrize(
    "bad",
    [
        {**SCENE, "persons": {"A": [[0, 0.2, "speak"]], "B": SCENE["persons"]["B"]}},  # A does not tile
        {**SCENE, "persons": {"A": [[0, 0.4, "dance"]], "B": SCENE["persons"]["B"]}},  # unknown kind
        {**SCENE, "duration_s": 0.41},  # not a frame multiple
        {**SCENE, "target": [[0.1, "A"]]},  # target must start at 0
    ],
)
def test_bad_scenes_raise(bad):
    with pytest.raises(ValueError):
        ae.parse_scene(bad)


def test_render_audio_and_frames():
    scene = ae.parse_scene(SCENE)
    talkers = {"s1": _talker(1), "s4": _talker(4)}
    r = ae.render(scene, "ab", talkers, snr_db=100.0, noise="pink", seed=0)
    n = scene.n_frames * ae.SPF
    assert r.n == n and len(r.stems["A"]) == n
    half = n // 2
    assert ae.active(r.stems["A"][:half]) and ae.active(r.stems["A"][half:])  # speak, then occluded (heard)
    assert not ae.active(r.stems["B"])  # mouth (muted), then absent
    f = r.frame(0)
    assert f.shape == (360, 640, 3) and f.dtype == np.uint8
    covered = r.frame(scene.n_frames - 1)
    assert (covered[:, :320] == 30).mean() > 0.2  # A's face (left) is under the occluder
    assert len(np.unique(covered[:, 320:].reshape(-1, 3), axis=0)) == 1  # B left: background only
    assert list(r.target_at()[[0, n - 1]]) == [0, 1]
    assert ae.return_events(scene) == []  # A's occlusion ends with the scene; B is target then
    assert ae.target_segments(scene) == [(0.0, 0.2, "A"), (0.2, 0.4, "B")]


def test_return_events_only_for_current_target():
    scene = ae.parse_scene(
        {
            "name": "r",
            "duration_s": 0.8,
            "persons": {"A": [[0, 0.2, "speak"], [0.2, 0.4, "away"], [0.4, 0.8, "speak"]],
                        "B": [[0, 0.4, "absent"], [0.4, 0.8, "speak"]]},
            "target": [[0, "A"]],
        }
    )
    assert ae.return_events(scene) == [0.4]


def test_window_correct_and_ls_shares():
    rng = np.random.default_rng(0)
    t = (ae.LEVEL * rng.standard_normal(16000)).astype(np.float32)
    o = (ae.LEVEL * rng.standard_normal(16000)).astype(np.float32)
    assert ae.window_correct(t + 0.1 * o, None, t, o)
    assert not ae.window_correct(o + 0.1 * t, None, t, o)
    silent = np.zeros_like(o)
    streams = np.stack([0.01 * o, t])
    assert ae.window_correct(streams[1], streams, t, silent)  # other silent: the oracle stream
    assert not ae.window_correct(streams[0], streams, t, silent)
    e_t, e_o = ae.ls_shares(2 * t + 0.5 * o, [t, o])
    assert e_t == pytest.approx(4 * np.sum(t.astype(np.float64) ** 2), rel=1e-3)
    assert e_o == pytest.approx(0.25 * np.sum(o.astype(np.float64) ** 2), rel=1e-3)


def test_oracle_lock_holds_then_relocks():
    from hearing.types import FaceTrack

    def tr(i, x, ts):
        return FaceTrack(i, (x, 0.3, 0.2, 0.4), None, 0.0, ts)

    lock = ae.OracleLock()
    assert lock.update([tr(1, 0.1, 0.0), tr(2, 0.6, 0.0)], "left", 0.0) == 1
    assert lock.update([tr(2, 0.6, 0.5)], "left", 0.5) == 1  # held
    assert lock.update([tr(2, 0.6, 1.1)], "left", 1.1) is None  # held 1 s, then dropped
    assert lock.update([tr(3, 0.1, 1.2), tr(2, 0.6, 1.2)], "left", 1.2) == 3  # re-locked


def test_score_on_a_scripted_run():
    """A run that extracts A exactly while A is target and B after the switch scores perfectly."""
    scene = ae.parse_scene(
        {
            "name": "s",
            "duration_s": 4.0,
            "persons": {"A": [[0, 4, "speak"]], "B": [[0, 4, "speak"]]},
            "target": [[0, "A"], [2, "B"]],
        }
    )
    r = ae.render(scene, "ab", {"s1": _talker(1), "s4": _talker(4)}, snr_db=100.0, noise="pink", seed=0)
    W, H = 16000, 8000
    windows, out = [], np.zeros(r.n, np.float32)
    for k in range((r.n - W) // H + 1):
        a = k * H
        who = "A" if a + W // 2 < 2 * ae.SR else "B"
        y = r.stems[who][a : a + W]
        windows.append(ae.WindowRec(k, a, "ok", "", y, None, 0, [0.9, 0.1]))
    who = r.target_at()
    out = np.where(who == 0, r.stems["A"], r.stems["B"])
    run = ae.TseRun(out[: r.n - W], W, W, H, windows, [], 0.0)
    p = ae.score(r, run, {})
    assert p["acc"] == 1.0 and p["switches"] == 0
    assert p["wrong_s"] == 0.0 and p["unfocused_s"] == 0.0
    assert p["lock_s"] == [1.0, 1.0]
    s = ae.summarize([p])
    assert s["go"] and s["acc_pooled"] == 1.0


def test_pairs_pick_the_talkers_per_variant():
    assert ae.speakers("ab") == ("s1", "s4") and ae.speakers("ba") == ("s4", "s1")  # T-061 default
    assert ae.speakers("ab", "mm") == ("s1", "s2") and ae.speakers("ba", "ff") == ("s7", "s4")
    scene = ae.parse_scene(SCENE)
    talkers = {"s1": _talker(1), "s2": _talker(2)}
    r = ae.render(scene, "ba", talkers, snr_db=100.0, noise="pink", seed=0, pair="mm")
    assert [t.speaker for t in r.talkers] == ["s2", "s1"] and r.a_side == "right"


def test_partial_results_are_written(tmp_path):
    import argparse
    import json

    out = tmp_path / "a.json"
    ae.write_results(out, argparse.Namespace(pair="ff"), {}, 0.0, {"final": []}, partial=True)
    d = json.loads(out.read_text())
    assert d["partial"] and d["pair"]["speakers"] == ["s4", "s7"] and d["summary"] == {}


def test_av_path_feeds_lip_crops_one_frame_late(monkeypatch):
    """--candidate av_mossformer2 runs the AV stage on the vision pass's crops (mock backend, no weights)."""
    from hearing.tse import av_mossformer as avm

    monkeypatch.setattr(avm.models, "load", lambda name: avm.MockBackend())
    scene = ae.parse_scene({"name": "av", "duration_s": 2.4, "persons": {"A": [[0, 2.4, "speak"]], "B": [[0, 2.4, "mouth"]]},
                            "target": [[0, "A"]]})
    r = ae.render(scene, "ab", {"s1": _talker(1), "s4": _talker(4)}, snr_db=100.0, noise="pink", seed=0)
    crop = np.full((88, 88), 128, np.uint8)
    states = []
    for i in range(scene.n_frames):
        ts = (i + 1) / ae.FPS - 0.005
        tracks = [ae.FaceTrack(1, (0.1, 0.3, 0.2, 0.4), None, 0.9, ts), ae.FaceTrack(2, (0.7, 0.3, 0.2, 0.4), None, 0.1, ts)]
        states.append(ae.FrameState(tracks, 1, {1: "left", 2: "right"}, [(ts, 1, crop, 0.9), (ts, 2, crop, 0.1)]))
    run = ae.run_av(r, states, window_s=1.0, hop_s=0.5)
    ok = [w for w in run.windows if w.status == "ok"]
    assert ok and all(w.y is not None and w.streams is None for w in ok)
    assert run.clips == [] and len(run.out) == r.n - run.delay
    p = ae.score(r, run, {1: "left", 2: "right"})
    assert p["n_scored"] == len(ok) and p["purity"] is None


def test_out_accepts_a_directory(tmp_path):
    d = tmp_path / "res"
    d.mkdir()
    out = ae.out_path(d)
    assert out.parent == d and out.name.startswith("association_") and out.suffix == ".json"
    new = ae.out_path(tmp_path / "new-dir", "ff")  # no .json: a directory, created
    assert new.parent.is_dir() and new.name.startswith("association_ff_")
    assert ae.out_path(tmp_path / "x" / "run.json") == tmp_path / "x" / "run.json"


def test_out_that_is_a_non_json_file_is_rejected_up_front(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(SystemExit) as e:
        ae.main(["--out", str(f)])
    assert e.value.code == 2
