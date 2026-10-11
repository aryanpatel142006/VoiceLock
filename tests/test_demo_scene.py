"""T-097: demo-shaped scenes (``make_fixtures --demo``): schedule, manifest structure, scorer compatibility."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from eval import demo_scene, session_nr
from eval.fixtures import video as fxvideo
from scripts import session_score as ss


def test_schedule_has_the_demo_shape():
    lengths = {"target": [40, 45, 38], "other": [42, 44]}  # ~1.7 s sentences, in frames
    placements, sections = demo_scene.schedule(lengths)
    assert [s["name"] for s in sections] == ["target_alone", "alternate", "overlap", "target_alone_2"]
    fps = demo_scene.FPS

    def who_in(s):
        a, b = s["start_s"] * fps, s["end_s"] * fps
        return [p["who"] for p in placements if a <= p["start"] < b]

    first, alt, both, last = (who_in(s) for s in sections)
    assert set(first) == {"target"} and len(first) >= 2
    assert set(last) == {"target"}
    assert alt[0] == "target" and all(x != y for x, y in itertools.pairwise(alt)) and "other" in alt
    assert {"target", "other"} <= set(both)
    assert sections[0]["end_s"] <= 5 + demo_scene.SLACK_S
    for who in ("target", "other"):  # one talker's sentences never overlap and keep a pause
        mine = sorted((p for p in placements if p["who"] == who), key=lambda p: p["start"])
        assert all(b["start"] - a["end"] >= demo_scene.GAP_FRAMES for a, b in itertools.pairwise(mine))
    assert max(p["end"] for p in placements) <= (demo_scene.DURATION_S + demo_scene.SLACK_S) * fps


def test_variants_named_as_the_demo_picker_expects():
    names = [v.name for v in demo_scene.VARIANTS]
    assert names == ["demo-mf-easy", "demo-mf-hard", "demo-mm", "demo-ff"]
    assert demo_scene.DEFAULT_CLIP == "demo-mf-easy"
    from hearing.replay import DEFAULT_CLIP

    assert DEFAULT_CLIP == demo_scene.DEFAULT_CLIP
    snr = {v.name: v.snr_db for v in demo_scene.VARIANTS}
    assert snr["demo-mf-easy"] == 10 and snr["demo-mf-hard"] == 5


@pytest.mark.skipif(not fxvideo.available(), reason="ffmpeg missing")
def test_build_writes_stems_timeline_and_scorers_read_it(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    spf = demo_scene.SPF

    def fake(_corpus, speaker):
        tone = {"s1": 0.1, "s4": 0.2}[speaker]
        out = []
        for i, n in enumerate((10, 12)):
            audio = (tone * rng.standard_normal(n * spf)).astype(np.float32)
            frames = np.full((n, 36, 32, 3), 40 * i, np.uint8)
            out.append(demo_scene.Sentence(f"u{i}", audio, frames, f"{speaker} sentence {i}"))
        return out

    monkeypatch.setattr(demo_scene, "load_sentences", fake)
    monkeypatch.setattr(demo_scene, "SECTIONS", (("target_alone", 1.0, "target"), ("alternate", 2.5, "alternate"),
                                                 ("overlap", 3.5, "both"), ("target_alone_2", 4.0, "target")))
    monkeypatch.setattr(demo_scene, "DURATION_S", 4.0)
    variant = demo_scene.Variant("demo-test", "s1", "s4", 10.0)
    path = demo_scene.build(variant, corpus=None, out_root=tmp_path)
    m = json.loads(path.read_text())

    assert (path.parent / m["clip"]).is_file()
    assert set(m["stems"]) == {"mix", "target", "other", "noise"}
    assert 4.0 <= m["duration_s"] <= 4.0 + demo_scene.SLACK_S + 0.05 and m["snr_db"] == 10.0
    assert [f["side"] for f in m["faces"]] == ["left", "right"]
    target = next(f for f in m["faces"] if f["track_id"] == m["target"]["track_id"])
    assert target["role"] == "target" and target["side"] == "left" and target["speaking"]
    other = next(f for f in m["faces"] if f["role"] == "other")
    assert other["speaking"]
    assert {s["who"] for s in m["sentences"]} == {"target", "other"}
    mine = [x["text"] for x in sorted(m["sentences"], key=lambda x: x["start_s"]) if x["who"] == "target"]
    assert m["transcript"] == " ".join(mine)  # T-117: the target's words, for caption WER

    stems = ss.resolve_stems(path)  # scripts/session_score.py
    assert len(stems.mix) == round(m["duration_s"] * demo_scene.SR)
    np.testing.assert_allclose(stems.mix, stems.target + stems.other + stems.noise, atol=2e-4)
    assert np.max(np.abs(stems.mix)) <= demo_scene.PEAK + 1e-3
    # target silent outside its timeline, babble at the requested SNR
    on = np.zeros(len(stems.target), bool)
    for s, e in target["speaking"]:
        on[int(s * demo_scene.SR): int(e * demo_scene.SR)] = True
    assert np.max(np.abs(stems.target[~on])) < 1e-4
    snr = 20 * np.log10(np.sqrt(np.mean(stems.target[on] ** 2)) / np.sqrt(np.mean(stems.noise ** 2)))
    assert snr == pytest.approx(10.0, abs=0.3)

    masks = session_nr.masks(m, n_frames=200)  # scripts/session_score_nr.py
    assert masks["target_only"].any() and masks["other_only"].any() and masks["both"].any()

    from hearing.replay import list_clips

    assert any(c["clip"] == "demo-test/clip.mp4" and c["target"] == "left" for c in list_clips(tmp_path))
