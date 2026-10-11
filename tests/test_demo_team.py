"""T-126: team demo scenes from solo takes, and the take recorder with fake devices."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from eval import demo_team
from eval.fixtures import video as fxvideo
from hearing.audio import io as aio

SR = 16000
ROOT = Path(__file__).resolve().parents[1]
needs_ffmpeg = pytest.mark.skipif(not fxvideo.available(), reason="ffmpeg (imageio-ffmpeg) missing")


def _rt():
    spec = importlib.util.spec_from_file_location("record_take", ROOT / "scripts" / "record_take.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def talk(f0: float, n_sent: int = 12, sent_s: float = 1.6, pause_s: float = 0.6, seed: int = 0) -> np.ndarray:
    """Harmonic 'sentences' with pauses and a faint noise floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sent_s * SR)) / SR
    voiced = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * np.hanning(len(t))
    parts = [np.zeros(int(0.5 * SR))]
    for _ in range(n_sent):
        parts += [0.2 * voiced, np.zeros(int(pause_s * SR))]
    x = np.concatenate(parts).astype(np.float32)
    return x + 1e-4 * rng.standard_normal(len(x)).astype(np.float32)


def test_speech_runs_splits_at_pauses_and_long_runs():
    x = talk(120, n_sent=5)
    runs = demo_team.speech_runs(demo_team.frame_db(x))
    assert len(runs) == 5
    assert all(30 <= b - a <= 45 for a, b in runs)  # ~1.6 s of 40 ms frames
    long = np.concatenate([np.full(250, -20.0), np.full(200, -90.0)])
    long[120] = -60.0  # the quietest frame inside a 10 s run
    pieces = demo_team.speech_runs(long)
    assert all(b - a <= demo_team.MAX_PIECE_FRAMES for a, b in pieces) and len(pieces) >= 3


def test_variants_add_same_only_for_one_shared_bucket():
    assert demo_team.variants(("male_adult", "female_adult")) == ["demo-team-easy", "demo-team-hard"]
    assert demo_team.variants((None, None))[-1] == "demo-team-hard"
    assert demo_team.variants(("male_adult", "male_adult"))[-1] == "demo-team-same"


def test_profile_and_fixture_must_be_different_files(tmp_path):
    d = tmp_path / "team" / "ana"
    d.mkdir(parents=True)
    x = talk(200)
    aio.save(d / "fixture-01.wav", x, SR)
    aio.save(d / "profile-01.wav", x, SR)
    with pytest.raises(ValueError, match="same file"):
        demo_team.discover(tmp_path / "team")
    aio.save(d / "profile-01.wav", talk(200, seed=1), SR)
    (t,) = demo_team.discover(tmp_path / "team")
    assert [x["stem"] for x in t.fixture] == ["fixture-01"] and [p.name for p in t.profile] == ["profile-01.wav"]


def _m4a(wav: Path, out: Path) -> Path:
    """AAC in an MP4 container, as QuickTime / Voice Memos save it."""
    fxvideo._run([fxvideo.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                  "-c:a", "aac", "-b:a", "128k", str(out)])
    return out


@needs_ffmpeg
def test_quicktime_takes_mov_and_m4a(tmp_path):
    """RECORDING_SCRIPT.md names (profile.wav, fixture.mov, ...) and QuickTime formats are takes;
    noisy.wav and team/conversation/ (Take 4) are not."""
    d = tmp_path / "team" / "ana"
    d.mkdir(parents=True)
    aio.save(tmp_path / "a.wav", talk(210, n_sent=4, seed=3), SR)
    frames = np.zeros((25 * 9, 48, 64, 3), np.uint8)
    frames[:, 10:30, 20:44] = 200
    fxvideo.write_video(d / "fixture.mov", frames, 25, audio_wav=tmp_path / "a.wav")
    aio.save(tmp_path / "b.wav", talk(210, n_sent=3, seed=4), SR)
    _m4a(tmp_path / "b.wav", d / "fixture-02.m4a")
    aio.save(tmp_path / "p.wav", talk(210, n_sent=3, seed=5), SR)
    _m4a(tmp_path / "p.wav", d / "profile.m4a")
    aio.save(d / "noisy.wav", talk(210, n_sent=2, seed=6), SR)
    conv = tmp_path / "team" / "conversation"
    conv.mkdir()
    _m4a(tmp_path / "a.wav", conv / "fixture-ana_close.m4a")  # even a fixture-like name in there is skipped

    (t,) = demo_team.discover(tmp_path / "team")
    takes = {x["stem"]: x for x in t.fixture}
    assert set(takes) == {"fixture", "fixture-02"} and [p.name for p in t.profile] == ["profile.m4a"]
    assert takes["fixture"]["audio"] == takes["fixture"]["video"] == d / "fixture.mov"  # .mov: video + its audio
    assert takes["fixture-02"]["audio"].suffix == ".m4a" and takes["fixture-02"]["video"] is None  # audio only
    sents = demo_team.teammate_sentences(t)
    assert len(sents) == 7 and len({s.frames.shape[1:] for s in sents}) == 1  # one face size per teammate
    by_take = {u: [s for s in sents if s.utt.startswith(u + "#")] for u in ("fixture", "fixture-02")}
    assert [len(v) for v in by_take.values()] == [4, 3]
    assert by_take["fixture"][0].frames.max() > 150 and by_take["fixture-02"][0].frames.max() == 96  # webcam, card
    assert all(len(s.audio) == len(s.frames) * demo_team.SPF for s in sents)

    import shutil
    shutil.copyfile(d / "profile.m4a", d / "fixture-03.m4a")  # the same QuickTime file as a profile take
    with pytest.raises(ValueError, match="same file"):
        demo_team.discover(tmp_path / "team")


@needs_ffmpeg
def test_long_take_keeps_a_speech_budget_and_decodes_only_that_much(tmp_path, monkeypatch):
    d = tmp_path / "team" / "ana"
    d.mkdir(parents=True)
    aio.save(tmp_path / "long.wav", talk(150, n_sent=60, seed=2), SR)  # ~2.3 min, 60 pieces
    fxvideo.write_video(d / "fixture.mov", np.zeros((25 * 4, 32, 32, 3), np.uint8), 25,
                        audio_wav=tmp_path / "long.wav")
    decoded = []
    read = fxvideo.read_video
    monkeypatch.setattr(fxvideo, "read_video", lambda *a, **k: decoded.append(k.get("duration_s")) or read(*a, **k))
    (t,) = demo_team.discover(tmp_path / "team")
    sents = demo_team.teammate_sentences(t, budget_s=12.0)
    speech = sum(len(s.audio) for s in sents) / SR
    assert 9.0 <= speech <= 13.0 and len(sents) < 10
    assert decoded and decoded[0] < 20.0  # not the whole 2+ minutes


@needs_ffmpeg
def test_build_from_recorded_takes(tmp_path, monkeypatch):
    team = tmp_path / "team"
    for name, f0, bucket in (("ana", 210.0, "female_adult"), ("ben", 110.0, "female_adult")):
        d = team / name
        d.mkdir(parents=True)
        x = talk(f0, seed=int(f0))
        aio.save(d / "fixture-01.wav", x, SR)
        aio.save(d / "profile-01.wav", talk(f0, n_sent=3, seed=7), SR)
        (d / "meta.json").write_text(json.dumps({"bucket": bucket}))
    # ana also has a webcam take (video + its own audio track, no wav)
    frames = np.zeros((25 * 8, 48, 64, 3), np.uint8)
    frames[:, 10:30, 20:44] = 200
    aio.save(tmp_path / "a.wav", talk(210, n_sent=3, seed=3), SR)
    fxvideo.write_video(team / "ana" / "fixture-02.mp4", frames, 25, audio_wav=tmp_path / "a.wav")
    aio.save(team / "venue_noise.wav", 0.01 * np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32), SR)

    seen = []
    every = demo_team.variants
    monkeypatch.setattr(demo_team, "variants", lambda b: seen.extend(every(b)) or seen[-1:])  # encode one clip (fast)
    paths = demo_team.build(tmp_path / "demo", team, target="ben")
    assert seen == ["demo-team-easy", "demo-team-hard", "demo-team-same"]  # both female_adult
    assert [p.parent.name for p in paths] == ["demo-team-same"]
    m = json.loads(paths[0].read_text())
    assert m["standin"] is False and m["kind"] == "team_scene" and m["snr_db"] == 10.0
    assert m["target"]["actor"] == "team-ben" and m["source"]["noise"]["kind"] == "venue"
    assert m["source"]["team"]["ana"]["fixture_takes"] == ["fixture-01", "fixture-02"]
    assert m["source"]["team"]["ana"]["profile_takes"] == ["profile-01.wav"]
    assert [s["name"] for s in m["sections"]] == ["target_alone", "alternate", "overlap", "target_alone_2"]
    stems = {k: aio.load(paths[0].parent / v, SR)[0] for k, v in m["stems"].items()}
    assert np.allclose(stems["mix"], stems["target"] + stems["other"] + stems["noise"], atol=2e-4)
    assert len(stems["mix"]) / SR == pytest.approx(m["duration_s"], abs=0.05)
    assert (paths[0].parent / "clip.mp4").exists()
    assert fxvideo.probe(paths[0].parent / "clip.mp4").width == 640


@pytest.fixture
def rt():
    return _rt()


class FakeMic:
    def __init__(self, take: np.ndarray, check: np.ndarray | None = None):
        self.queue = [x for x in (check, take) if x is not None]
        self.asked: list[float] = []

    def record(self, seconds: float) -> np.ndarray:
        self.asked.append(seconds)
        x = self.queue.pop(0)
        return np.resize(x, round(seconds * SR))


class FakeCamera:
    started = stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        return [(i / 30, np.full((72, 128, 3), i % 255, np.uint8)) for i in range(30 * 6)]  # 30 fps webcam


@needs_ffmpeg
def test_record_take_writes_numbered_wav_and_mp4(rt, tmp_path):
    lines = []
    mic = FakeMic(talk(150, n_sent=2), check=talk(150, n_sent=1))
    out = rt.record("ana", "fixture", 6.0, tmp_path, mic, FakeCamera(), bucket="female_adult",
                    sleep=lambda s: None, say=lines.append)
    assert mic.asked == [rt.CHECK_S, 6.0]
    assert out["wav"] == tmp_path / "team" / "ana" / "fixture-01.wav" and out["mp4"].name == "fixture-01.mp4"
    assert aio.load(out["wav"], SR)[0].shape == (6 * SR,)
    info = fxvideo.probe(out["mp4"])
    assert (info.width, info.height) == (128, 72) and info.fps == pytest.approx(25)
    assert json.loads((tmp_path / "team" / "ana" / "meta.json").read_text()) == {"bucket": "female_adult"}
    assert any("3..." in s for s in lines) and out["problem"] == "too little speech"  # 6 s < 20 s
    again = rt.record("ana", "profile", 2.0, tmp_path, FakeMic(talk(150, n_sent=1)), None, check=False,
                      sleep=lambda s: None, say=lines.append)
    assert again["wav"].name == "profile-01.wav" and again["mp4"] is None


def test_record_take_level_check_refuses_silence_and_clipping(rt, tmp_path):
    quiet = np.zeros(SR, np.float32)
    with pytest.raises(rt.LevelError, match="too quiet"):
        rt.record("ana", "fixture", 2.0, tmp_path, FakeMic(quiet, check=quiet), None, sleep=lambda s: None,
                  say=lambda s: None)
    loud = np.clip(talk(150, n_sent=1) * 20, -1, 1)
    with pytest.raises(rt.LevelError, match="clipping"):
        rt.record("ana", "fixture", 2.0, tmp_path, FakeMic(loud, check=loud), None, sleep=lambda s: None,
                  say=lambda s: None)
    assert not (tmp_path / "team" / "ana").exists()


def test_record_take_venue_noise(rt, tmp_path):
    noise = 0.02 * np.random.default_rng(1).standard_normal(SR).astype(np.float32)
    out = rt.record(None, "venue", 3.0, tmp_path, FakeMic(noise), FakeCamera(), sleep=lambda s: None,
                    say=lambda s: None)
    assert out["wav"] == tmp_path / "team" / "venue_noise.wav" and out["mp4"] is None and out["problem"] is None
