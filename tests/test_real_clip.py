"""Real-room clips (T-090): ingest → manifest, no-reference session score, replay of clip dirs."""
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eval import real_clip, session_nr
from eval.fixtures import video
from hearing.audio.io import float_to_pcm16, save
from hearing.replay import Clip, clip_manifest, list_clips

SR = 16000


def phone_clip(path: Path, seconds: float = 3.0, fps: int = 30, size=(320, 240)) -> Path:
    """Two marker faces (left, right) at a phone-ish rate/aspect; left talks 0-1 s, right 1-2 s, both 2-3 s."""
    w, h = size
    n = round(seconds * fps)
    frames = np.full((n, h, w, 3), 128, dtype=np.uint8)
    faces = {"left": (40, 110), "right": (200, 270)}
    for x0, x1 in faces.values():
        frames[:, 60:160, x0:x1] = (0, 255, 0)
    t = np.arange(round(seconds * SR)) / SR
    left = (t < 1) | (t >= 2)
    right = t >= 1
    for i in range(n):
        ts = i / fps
        for side, (x0, x1) in faces.items():
            talking = (ts < 1 or ts >= 2) if side == "left" else ts >= 1
            if talking and i % 4 < 2:
                frames[i, 120:140, x0 + 15:x1 - 15] = 0  # open mouth
    audio = 0.05 * (left * np.sin(2 * np.pi * 220 * t) + right * np.sin(2 * np.pi * 330 * t))
    save(path.with_suffix(".wav"), audio.astype(np.float32), SR)
    return video.write_video(path, frames, fps, audio_wav=path.with_suffix(".wav"))


def test_segments_bridge_short_pauses_and_drop_blips():
    active = np.zeros(100, dtype=bool)
    active[10:30] = True
    active[32:50] = True  # 2-frame pause: bridged
    active[70:72] = True  # 2-frame blip: dropped
    assert real_clip.segments(active, rate=25) == [[0.4, 2.0]]


def test_normalise_level_matches_target_without_clipping():
    t = np.arange(SR) / SR
    x = (0.01 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    y, info = real_clip.normalise_level(x, -20.0)
    assert real_clip.speech_level_db(y) == pytest.approx(-20.0, abs=0.2)
    assert info["gain_db"] > 0
    loud = np.r_[x, np.float32(0.9)]  # a click: the peak caps the gain
    z, _ = real_clip.normalise_level(loud, 0.0)
    assert np.abs(z).max() <= 10 ** (real_clip.PEAK_DB / 20) + 1e-6


def analysis_two_faces(n=100):
    frames = list(range(n))
    p_left = [1.0 if i < 40 else 0.0 for i in frames]
    p_right = [1.0 if 40 <= i < 80 else 0.0 for i in frames]
    return {"detector": "mock", "asd": "mock", "tracks": {
        1: {"frames": frames, "boxes": [[0.1, 0.2, 0.2, 0.4]] * n, "p": p_left},
        2: {"frames": frames, "boxes": [[0.6, 0.2, 0.2, 0.4]] * n, "p": p_right},
        3: {"frames": [5], "boxes": [[0.4, 0.4, 0.05, 0.05]], "p": [0.0]},  # glitch: < 20 % of frames
    }}


@pytest.mark.parametrize(("rule", "track"), [("left", 1), ("right", 2), ("track:2", 2), ("largest", 1)])
def test_manifest_picks_target_and_keeps_timelines(rule, track):
    m = real_clip.build_manifest("x", "x.mp4", analysis_two_faces(), 100, rule, {})
    assert [f["track_id"] for f in m["faces"]] == [1, 2]
    assert m["target"]["track_id"] == track
    assert m["target"]["side"] == {1: "left", 2: "right"}[track]
    assert m["faces"][0]["speaking"] == [[0.0, 1.6]]
    assert m["faces"][1]["speaking"] == [[1.6, 3.2]]
    assert set(m["boxes"]) == {"1", "2"}
    assert json.loads(real_clip.dump_manifest(m)) == m
    with pytest.raises(ValueError):
        real_clip.build_manifest("x", "x.mp4", analysis_two_faces(), 100, "track:9", {})


def test_ingest_normalises_a_phone_clip_into_a_replayable_dir(tmp_path):
    src = phone_clip(tmp_path / "phone.mp4")
    path = real_clip.ingest(src, "room", "left", tmp_path / "demo", detector="mock", asd="lipmotion")
    assert path == tmp_path / "demo" / "room" / "manifest.json"
    info = video.probe(path.parent / "clip.mp4")
    assert (info.width, info.height) == (640, 360)
    assert info.fps == pytest.approx(25)
    m = real_clip.load_manifest(path.parent)
    assert m["duration_s"] == pytest.approx(3.0, abs=0.1)
    assert len(m["faces"]) == 2
    assert m["target"]["side"] == "left"
    assert m["loudness"]["output_db"] == pytest.approx(real_clip.GRID_LEVEL_DB, abs=1.5)
    assert all(len(b) > 50 for b in m["boxes"].values())  # per-frame boxes
    # the replay path loads the directory and finds the manifest
    clip = Clip.load(path.parent)
    assert clip.duration == pytest.approx(3.0, abs=0.1)
    assert clip_manifest(path.parent)["target"]["track_id"] == m["target"]["track_id"]
    assert list_clips(tmp_path / "demo")[0]["clip"] == "room/clip.mp4"


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(float_to_pcm16(x))


def fake_session(tmp_path: Path, lag_s=0.1, leak=0.0, hole=None) -> Path:
    """Target speaks 0-2 s, the other 2-4 s, both 4-5 s; the 'engine' delays by lag_s and scales the other."""
    t = np.arange(5 * SR) / SR
    target = 0.1 * np.sin(2 * np.pi * 200 * t) * (t < 2) + 0.1 * np.sin(2 * np.pi * 200 * t) * (t >= 4)
    other = 0.1 * np.sin(2 * np.pi * 310 * t) * (t >= 2)
    x = (target + other).astype(np.float32)
    y = (target + leak * other).astype(np.float32)
    if hole:
        y[int(hole[0] * SR): int(hole[1] * SR)] = 0
    y = np.r_[np.zeros(int(lag_s * SR), np.float32), y]
    d = tmp_path / "session"
    d.mkdir()
    write_wav(d / "in.wav", x)
    write_wav(d / "out.wav", y)
    rows = [{"t": 1.0, "dir": "in", "event": {"type": "binary", "kind": 1, "seq": 0}}]
    for i, (state, tier) in enumerate([("searching", "gate"), ("locked-visual", "gate"), ("locked", "focus_hq")]):
        rows.append({"t": 1.5 + i, "dir": "out", "event": {"type": "telemetry", "tier": tier,
                                                             "acquisition": {"state": state}}})
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


MANIFEST = {"name": "fake", "duration_s": 5.0, "target": {"track_id": 1},
            "faces": [{"track_id": 1, "speaking": [[0.0, 2.0], [4.0, 5.0]]},
                      {"track_id": 2, "speaking": [[2.0, 5.0]]}]}


def test_session_score_measures_leak_preservation_lag_and_acquisition(tmp_path):
    r = session_nr.score(fake_session(tmp_path, leak=0.1), MANIFEST, dnsmos="mock")
    assert r["output_lag_ms"] == pytest.approx(100, abs=20)
    assert r["other_leak_db"] == pytest.approx(-20.0, abs=1.0)  # 0.1 amplitude = -20 dB
    assert r["target_preserved_db"] == pytest.approx(0.0, abs=0.5)
    assert r["dropouts"]["frames"] == 0
    assert r["segments_s"]["target_only"] == pytest.approx(2.0, abs=0.05)
    assert r["time_to_isolated_s"] == pytest.approx(2.5)
    assert r["time_to_locked_visual_s"] == pytest.approx(1.5)
    assert [tier for _, tier in r["tier_timeline"]] == ["gate", "focus_hq"]
    text = session_nr.format_report(r)
    for label in ("DNSMOS", "other-talker leak", "target preserved", "dropouts", "time to Isolated"):
        assert label in text


def test_session_score_counts_true_dropouts_only_in_target_speech(tmp_path):
    r = session_nr.score(fake_session(tmp_path, lag_s=0.0, hole=(0.5, 0.9)), MANIFEST, dnsmos="mock")
    assert r["dropouts"]["longest_ms"] == pytest.approx(400, abs=40)
    assert r["target_preserved_db"] < -0.5


def test_score_cli_prints_five_numbers(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import session_score_nr

    (tmp_path / "clip").mkdir()
    (tmp_path / "clip" / "manifest.json").write_text(json.dumps(MANIFEST))
    session = fake_session(tmp_path, leak=0.0)
    assert session_score_nr.main([str(session), "--clip", str(tmp_path / "clip"), "--dnsmos", "mock"]) == 0
    out = capsys.readouterr().out
    assert "other-talker leak" in out and "time to Isolated       2.5 s" in out


def test_oracle_lock_picks_the_face_at_the_manifest_target():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from replay import OracleLock

    oracle = OracleLock({"target": {"track_id": 1, "box": [0.1, 0.2, 0.2, 0.4]}})
    faces = [{"track_id": 7, "box": [0.6, 0.2, 0.2, 0.4]}, {"track_id": 9, "box": [0.12, 0.2, 0.2, 0.4]}]
    assert oracle.on_message({"type": "telemetry", "faces": faces}) == {
        "type": "control", "target": {"lock_track_id": 9}}
    assert oracle.on_message({"type": "telemetry", "faces": faces}) is None  # still visible: no re-lock
    assert OracleLock(None).on_message({"type": "telemetry", "faces": faces}) is None


def test_replay_endpoint_lists_and_accepts_ingested_clip_dirs(tmp_path, monkeypatch):
    from hearing.server.app import create_app
    from hearing.server.session import ServerConfig

    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    (tmp_path / "room").mkdir()
    phone_clip(tmp_path / "room" / "clip.mp4", seconds=0.2, fps=25)
    (tmp_path / "room" / "manifest.json").write_text(json.dumps({"duration_s": 0.2, "target": {"side": "left"}}))
    config = ServerConfig(replay_dir=tmp_path, pipelines={"natural": "passthrough"}, faces="mock",
                          asd="lipmotion", vision_threaded=False)
    with TestClient(create_app(config)) as client, client.websocket_connect("/ws/session?replay=1") as ws:
        clips = client.get("/replay/clips").json()["clips"]
        assert clips == [{"clip": "room/clip.mp4", "label": "room", "duration_s": 0.2, "target": "left"}]
        session_id = ws.receive_json()["session_id"]
        for name in ("room/clip.mp4", "room"):
            assert client.post("/replay", params={"session_id": session_id, "clip": name}).status_code == 200
        assert client.post("/replay", params={"session_id": session_id, "clip": "../x/clip.mp4"}).status_code == 400
        assert client.post("/replay", params={"session_id": session_id, "clip": "room/../../x.mp4"}).status_code == 400
        assert client.post("/replay", params={"session_id": session_id, "action": "stop"}).status_code == 200


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("HEARING_RUN_AV") != "1",
    reason="AV two-talker test is flaky under load; camera/vision is out of the Oct 11 audio-only demo. "
    "Set HEARING_RUN_AV=1 to run it.",
)
def test_synthesised_av_full_two_talker_clip_ingests_with_the_real_vision_stack(tmp_path):
    """Stand-in for the phone recording: two GRID talkers side by side (needs av_full + MediaPipe)."""
    from hearing.server.vision import mediapipe_available

    if not mediapipe_available():
        pytest.skip("MediaPipe weights missing")
    try:
        truth = real_clip.synth_phone_clip(tmp_path / "two.mp4")
    except RuntimeError as e:
        pytest.skip(str(e))
    m = real_clip.load_manifest(real_clip.ingest(tmp_path / "two.mp4", "two", "left", tmp_path / "demo"))
    assert [f["side"] for f in m["faces"]] == ["left", "right"]
    assert m["target"]["side"] == "left"
    left, right = m["faces"]
    # each talker's detected speech falls (mostly) inside the windows where only they or both talk
    for face, alone in ((left, truth["left"]), (right, truth["right"])):
        inside = sum(max(0.0, min(e, b) - max(s, a)) for s, e in face["speaking"] for a, b in alone)
        assert face["speaking_s"] > 1.0
        assert inside >= 0.7 * face["speaking_s"]
