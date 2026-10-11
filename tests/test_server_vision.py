"""Vision in the session: JPEG frames -> faces, p(speaking), target and tier in telemetry.

Faces are drawn as pure-green rectangles with a black mouth bar and found by the
``mock`` detector (:class:`hearing.server.vision.MarkerFaces`), so the whole path
(decode, detect, track, lip-motion ASD, target selection, gate) runs without weights.
"""

import io
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.server import vision
from hearing.server.app import create_app
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    KIND_VIDEO_IN,
    pack_frame,
    unpack_frame,
)
from hearing.server.session import ServerConfig, Session, load_stage_modules
from hearing.types import AudioChunk

SR = 16000
FPS = 25
W, H = 640, 360
CENTRE = (0.42, 0.3, 0.16, 0.4)  # x, y, w, h normalised: a face in the middle
SIDE = (0.05, 0.3, 0.16, 0.4)


def jpeg(faces=(), size=(W, H), quality=90) -> bytes:
    """A grey frame with each ``(box, mouth)`` drawn as a green face with a black mouth
    ``mouth`` (fraction of the face height) tall."""
    w, h = size
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    for (x, y, bw, bh), mouth in faces:
        x0, y0, x1, y1 = round(x * w), round(y * h), round((x + bw) * w), round((y + bh) * h)
        img[y0:y1, x0:x1] = (0, 255, 0)
        m = round(mouth * (y1 - y0))
        if m:
            my = y0 + round(0.65 * (y1 - y0))
            img[my : my + m, x0 + (x1 - x0) // 4 : x1 - (x1 - x0) // 4] = 0
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def syllables(t: float) -> bool:
    """4 Hz talking: 125 ms open/loud, 125 ms closed/quiet."""
    return int(t * 8) % 2 == 0


def audio(n: int, loud: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32) * (0.2 if loud else 0.002)
    return pcm16_to_float(float_to_pcm16(x))


def mock_session(**config) -> Session:
    load_stage_modules()  # registers passthrough/gate/synth, as create_app does
    config.setdefault("faces", "mock")
    config.setdefault("asd", "lipmotion")
    config.setdefault("vision_threaded", False)
    return Session(ServerConfig(**config), device="cpu")


def run_scene(session: Session, seconds: float, speaker=SIDE, silent=CENTRE, talking=True,
              t0: float = 0.0) -> list[bytes | dict]:
    """Feed ``seconds`` of 25 fps video and 40 ms audio: ``speaker``'s mouth and the room
    audio follow :func:`syllables` (when ``talking``), ``silent`` keeps its mouth shut."""
    out = []
    for i in range(round(seconds * FPS)):
        t = t0 + i / FPS
        loud = talking and syllables(t)
        faces = [(b, m) for b, m in ((speaker, 0.15 if loud else 0.0), (silent, 0.0)) if b]
        out += session.handle_binary(pack_frame(KIND_VIDEO_IN, i, t, jpeg(faces)))
        out += session.handle_binary(pack_frame(KIND_AUDIO_IN, i, t + 0.04,
                                                float_to_pcm16(audio(640, loud, i))))
    return out


# ---------------------------------------------------------------- decode and detect


def test_decode_jpeg_keeps_small_frames_and_shrinks_big_ones():
    small = vision.decode_jpeg(jpeg(size=(320, 180)))
    assert small.shape == (180, 320, 3) and small.dtype == np.uint8
    big = vision.decode_jpeg(jpeg(size=(1280, 720)))
    assert big.shape == (360, 640, 3)
    with pytest.raises(ValueError, match="cannot decode JPEG"):
        vision.decode_jpeg(b"\xff\xd8\xff\xd9")


def test_marker_detector_finds_faces_and_mouth_opening():
    image = vision.decode_jpeg(jpeg([(CENTRE, 0.2), (SIDE, 0.0)]))
    dets = sorted(vision.MarkerFaces().detect(image, 0.0), key=lambda d: d.box[0])
    assert len(dets) == 2
    np.testing.assert_allclose(dets[0].box, SIDE, atol=0.01)
    np.testing.assert_allclose(dets[1].box, CENTRE, atol=0.01)
    open_lm, shut_lm = dets[1].landmarks, dets[0].landmarks
    assert open_lm.shape == (478, 2)
    face_h = open_lm[152, 1] - open_lm[10, 1]
    assert (open_lm[14, 1] - open_lm[13, 1]) / face_h == pytest.approx(0.2, abs=0.02)
    assert shut_lm[14, 1] == shut_lm[13, 1]


def test_unknown_detector_name_is_rejected():
    with pytest.raises(ValueError, match="unknown face detector"):
        vision.get_face_detector("bogus")
    with pytest.raises(ValueError, match="unknown face detector"):
        ServerConfig(faces="bogus")


# ---------------------------------------------------------------- fake client over /ws/session


def test_fake_client_sends_jpeg_frames_and_gets_faces_in_telemetry(tmp_path):
    client = TestClient(create_app(ServerConfig(console_dir=tmp_path, faces="mock", asd="lipmotion",
                                                pipelines={"natural": "passthrough>gate"})))
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "hello", "sample_rate": 16000, "frame_ms": 40,
                                 "video": True, "client": "test"}))
        telemetry = None
        for i in range(60):
            ws.send_bytes(pack_frame(KIND_VIDEO_IN, i, i / FPS, jpeg([(CENTRE, 0.0)])))
            ws.send_bytes(pack_frame(KIND_AUDIO_IN, i, i / FPS, float_to_pcm16(audio(640, False, i))))
            time.sleep(0.005)  # let the vision worker finish so frames are not all dropped
            if i >= 10:
                ws.send_text(json.dumps({"type": "control", "suppression": 50}))
                msg = _recv_json(ws, "telemetry")
                if msg["faces"]:
                    telemetry = msg
                    break
    assert telemetry is not None, "no faces in telemetry after 60 frames"
    (face,) = telemetry["faces"]
    assert set(face) == {"track_id", "box", "p_speaking"}
    np.testing.assert_allclose(face["box"], CENTRE, atol=0.02)
    assert 0.0 <= face["p_speaking"] <= 1.0
    assert telemetry["tier"] == "gate"
    assert telemetry["vision"]["status"] == "live"
    assert telemetry["vision"]["detector"] == "mock"
    assert telemetry["latency_ms"]["vision"] >= 0.0


def _recv_json(ws, kind: str, limit: int = 400) -> dict:
    for _ in range(limit):
        raw = ws.receive()
        if raw.get("text") is not None:
            msg = json.loads(raw["text"])
            if msg.get("type") == kind:
                return msg
    raise AssertionError(f"no {kind} message")


# ---------------------------------------------------------------- target, ASD and tier


def test_speaking_face_becomes_target_and_silent_centre_face_does_not():
    session = mock_session()
    run_scene(session, 2.0)
    t = session.telemetry()
    faces = {f["track_id"]: f for f in t["faces"]}
    assert len(faces) == 2
    speaker = min(faces.values(), key=lambda f: f["box"][0])  # SIDE is on the left
    silent = max(faces.values(), key=lambda f: f["box"][0])
    assert speaker["p_speaking"] > 0.5 > silent["p_speaking"]
    assert t["target"]["track_id"] == speaker["track_id"]
    assert t["target"]["locked"] is False
    assert t["tier"] == "av"  # default natural pipeline is plain dfn_ll, no gate (T-103)
    assert t["vision"]["status"] == "live" and t["vision"]["asd"] == "lipmotion"


def test_gate_ducks_the_room_when_the_target_stops_talking():
    session = mock_session(pipelines={"natural": "passthrough>gate"})
    session.handle_text(json.dumps({"type": "control", "suppression": 100}))
    run_scene(session, 2.0)
    assert session.telemetry()["target"]["track_id"] is not None
    out = run_scene(session, 1.5, talking=False, t0=2.0)
    frames = [unpack_frame(m) for m in out if isinstance(m, bytes) and m[0] == KIND_AUDIO_OUT]
    tail = np.concatenate([pcm16_to_float(f.payload) for f in frames[-10:]])
    # target silent, slider at 100: the room is ducked to the -30 dB floor (T-096), not to silence
    assert np.max(np.abs(tail)) <= 10 ** (-30 / 20) + 1e-3
    assert session.telemetry()["tier"] == "gate"


def test_lock_from_the_console_pins_the_target():
    session = mock_session()
    run_scene(session, 1.0)
    faces = session.telemetry()["faces"]
    silent = max(faces, key=lambda f: f["box"][0])["track_id"]
    session.handle_text(json.dumps({"type": "control", "target": {"lock_track_id": silent}}))
    run_scene(session, 1.0, t0=1.0)
    target = session.telemetry()["target"]
    assert target == {"track_id": silent, "confidence": pytest.approx(1.0), "locked": True}
    session.handle_text(json.dumps({"type": "control", "target": None}))
    assert session.telemetry()["target"]["locked"] is False


def test_voice_estimate_only_hears_the_target_while_its_face_speaks():
    session = mock_session()
    run_scene(session, 2.0)  # SIDE talks and becomes the target
    target = session.ctx.target.track_id
    flags = []
    session._voice_feed = type("Feed", (), {"update": lambda self, c, speaking: flags.append(speaking),
                                            "reset": lambda self: None})()
    session._voice_worker = type("Worker", (), {"track": None, "poll": lambda self, t=0.0: None})()
    loud = AudioChunk(audio(160, True, 0), SR, 0, 0.0)
    for p in (0.9, 0.1):
        for track in session.ctx.tracks:
            track.p_speaking = p if track.track_id == target else 0.0
        session._feed_voice(loud, [])
    session.vision.live = False  # no camera: the energy VAD alone decides
    session._feed_voice(loud, [])
    assert flags == [True, False, True]


# ---------------------------------------------------------------- degrade


def test_camera_stop_degrades_to_audio_only(monkeypatch):
    # Only advance camera liveness when the test deliberately stops the camera.
    # A loaded host may take >50 ms to process audio between synthetic frames.
    clock = [0.0]
    monkeypatch.setattr(vision, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))
    monkeypatch.setattr(vision, "LIVE_S", 0.05)
    session = mock_session(pipelines={"natural": "passthrough>gate"})
    run_scene(session, 1.0)
    assert session.telemetry()["tier"] == "gate"
    clock[0] += 0.1
    session.handle_binary(pack_frame(KIND_AUDIO_IN, 99, 9.0, float_to_pcm16(audio(640, True, 0))))
    t = session.telemetry()
    assert t["tier"] == "audio-only"
    assert t["faces"] == [] and t["target"]["track_id"] is None
    assert t["vision"]["status"] == "off"


def test_camera_coming_back_restarts_tracks_and_the_gate(monkeypatch):
    # Only advance camera liveness when the test deliberately stops the camera.
    # A loaded host may take >50 ms to process audio between synthetic frames.
    clock = [0.0]
    monkeypatch.setattr(vision, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))
    monkeypatch.setattr(vision, "LIVE_S", 0.05)
    session = mock_session(pipelines={"natural": "passthrough>gate"})
    run_scene(session, 1.0)
    before = {f["track_id"] for f in session.telemetry()["faces"]}
    clock[0] += 0.1
    session.handle_binary(pack_frame(KIND_AUDIO_IN, 99, 9.0, float_to_pcm16(audio(640, True, 0))))
    assert session.telemetry()["tier"] == "audio-only"
    run_scene(session, 2.0, t0=10.0)
    t = session.telemetry()
    assert t["tier"] == "gate" and t["vision"]["status"] == "live"
    assert {f["track_id"] for f in t["faces"]} == before  # the tracker restarted its ids
    assert t["target"]["track_id"] == min(t["faces"], key=lambda f: f["box"][0])["track_id"]


def test_tracking_and_asd_run_on_the_vision_worker(monkeypatch):
    threads = []
    real_get_asd = vision.get_asd

    def spying_get_asd(name):
        asd = real_get_asd(name)
        update = asd.update

        def spy(*args):
            threads.append(threading.current_thread().name)
            return update(*args)

        asd.update = spy
        return asd

    monkeypatch.setattr(vision, "get_asd", spying_get_asd)
    session = mock_session(vision_threaded=True)
    for i in range(5):
        session.handle_binary(pack_frame(KIND_VIDEO_IN, i, i / FPS, jpeg([(CENTRE, 0.0)])))
        session.wait_vision(1.0)
    assert len(threads) == 5 and all(n.startswith("vision") for n in threads)
    assert session.telemetry()["faces"]
    session.close()


def test_failing_asd_falls_back_to_lipmotion(monkeypatch):
    class Broken:
        name = "broken"

        def update(self, *args):
            raise RuntimeError("MPS out of memory")

        def reset(self):
            pass

    real_get_asd = vision.get_asd
    monkeypatch.setattr(vision, "get_asd",
                        lambda name: Broken() if name == "broken" else real_get_asd(name))
    session = mock_session(asd="broken")
    run_scene(session, 0.4)
    t = session.telemetry()
    assert t["vision"]["asd"] == "lipmotion" and t["vision"]["status"] == "live"
    assert len(t["faces"]) == 2


def test_no_detector_reports_once_and_stays_audio_only():
    session = mock_session(faces="none", pipelines={"natural": "passthrough>gate"})
    out = run_scene(session, 0.4)
    errors = [m for m in out if isinstance(m, dict) and m["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["stage"] == "vision" and errors[0]["degraded_to"] == "audio-only"
    t = session.telemetry()
    assert t["tier"] == "audio-only" and t["faces"] == []
    assert t["vision"]["status"] == "unavailable" and t["vision"]["frames"] == 10


def test_bad_jpeg_is_reported_once_after_the_audio_reply():
    session = mock_session(pipelines={"natural": "passthrough>gate"})
    replies = []
    for i in range(3):
        session.handle_binary(pack_frame(KIND_VIDEO_IN, i, 0.0, b"not a jpeg"))
        replies += session.handle_binary(pack_frame(KIND_AUDIO_IN, i, 0.0,
                                                    float_to_pcm16(audio(320, False, i))))
    assert isinstance(replies[0], bytes) and replies[0][0] == KIND_AUDIO_OUT
    errors = [m for m in replies if isinstance(m, dict) and m["type"] == "error"]
    assert len(errors) == 1 and "cannot decode JPEG" in errors[0]["message"]
    assert session.telemetry()["vision"]["status"] == "off"


def test_server_mediapipe_uses_the_cpu_delegate_unless_opted_in(monkeypatch):
    seen = []

    class Recorder:
        def __init__(self, running_mode, max_faces, delegate):
            seen.append((running_mode, delegate))

    monkeypatch.setattr(vision.faces, "MediaPipeBackend", Recorder)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.delenv("HEARING_FACES_DELEGATE", raising=False)
    assert vision.get_face_detector("mediapipe").delegate == "cpu"
    monkeypatch.setenv("HEARING_FACES_DELEGATE", "gpu")
    assert vision.get_face_detector("mediapipe").delegate == "gpu"
    assert seen == [("video", "cpu"), ("video", "gpu")]


def test_busy_worker_drops_frames_instead_of_queueing(monkeypatch):
    class Slow(vision.MarkerFaces):
        def detect(self, image, ts):
            time.sleep(0.05)
            return super().detect(image, ts)

    monkeypatch.setattr(vision, "get_face_detector", lambda name: Slow())
    session = mock_session(vision_threaded=True)
    frame = jpeg([(CENTRE, 0.0)])
    for i in range(10):
        session.handle_binary(pack_frame(KIND_VIDEO_IN, i, i / FPS, frame))
    session.wait_vision(1.0)
    t = session.telemetry()
    assert t["vision"]["dropped"] >= 5
    assert t["vision"]["frames"] == 10
    session.close()


def test_apparent_face_hook_runs_at_most_once_per_second_per_track():
    session = mock_session()
    calls = []
    session.vision.on_face = lambda tid, image, box: calls.append((tid, image.shape, box)) or []
    run_scene(session, 2.0)
    per_track = {}
    for tid, shape, _ in calls:
        per_track[tid] = per_track.get(tid, 0) + 1
        assert shape == (H, W, 3)
    assert len(per_track) == 2 and all(n == 2 for n in per_track.values())


# ---------------------------------------------------------------- real detector


@pytest.mark.slow
def test_mediapipe_finds_a_real_face():
    if not vision.mediapipe_available():
        pytest.skip("mediapipe or its face models missing (python -m hearing.vision.faces --download)")
    from matplotlib import cbook

    with open(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False), "rb") as f:
        image = vision.decode_jpeg(f.read())
    detector = vision.get_face_detector("mediapipe")
    try:
        dets = detector.detect(image, 0.0)
        t0 = time.perf_counter()
        for i in range(20):
            detector.detect(image, (i + 1) / FPS)
        ms = 1000 * (time.perf_counter() - t0) / 20
    finally:
        detector.close()
    assert len(dets) == 1
    x, _, w, _ = dets[0].box
    assert dets[0].landmarks.shape == (478, 3)
    assert 0.2 < x + w / 2 < 0.8 and 0.1 < w < 0.9
    print(f"mediapipe detector + landmarker {image.shape[1]}x{image.shape[0]}: {ms:.1f} ms/frame")


def test_same_bucket_signal_reaches_the_ladder_context(monkeypatch):
    """T-070: both faces get the mock face estimate (male_adult), so the session publishes per-track
    buckets and same_bucket=True for the tier ladder; never on the wire."""
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    session = mock_session()
    out = run_scene(session, 2.0)
    extras = session.ctx.telemetry.extras
    assert session.ctx.target.track_id is not None
    assert len(extras["voice_buckets"]) == 2 and set(extras["voice_buckets"].values()) == {"male_adult"}
    assert extras["same_bucket"] is True
    telemetry = [m for m in out if isinstance(m, dict) and m.get("type") == "telemetry"]
    assert all("voice_buckets" not in json.dumps(m) for m in telemetry)


def test_no_face_estimator_means_no_same_bucket(monkeypatch):
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    session = mock_session()
    session.face_estimator = None  # buckets stay neutral (unknown) -> False
    run_scene(session, 1.0)
    assert session.ctx.telemetry.extras["same_bucket"] is False


def test_busy_gpu_runs_light_asd_on_the_cpu_instead_of_skipping(monkeypatch):
    """T-083: a held TORCH_LOCK must not leave p_speaking stale."""
    from hearing.tse.gpu import TORCH_LOCK
    from hearing.vision import light_asd

    class Gpu:
        device = "mps"
        net = object()

    class Cpu:
        device = "cpu"

    class Asd:
        name = "light_asd"
        def __init__(self):
            self._backend = Gpu()
            self.seen: list[str] = []

        @property
        def backend(self):
            return self._backend

        def update(self, tracks, ts, audio, frame):
            self.seen.append(self._backend.device)
            for t in tracks:
                t.p_speaking = 0.9

    monkeypatch.setattr(light_asd, "TorchBackend", lambda dev: Cpu())
    eng = vision.VisionEngine(detector="none", asd="lipmotion", threaded=False)
    eng.asd = Asd()
    eng._last_update -= 1.0  # the last fresh update is old enough for the CPU gap
    track = SimpleNamespace(track_id=1, p_speaking=0.0)
    held, release = threading.Event(), threading.Event()

    def hog():
        with TORCH_LOCK:
            held.set()
            release.wait(5)

    th = threading.Thread(target=hog)
    th.start()
    held.wait(5)
    try:
        eng._run_asd([track], 0.0, None, None)
    finally:
        release.set()
        th.join()
    assert eng.asd.seen == ["cpu"] and eng.asd._backend.device == "mps"  # restored
    assert track.p_speaking == 0.9 and eng.asd_cpu == 1 and eng.asd_skipped == 0
    eng._run_asd([track], 0.04, None, None)  # lock free again: the GPU runs it
    assert eng.asd.seen == ["cpu", "mps"] and len(eng._age_ms) == 2


def test_lock_survives_vision_going_dark_and_reattaches_by_position():
    """T-107: a vision stall (no applied frame for LIVE_S) restarts track ids; it used to reset the
    selector, dropping the explicit lock, and free selection then picked the other, more central face."""
    from hearing.types import FaceTrack, StageContext

    def faces(ts, left_id, right_id):
        return [FaceTrack(left_id, (0.14, 0.4, 0.12, 0.2), None, 0.9, ts),
                FaceTrack(right_id, (0.56, 0.4, 0.12, 0.2), None, 0.9, ts)]

    eng = vision.VisionEngine(detector="none", asd="lipmotion", threaded=False)
    ctx = StageContext.default()
    eng.selector.update(faces(0.0, 1, 2), 0.0)
    eng.lock(1)
    assert eng.selector.update(faces(0.04, 1, 2), 0.04).track_id == 1
    eng.live = True
    eng._go_dark(ctx)
    assert ctx.target.track_id is None and ctx.target.locked
    # the restarted tracker numbers the faces the other way round
    for i in range(50):
        state = eng.selector.update(faces(2.0 + i * 0.04, 2, 1), 2.0 + i * 0.04)
        assert state.track_id == 2 and state.locked


def _background_tab(session: Session, clock: list[float], seconds: int, gap_s: float, t0: float,
                    loud: bool = False) -> list[dict]:
    """Video at one frame per ``gap_s`` (Chrome throttles a hidden tab's timers to 1 Hz) with
    40 ms audio in between (``loud``: someone talks); the wall clock follows the capture clock.
    Telemetry after each frame."""
    seen = []
    t = t0
    for i in range(round(seconds / gap_s)):
        session.handle_binary(pack_frame(KIND_VIDEO_IN, 1000 + i, t, jpeg([(SIDE, 0.0), (CENTRE, 0.0)])))
        for k in range(round(gap_s / 0.04)):
            clock[0] += 0.04
            session.handle_binary(pack_frame(KIND_AUDIO_IN, 1000 + i, t + 0.04 * (k + 1),
                                             float_to_pcm16(audio(640, loud, k))))
        seen.append(session.telemetry())
        t += gap_s
    return seen


def test_explicit_lock_survives_a_backgrounded_tab_at_one_fps(monkeypatch):
    """T-138: live session 20261009T174419Z-2 got exactly 1 fps while the judge tab was hidden. Each
    1.0 s gap went past LIVE_S, so vision flapped off/live, the tracker restarted its ids (target
    1 -> 3 -> 1) and enrollment failed. With an explicit lock a gap up to LOCK_GAP_S keeps it all."""
    clock = [0.0]
    monkeypatch.setattr(vision, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))
    session = mock_session(pipelines={"natural": "passthrough>gate"})
    run_scene(session, 1.0)
    side = min(session.telemetry()["faces"], key=lambda f: f["box"][0])["track_id"]
    session.handle_text(json.dumps({"type": "control", "target": {"lock_track_id": side}}))
    run_scene(session, 0.4, t0=1.0)
    assert session.telemetry()["target"] == {**session.telemetry()["target"], "track_id": side, "locked": True}
    # the judge's calibration window (the live session's enrollment.state=failed) spans the gap
    session.handle_text(json.dumps({"type": "control", "enroll": {"action": "start", "seconds": 3.0}}))
    seen = _background_tab(session, clock, 12, 1.2, t0=2.0, loud=True)  # 1 Hz timer + JPEG/upload jitter
    assert [t["vision"]["status"] for t in seen] == ["live"] * 10
    assert {t["target"]["track_id"] for t in seen} == {side}
    assert all(t["target"]["locked"] for t in seen)
    assert "failed" not in {t["enrollment"]["state"] for t in seen}
    assert seen[-1]["enrollment"]["state"] == "calibrated" and session.enroller.track_id == side
    # a longer gap still goes dark (the lock itself is kept and re-attached by position, T-107)
    clock[0] += vision.LOCK_GAP_S + 0.1
    session.handle_binary(pack_frame(KIND_AUDIO_IN, 2000, 20.0, float_to_pcm16(audio(640, False, 0))))
    assert session.telemetry()["vision"]["status"] == "off"


def test_one_fps_without_a_lock_still_goes_dark(monkeypatch):
    """Without an explicit lock the 1 s camera timeout is unchanged (audio-only is the safe tier)."""
    clock = [0.0]
    monkeypatch.setattr(vision, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))
    session = mock_session(pipelines={"natural": "passthrough>gate"})
    run_scene(session, 1.0)
    clock[0] += vision.LIVE_S + 0.1
    session.handle_binary(pack_frame(KIND_AUDIO_IN, 99, 9.0, float_to_pcm16(audio(640, False, 0))))
    assert session.telemetry()["vision"]["status"] == "off"
