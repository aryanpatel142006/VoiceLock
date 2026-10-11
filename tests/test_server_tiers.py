"""Focus HQ in the session: the tier ladder, lip crops from vision, crossfaded mode switches."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from hearing import registry
from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.fusion.tiers import fade_gains
from hearing.server import vision
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    KIND_VIDEO_IN,
    pack_frame,
    unpack_frame,
)
from hearing.server.session import (
    DEFAULT_PIPELINES,
    SWITCH_FADE_MS,
    ServerConfig,
    Session,
    load_stage_modules,
)
from hearing.tse.base import register_candidate

SR = 16000
FPS = 25
W, H = 640, 360
SPEAKER = (0.05, 0.3, 0.16, 0.4)
SILENT = (0.42, 0.3, 0.16, 0.4)
CALLS = []


@register_candidate("t019_lips")
class LipsCandidate:
    """Identity that records the lip crops it was given (needs lips, like the AV models)."""

    name = "t019_lips"
    licence = "test"
    needs = frozenset({"lips"})
    device = "cpu"

    def load(self):
        pass

    def extract(self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None):
        CALLS.append(lips)
        return np.asarray(mix, np.float32).copy()


def jpeg(faces) -> bytes:
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    for (x, y, bw, bh), mouth in faces:
        x0, y0, x1, y1 = round(x * W), round(y * H), round((x + bw) * W), round((y + bh) * H)
        img[y0:y1, x0:x1] = (0, 255, 0)
        m = round(mouth * (y1 - y0))
        if m:
            my = y0 + round(0.65 * (y1 - y0))
            img[my : my + m, x0 + (x1 - x0) // 4 : x1 - (x1 - x0) // 4] = 0
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def audio(n: int, loud: bool, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32) * (0.2 if loud else 0.002)
    return pcm16_to_float(float_to_pcm16(x))


def session_with(tmp_path, tiers_yaml: str | None = None, **config) -> Session:
    load_stage_modules()
    pipelines = config.pop("pipelines", {})
    if tiers_yaml is not None:
        path = tmp_path / "tiers.yaml"
        path.write_text(tiers_yaml)
        pipelines.setdefault("focus_hq", f'tiers(config="{path}")')
    config.update(faces="mock", asd="lipmotion", vision_threaded=False)
    return Session(ServerConfig(pipelines=pipelines, **config), device="cpu")


def run_scene(session: Session, seconds: float, t0: float = 0.0, camera: bool = True) -> list:
    """25 fps marker faces (SPEAKER talks at 4 Hz, SILENT does not) and 40 ms of audio per frame."""
    out = []
    for i in range(round(seconds * FPS)):
        t = t0 + i / FPS
        loud = int(t * 8) % 2 == 0
        if camera:
            faces = [(SPEAKER, 0.15 if loud else 0.0), (SILENT, 0.0)]
            out += session.handle_binary(pack_frame(KIND_VIDEO_IN, i, t, jpeg(faces)))
        out += session.handle_binary(pack_frame(KIND_AUDIO_IN, i, t + 0.04, float_to_pcm16(audio(640, loud, i))))
    return out


LADDER = """
crossfade_ms: 50
upgrade_after_s: 0.2
tiers:
  - name: focus_hq
    pipeline: 'tse(candidate="t019_lips", window_s=0.4, hop_s=0.2, threaded=false)>mixer'
    needs: [camera, target, healthy]
  - name: gate
    pipeline: 'passthrough>gate>mixer'
    needs: [camera, target]
  - name: audio-only
    pipeline: passthrough
"""


def test_focus_hq_defaults_to_the_tier_ladder():
    assert DEFAULT_PIPELINES["focus_hq"] == "tiers"
    load_stage_modules()
    assert "tiers" in registry.names() and "tse" in registry.names()


def test_focus_hq_reports_the_active_tier_and_the_ladder(tmp_path, monkeypatch):
    # Camera liveness is wall clock: a 50 ms LIVE_S went dark mid-scene under load (Q-244). Keep the
    # camera live while frames flow, then make "no new frame" mean dark at once, with no sleep.
    monkeypatch.setattr(vision, "LIVE_S", 3600.0)
    CALLS.clear()
    s = session_with(tmp_path, LADDER)
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq", "suppression": 100}))
    run_scene(s, 2.0)
    t = s.telemetry()
    assert t["pipeline"].startswith("tiers(")
    assert t["tier"] == "focus_hq"
    ladder = t["tiers"]
    assert ladder["active"] == "focus_hq" and ladder["delay_ms"] == 400.0
    assert [x["name"] for x in ladder["tiers"]] == ["focus_hq", "gate", "audio-only"]
    hq_health = ladder["tiers"][0]["health"][0]
    assert hq_health["candidate"] == "t019_lips" and hq_health["ok"] and hq_health["lips"] == 1.0
    assert t["latency_ms"]["algorithmic"] == 400.0
    json.dumps(t)  # wire-safe

    # The lip crops came from the vision path: 88x88 per face per frame, target's into the model.
    assert len(s.lip_feed) > 0 and s.lip_feed.since(0)[-1].crop.shape == (88, 88)
    assert CALLS and all(c.shape == (10, 88, 88) for c in CALLS)

    # Camera gone: the AV tiers drop out, the ladder plays audio-only.
    monkeypatch.setattr(vision, "LIVE_S", -1.0)
    run_scene(s, 0.4, t0=2.0, camera=False)
    t = s.telemetry()
    assert t["tier"] == "audio-only" and t["tiers"]["switches"] >= 1
    assert t["tiers"]["tiers"][0]["reason"] == "no camera"


TWO_DELAYS = """
crossfade_ms: 50
tiers:
  - name: enrolled
    pipeline: 'delay(ms=1500)>mixer'
  - name: focus_hq
    pipeline: 'delay(ms=3000)>mixer'
    needs: [camera, target]
  - name: audio-only
    pipeline: passthrough
"""


def test_focus_hq_algorithmic_latency_is_the_playing_tiers_own(tmp_path):
    """The Mac ladder plays enrolled at 1.5 s; the cold-start AV tier's 3 s must not hold it back."""
    s = session_with(tmp_path, TWO_DELAYS)
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    run_scene(s, 1.0)
    t = s.telemetry()
    assert t["tier"] == "enrolled"
    assert t["latency_ms"]["algorithmic"] == 1500.0  # not 3000
    assert t["tiers"]["delay_ms"] == 1500.0
    assert [(x["name"], x["delay_ms"]) for x in t["tiers"]["tiers"]] == [
        ("enrolled", 1500.0), ("focus_hq", 3000.0), ("audio-only", 0.0)]


HEALTH_T072 = {}  # tier key -> (ok, reason)


@registry.register("t072_health")
class HealthFlag:
    """Pass-through whose health the test sets (stands in for the tse stage's health)."""

    sample_rate = SR
    latency_ms = 0.0

    def __init__(self, key: str = "enr"):
        self.key, self.name = key, f"t072_health({key})"

    def process(self, chunk, ctx):
        return chunk

    def reset(self):
        pass

    def health(self):
        ok, reason = HEALTH_T072.get(self.key, (True, ""))
        return {"ok": ok, "reason": reason}


MAC_LADDER = """
crossfade_ms: 50
upgrade_after_s: 0.2
tiers:
  - name: enrolled
    pipeline: 't072_health(key="enr")>delay(ms=1500)>mixer'
    needs: [healthy]
  - name: focus_hq
    pipeline: 't072_health(key="av")>delay(ms=3000)>mixer'
    needs: [camera, target, healthy]
  - name: gate
    pipeline: 'delay(ms=13)>gate>mixer'
    needs: [camera, target]
  - name: audio-only
    pipeline: 'delay(ms=13)'
"""


def test_reported_delay_is_the_playing_tiers_after_the_first_switch(tmp_path, monkeypatch):
    """T-072 (live run on the Mac): with enrolled still enrolling and the AV tier overrunning, the
    ladder plays gate, and telemetry must say ~13 ms (not the 3000 ms of the slowest built tier)
    from the first switch on; once enrolled plays it says 1500 ms."""
    monkeypatch.setattr(vision, "LIVE_S", 0.5)
    HEALTH_T072.update(enr=(False, "enrolling"), av=(False, "overrun"))
    s = session_with(tmp_path, MAC_LADDER)
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq", "suppression": 100}))
    seen = []
    for k in range(10):  # 2 s in 0.2 s slices, telemetry after each
        run_scene(s, 0.2, t0=0.2 * k)
        t = s.telemetry()
        seen.append((t["tier"], t["tiers"]["delay_ms"], t["latency_ms"]["algorithmic"]))
    assert seen[-1][0] == "gate" and s._built[s.active_spec].stages[0].switches >= 1
    for tier, delay, algorithmic in seen:
        assert tier in {"audio-only", "gate"}
        assert delay == algorithmic == 13.0, seen  # never the AV tier's 3000 ms
    assert t["latency_ms"]["total"] < 13.0 + 100.0  # + server + one 40 ms client frame
    assert {x["name"]: x["delay_ms"] for x in t["tiers"]["tiers"]} == {
        "enrolled": 1500.0, "focus_hq": 3000.0, "gate": 13.0, "audio-only": 13.0}

    HEALTH_T072["enr"] = (True, "")  # enrollment complete
    run_scene(s, 1.0, t0=2.0)
    t = s.telemetry()
    assert t["tier"] == "enrolled"
    assert t["tiers"]["delay_ms"] == t["latency_ms"]["algorithmic"] == 1500.0
    HEALTH_T072.clear()
    s.close()


def test_tier_without_ladder_keeps_the_old_rules(tmp_path):
    s = session_with(tmp_path, pipelines={"natural": "passthrough>gate"})
    run_scene(s, 1.0)
    t = s.telemetry()
    assert t["tier"] == "gate" and t["tiers"] is None and t["transition"] is None


def _hop_audio(session: Session, n_frames: int, start: int = 0, value: float = 0.25) -> np.ndarray:
    """Feed constant 20 ms frames; return the concatenated audio out."""
    ys = []
    x = np.full(320, value, np.float32)
    for i in range(start, start + n_frames):
        for m in session.handle_binary(pack_frame(KIND_AUDIO_IN, i, i * 0.02, float_to_pcm16(x))):
            if isinstance(m, bytes) and m[0] == KIND_AUDIO_OUT:
                ys.append(pcm16_to_float(unpack_frame(m).payload))
    return np.concatenate(ys)


def test_mode_switch_is_make_before_break(tmp_path):
    s = session_with(tmp_path, pipelines={"natural": "gain(db=-6)", "focus_hq": "delay(ms=300)"})
    _hop_audio(s, 25)  # natural, warmed up
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    y = _hop_audio(s, 1, start=25)  # the switch happens on the next audio frame
    assert s.telemetry()["transition"] == {"from": "gain(db=-6)", "priming_ms": 280.0, "fade": 0.0,
                                           "fade_kind": None}
    y = np.concatenate([y, _hop_audio(s, 39, start=26)])
    half = 0.25 * 10 ** (-6 / 20)
    prime, fade = round(0.3 * SR), round(SWITCH_FADE_MS / 1000 * SR)
    # No gap: the old pipeline plays while the new one fills its 300 ms delay ...
    np.testing.assert_allclose(y[:prime], half, atol=1e-4)
    # ... then the ladder's fade for a 300 ms jump in delay (an equal-power splice) ...
    old_gain, new_gain = fade_gains("splice", np.arange(fade), fade)
    np.testing.assert_allclose(y[prime : prime + fade], old_gain * half + new_gain * 0.25, atol=1e-4)
    # ... and the new pipeline alone afterwards.
    np.testing.assert_allclose(y[prime + fade :], 0.25, atol=1e-4)
    assert s.telemetry()["transition"] is None


def test_switching_back_while_priming_keeps_the_old_pipeline_seamless(tmp_path):
    s = session_with(tmp_path, pipelines={"natural": "gain(db=-6)", "focus_hq": "delay(ms=300)"})
    _hop_audio(s, 10)
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    _hop_audio(s, 5, start=10)
    s.handle_text(json.dumps({"type": "control", "mode": "natural"}))
    y = _hop_audio(s, 10, start=15)
    np.testing.assert_allclose(y, 0.25 * 10 ** (-6 / 20), atol=1e-4)
    assert s.telemetry()["transition"] is None


def test_failed_pipeline_degrades_with_a_crossfade_from_the_old_mode(tmp_path):
    s = session_with(tmp_path, pipelines={"natural": "gain(db=-6)", "focus_hq": "no_such_stage"})
    _hop_audio(s, 5)
    out = s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    y = _hop_audio(s, 5, start=5)
    assert not [m for m in out if isinstance(m, dict) and m["type"] == "error"]  # built lazily
    fade = round(SWITCH_FADE_MS / 1000 * SR)
    assert y[0] < 0.13 and np.all(np.diff(y) >= -1e-4)  # old mode gliding into the fallback
    np.testing.assert_allclose(y[fade:], 0.25, atol=1e-4)
    t = s.telemetry()
    assert t["pipeline"] == "passthrough" and t["transition"] is None


class _Client:
    connected = True

    def __init__(self):
        self.video = []

    def send_video(self, seq, ts, jpeg):
        self.video.append(seq)


@registry.register("t019_fake_remote")
class _FakeRemote:
    name = "t019_fake_remote"
    sample_rate = SR
    latency_ms = 0.0

    def __init__(self):
        self.client = _Client()

    def process(self, chunk, ctx):
        return chunk

    def reset(self):
        pass


def test_camera_frames_are_forwarded_to_remote_stages(tmp_path):
    s = session_with(tmp_path, pipelines={"focus_hq": "t019_fake_remote"})
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    run_scene(s, 0.2)
    stage = s._built["t019_fake_remote"].stages[0]
    assert stage.client.video == [1, 2, 3, 4]  # frame 0 came before the pipeline was built


@pytest.mark.slow
def test_default_ladder_builds_with_real_tiers():
    from hearing.fusion.tiers import TIERS, TierLadder

    load_stage_modules()
    lad = TierLadder()
    assert lad.order == TIERS[: len(lad.order)] or set(lad.order) <= set(TIERS)
    assert lad.built["enrolled"].delay_ms == 1500.0  # each tier plays at its own delay
    # Before the first chunk: the slowest built tier (AV-MossFormer2 on CUDA only, T-072).
    assert lad.latency_ms == (3000.0 if "focus_hq" in lad.built else 1500.0)
    lad.close()


def test_frames_dropped_by_a_busy_worker_still_feed_lip_crops():
    from concurrent.futures import Future

    from hearing.tse.lips import FEED_KEY, LipFeed
    from hearing.types import StageContext

    eng = vision.VisionEngine("mock", "lipmotion", threaded=True)
    ctx = StageContext.default()
    feed = ctx.telemetry.extras[FEED_KEY] = LipFeed()
    faces = [(SPEAKER, 0.1), (SILENT, 0.0)]
    for i in range(5):  # tracks confirmed and applied
        eng.submit(i, i / FPS, jpeg(faces))
        eng.wait(2.0)
        eng.poll(ctx)
    before = len(feed)
    assert before > 0 and ctx.tracks
    eng._jobs.extend(Future() for _ in range(vision.MAX_IN_FLIGHT))  # busy (the ASD waits for the GPU)
    for i in range(5, 8):
        eng.submit(i, i / FPS, jpeg(faces))
    assert eng.dropped == 3 and eng.lip_only == 3
    new = feed.since(0)[before:]
    assert len(new) == 3 * len(ctx.tracks) and {round(f.ts, 2) for f in new} == {0.2, 0.24, 0.28}
    eng._jobs.clear()
    eng.close()


def test_latency_preset_switches_the_ladder_spec_and_shows_in_telemetry(tmp_path):
    session = session_with(tmp_path, LADDER)
    session.handle_text(json.dumps({"type": "control", "mode": "focus_hq"}))
    base = session.active_spec
    assert "preset=" not in base
    out = session.handle_text(json.dumps({"type": "control", "latency_preset": "speed"}))
    assert session.latency_preset == "speed" and not [m for m in out if m.get("type") == "error"]
    assert session.active_spec.endswith(', preset="speed")')
    bad = session.handle_text(json.dumps({"type": "control", "latency_preset": "warp"}))
    assert any(m.get("type") == "error" for m in bad) and session.latency_preset == "speed"


def test_latency_preset_balanced_keeps_the_direct_enrolled_gpu_spec(tmp_path):
    s = session_with(tmp_path, pipelines={"focus_hq":
        'tse(candidate="enrolled", window_s=0.5, hop_s=0.25)>mixer'})
    s.handle_text(json.dumps({"type": "control", "mode": "focus_hq",
                             "latency_preset": "accuracy"}))
    assert "window_s=1.5, hop_s=0.75" in s.active_spec
    s.handle_text(json.dumps({"type": "control", "latency_preset": "balanced"}))
    # T-078: balanced is the configured GPU spec (0.5 s / 0.25 s -> 750 ms), not 1.0 / 0.5.
    assert s.active_spec == 'tse(candidate="enrolled", window_s=0.5, hop_s=0.25)>mixer'
