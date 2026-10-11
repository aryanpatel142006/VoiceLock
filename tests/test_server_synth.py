"""Session integration of Synthetic/Text modes: synth stage, captions and voice matching.

Engines are the offline mocks (``HEARING_ASR/TTS=mock``, ``HEARING_VOICE_MATCH=mock``):
the mock ASR reveals one word per 0.4 s of voiced audio and ends a phrase on 0.4 s of
silence; the mock TTS speaks one tone per word at the bucket's ``mock_hz`` from
``config/voices.yaml`` (neutral 170 Hz, female_adult 220 Hz, male_adult 120 Hz).
"""

import json
import time
from itertools import pairwise

import numpy as np
import pytest
from conftest import scaled
from fastapi.testclient import TestClient

from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.server.app import create_app
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    pack_frame,
    unpack_frame,
)
from hearing.server.session import ServerConfig, Session
from hearing.synth.tts import resolve_voice
from hearing.types import TargetState

SR = 16000
FRAME = 320  # 20 ms client frames
LONG_TRANSCRIPT = " ".join(f"w{i}" for i in range(40))


@pytest.fixture(autouse=True)
def _mock_engines(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_ASR", "mock")
    monkeypatch.setenv("HEARING_TTS", "mock")
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    monkeypatch.setenv("HEARING_SYNTH_FEED", "mixture")  # the mock ASR reads tones, not enhanced speech
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)


@pytest.fixture
def long_transcript(monkeypatch):
    monkeypatch.setattr("hearing.synth.asr.DEFAULT_TRANSCRIPT", LONG_TRANSCRIPT)


def _hz(bucket: str) -> float:
    return float(resolve_voice(bucket, "mock"))


def _voiced(seconds: float, hz: float = 300.0) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return pcm16_to_float(float_to_pcm16((0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)))


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SR), np.float32)


def _segments(n: int, voiced_s: float = 1.2, pause_s: float = 0.6) -> np.ndarray:
    """``n`` × (voiced, pause): one 3-word phrase per segment with the mock ASR."""
    return np.concatenate([np.concatenate([_voiced(voiced_s), _silence(pause_s)]) for _ in range(n)])


class _Feeder:
    """Drives a :class:`Session` frame by frame, letting the TTS worker catch up after each
    frame so output is deterministic; collects audio out, captions and errors."""

    def __init__(self, session: Session):
        self.session = session
        self.seq = 0
        self.audio: list[np.ndarray] = []  # one array per frame, in order
        self.captions: list[dict] = []
        self.errors: list[dict] = []

    @property
    def ts(self) -> float:
        return self.seq * FRAME / SR

    def feed(self, x: np.ndarray) -> list[dict]:
        captions = []
        for i in range(0, len(x) - FRAME + 1, FRAME):
            data = pack_frame(KIND_AUDIO_IN, self.seq, self.ts, float_to_pcm16(x[i : i + FRAME]))
            for reply in self.session.handle_binary(data):
                if isinstance(reply, bytes):
                    frame = unpack_frame(reply)
                    assert frame.kind == KIND_AUDIO_OUT and frame.seq == self.seq
                    self.audio.append(pcm16_to_float(frame.payload))
                elif reply["type"] == "caption":
                    captions.append(reply)
                elif reply["type"] == "error":
                    self.errors.append(reply)
            self.seq += 1
            self.errors += self.session.wait_voice(2.0)
            synth = self.session._synth()
            if synth is not None:
                assert synth.wait_idle(scaled(2.0))
        self.captions += captions
        return captions

    def control(self, **fields) -> list[dict]:
        return self.session.handle_text(json.dumps({"type": "control", **fields}))

    def output(self, start_frame: int = 0) -> np.ndarray:
        return np.concatenate(self.audio[start_frame:])


def _session(**config) -> Session:
    return Session(ServerConfig(**config), device="cpu")


def _pitch(y: np.ndarray) -> float:
    """Dominant frequency of the non-silent samples (the mock TTS is one tone per voice)."""
    y = y[np.abs(y) > 1e-3]
    assert len(y) > SR // 10, "no synthesized audio in this segment"
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.argmax(spectrum)) * SR / len(y)


def _phrase_pitches(feeder: _Feeder, start_frame: int) -> list[float]:
    """Pitch of every spoken phrase, split at each phrase's first output sample."""
    out = feeder.output(start_frame)
    starts = [r.out_start for r in feeder.session._synth().phrases()]
    bounds = [*starts, len(out)]
    return [_pitch(out[a:b]) for a, b in pairwise(bounds)]


# ---------------------------------------------------------------- websocket


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(ServerConfig(console_dir=tmp_path)))


def _ws_stream(ws, x: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, dict]]]:
    """Send 20 ms frames one at a time; returns per-frame audio and (frame index, caption)
    pairs where the index is the frame whose processing produced the caption."""

    audio, captions = [], []
    n = len(x) // FRAME

    def read_until_audio(seq):
        while True:
            raw = ws.receive()
            if raw.get("bytes") is not None:
                frame = unpack_frame(raw["bytes"])
                if frame.kind == KIND_AUDIO_OUT and frame.seq == seq:
                    audio.append(pcm16_to_float(frame.payload))
                    return
            else:
                msg = json.loads(raw["text"])
                if msg["type"] == "caption":
                    captions.append((seq - 1, msg))  # trails the previous frame's audio
                assert msg["type"] != "error", msg

    for seq in range(n):
        ws.send_bytes(pack_frame(KIND_AUDIO_IN, seq, seq * FRAME / SR,
                                 float_to_pcm16(x[seq * FRAME : (seq + 1) * FRAME])))
        read_until_audio(seq)
        if any(c["final"] for _, c in captions):
            time.sleep(0.005)  # realtime-ish pacing: the TTS worker runs beside the stream
    ws.send_text(json.dumps({"type": "control"}))  # flush: captions of the last frame
    while True:
        msg = json.loads(ws.receive()["text"])
        if msg["type"] == "caption":
            captions.append((n - 1, msg))
        if msg["type"] == "telemetry":
            return audio, captions


def test_synthetic_mode_outputs_tts_only_after_the_first_final_caption(tmp_path):
    x = np.concatenate([_voiced(1.2), _silence(2.0)])
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "control", "mode": "synthetic", "synth_feed": "mixture"}))
        assert json.loads(ws.receive()["text"])["pipeline"] == "passthrough>synth(speak=true)"
        audio, captions = _ws_stream(ws, x)
    assert captions and all(set(c) == {"type", "text", "final", "t0", "t1"} for _, c in captions)
    first_final = next(i for i, c in captions if c["final"])
    assert captions[0][1]["text"] == "See"
    assert next(c for _, c in captions if c["final"])["text"] == "See you at"
    assert not np.concatenate(audio[: first_final + 1]).any()  # no room audio, ever
    assert np.concatenate(audio[first_final + 1 :]).any()  # the phrase is spoken


def test_text_mode_outputs_silence_and_the_same_captions(tmp_path):
    x = np.concatenate([_voiced(1.2), _silence(2.0)])
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "control", "mode": "text", "synth_feed": "mixture"}))
        assert json.loads(ws.receive()["text"])["pipeline"] == "passthrough>synth(speak=false)"
        audio, captions = _ws_stream(ws, x)
    assert not np.concatenate(audio).any()
    finals = [c for _, c in captions if c["final"]]
    assert [c["text"] for c in finals] == ["See you at"]
    assert set(finals[0]) == {"type", "text", "final", "t0", "t1"}
    assert 0.0 <= finals[0]["t0"] < finals[0]["t1"] <= 1.3


def test_unknown_voice_override_is_an_error_and_keeps_the_voice(tmp_path):
    with _client(tmp_path).websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "control", "voice_override": "female_adult"}))
        before = json.loads(ws.receive()["text"])["voice"]
        ws.send_text(json.dumps({"type": "control", "voice_override": "nonsense"}))
        err = json.loads(ws.receive()["text"])
        after = json.loads(ws.receive()["text"])
    assert err["type"] == "error" and err["stage"] == "voice"
    assert "nonsense" in err["message"]
    assert after["type"] == "telemetry"
    assert before == after["voice"] == {"bucket": "female_adult", "source": "override",
                                        "override": "female_adult"}


def test_health_lists_synth_ready(tmp_path):
    stages = _client(tmp_path).get("/health").json()["stages"]
    assert stages["synth"] == "ready"


# ---------------------------------------------------------------- session


def test_voice_override_switches_the_spoken_voice_for_the_next_phrase():
    f = _Feeder(_session())
    f.control(mode="synthetic")
    f.feed(_segments(1))
    t = f.control(voice_override="female_adult")[-1]
    assert t["voice"] == {"bucket": "female_adult", "source": "override", "override": "female_adult"}
    f.feed(_segments(1, pause_s=1.5))
    pitches = _phrase_pitches(f, 0)
    assert len(pitches) == 2
    assert pitches[0] == pytest.approx(_hz("neutral"), abs=5)  # no target: neutral
    assert pitches[1] == pytest.approx(_hz("female_adult"), abs=5)
    t = f.session.telemetry()
    assert t["latency_ms"]["synth_phrase_p50"] > 0
    assert "see" not in json.dumps(t)  # caption text never rides on telemetry


def test_target_change_switches_voice_bucket_and_pitch(long_transcript):
    f = _Feeder(_session())
    f.control(mode="synthetic")
    f.session.ctx.target = TargetState(1, 0.9, False, 0.0)
    f.feed(_segments(3))  # >= 3 s of voiced target audio -> one voice estimate for track 1
    one = f.session.telemetry()["voice"]
    assert one == {"bucket": "male_adult", "source": "voice", "override": None}
    f.session.ctx.target = TargetState(2, 0.9, False, f.ts)  # e.g. the vision task retargets
    two = f.session.telemetry()["voice"]
    assert two == {"bucket": "neutral", "source": "default", "override": None}  # nothing on 2 yet
    f.feed(_segments(1, pause_s=1.5))
    pitches = _phrase_pitches(f, 0)
    assert len(pitches) == 4
    assert pitches[2] == pytest.approx(_hz("male_adult"), abs=5)
    assert pitches[3] == pytest.approx(_hz("neutral"), abs=5)


def test_face_estimates_pick_the_voice_of_the_target_track_only():
    session = _session()
    image = np.zeros((360, 640, 3), np.uint8)
    assert session.on_face(5, image, (0.4, 0.3, 0.2, 0.3)) == []
    assert session.telemetry()["voice"]["source"] == "default"  # 5 is not the target
    session.ctx.target = TargetState(5, 0.8, False, 0.0)
    assert session.telemetry()["voice"] == {"bucket": "male_adult", "source": "face",
                                            "override": None}
    session.handle_text(json.dumps({"type": "control", "voice_override": "neutral"}))
    assert session.telemetry()["voice"] == {"bucket": "neutral", "source": "override",
                                            "override": "neutral"}


def test_no_voice_estimators_means_neutral_default(monkeypatch):
    monkeypatch.delenv("HEARING_VOICE_MATCH")
    monkeypatch.setattr("hearing.synth.voice_match.get_face_estimator", lambda: None)
    monkeypatch.setattr("hearing.synth.voice_match.get_voice_estimator", lambda: None)
    f = _Feeder(_session())
    f.control(mode="synthetic")
    f.session.ctx.target = TargetState(1, 0.9, False, 0.0)
    f.feed(_segments(1))
    assert f.session.on_face(1, np.zeros((360, 640, 3), np.uint8), (0.4, 0.3, 0.2, 0.3)) == []
    assert f.session.telemetry()["voice"] == {"bucket": "neutral", "source": "default",
                                              "override": None}


def test_mode_switch_drops_stale_phrases_and_captions():
    f = _Feeder(_session())
    f.control(mode="synthetic")
    f.feed(np.concatenate([_voiced(1.2), _silence(0.42)]))  # final: phrase queued and buffered
    f.feed(_voiced(0.8))  # the next phrase is in progress
    assert [c["text"] for c in f.captions if c["final"]] == ["See you at"]
    f.control(mode="natural")
    f.feed(_silence(0.3))
    f.control(mode="synthetic")
    back = len(f.audio)
    stale = f.feed(_silence(1.5))
    assert stale == []  # neither the in-progress phrase nor anything older comes back
    assert not f.output(back).any()  # the buffered phrase was dropped, not played late
    fresh = f.feed(np.concatenate([_voiced(0.8), _silence(0.5)]))
    assert fresh and all(c["t0"] >= back * FRAME / SR for c in fresh)


def test_synthetic_never_degrades_to_room_audio():
    f = _Feeder(_session(pipelines={"natural": "no_such_stage"}))
    f.control(mode="synthetic", synth_feed="natural")
    f.feed(_voiced(0.4))
    assert [e["degraded_to"] for e in f.errors] == ["synth(speak=true)"]
    assert not f.output().any()
    assert f.session.telemetry()["pipeline"] == "synth(speak=true)"


def test_config_engines_shape_the_synth_pipelines_and_the_mixture_stays_raw():
    config = ServerConfig(pipelines={"natural": "gain(db=-6)"}, asr="mock", tts="mock")
    assert config.pipelines["synthetic"] == 'passthrough>synth(speak=true,asr="mock",tts="mock")'
    assert config.pipelines["text"] == 'passthrough>synth(speak=false,asr="mock",tts="mock")'
    assert config.pipelines["quiet"] == "silence"
    explicit = ServerConfig(pipelines={"synthetic": "silence"})
    assert explicit.pipelines["synthetic"] == "silence"


def test_close_stops_the_tts_worker():
    session = _session()
    session.handle_text(json.dumps({"type": "control", "mode": "synthetic"}))
    session.handle_binary(pack_frame(KIND_AUDIO_IN, 0, 0.0, float_to_pcm16(_voiced(0.02))))
    synth = session._synth()
    assert synth._thread is not None and synth._thread.is_alive()
    thread = synth._thread
    session.close()
    assert not thread.is_alive()
