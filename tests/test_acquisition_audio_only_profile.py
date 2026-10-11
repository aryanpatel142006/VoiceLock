"""T-329: under the audio_only team preset the saved-voice lock is the acquisition, not a face."""

import numpy as np
import pytest

from hearing.registry import Pipeline
from hearing.server.session import ServerConfig, Session
from hearing.tse.stage import FocusHQ
from hearing.types import FaceTrack


class _EnrolledCandidate:
    """Offline extractor boundary; the enroller and the session's acquisition stay real."""

    name = "test_enrolled"
    needs = frozenset({"enroll"})
    licence = "test-only"

    def load(self):
        pass

    def extract(self, mix, sr, **kwargs):
        return mix.copy()


class _Ladder:
    """The ladder's ``status()`` shape, with the Enrolled tier's eligibility reason under test."""

    active = "gate"

    def __init__(self, reason: str):
        self.reason = reason

    def status(self) -> dict:
        return {"active": self.active, "tiers": [{"name": "enrolled", "eligible": not self.reason,
                                                   "reason": self.reason}]}


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    s = Session(ServerConfig(pipelines={"natural": "passthrough"}, faces="mock"), "cpu")
    stage = FocusHQ(_candidate=_EnrolledCandidate(), threaded=False)
    s._built[s.active_spec] = Pipeline(s.active_spec, (stage,))
    yield s, stage
    s.close()


def _profile(stage):
    stage.enroller.enroll_profile("aryan", [np.full(16000, 0.1, np.float32)] * 3)


def test_audio_only_profile_loads_then_locks_with_the_name(session, monkeypatch):
    s, stage = session
    s.preset = {"audio_only": True}
    assert s.telemetry()["acquisition"]["detail"] == "looking for a speaking face"  # no profile yet
    _profile(stage)
    ladder = _Ladder("loading")
    monkeypatch.setattr("hearing.server.session._ladder", lambda pipeline: ladder)
    s._pipeline_tier = "gate"
    for reason in ("", "loading", "warming up"):  # the ~4 s cold start: truthful, never "look"
        ladder.reason = reason
        a = s.telemetry()["acquisition"]
        assert (a["state"], a["detail"]) == ("acquiring", "loading saved voice: aryan")
        assert not s._isolated
    ladder.reason = "overrun"  # held out by the ladder: not isolating, says why
    a = s.telemetry()["acquisition"]
    assert (a["state"], a["detail"]) == ("acquiring", "saved voice aryan not playing: overrun")
    s._pipeline_tier = "enrolled"  # Enrolled plays the saved voice: the lock, with no face target
    a = s.telemetry()["acquisition"]
    assert s.ctx.target.track_id is None
    assert (a["state"], a["detail"]) == ("locked", "known voice: aryan")
    assert s._isolated
    s._pipeline_tier = "gate"  # demoted: Isolated is no longer claimed
    ladder.reason = ""
    assert s.telemetry()["acquisition"]["state"] == "acquiring"


def test_audio_only_without_a_ladder_reads_loading(session):
    s, stage = session
    s.preset = {"audio_only": True}
    _profile(stage)
    a = s.telemetry()["acquisition"]
    assert (a["state"], a["detail"]) == ("acquiring", "loading saved voice: aryan")


def test_face_path_unchanged_without_audio_only(session):
    """A camera session with a saved voice and no face still asks for the person (T-107 / T-125 path)."""
    s, stage = session
    _profile(stage)
    s._pipeline_tier = "enrolled"
    assert not s.audio_only
    a = s.telemetry()["acquisition"]
    assert (a["state"], a["detail"]) == ("searching", "looking for a speaking face")
    selector = s.vision.selector
    face = FaceTrack(1, (0.4, 0.4, 0.2, 0.2), None, 0.9, 10.0)
    s.ctx.tracks = [face]
    s.ctx.target = selector.update([face], 10.0)
    s.ctx.target = selector.update([face], 10.3)
    assert s.ctx.target.track_id == 1
    stage.enroller.track_id = 1
    a = s.telemetry()["acquisition"]
    assert (a["state"], a["detail"]) == ("locked", "known voice: aryan")
