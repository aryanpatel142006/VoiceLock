"""Acquisition reflects real selector dwell, enrollment and played-tier changes."""

import numpy as np
import pytest

from hearing.registry import Pipeline
from hearing.server.session import ServerConfig, Session
from hearing.tse.stage import FocusHQ
from hearing.types import FaceTrack


class _EnrolledCandidate:
    """Offline extractor boundary; enrollment and selection remain real."""

    name = "test_enrolled"
    needs = frozenset({"enroll"})
    licence = "test-only"

    def load(self):
        pass

    def extract(self, mix, sr, **kwargs):
        return mix.copy()


def test_scripted_acquisition_scene(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    clock = [10.0]
    monkeypatch.setattr("hearing.server.session.time.monotonic", lambda: clock[0])
    s = Session(ServerConfig(pipelines={"natural": "passthrough"},
                             faces="mock"), "cpu")
    stage = FocusHQ(_candidate=_EnrolledCandidate(), threaded=False)
    selector = s.vision.selector
    selector.set_attention(hold_s=1.0)  # scene timings below assume a 1 s hold
    need_s = stage.enroller.need / stage.enroller.sr

    def scene(ts, tracks):
        clock[0] = ts
        s.ctx.tracks = tracks
        s.ctx.target = selector.update(tracks, ts)
        return s.telemetry()["acquisition"]

    def face(track_id=1, x=0.4):
        return FaceTrack(track_id, (x, 0.4, 0.2, 0.2), None, 0.9, clock[0])

    try:
        assert scene(10, []) == {"state": "searching", "since_s": 0.0,
                                  "detail": "looking for a speaking face"}
        assert scene(10.1, [face()])["state"] == "candidate"
        assert scene(10.3, [face()])["state"] == "locked-visual"
        spec = s.active_spec
        s._built[spec] = Pipeline(spec, (stage,))
        stage.enroller.update(np.full(24000, 0.1, np.float32), s.ctx)
        a = scene(10.4, [face()])
        assert a["state"] == "acquiring"
        assert a["detail"] == f"enrolling 1.5/{need_s:.1f} s"
        tracks = s.ctx.tracks
        s.ctx.tracks = [face(), face(2, x=0.7)]  # both talk: the 10 ms chunks are rejected
        for _ in range(3):
            assert not stage.enroller.update(np.full(160, 0.1, np.float32), s.ctx)
        s.ctx.tracks = tracks
        assert scene(10.4, [face()])["detail"] == f"enrolling 1.5/{need_s:.1f} s · need them to talk alone"
        assert scene(10.9, [face()])["since_s"] == pytest.approx(0.5)
        s._pipeline_tier = "focus_hq"
        assert scene(10.95, [face()])["state"] == "locked"
        s._pipeline_tier = "gate"
        stage.enroller.update(np.full(19200, 0.1, np.float32), s.ctx)
        s._pipeline_tier = "enrolled"
        assert scene(11, [face()])["state"] == "locked"
        a = scene(11.4, [])
        assert a["state"] == "reacquiring"
        assert a["detail"] == "target lost 0.4 s"
        assert scene(12.1, [])["state"] == "searching"
        assert scene(12.2, [face(2)])["state"] == "candidate"
        s._pipeline_tier = "gate"
        scene(12.4, [face(2)])
        stage.enroller.update(np.full(1600, 0.1, np.float32), s.ctx)
        assert scene(12.5, [face(2)])["state"] == "reacquiring"
        stage.enroller.update(np.full(46400, 0.1, np.float32), s.ctx)
        s._pipeline_tier = "focus_hq"
        assert scene(13, [face(2)])["state"] == "locked"
        stage.enroller.reset()
        stage.enroller.update(np.full(1600, 0.1, np.float32), s.ctx)
        s._pipeline_tier = "gate"
        assert scene(13.1, [face(2)])["state"] == "reacquiring"
        assert scene(13.2, [face(2)])["state"] == "reacquiring"
    finally:
        s.close()
