import numpy as np


def test_stage_is_structural_and_runtime_checkable():
    from hearing.types import AudioChunk, Stage, StageContext

    class Passthrough:
        name = "passthrough"
        sample_rate = 16000
        latency_ms = 0.0

        def process(self, chunk, ctx):
            return chunk

        def reset(self):
            pass

    stage = Passthrough()
    assert isinstance(stage, Stage)
    assert not isinstance(object(), Stage)
    chunk = AudioChunk(np.zeros(160, dtype=np.float32), 16000, 7, 1.5)
    assert stage.process(chunk, StageContext.default()) is chunk


def test_default_context_has_independent_mutable_state():
    from hearing.types import StageContext

    first, second = StageContext.default(), StageContext.default()
    assert first.mode == "natural"
    assert first.suppression == 0.5
    assert first.target.track_id is None
    assert first.target.confidence == 0.0
    assert not first.target.locked
    assert first.target.since_ts == 0.0
    assert first.tracks == []
    assert first.telemetry.seq == 0
    assert first.telemetry.tier == ""
    assert first.telemetry.device == ""
    first.target.locked = True
    first.tracks.append(None)
    first.telemetry.latency_ms["test"] = 1.0
    first.telemetry.extras["test"] = True
    assert not second.target.locked
    assert second.tracks == []
    assert second.telemetry.latency_ms == {}
    assert second.telemetry.extras == {}
