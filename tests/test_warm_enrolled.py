"""T-339: HEARING_WARM_ENROLLED warms the enrolled tier at server start; unset keeps today's behaviour."""

import time

import numpy as np
import pytest
import soundfile as sf

from hearing.server import warm
from hearing.tse import enrolled
from hearing.tse.blind import BlindSeparator
from hearing.tse.blind import MockBackend as SeparatorMock
from hearing.tse.enroll import profile_clip
from hearing.tse.enrolled import WARM_ENV, EnrolledCandidate, MockBackend
from hearing.tse.stage import FocusHQ
from hearing.types import AudioChunk, StageContext

SR = 16000
HOP = 160


class Instant:
    """Candidate whose load and extract cost nothing; counts loads."""

    name = "instant"
    licence = "test"
    needs = frozenset()
    device = "cpu"

    def __init__(self):
        self.loads = 0

    def load(self):
        self.loads += 1

    def extract(self, mix, sr, **_):
        return 0.5 * mix


class CountingEmbedder(MockBackend):
    def __init__(self):
        self.calls = 0

    def embed(self, audio):
        self.calls += 1
        return super().embed(audio)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(WARM_ENV, raising=False)
    monkeypatch.setattr(enrolled, "WARM_PROFILES", {})


def _run(stage, seconds):
    """Feed ``seconds`` of noise in 10 ms chunks; returns the health after each chunk."""
    x = (0.1 * np.random.default_rng(0).standard_normal(round(seconds * SR))).astype(np.float32)
    ctx = StageContext.default()
    seen = []
    for i in range(0, len(x) - HOP + 1, HOP):
        stage.process(AudioChunk(x[i : i + HOP], SR, i // HOP, i / SR), ctx)
        if stage._load is not None:  # the load waits on TORCH_LOCK, which other threads may hold
            stage._load.result(timeout=5.0)
        time.sleep(0.001)
        seen.append(dict(ctx.telemetry.extras["tse"]))
    return seen


def test_knob_off_loads_only_when_the_first_window_is_due():
    cand = Instant()
    st = FocusHQ(_candidate=cand, window_s=1.0, hop_s=0.5, threaded=True)
    assert st.eager_load is False
    seen = _run(st, 0.9)  # no window due yet: nothing loads (today's behaviour)
    assert cand.loads == 0 and st._load is None
    seen += _run(st, 1.1)  # 2.0 s in: window 0 was skipped as loading, 4 clean windows needed to outlive it
    st.close()
    assert cand.loads == 1
    assert seen[-1]["ok"] is False and seen[-1]["reason"] == "loading"


def test_knob_on_loads_with_the_first_chunk_and_is_healthy_after_window_0(monkeypatch):
    monkeypatch.setenv(WARM_ENV, "1")
    cand = Instant()
    st = FocusHQ(_candidate=cand, window_s=1.0, hop_s=0.5, threaded=True)
    assert st.eager_load is True
    _run(st, 0.05)
    assert cand.loads == 1 and st.loaded
    seen = _run(st, 1.55)  # window 0 (due at 1.0 s) is computed, not skipped
    st.close()
    assert seen[-1]["ok"] is True and seen[-1]["reason"] == ""
    assert seen[-1]["windows"] >= 1 and seen[-1]["overruns"] == 0


def test_knob_on_never_loads_an_inline_stage_early(monkeypatch):
    monkeypatch.setenv(WARM_ENV, "1")
    cand = Instant()
    st = FocusHQ(_candidate=cand, window_s=1.0, hop_s=0.5, threaded=False)
    assert st.eager_load is False
    _run(st, 0.5)
    assert cand.loads == 0


def _profile_pair():
    rng = np.random.default_rng(3)
    target = profile_clip([(0.1 * rng.standard_normal(SR)).astype(np.float32) for _ in range(3)])
    other = profile_clip([(0.1 * rng.standard_normal(SR)).astype(np.float32) for _ in range(3)])
    return target, other


def _candidate(embedder):
    return EnrolledCandidate(BlindSeparator(SeparatorMock()), embedder, clean_enroll=False)


def test_knob_off_every_candidate_embeds_the_profiles_itself():
    target, other = _profile_pair()
    emb = CountingEmbedder()
    _candidate(emb).prepare(target, other)
    first = emb.calls
    _candidate(emb).prepare(target, other)
    assert first == 6 and emb.calls == 2 * first
    assert enrolled.WARM_PROFILES == {}


def test_knob_on_a_second_candidate_reuses_the_warmed_embeddings(monkeypatch):
    monkeypatch.setenv(WARM_ENV, "1")
    target, other = _profile_pair()
    emb = CountingEmbedder()
    warmed = _candidate(emb)
    warmed.prepare(target, other)
    assert emb.calls == 6 and len(enrolled.WARM_PROFILES) == 2
    session = _candidate(emb)
    session.prepare(target, other)
    assert emb.calls == 6  # cache hit: no ECAPA job in the session
    np.testing.assert_array_equal(session.target_embedding(target), warmed.target_embedding(target))
    assert session._closed_set() is not None  # the profile pieces came along: the LDA pick still applies
    other_encoder = CountingEmbedder()
    _candidate(other_encoder).prepare(target, other)
    assert other_encoder.calls == 6  # a different encoder never reads another encoder's embeddings


def test_identical_picks_with_and_without_the_knob(monkeypatch):
    target, other = _profile_pair()
    mix = (0.1 * np.random.default_rng(9).standard_normal(SR)).astype(np.float32)
    y_off = _candidate(MockBackend()).extract(mix, SR, enroll=target, other=other)
    monkeypatch.setenv(WARM_ENV, "1")
    emb = MockBackend()
    _candidate(emb).prepare(target, other)
    y_on = _candidate(emb).extract(mix, SR, enroll=target, other=other)
    np.testing.assert_array_equal(y_on, y_off)


def test_warm_enrolled_skips_without_the_knob():
    assert warm.warm_enrolled().startswith("skipped: HEARING_WARM_ENROLLED")


def test_warm_enrolled_embeds_every_saved_profile(monkeypatch, tmp_path):
    from hearing.tse import base

    rng = np.random.default_rng(1)
    for name in ("aryan", "joseph"):
        (tmp_path / name).mkdir()
        sf.write(tmp_path / name / "a.wav", (0.1 * rng.standard_normal(3 * SR)).astype(np.float32), SR)
    monkeypatch.setenv("HEARING_PROFILES_DIR", str(tmp_path))
    monkeypatch.setenv(WARM_ENV, "1")
    monkeypatch.setattr(enrolled, "available", lambda: True)
    emb = MockBackend()
    monkeypatch.setattr(base, "get_candidate", lambda name, **kw: _candidate(emb))
    done = warm.warm_enrolled()
    assert done.startswith("ok") and done.endswith("(2 profiles)")
    assert {k[0] for k in enrolled.WARM_PROFILES} == {id(emb)} and len(enrolled.WARM_PROFILES) == 2


def test_the_start_up_runs_the_profile_warm_up_only_with_the_knob(monkeypatch):
    calls = []
    monkeypatch.setattr(warm, "warm", dict)
    monkeypatch.setattr(warm, "warm_inference", lambda: "ok")
    monkeypatch.setattr(warm, "warm_enrolled", lambda: calls.append(1) or "ok")
    assert warm.warm_all() == {"enrolled_inference": "ok"} and calls == []
    monkeypatch.setenv(WARM_ENV, "1")
    assert warm.warm_all() == {"enrolled_inference": "ok", "enrolled_profiles": "ok"} and calls == [1]
