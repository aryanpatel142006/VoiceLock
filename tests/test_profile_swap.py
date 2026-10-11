"""T-155: a saved-profile swap (/judge keys 1/2) keeps the Enrolled tier with two same-gender faces.

Live Oct 9 (team preset, Aryan + Joseph, both male): the session-start pair set ``voice_profile.other``,
which exempted the saved pair from the same-gender guard; a key 1/2 swap loads every other profile as
the negative but left ``other`` out, so the guard fired and Focus HQ sat on the gate for 40 s. This
replays a session with two same-bucket faces and scripted swap controls on the audio clock (every
window and preparation job finished before the next 20 ms frame, so the numbers do not move with load).
"""

import json

import numpy as np
import pytest

from hearing.audio.io import float_to_pcm16, save
from hearing.fusion.tiers import TierLadder
from hearing.server.protocol import KIND_AUDIO_IN, pack_frame
from hearing.server.session import ServerConfig, Session, load_stage_modules
from hearing.synth.voice_match import SAME_BUCKET_KEY
from hearing.tse import stage as tse_stage
from hearing.tse.enroll import (
    PROFILE_TARGET_ENV,
    PROFILES_ENV,
    load_profile,
    profile_clip,
)

SR = 16000
FRAME = 320  # 20 ms, as /judge sends
SWAP_BUDGET_S = 1.5  # acceptance: a swap plays the new profile in the Enrolled tier within 1.5 s

# The judge ladder's policy (upgrade 0.5 s, at most one promotion per 3 s) and the Mac enrolled windows.
LADDER = """
crossfade_ms: 150
upgrade_after_s: 0.5
min_switch_s: 3.0
downgrade_after_s: 10.0
tiers:
  - name: enrolled
    pipeline: 'tse(candidate="enrolled", window_s=1.0, hop_s=0.5, threaded=true, residual=false)'
    needs: [healthy]
    avoid_when: [same_bucket]
  - name: gate
    pipeline: passthrough
"""


class _ProfileEnrolled:
    """Stands in for the enrolled candidate: prepares (enroll, other) and records which clip each
    window was separated with."""

    name = "t155_enrolled"
    licence = "test"
    needs = frozenset({"enroll", "other"})
    device = "cpu"

    def __init__(self):
        self.used: list[int] = []  # len(enroll) per extracted window

    def load(self):
        pass

    def prepare(self, enroll, other, sr):
        pass

    def extract(self, mix, sr, enroll=None, **kwargs):
        self.used.append(len(enroll))
        return 0.5 * np.asarray(mix, np.float32)


@pytest.fixture
def session(tmp_path, monkeypatch):
    for name, seconds, seed in (("aryan", 6.0, 1), ("joseph", 7.0, 2)):  # lengths tell the clips apart
        (tmp_path / name).mkdir()
        save(tmp_path / name / "a.wav",
             (0.1 * np.random.default_rng(seed).standard_normal(round(seconds * SR))).astype(np.float32), SR)
    monkeypatch.setenv(PROFILES_ENV, str(tmp_path))
    monkeypatch.delenv(PROFILE_TARGET_ENV, raising=False)
    for var, value in (("HEARING_DEVICE", "cpu"), ("HEARING_ASR", "mock"), ("HEARING_TTS", "mock"),
                       ("HEARING_VOICE_MATCH", "mock")):
        monkeypatch.setenv(var, value)
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)
    cand = _ProfileEnrolled()
    real = tse_stage.get_candidate
    monkeypatch.setattr(tse_stage, "get_candidate", lambda name, **kw: cand if name == "enrolled" else real(name, **kw))
    load_stage_modules()
    path = tmp_path / "tiers.yaml"
    path.write_text(LADDER)
    s = Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": f'tiers(config="{path}")'},
                             synth_feed="mixture", faces="mock", vision_threaded=False), "cpu")
    s.vision.poll = lambda ctx, timeout=0.0: []
    # Two male faces on camera (Aryan and Joseph): the guard is on and sees a same-gender pair.
    s.same_bucket.publish = lambda ctx: ctx.telemetry.extras.__setitem__(SAME_BUCKET_KEY, True)
    s.cand = cand
    s.clip_len = {n: len(profile_clip(load_profile(n, tmp_path))) for n in ("aryan", "joseph")}
    yield s
    s.close()


def _control(s: Session, **fields) -> list[dict]:
    out = s.handle_text(json.dumps({"type": "control", **fields}))
    assert [m for m in out if m.get("type") == "error"] == []
    return out


def _ladder(s: Session) -> TierLadder:
    return next(st for st in s._built[s._active_spec].stages if isinstance(st, TierLadder))


def _run(s: Session, seconds: float, frame: int, swaps: dict[int, str]) -> list[tuple[float, str, int | None]]:
    """Feed ``seconds`` of 20 ms frames from ``frame``, sending ``swaps[frame]`` (a profile name) before
    that frame; ``(audio time, playing tier, enroll length of the last window)`` after every frame."""
    x = float_to_pcm16(np.full(FRAME, 0.25, np.float32))
    trace = []
    for i in range(frame, frame + round(seconds * SR / FRAME)):
        if i in swaps:
            _control(s, enroll={"action": "profile", "name": swaps[i]})
        s.handle_binary(pack_frame(KIND_AUDIO_IN, i, i * FRAME / SR, x))
        tse = _ladder(s).built["enrolled"].pipeline.stages[0]
        if tse._pool is not None:
            tse._pool.submit(lambda: None).result()  # one worker: every queued window/preparation is done
        trace.append(((i + 1) * FRAME / SR, s.ctx.telemetry.tier, s.cand.used[-1] if s.cand.used else None))
    return trace


def _start(s: Session) -> None:
    """The judge session start: Aryan with Joseph as the known negative (the team preset pair)."""
    _control(s, mode="focus_hq")
    _control(s, enroll={"action": "profile", "name": "aryan", "other": "joseph"})
    s.handle_text(json.dumps({"type": "hello", "sample_rate": SR, "frame_ms": 20}))  # preloads both pairs


def _reached(trace, t_swap: float, clip_len: int) -> float | None:
    """Seconds from ``t_swap`` until the Enrolled tier plays windows separated with ``clip_len``'s clip
    and keeps doing so to the end of ``trace``; None if it never settles there."""
    after = [(t, tier, used) for t, tier, used in trace if t > t_swap]
    good = [tier == "enrolled" and used == clip_len for _, tier, used in after]
    if not good or not good[-1]:
        return None
    i = len(good) - 1
    while i > 0 and good[i - 1]:
        i -= 1
    return after[i][0] - t_swap


@pytest.mark.parametrize("run", range(5))
def test_swap_at_20s_reaches_enrolled_with_the_new_profile_within_1_5_s(session, run):
    s = session
    _start(s)
    frames = 1000 + 7 * run  # swap at 20 s, shifted 0-0.56 s against the 0.5 s hop grid
    trace = _run(s, frames * FRAME / SR, 0, {})
    assert trace[-1][1:] == ("enrolled", s.clip_len["aryan"])  # the pair exempts the same-gender guard
    t_swap = trace[-1][0]
    trace = _run(s, 5.0, frames, {frames: "joseph"})
    assert s.enroller.profile == "joseph" and s.ctx.telemetry.extras[SAME_BUCKET_KEY] is False
    assert all(tier == "enrolled" for _, tier, _ in trace)  # never drops to the gate
    took = _reached(trace, t_swap, s.clip_len["joseph"])
    assert took is not None and took <= SWAP_BUDGET_S, took


def test_three_rapid_swaps_end_enrolled_with_the_last_profile(session):
    s = session
    _start(s)
    _run(s, 20.0, 0, {})
    swaps = {1000: "joseph", 1150: "aryan", 1400: "joseph"}  # 20 s, 23 s, 28 s: three swaps in 10 s
    trace = _run(s, 12.0, 1000, swaps)
    assert all(tier == "enrolled" for _, tier, _ in trace)
    took = _reached(trace, 1400 * FRAME / SR, s.clip_len["joseph"])
    assert took is not None and took <= SWAP_BUDGET_S, took
    assert s.enroller.profile == "joseph"


def test_guard_still_applies_without_a_saved_negative(session):
    """A profile with no other saved voice (or an auto/Name Call clip) keeps the same-gender guard."""
    s = session
    s.enroller.enroll_profile("aryan", load_profile("aryan"), None)
    s._poll_vision()
    assert s.ctx.telemetry.extras[SAME_BUCKET_KEY] is True
    s.enroller.enroll_profile("aryan", load_profile("aryan"), load_profile("joseph"))
    s._poll_vision()
    assert s.ctx.telemetry.extras[SAME_BUCKET_KEY] is False
