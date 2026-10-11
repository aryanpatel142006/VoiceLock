"""Focus HQ ``tse`` stage: streaming windows, lips from the feed, fallbacks, enrollment."""

import time

import numpy as np
import pytest

from hearing import registry
from hearing.fusion.mixer import RESIDUAL_KEY, Mixer
from hearing.tse import stage as tse_stage
from hearing.tse.enroll import AutoEnroller
from hearing.tse.lips import FEED_KEY, LipFeed, feed_of, lip_crop
from hearing.tse.stage import FocusHQ
from hearing.tse.windowed import SlidingWindowExtractor
from hearing.types import AudioChunk, FaceTrack, StageContext, TargetState

SR = 16000
HOP = 160


class Shaper:
    """Window-dependent nonlinear candidate: catches any misplaced or mis-weighted window."""

    name = "shaper"
    licence = "test"
    needs = frozenset()
    device = "cpu"

    def __init__(self, sleep_s: float = 0.0, needs=frozenset()):
        self.sleep_s = sleep_s
        self.needs = frozenset(needs)
        self.calls = []

    def load(self):
        pass

    def extract(self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None):
        self.calls.append({"lips": None if lips is None else lips.shape, "enroll": enroll})
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return np.tanh(3 * mix) * (1 + np.mean(np.abs(mix)))


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    return (0.3 * np.random.default_rng(seed).standard_normal(round(seconds * SR))).astype(np.float32)


def stream(stage, x, ctx=None, tail=None, sleep=0.0, before=None):
    """Feed ``x`` (plus ``tail`` s of zeros) in 10 ms chunks; returns the output aligned to x."""
    ctx = ctx or StageContext.default()
    pad = stage.delay + HOP if tail is None else round(tail * SR)
    padded = np.concatenate([x, np.zeros(pad, np.float32)])
    outs = []
    for i in range(0, len(padded) - HOP + 1, HOP):
        if before is not None:
            before(i, ctx)
        outs.append(stage.process(AudioChunk(padded[i : i + HOP], SR, i // HOP, i / SR), ctx).samples)
        if sleep:
            time.sleep(sleep)
    y = np.concatenate(outs)
    return y[stage.delay : stage.delay + len(x)], ctx


@pytest.mark.parametrize("threaded", [False, True])
def test_chunked_stream_equals_the_whole_file_extractor(threaded):
    x = noise(4.0)
    ref = SlidingWindowExtractor(Shaper(), 1.0, 0.5).extract(x, SR)
    st = FocusHQ(_candidate=Shaper(), window_s=1.0, hop_s=0.5, threaded=threaded)
    st.load()
    y, _ = stream(st, x, sleep=0.0005 if threaded else 0.0)
    st.close()
    assert st.latency_ms == (1500.0 if threaded else 1000.0)
    complete = 3 * SR  # the last full window of x ends at 3.5 s
    np.testing.assert_allclose(y[:complete], ref[:complete], atol=1e-6)


def test_identity_is_an_exact_delay_and_registers_as_a_stage():
    st = registry.build_pipeline('tse(candidate="identity", window_s=0.5, hop_s=0.25, threaded=false)')
    assert st.latency_ms == 500.0
    x = noise(2.0, seed=1)
    y, _ = stream(st.stages[0], x)
    np.testing.assert_array_equal(y, x)


def test_auto_picks_an_installed_candidate_or_identity(monkeypatch):
    monkeypatch.setattr(tse_stage, "candidate_available", lambda name: False)
    assert tse_stage.resolve_candidate("auto") == "identity"
    monkeypatch.setattr(tse_stage, "candidate_available", lambda name: name == "dolphin")
    assert tse_stage.resolve_candidate("auto") == "dolphin"
    assert tse_stage.resolve_candidate("enrolled") == "enrolled"
    assert tse_stage.AUTO_ORDER[0] == "av_mossformer2"  # the bench winner


def _target_ctx(track_id: int = 7) -> StageContext:
    ctx = StageContext.default()
    ctx.target = TargetState(track_id, 0.9, False, 0.0)
    ctx.tracks = [FaceTrack(track_id, (0.4, 0.3, 0.2, 0.4), None, 0.9, 0.0)]
    ctx.telemetry.extras[FEED_KEY] = LipFeed()
    return ctx


def _video(track_id: int, fps: float = 25.0):
    """Push one 88x88 crop per video frame of ``track_id`` as its capture time passes."""
    crop = np.full((88, 88), 90, np.uint8)
    state = {"next": 0}

    def before(i, ctx):
        while state["next"] / fps < (i + HOP) / SR:
            feed_of(ctx).append(state["next"] / fps, track_id, crop, 0.9)
            state["next"] += 1

    return before


def test_av_candidate_without_lips_plays_the_delayed_mix():
    cand = Shaper(needs={"lips"})
    st = FocusHQ(_candidate=cand, window_s=0.5, hop_s=0.25, threaded=False)
    x = noise(2.0, seed=2)
    y, ctx = stream(st, x, ctx=_target_ctx())
    # No window ran: the fallback is the input itself, faded in over 10 ms from the silent warm-up.
    np.testing.assert_array_equal(y[HOP:], x[HOP:])
    assert np.all(np.abs(y[:HOP]) <= np.abs(x[:HOP]))
    assert cand.calls == []
    health = ctx.telemetry.extras["tse"]
    assert health["ok"] is False and health["reason"] == "no lips"


def test_lips_of_the_target_reach_the_candidate_on_the_window_grid():
    cand = Shaper(needs={"lips"})
    st = FocusHQ(_candidate=cand, window_s=0.5, hop_s=0.25, threaded=False)
    x = noise(2.0, seed=3)
    y, ctx = stream(st, x, ctx=_target_ctx(7), before=_video(7), tail=0.5)
    assert cand.calls and all(c["lips"] == (13, 88, 88) for c in cand.calls)  # round(12.5) up, as frame_span
    ref = SlidingWindowExtractor(Shaper(), 0.5, 0.25).extract(x, SR)
    np.testing.assert_allclose(y[: SR], ref[: SR], atol=1e-6)
    assert ctx.telemetry.extras["tse"]["ok"] is True
    assert ctx.telemetry.extras["tse"]["lips"] == 1.0


def test_lips_of_another_face_are_ignored():
    cand = Shaper(needs={"lips"})
    st = FocusHQ(_candidate=cand, window_s=0.5, hop_s=0.25, threaded=False)
    stream(st, noise(1.5), ctx=_target_ctx(7), before=_video(8))
    assert cand.calls == []


def test_residual_feeds_the_mixer_slider():
    st = FocusHQ(_candidate=Shaper(), window_s=0.5, hop_s=0.25, threaded=False)
    mixer = Mixer()
    x = noise(2.0, seed=4)
    outs = {}
    for s in (0.0, 1.0):
        st.reset()
        ctx = StageContext.default()
        ctx.suppression = s
        ys = []
        padded = np.concatenate([x, np.zeros(st.delay + HOP, np.float32)])
        for i in range(0, len(padded) - HOP + 1, HOP):
            chunk = st.process(AudioChunk(padded[i : i + HOP], SR, i // HOP, i / SR), ctx)
            assert RESIDUAL_KEY in ctx.telemetry.extras
            ys.append(mixer.process(chunk, ctx).samples)
            assert RESIDUAL_KEY not in ctx.telemetry.extras  # popped every chunk
        outs[s] = np.concatenate(ys)[st.delay : st.delay + len(x)]
    np.testing.assert_allclose(outs[0.0], x, atol=1e-6)  # slider 0: the room untouched (delayed)
    ref = SlidingWindowExtractor(Shaper(), 0.5, 0.25).extract(x, SR)
    np.testing.assert_allclose(outs[1.0][: SR], ref[: SR], atol=1e-6)  # slider 1: only the target


def test_slow_candidate_overruns_without_stalling_the_stream():
    st = FocusHQ(_candidate=Shaper(sleep_s=0.25), window_s=0.2, hop_s=0.1, threaded=True, adapt_hop=False)
    st.load()
    x = noise(4.0, seed=5)  # long enough to outlive the cold first window (it blocks as "warming up")
    y, ctx = stream(st, x, sleep=0.001)  # ~0.1 s of compute budget per 0.25 s window
    st.close()
    assert len(y) == len(x) and np.isfinite(y).all()
    health = ctx.telemetry.extras["tse"]
    assert st.overruns > 0
    assert health["ok"] is False and health["reason"] in ("overrun", "late")
    assert np.max(np.abs(np.diff(y))) < 2.5  # ramps, no full-scale clicks between sources


def test_threaded_load_failure_is_reported_and_the_mix_plays():
    class Broken(Shaper):
        def load(self):
            raise RuntimeError("weights missing")

    st = FocusHQ(_candidate=Broken(), window_s=0.2, hop_s=0.1, threaded=True)
    x = noise(1.0, seed=6)

    def load_done(i, ctx):  # the load waits on TORCH_LOCK, which other threads may hold
        if st._load is not None:
            st._load.result(timeout=5.0)

    y, ctx = stream(st, x, sleep=0.001, before=load_done)
    st.close()
    np.testing.assert_array_equal(y[HOP:], x[HOP:])
    assert "weights missing" in ctx.telemetry.extras["tse"]["reason"]


def test_require_gpu_on_cpu_disables_the_tier(monkeypatch):
    monkeypatch.setattr(tse_stage.hdevice, "pick", lambda: "cpu")
    st = FocusHQ(_candidate=Shaper(), window_s=0.2, hop_s=0.1, threaded=False, require_gpu=True)
    _, ctx = stream(st, noise(0.6))
    assert ctx.telemetry.extras["tse"]["ok"] is False
    assert "no GPU" in ctx.telemetry.extras["tse"]["reason"]


def test_enrolled_candidate_waits_for_automatic_enrollment():
    cand = Shaper(needs={"enroll"})
    st = FocusHQ(_candidate=cand, window_s=0.5, hop_s=0.25, threaded=False, enroll_s=1.0)
    ctx = _target_ctx(7)
    x = noise(3.0, seed=7)
    stream(st, x, ctx=ctx)
    assert st.enroller.ready
    first = next(i for i, c in enumerate(cand.calls) if c["enroll"] is not None)
    assert len(cand.calls[first]["enroll"]) == SR
    assert ctx.telemetry.extras["tse"]["enrolled_s"] == 1.0


def test_enroller_keeps_only_the_target_talking_alone():
    enr = AutoEnroller(seconds=0.05)
    ctx = _target_ctx(7)
    loud = np.full(HOP, 0.1, np.float32)
    ctx.tracks = [FaceTrack(7, (0, 0, 1, 1), None, 0.9, 0.0), FaceTrack(8, (0, 0, 1, 1), None, 0.8, 0.0)]
    assert not enr.update(loud, ctx)  # someone else talks too
    ctx.tracks[1].p_speaking = 0.1
    assert not enr.update(np.zeros(HOP, np.float32), ctx)  # silence
    assert enr.update(loud, ctx)
    ctx.tracks[0].p_speaking = 0.35
    assert not enr.update(loud, ctx)  # target not clearly talking (0.4-0.7 counts only next to a quiet face, T-092)
    ctx.tracks[0].p_speaking = 0.9
    for _ in range(4):
        enr.update(loud, ctx)
    assert enr.ready and len(enr.clip()) == 800
    ctx.target = TargetState(8, 0.9, False, 0.0)
    enr.update(loud, ctx)
    assert not enr.ready and enr.track_id == 8  # new person: start over


def test_lip_feed_and_crop():
    feed = LipFeed(keep_s=1.0)
    crop = lip_crop(np.zeros((360, 640, 3), np.uint8), (0.4, 0.3, 0.2, 0.4))
    assert crop.shape == (88, 88) and crop.dtype == np.uint8
    for i in range(50):
        feed.append(i * 0.04, 1, crop, 0.5)
    assert len(feed) == 26  # 1 s of history at 25 fps (inclusive)
    last = feed.since(0)[-1]
    assert feed.since(last.index + 1) == []
    with pytest.raises(ValueError):
        feed.append(0.0, 1, np.zeros((88, 88), np.float32))


# ------------------------------------------------------------ pause kwargs (T-070)


def test_enrolled_pause_kwargs_parse_from_the_spec_string():
    from hearing.tse.enrolled import EnrolledCandidate

    spec = 'tse(candidate="enrolled", window_s=1.0, hop_s=0.5, pause_floor=0.25, pause_p=0.5, pause_action="duck")'
    stage = registry.build_pipeline(spec).stages[0]
    cand = stage.cand
    assert isinstance(cand, EnrolledCandidate)
    assert (cand.pause_floor, cand.pause_p, cand.pause_action, cand.pause_max_s) == (0.25, 0.5, "duck", 2.0)
    assert "asd" in cand.needs and stage._want_asd  # pause_p: the stage passes the target's ASD
    assert stage.candidate_kwargs == {"pause_floor": 0.25, "pause_p": 0.5, "pause_action": "duck"}

    hold = registry.build_pipeline('tse(candidate="enrolled", pause_floor=0.3, pause_max_s=1.5)').stages[0]
    assert (hold.cand.pause_floor, hold.cand.pause_p, hold.cand.pause_action, hold.cand.pause_max_s) == (
        0.3, None, "hold", 1.5)
    assert "asd" not in hold.cand.needs

    default = registry.build_pipeline('tse(candidate="enrolled")').stages[0]
    assert default.cand.pause_floor is None and default.candidate_kwargs == {}  # off by default (T-069)

    with pytest.raises(ValueError, match="pause_action"):
        registry.build_pipeline('tse(candidate="enrolled", pause_floor=0.25, pause_action="shout")')
    with pytest.raises(ValueError, match="only apply to candidate 'enrolled'"):
        registry.build_pipeline('tse(candidate="identity", pause_floor=0.25)')


class Preparer(Shaper):
    """Enrolled-like candidate whose per-clip preparation (embeddings) is slow, like ECAPA + separation."""

    def __init__(self, prepare_s: float):
        super().__init__(needs={"enroll"})
        self.prepare_s = prepare_s
        self.prepared = []

    def prepare(self, enroll, other=None, sr=SR):
        time.sleep(self.prepare_s)
        self.prepared.append(len(enroll))


def test_new_enrollment_is_prepared_off_the_window_path_and_reads_enrolling():
    """T-088: the first window on a new clip used to pay ~1 s of embedding and overrun; the stage
    now prepares the clip in its own job and reports ``enrolling`` (not ``overrun``) meanwhile."""
    cand = Preparer(prepare_s=0.3)
    st = FocusHQ(_candidate=cand, window_s=0.2, hop_s=0.1, threaded=True, enroll_s=0.5)
    st.load()
    stream(st, noise(2.0, seed=9), ctx=_target_ctx(7), sleep=0.0015)
    st.close()
    assert cand.prepared == [round(0.5 * SR)]  # once per clip
    first = next(i for i, c in enumerate(cand.calls) if c["enroll"] is not None)
    assert first == 0  # no window ran before the clip was prepared
    reasons = [w.reason for w in st._recent if w.status == "skip"]
    assert "overrun" not in reasons and "enrolling" in reasons


def test_health_is_ok_once_a_window_succeeds_after_enrolling():
    """T-088: windows skipped while the enrollment was collected do not hold the tier back for
    HEALTH_WINDOWS more windows; one good window after them is enough (the ladder's
    upgrade_after_s then covers the playout delay)."""
    cand = Shaper(needs={"enroll"})
    st = FocusHQ(_candidate=cand, window_s=0.5, hop_s=0.25, threaded=False, enroll_s=1.0)
    ctx = _target_ctx(7)
    seen = []

    def watch(i, c):
        h = c.telemetry.extras.get("tse")
        if h is not None:
            seen.append((sum(1 for x in cand.calls if x["enroll"] is not None), h["ok"]))

    stream(st, noise(3.0, seed=7), ctx=ctx, before=watch)
    assert (1, True) in seen  # ok right after the first enrolled window
    assert all(not ok for n, ok in seen if n == 0)  # never before
