"""Association guards of the Enrolled tier (T-061): margin hold, similarity floor, purity filter,
settled and sticky auto-enrollment."""

import numpy as np
import pytest

from hearing.tse.blind import BlindSeparator
from hearing.tse.blind import MockBackend as SeparatorMock
from hearing.tse.enroll import AutoEnroller
from hearing.tse.enrolled import EnrolledCandidate, _overlap
from hearing.types import FaceTrack, StageContext, TargetState

SR = 16000
HOP = 160


class SignEmbedder:
    """Embedding = (positive share, negative share): a "voice" is the sign of the samples."""

    device = "cpu"

    def __init__(self):
        self.calls = 0

    def embed(self, audio):
        self.calls += 1
        return np.array([np.mean(audio > 0) + 1e-3, np.mean(audio < 0) + 1e-3])


class Streams:
    """Separator returning scripted streams per call (index permutation varies per window)."""

    device = "cpu"

    def __init__(self, fn):
        self.fn = fn

    def separate(self, mix):
        return self.fn(mix)


def _ctx(target, tracks, switch_seq=0):
    ctx = StageContext.default()
    ctx.target = TargetState(target, 1.0, True, 0.0, switch_seq)
    ctx.tracks = [FaceTrack(i, (0, 0, 0.2, 0.2), None, p, 0.0) for i, p in tracks]
    return ctx


def test_overlap_finds_the_previous_window():
    x = np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32)
    assert _overlap(x[:SR], x[SR // 2 : SR // 2 + SR]) == SR // 2
    assert _overlap(x[:SR], x[2 * SR : 3 * SR]) is None  # no shared samples


def test_floor_returns_silence_when_no_stream_matches():
    a = np.ones(SR, np.float32)
    streams = np.stack([0.5 * a, 0.2 * a])
    # Isolate the floor: both streams have identical sign embeddings, so the default
    # margin tracker could follow either waveform after the first (gated) window.
    c = EnrolledCandidate(BlindSeparator(SeparatorMock(streams)), SignEmbedder(), floor=0.5, margin=0.0)
    y = c.extract(a, SR, enroll=-np.ones(SR, np.float32))  # enrolled voice is "negative"
    assert not y.any() and c.gated
    y = c.extract(a, SR, enroll=np.ones(SR, np.float32))
    assert c.similarity > 0.99 and not c.gated
    # Both positive streams have identical sign embeddings; either tie winner is valid.
    # The floor must pass through a separated stream, rescaled to sum to the mix.
    assert any(np.allclose(y, stream / 0.7, rtol=1e-5) for stream in streams)


def test_margin_holds_the_previous_stream_across_permuted_windows():
    rng = np.random.default_rng(1)
    pos = np.abs(rng.standard_normal(2 * SR)).astype(np.float32)  # the target ("positive" voice)
    other = rng.standard_normal(2 * SR).astype(np.float32)  # sign-balanced: ambiguous
    mix = pos + other
    calls = []

    def sep(window):
        start = int(np.flatnonzero(mix == window[0])[0])
        t, o = pos[start : start + len(window)], other[start : start + len(window)]
        calls.append(start)
        if len(calls) == 1:
            return np.stack([t, o])
        # second window: streams swapped and the target made ambiguous for the embedder
        return np.stack([o, t - np.abs(t).mean()])

    c = EnrolledCandidate(BlindSeparator(Streams(sep)), SignEmbedder(), margin=0.9, clean_enroll=False)
    enroll = np.ones(SR, np.float32)
    y0 = c.extract(mix[:SR], SR, enroll=enroll)
    assert c.selected_stream == 0 and np.array_equal(y0, pos[:SR])
    c.extract(mix[SR // 2 : SR // 2 + SR], SR, enroll=enroll)
    assert c.selected_stream == 1 and c.held  # followed the waveform, not the noisy embedding
    plain = EnrolledCandidate(BlindSeparator(Streams(sep)), SignEmbedder(), margin=0.0, clean_enroll=False)
    calls.clear()
    plain.extract(mix[:SR], SR, enroll=enroll)
    plain.extract(mix[SR // 2 : SR // 2 + SR], SR, enroll=enroll)
    assert plain.selected_stream == 0  # without the margin the ambiguous window flips


def test_purity_filter_drops_foreign_pieces_and_caches():
    emb = SignEmbedder()
    c = EnrolledCandidate(BlindSeparator(SeparatorMock()), emb, purify=0.9)
    piece = SR // 2
    enroll = np.ones(6 * piece, np.float32)
    enroll[piece : 2 * piece] = -1.0  # one half-second of someone else
    t = c.target_embedding(enroll)
    assert c.kept_share == pytest.approx(5 / 6)
    assert t[0] > 0.99  # the clean "positive" voice
    calls = emb.calls
    c.target_embedding(enroll)
    assert emb.calls == calls  # cached per clip
    unfiltered = EnrolledCandidate(BlindSeparator(SeparatorMock()), SignEmbedder()).target_embedding(enroll)
    assert unfiltered[1] > t[1]


def test_prepare_fills_the_cache_extract_uses():
    """T-088: the tse stage prepares a new clip off the window path; extract must then embed nothing
    but the window's streams."""
    emb = SignEmbedder()
    c = EnrolledCandidate(BlindSeparator(SeparatorMock()), emb, purify=0.9)
    enroll, other = np.ones(3 * SR, np.float32), -np.ones(2 * SR, np.float32)
    c.prepare(enroll, other, SR)
    before = emb.calls
    c.extract(np.ones(SR, np.float32), SR, enroll=enroll, other=other)
    streams = len(BlindSeparator(SeparatorMock()).separate(np.ones(SR, np.float32), SR))
    assert emb.calls - before == streams


def test_negative_enrollment_mirrors_the_gate_and_follows_a_switch():
    enr = AutoEnroller(seconds=0.02, other_s=0.02)
    loud = np.full(HOP, 0.1, np.float32)
    enr.update(loud, _ctx(1, [(1, 0.1), (2, 0.9)]))  # only the other talks: negative
    assert enr.other_clip() is None and enr.collected_s == 0
    enr.update(loud, _ctx(1, [(1, 0.1), (2, 0.9), (3, 0.8)]))  # two others at once: neither
    enr.update(loud, _ctx(1, [(1, 0.1), (2, 0.9)]))
    assert len(enr.other_clip()) == 320 and enr.other_spans == [(0, HOP), (2 * HOP, HOP)]
    enr.update(loud, _ctx(1, [(1, 0.9), (2, 0.6)]))  # other below OTHER_P (0.7): target kept
    enr.update(loud, _ctx(1, [(1, 0.9), (2, 0.1)]))
    clip = enr.clip()
    assert enr.ready and len(clip) == 320
    enr.update(loud, _ctx(2, [(1, 0.0), (2, 0.0)]))  # deliberate switch to the visible track 2
    assert enr.clip() is None and np.array_equal(enr.other_clip(), clip)  # old target = negative


def test_sticky_keeps_the_clip_when_the_track_was_redetected():
    enr = AutoEnroller(seconds=0.02)
    loud = np.full(HOP, 0.1, np.float32)
    ctx = _ctx(1, [(1, 0.9), (2, 0.0)])
    enr.update(loud, ctx)
    enr.update(loud, ctx)
    clip = enr.clip()
    assert enr.ready
    # occlusion: track 1 is gone, the face comes back as track 3 -> keep serving the old clip
    enr.update(loud, _ctx(3, [(3, 0.0), (2, 0.0)]))
    assert enr.holding and np.array_equal(enr.clip(), clip) and enr.track_id == 3
    enr.update(loud, _ctx(3, [(3, 0.9), (2, 0.0)]))
    enr.update(loud, _ctx(3, [(3, 0.9), (2, 0.0)]))
    assert enr.ready and not enr.holding  # the new collection replaces it
    # deliberate switch to a track visible next to the old one: start over, nothing held
    enr.update(loud, _ctx(2, [(3, 0.0), (2, 0.0)]))
    assert enr.clip() is None and not enr.holding


def test_deliberate_switch_resets_even_when_the_old_face_is_off_screen():
    enr = AutoEnroller(seconds=0.02)
    loud = np.full(HOP, 0.1, np.float32)
    enr.update(loud, _ctx(1, [(1, 0.9)]))
    enr.update(loud, _ctx(1, [(1, 0.9)]))
    assert enr.ready
    old = enr.clip()
    # lock / challenger win onto track 5 while track 1 is off screen: switch_seq moved
    enr.update(loud, _ctx(5, [(5, 0.0)], switch_seq=1))
    assert enr.clip() is None and not enr.holding and enr.track_id == 5
    assert np.array_equal(enr.other_clip(), old)  # the absent person becomes the negative
    # an automatic re-detection (same switch_seq) still holds the clip
    enr.update(loud, _ctx(5, [(5, 0.9)], switch_seq=1))
    enr.update(loud, _ctx(5, [(5, 0.9)], switch_seq=1))
    assert enr.ready
    clip = enr.clip()
    enr.update(loud, _ctx(6, [(6, 0.0)], switch_seq=1))
    assert enr.holding and np.array_equal(enr.clip(), clip)


def test_selector_counts_locks_and_challenger_wins_but_not_reacquisition():
    from hearing.vision.target import TargetSelector

    sel = TargetSelector()
    box = (0.4, 0.3, 0.2, 0.3)
    mk = lambda i: FaceTrack(i, box, None, 0.9, 0.0)
    sel.lock(1, 0.0)  # nothing held yet: acquiring is not a switch
    assert sel.state.switch_seq == 0
    sel.lock(2, 1.0)
    assert sel.state.switch_seq == 1
    sel.unlock()
    assert sel.state.switch_seq == 1
    sel.update([mk(2)], 2.0)
    sel.update([], 10.0)  # held target lost for longer than hold_s: dropped
    sel.update([mk(7)], 10.1)
    sel.update([mk(7)], 12.0)
    assert sel.state.track_id == 7 and sel.state.switch_seq == 1  # re-acquired, not switched


def test_sticky_can_be_turned_off():
    enr = AutoEnroller(seconds=0.01, sticky=False)
    loud = np.full(HOP, 0.1, np.float32)
    enr.update(loud, _ctx(1, [(1, 0.9)]))
    assert enr.ready
    enr.update(loud, _ctx(3, [(3, 0.0)]))
    assert enr.clip() is None


def test_separator_streams_are_put_back_on_the_mix_scale():
    rng = np.random.default_rng(3)
    a, b = (0.05 * rng.standard_normal(SR)).astype(np.float32), (0.01 * rng.standard_normal(SR)).astype(np.float32)
    out = BlindSeparator(SeparatorMock(np.stack([40 * a, 40 * b]))).separate(a + b)
    np.testing.assert_allclose(out[0], a, atol=1e-5)
    np.testing.assert_allclose(out[1], b, atol=1e-5)  # relative level kept, not lifted to the mix RMS
    silent = BlindSeparator(SeparatorMock()).separate(np.zeros(SR, np.float32))
    assert not silent.any()


def _voice(pos_share, n=SR):
    """A "voice" for SignEmbedder: ``pos_share`` of the samples +1, the rest -1."""
    k = round(pos_share * n)
    return np.concatenate([np.ones(k), -np.ones(n - k)]).astype(np.float32)


def test_clean_enroll_embeds_the_louder_separated_stream():
    sep = Streams(lambda x: np.stack([0.1 * _voice(0.0, len(x)), _voice(1.0, len(x))]))
    c = EnrolledCandidate(BlindSeparator(sep), SignEmbedder())
    t = c.target_embedding(_voice(0.7))  # the room clip itself is mixed (0.7 "positive")
    assert t[0] > 0.99  # the reference is the louder stream's voice


def test_negative_enrollment_silences_the_other_voice():
    other_voice, noise = _voice(0.4), _voice(0.1)
    streams = np.stack([other_voice, noise])
    kw = {"clean_enroll": False, "margin": 0.0}
    c = EnrolledCandidate(BlindSeparator(SeparatorMock(streams)), SignEmbedder(), **kw)
    target, other = _voice(1.0), _voice(0.0)
    y = c.extract(other_voice + noise, SR, enroll=target)
    assert c.selected_stream == 0 and y.any()  # verification alone plays the other person
    y = c.extract(other_voice + noise, SR, enroll=target, other=other)
    assert c.gated and not y.any() and c.scores[0] < 0  # closer to "other" than to the target
    c = EnrolledCandidate(BlindSeparator(SeparatorMock(streams)), SignEmbedder(), **kw)
    c.extract(other_voice + noise, SR, enroll=target, other=_voice(0.95))  # polluted negative
    assert not c.gated and c.scores == c.similarities  # ignored (cos > SAME_VOICE)
    c = EnrolledCandidate(BlindSeparator(SeparatorMock(streams)), SignEmbedder(), same_voice=1.0, **kw)
    c.extract(other_voice + noise, SR, enroll=target, other=_voice(0.95))
    assert c.scores != c.similarities  # same_voice raised (T-069, same-gender pairs): the negative is kept


def test_stage_passes_the_negative_to_candidates_that_ask():
    from hearing.tse.stage import FocusHQ

    seen = []

    class Cand:
        name, licence, device = "c", "test", "cpu"
        needs = frozenset({"enroll", "other"})

        def load(self):
            pass

        def extract(self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None, other=None):
            seen.append(other)
            return mix

    st = FocusHQ(_candidate=Cand(), window_s=0.1, hop_s=0.05, threaded=False, enroll_s=0.02)
    st.enroller = AutoEnroller(seconds=0.02, other_s=0.02)
    loud = np.full(HOP, 0.1, np.float32)
    from hearing.types import AudioChunk

    for i, (p1, p2) in enumerate([(0.9, 0.1)] * 2 + [(0.1, 0.9)] * 2 + [(0.9, 0.1)] * 20):
        st.process(AudioChunk(loud, SR, i, i * 0.01), _ctx(1, [(1, p1), (2, p2)]))
    assert seen and seen[-1] is not None and len(seen[-1]) == 320


def test_other_hold_needs_the_other_face_quiet_for_a_while():
    enr = AutoEnroller(seconds=0.05, other_hold_s=0.03)
    loud = np.full(HOP, 0.1, np.float32)
    assert not enr.update(loud, _ctx(1, [(1, 0.9), (2, 0.9)]))
    assert not enr.update(loud, _ctx(1, [(1, 0.9), (2, 0.1)]))  # a one-frame dip of the other face
    assert not enr.update(loud, _ctx(1, [(1, 0.9), (2, 0.1)]))
    assert enr.update(loud, _ctx(1, [(1, 0.9), (2, 0.1)]))  # quiet for 30 ms: kept


def test_enroller_reasons_and_hint():
    """T-092: every chunk says why it was not kept, and the pill hint names the commonest reason."""
    voice, hush = np.full(160, 0.1, np.float32), np.zeros(160, np.float32)
    e = AutoEnroller(seconds=1.0, other_s=0.0)
    e.update(voice, _ctx(None, []))
    assert e.reason == "no_target" and e.hint == ""
    for _ in range(60):
        e.update(voice, _ctx(1, [(1, 0.9), (2, 0.95)]))
    assert e.reason == "other_talking" and e.hint == "need them to talk alone"
    for _ in range(60):
        e.update(voice, _ctx(1, [(1, 0.2), (2, 0.9)]))
    assert e.reason == "target_quiet" and e.hint == "other person talking"
    for _ in range(100):
        e.update(voice, _ctx(1, [(1, 0.2), (2, 0.1)]))
    assert e.reason == "target_quiet" and e.hint == "waiting for them to talk"
    for _ in range(100):
        e.update(hush, _ctx(1, [(1, 0.9)]))
    assert e.reason == "unvoiced" and e.hint == "too quiet"
    assert e.update(voice, _ctx(1, [(1, 0.9)])) and e.reason == "kept"  # no other face: kept
    for _ in range(100):
        e.update(voice, _ctx(1, [(1, 0.9)]))
    assert e.ready and e.reason == "ready" and e.hint == ""


def test_enroller_relaxed_gate():
    """A target at 0.5-0.7 counts only while every other face is clearly quiet (< relax_other_p)."""
    voice = np.full(160, 0.1, np.float32)
    strict = AutoEnroller(seconds=1.0, other_s=0.0, relax_target_p=None)
    relaxed = AutoEnroller(seconds=1.0, other_s=0.0, relax_target_p=0.5, relax_other_p=0.3)
    quiet_other, busy_other = _ctx(1, [(1, 0.6), (2, 0.1)]), _ctx(1, [(1, 0.6), (2, 0.4)])
    assert not strict.update(voice, quiet_other) and strict.reason == "target_quiet"
    assert relaxed.update(voice, quiet_other)
    assert not relaxed.update(voice, busy_other) and relaxed.reason == "target_quiet"
    assert not relaxed.update(voice, _ctx(1, [(1, 0.45), (2, 0.0)]))
