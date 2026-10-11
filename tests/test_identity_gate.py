"""T-413: identity bouncer (hearing.tse.identity_gate) with a tone-band embedder."""

import numpy as np

from hearing.tse import identity_gate as ig

SR = 16000


class BandEmbedder:
    """Energy in 8 bands up to 4 kHz: tones of different pitch are different "speakers"."""

    def embed(self, audio):
        spec = np.abs(np.fft.rfft(np.asarray(audio, np.float64), n=2048))[:512]
        return np.array([np.linalg.norm(b) for b in np.array_split(spec, 8)])


def tone(freq, seconds=2.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


A, B = 300.0, 2500.0  # band 0 vs band 4


def gate(**kw):
    return ig.IdentityGate(BandEmbedder(), tone(A), others=[tone(B)], accept_cos=0.8, margin_cos=0.1, **kw)


def test_score_passes_target_blocks_other_and_ignores_silence():
    g = gate()
    assert g.score(tone(A, 1.0)).passed
    s = g.score(tone(B, 1.0))
    assert not s.passed and s.reason == "low_similarity" and s.margin < 0
    quiet = g.score(np.zeros(SR, np.float32))
    assert quiet.passed and quiet.reason == "silent"


def test_closed_set_margin_rejects_ambiguous_output():
    g = ig.IdentityGate(BandEmbedder(), tone(A), others=[tone(B)], accept_cos=0.5, margin_cos=0.1)
    s = g.score(tone(A, 1.0) + tone(B, 1.0))
    assert not s.passed and s.reason == "closer_to_other"


def test_bounce_retries_on_residual_after_wrong_lock():
    mix = tone(A) + tone(B)
    calls = []

    def wrong_then_identity(x):  # first call locks onto B, then returns what it is given
        calls.append(x)
        return tone(B) if len(calls) == 1 else x

    r = ig.bounce(mix, wrong_then_identity, gate())
    assert r.action == "retry_pass" and len(r.scores) == 2
    np.testing.assert_allclose(r.audio, tone(A), atol=1e-5)


def test_bounce_blocks_to_room_floor_never_the_wrong_voice():
    mix = tone(A) + tone(B)
    r = ig.bounce(mix, lambda x: tone(B), gate(), retries=0, floor_db=20.0)
    assert r.action == "blocked"
    np.testing.assert_allclose(r.audio, mix * 0.1, atol=1e-6)
    assert ig.bounce(mix, lambda x: tone(B), gate(), retries=0, floor_db=None).audio.max() == 0.0


def test_bounce_passes_first_try():
    r = ig.bounce(tone(A) + tone(B), lambda x: tone(A), gate())
    assert r.action == "pass" and len(r.scores) == 1


def test_stream_blocks_wrong_voice_after_one_window_and_reopens():
    g = gate()
    st = ig.IdentityGateStream(g, window_s=0.5, hop_s=0.1, floor_db=20.0, fail_hops=1, pass_hops=2)
    hop = 160
    seq = [tone(A, 1.0), tone(B, 1.5), tone(A, 1.5)]
    out = []
    for seg in seq:
        mix = seg + 0.0
        for k in range(0, len(seg), hop):
            out.append(st.process(seg[k : k + hop], mix[k : k + hop]))
    y = np.concatenate(out)
    assert y.shape == (4 * SR,)
    # target section untouched; late in the wrong section only the 20 dB floor of the mix plays
    np.testing.assert_allclose(y[: SR], tone(A, 1.0), atol=1e-6)
    wrong_tail = y[int(2.0 * SR) : int(2.5 * SR)]
    np.testing.assert_allclose(wrong_tail, tone(B, 1.5)[SR : int(1.5 * SR)] * 0.1, atol=1e-5)
    # reopened by the end of the final target section
    np.testing.assert_allclose(y[-SR // 4 :], tone(A, 1.5)[-SR // 4 :], atol=1e-5)
    opened = [o for _, _, o in st.history]
    assert False in opened and opened[-1]


def test_stream_reset_restores_open_state():
    st = ig.IdentityGateStream(gate(), window_s=0.25, hop_s=0.1)
    for k in range(0, SR, 160):
        st.process(tone(B, 1.0)[k : k + 160], tone(B, 1.0)[k : k + 160])
    assert not st.open
    st.reset()
    assert st.open and st.gain == 1.0 and not st.history
