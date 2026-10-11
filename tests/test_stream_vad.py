"""T-337: per-stream speech activity on separated streams (overlap vs solo speech)."""

import numpy as np

from hearing.fusion.stream_vad import StreamVad, flatness

SR = 16000


def voice(f0, s=1.0, amp=0.2):
    t = np.arange(round(s * SR)) / SR
    return (amp * sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 8))).astype(np.float32)


def room(s=1.0, amp=0.003, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(round(s * SR))).astype(np.float32)


def test_flatness_tells_voice_from_noise():
    assert np.median(flatness(voice(120.0))) < 0.1
    assert np.median(flatness(room(amp=0.1))) > 0.5


def test_two_active_streams_are_overlap():
    a, b = voice(110.0) + room(seed=1), voice(170.0, amp=0.15) + room(seed=2)
    r = StreamVad().window(a + b, [a, b])
    assert r["overlap"] and r["both_s"] > 0.9


def test_one_voice_and_leakage_is_solo():
    a = voice(110.0) + room(seed=1)
    leak = 0.03 * voice(110.0) + room(seed=2)  # ~ -16 dB of the same voice in the other stream
    r = StreamVad().window(a + leak, [a, leak])
    assert not r["overlap"] and r["solo_s"] > 0.9 and r["both_s"] == 0


def test_noise_only_streams_are_not_speech():
    vad = StreamVad()
    for seed in range(4):  # the running floor learns the room
        x, y = room(seed=10 + seed, amp=0.05), room(seed=20 + seed, amp=0.05)
        r = vad.window(x + y, [x, y])
    assert not r["overlap"] and r["any_s"] == 0


def test_short_overlap_in_a_solo_window_stays_solo():
    a = voice(110.0) + room(seed=1)
    b = np.concatenate([voice(170.0, s=0.05), np.zeros(SR - 800, np.float32)]) + room(seed=2)
    r = StreamVad().window(a + b, [a, b])
    assert not r["overlap"]
