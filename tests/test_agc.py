"""T-135: output AGC + soft limiter (hearing/audio/agc.py)."""

import numpy as np

from hearing.audio.agc import (
    FLOOR_DB,
    FLOOR_RULE,
    LIMIT,
    MAX_GAIN_DB,
    TARGET_DB,
    Agc,
    AgcStage,
    soft_limit,
)
from hearing.testing import assert_streaming_equivalent

SR = 16000


def _speechlike(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    """Noise with a 4 Hz syllable envelope and 30 % pauses, at ``db`` dBFS RMS while voiced."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    env = np.clip(np.sin(2 * np.pi * 2.0 * t), 0, None) ** 0.5
    x = rng.standard_normal(n) * env
    voiced = env > 0
    x *= 10 ** (db / 20) / np.sqrt(np.mean(x[voiced] ** 2))
    return x.astype(np.float32)


def _rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def test_streaming_equals_whole_file():
    x = np.concatenate([_speechlike(1.5, -35), _speechlike(1.5, -8, seed=1)])
    assert_streaming_equivalent(AgcStage, x, atol=1e-5)
    whole = Agc().process_array(x)
    agc = Agc()
    sizes = [7, 160, 333, 1, 480]
    out, i, k = [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        out.append(agc.process_array(x[i:i + n]))
        i, k = i + n, k + 1
    np.testing.assert_allclose(np.concatenate(out), whole, atol=1e-6)


def test_levels_converge_to_target_within_2db():
    for db in (-34.0, -26.0, -20.0, -12.0, -6.0):
        y = Agc().process_array(_speechlike(4.0, db))
        tail = y[2 * SR:]
        x_tail = _speechlike(4.0, db)[2 * SR:]
        voiced = np.abs(x_tail) > 0
        got = _rms_db(tail[voiced])
        if db >= TARGET_DB - MAX_GAIN_DB:
            assert abs(got - TARGET_DB) < 2.5, (db, got)
        else:  # quieter than target - max gain: raised by the full 12 dB only
            assert abs(got - (db + MAX_GAIN_DB)) < 2.5, (db, got)


def test_silence_holds_the_gain():
    agc = Agc()
    agc.process_array(_speechlike(2.0, -30))
    g = agc.gain_db
    agc.process_array(np.full(SR, 10 ** ((FLOOR_DB - 10) / 20), dtype=np.float32))
    assert agc.gain_db == g


def test_pauses_of_an_isolated_voice_are_not_pumped_up():
    """T-135 A/B (demo_regress focus_hq, judge): the residual in the target's pauses, ~18 dB under its
    speech, was raised toward -20 dBFS (leak -18.0 -> -11.5 dB). Blocks HOLD_DB below the speech hold."""
    from hearing.audio.agc import HOLD_DB

    assert HOLD_DB <= 15.0
    agc = Agc()
    agc.process_array(_speechlike(3.0, -26))
    g = agc.gain_db
    rng = np.random.default_rng(1)
    residual = (rng.standard_normal(2 * SR) * 10 ** (-44 / 20)).astype(np.float32)  # 18 dB under the speech
    agc.process_array(residual)
    assert abs(agc.gain_db - g) < 0.5


def test_limiter_never_clips():
    x = np.linspace(-3, 3, 1001).astype(np.float32)
    y = soft_limit(x)
    assert np.all(np.abs(y) <= 1.0)
    assert np.all(np.abs(y[np.abs(x) <= 1.0]) < 0.96)  # a knee, not a wall
    small = np.abs(x) <= LIMIT
    np.testing.assert_array_equal(y[small], x[small])
    assert np.all(np.diff(y) >= 0)  # monotonic
    loud = Agc().process_array(_speechlike(1.0, 0.0) * 4)
    assert np.max(np.abs(loud)) <= 1.0


def test_registered_as_stage():
    from hearing import registry
    stage = registry.build_pipeline("agc(target_db=-20)")
    assert stage.latency_ms == 0.0


def test_a_long_pause_is_never_boosted():
    """T-169: the held speech gain used to play on through the room noise of every pause (live A100
    session 20261009T203531Z-2: quiet windows +4.5 dB median after one AGC, +8.5 dB after two). After
    PAUSE_HANG_MS of held blocks the applied gain relaxes to <= 0 dB; speech brings it back at once."""
    from hearing.audio.agc import PAUSE_HANG_MS

    agc = Agc()
    speech = _speechlike(3.0, -30)
    agc.process_array(speech)
    g = agc.gain_db
    assert g > 8.0
    rng = np.random.default_rng(2)
    room = (rng.standard_normal(2 * SR) * 10 ** (-47 / 20)).astype(np.float32)  # > HOLD_DB under the speech
    y = agc.process_array(room)
    assert agc.gain_db == g  # the speech gain is held for the next sentence
    tail = slice(int((PAUSE_HANG_MS + 300) * SR / 1000), None)
    assert _rms_db(y[tail]) <= _rms_db(room[tail]) + 0.1  # the room is not raised
    assert _rms_db(y) <= _rms_db(room) + 3.0  # over the whole 2 s pause
    back = agc.process_array(_speechlike(1.0, -30, seed=3))
    voiced = np.abs(back[SR // 10:]) > 0
    assert abs(_rms_db(back[SR // 10:][voiced]) - TARGET_DB) < 2.5  # the speech gain is back


def test_short_gaps_inside_speech_keep_the_gain():
    """Gaps shorter than PAUSE_HANG_MS (syllables, stops) never touch the gain: no pumping."""
    from hearing.audio.agc import PAUSE_HANG_MS

    assert PAUSE_HANG_MS >= 250
    x = _speechlike(4.0, -30)  # 250 ms silent half-periods
    agc = Agc()
    agc.process_array(x[:2 * SR])
    assert agc.pause_db == 0.0


def test_effective_floor_follows_the_room_noise():
    """T-179: the floor is max(FLOOR_DB, noise floor + NOISE_MARGIN_DB): -50 dBFS in a quiet room,
    above -45 dBFS cafe noise in a noisy one."""
    from hearing.audio.agc import NOISE_MARGIN_DB

    agc = Agc()
    assert agc.effective_floor_db() == FLOOR_DB
    rng = np.random.default_rng(4)
    agc.process_array((rng.standard_normal(2 * SR) * 10 ** (-70 / 20)).astype(np.float32))
    assert agc.effective_floor_db() == FLOOR_DB
    agc.process_array((rng.standard_normal(2 * SR) * 10 ** (-45 / 20)).astype(np.float32))
    assert agc.effective_floor_db() > -45.0
    assert abs(agc.effective_floor_db() - (agc.noise_db + NOISE_MARGIN_DB)) < 1e-9


def test_room_noise_alone_is_never_raised():
    """T-179: cafe noise at -45 dBFS (above the old fixed -50 floor) from the start is never boosted."""
    rng = np.random.default_rng(5)
    room = (rng.standard_normal(4 * SR) * 10 ** (-45 / 20)).astype(np.float32)
    agc = Agc(**FLOOR_RULE)
    gains = []
    for i in range(0, len(room), 160):
        agc.process_array(room[i:i + 160])
        gains.append(agc.applied_db)
    assert max(gains) <= 0.0


def test_under_the_floor_the_gain_only_moves_toward_zero():
    """T-179: a block under the effective floor never raises the applied gain; it holds or decays to 0 dB."""

    agc = Agc(pause_hang_ms=None, **FLOOR_RULE)  # no T-169 pause relax: the floor rule alone
    agc.process_array(_speechlike(3.0, -32))
    prev = agc.applied_db
    assert prev > 5.0
    rng = np.random.default_rng(6)
    room = (rng.standard_normal(2 * SR) * 10 ** (-60 / 20)).astype(np.float32)  # under -50 dBFS
    for i in range(0, len(room), 160):
        agc.process_array(room[i:i + 160])
        assert agc.applied_db <= max(prev, 0.0) + 1e-9
        prev = agc.applied_db
    # a cut (loud speech) relaxes toward 0 dB at most
    agc = Agc(pause_hang_ms=None, **FLOOR_RULE)
    agc.process_array(_speechlike(2.0, -8))
    assert agc.applied_db < -5.0
    for i in range(0, len(room), 160):
        agc.process_array(room[i:i + 160])
        assert agc.applied_db <= 0.0


def test_the_t169_agc_raised_the_room_where_this_one_does_not():
    """The before/after the replay compares (scripts/agc_replay.py): never_up=False is the T-169 rule."""
    rng = np.random.default_rng(7)
    x = np.concatenate([_speechlike(2.0, -34), (rng.standard_normal(SR // 4) * 10 ** (-46 / 20)).astype(np.float32)])
    old = Agc().process_array(x)
    new = Agc(**FLOOR_RULE).process_array(x)
    gap = slice(2 * SR, None)
    assert _rms_db(new[gap]) < _rms_db(old[gap])


def test_the_floor_rule_is_opt_in(monkeypatch):
    """Q-485: the shipped default stays the T-169 AGC (+12 dB, no never-up); HEARING_AGC_FLOOR_RULE=1 opts in."""
    from hearing.audio.agc import FLOOR_RULE_ENV, floor_rule_kwargs

    assert MAX_GAIN_DB == 12.0 and not Agc().never_up and Agc().status()["floor_rule"] is False
    assert FLOOR_RULE == {"max_gain_db": 9.0, "never_up": True}
    monkeypatch.delenv(FLOOR_RULE_ENV, raising=False)
    assert floor_rule_kwargs() == {}
    monkeypatch.setenv(FLOOR_RULE_ENV, "1")
    agc = Agc(**floor_rule_kwargs())
    assert agc.never_up and agc.max_gain_db == 9.0 and agc.status()["floor_rule"] is True
