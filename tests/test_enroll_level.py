"""T-119: the voiced gate of AutoEnroller at a real laptop mic's level.

Levels from the first live /judge session (results/sessions/20261007T010121Z-1, mic only, the demo
replay left out): chunk RMS p10 -60, median -51, p90 -41, p95 -34 dBFS; 92 % of chunks below the old
absolute -40 dBFS gate. Here: room noise at -58 dBFS, the target speaking in 400 ms words at -45 dBFS
with 150 ms pauses, ASD saying the target talks alone.
"""

import numpy as np

from hearing.tse import enroll
from hearing.tse.enroll import AutoEnroller, VoicedGate, level_db
from hearing.types import FaceTrack, StageContext, TargetState

SR, HOP = 16000, 160


def _at(db: float, n: int, rng) -> np.ndarray:
    x = rng.standard_normal(n).astype(np.float32)
    return x * np.float32(10 ** (db / 20) / np.sqrt(np.mean(x**2)))


def _laptop_mic(seconds: float, speech_db: float = -45.0, noise_db: float = -58.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = round(seconds * SR)
    x = _at(noise_db, n, rng)
    pos = SR  # 1 s of room first
    while pos < n:
        w = min(round(0.4 * SR), n - pos)
        x[pos:pos + w] += _at(speech_db, w, rng)
        pos += w + round(0.15 * SR)
    return x


def _ctx(p_target: float) -> StageContext:
    ctx = StageContext.default()
    ctx.target = TargetState(1, 1.0, False, 0.0)
    ctx.tracks = [FaceTrack(1, (0.4, 0.4, 0.2, 0.3), None, p_target, 0.0)]
    return ctx


def _run(x, ctx, en):
    hints = []
    for i in range(len(x) // HOP):
        en.update(x[i * HOP:(i + 1) * HOP], ctx)
        if i >= 150 and i % 10 == 0:  # after 1.5 s, a telemetry tick every 100 ms
            hints.append(en.hint)
    return hints


def test_level_db():
    assert abs(level_db(np.full(160, 0.01, np.float32)) + 40.0) < 1e-6
    assert level_db(np.zeros(0, np.float32)) == -120.0


def test_laptop_mic_speech_enrolls_without_too_quiet():
    x = _laptop_mic(6.0)
    en = AutoEnroller(other_s=0.0)
    hints = _run(x, _ctx(0.9), en)
    assert en.ready
    assert hints.count("too quiet") == 0, hints
    assert -60.0 < en.gate.floor_db < -54.0  # the room, not the speech


def test_absolute_gate_rejected_the_same_speech(monkeypatch):
    """Before T-119 (fixed -40 dBFS) this mic never enrolled and the pill said 'too quiet'."""
    monkeypatch.setattr(enroll, "VOICED_MIN_DB", -40.0)
    en = AutoEnroller(other_s=0.0)
    hints = _run(_laptop_mic(6.0), _ctx(0.9), en)
    assert not en.ready and hints.count("too quiet") > 0.9 * len(hints)


def test_room_noise_alone_is_still_too_quiet():
    """ASD says the target talks but the mic only hears the room: the hint is right to fire."""
    rng = np.random.default_rng(1)
    en = AutoEnroller(other_s=0.0)
    hints = _run(_at(-58.0, 4 * SR, rng), _ctx(0.9), en)
    assert not en.ready and hints[-1] == "too quiet"


def test_silence_while_nobody_talks_is_waiting_not_quiet():
    rng = np.random.default_rng(2)
    en = AutoEnroller(other_s=0.0)
    hints = _run(_at(-58.0, 3 * SR, rng), _ctx(0.1), en)
    assert hints[-1] == "waiting for them to talk"


def test_loud_room_keeps_the_absolute_gate():
    """A floor at -30 dBFS never raises the threshold above -40 dBFS (the old behaviour)."""
    rng = np.random.default_rng(3)
    g = VoicedGate()
    for _ in range(300):
        g.voiced(_at(-30.0, HOP, rng))
    assert g.threshold_db == -40.0
    assert g.voiced(_at(-38.0, HOP, rng))


def test_digital_silence_floor_clamps_at_minimum():
    g = VoicedGate()
    for _ in range(300):
        g.voiced(np.zeros(HOP, np.float32))
    assert g.threshold_db == enroll.VOICED_MIN_DB
    assert not g.voiced(_at(-60.0, HOP, np.random.default_rng(4)))


# --- T-145: the built-in MacBook mic (speech -50..-35 dBFS over a -58 dBFS room) ----------------------

def _words(seconds: float, speech_db: float, seed: int = 5) -> np.ndarray:
    """Words that decay (a 400 ms word whose second half is 8 dB softer: tails and soft consonants)."""
    rng = np.random.default_rng(seed)
    n = round(seconds * SR)
    x = _at(-58.0, n, rng)
    pos = SR
    while pos + round(0.4 * SR) < n:
        x[pos:pos + 3200] += _at(speech_db, 3200, rng)
        x[pos + 3200:pos + 6400] += _at(speech_db - 8.0, 3200, rng)
        pos += 6400 + round(0.15 * SR)
    return x


def test_hysteresis_keeps_word_tails_at_minus_45():
    """At -45 dBFS the soft half of each word sits at -53, under the -49 dB open threshold: held open."""
    en = AutoEnroller(other_s=0.0)
    _run(_words(6.0, -45.0), _ctx(0.9), en)
    assert en.ready and en.collected_s >= enroll.ENROLL_S


def _voiced_chunks(x: np.ndarray) -> int:
    g = VoicedGate()
    return sum(g.voiced(x[i * HOP:(i + 1) * HOP]) for i in range(len(x) // HOP))


def test_without_hysteresis_the_tails_are_lost(monkeypatch):
    x = _words(6.0, -45.0)
    held = _voiced_chunks(x)
    monkeypatch.setattr(enroll, "HOLD_DB", 0.0)
    plain = _voiced_chunks(x)
    assert held > 1.5 * plain  # the soft halves (-53 dBFS) only pass with the hold


def test_hold_never_opens_on_the_room_alone():
    """The hold only follows a voiced chunk and stays HOLD_FLOOR_DB above the floor: white room noise
    at -58 dBFS is (almost) never voiced."""
    g = VoicedGate()
    x = _at(-58.0, 10 * SR, np.random.default_rng(6))
    v = [g.voiced(x[i * HOP:(i + 1) * HOP]) for i in range(len(x) // HOP)]
    assert np.mean(v[300:]) < 0.01


def test_team_eval_helpers_on_synthetic_speech():
    """eval.enroll_eval --team: level scaling, the meter model and one replay, no weights."""
    from eval import enroll_eval as ee

    take = _words(14.0, -20.0, seed=7)
    act = ee.active_chunks(take)
    assert 0.3 < act.mean() < 0.9
    y = ee.scale_to_active(take, -45.0, act)
    c = y[: len(act) * HOP].reshape(-1, HOP)[act]
    assert abs(20 * np.log10(np.sqrt(np.mean(c.astype(np.float64) ** 2))) + 45.0) < 0.01
    # meter: a steady 0.01 peak reads -40 dBFS; after it stops the level falls 24 dB/s
    m = ee.meter_levels(np.concatenate([np.full(SR, 0.01, np.float32), np.zeros(SR // 2, np.float32)]))
    assert abs(m[19] + 40.0) < 0.01 and abs(m[29] + 52.0) < 1.5
    scene = ee.team_scene(take, -45.0, -58.0, 0.0, 8.0, seed=0)
    res = ee.team_replay(scene)
    assert res["ready_s"] is not None and res["ready_s"] <= 3.0
    assert res["kept_share"] >= 0.6
    assert -50.0 < res["meter_p10_db"] < -38.0
