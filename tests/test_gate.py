"""Target gate: streaming equivalence, duck depth, bounded envelope slope, bypass."""

from __future__ import annotations

import time

import numpy as np
import pytest

import hearing.fusion  # noqa: F401  (registers "gate")
from hearing import registry
from hearing.fusion.gate import Gate
from hearing.fusion.mixer import DUCK_FLOOR_DB, duck_gain, room_gain
from hearing.testing import assert_streaming_equivalent, make_test_signal
from hearing.types import AudioChunk, FaceTrack, Stage, StageContext, TargetState

SR = 16000
HOP = 160


def make_ctx(
    p_speaking: float | None = 0.9, suppression: float = 0.9, target_id: int | None = 7
) -> StageContext:
    """Context with one tracked face; ``p_speaking=None`` drops the track entirely."""
    ctx = StageContext.default()
    ctx.suppression = suppression
    ctx.target = TargetState(target_id, 0.9, False, 0.0)
    ctx.tracks = []
    if p_speaking is not None:
        ctx.tracks.append(FaceTrack(7, (0.4, 0.3, 0.2, 0.3), None, p_speaking, 0.0))
    return ctx


def run_scripted(stage: Stage, x: np.ndarray, p_of_t, suppression: float) -> tuple[np.ndarray, np.ndarray]:
    """Stream ``x`` in 10 ms hops, taking the target's p_speaking from ``p_of_t(seconds)``.

    Returns the output and the per-chunk gain reported in telemetry.
    """
    outs, gains = [], []
    for i in range(0, len(x), HOP):
        t = i / SR
        ctx = make_ctx(p_of_t(t), suppression)
        chunk = AudioChunk(x[i : i + HOP], SR, i // HOP, t)
        outs.append(stage.process(chunk, ctx).samples)
        gains.append(ctx.telemetry.extras["gate"]["gain"])
    return np.concatenate(outs), np.asarray(gains)


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def speak_silent_speak(t: float) -> float:
    return 0.0 if 1.0 <= t < 2.0 else 0.9


# ----------------------------------------------------------------- registration
def test_registered_as_stage_with_kwargs():
    stage = registry.get("gate")
    assert isinstance(stage, Stage)
    assert stage.sample_rate == SR and stage.latency_ms == 0
    custom = registry.get("gate", release_ms=400, attack_ms=10, hold_ms=0)
    assert custom.release_ms == 400 and custom.attack_ms == 10 and custom.hold_ms == 0
    pipeline = registry.build_pipeline("gate(release_ms=400)")
    assert pipeline.stages[0].release_ms == 400


def test_invalid_kwargs_raise():
    with pytest.raises(ValueError):
        Gate(attack_ms=-1)
    with pytest.raises(ValueError):
        Gate(on_threshold=0.3, off_threshold=0.5)


# ------------------------------------------------------- (a) streaming equivalence
@pytest.mark.parametrize("p_speaking", [0.9, 0.1, None])
def test_streaming_equivalent_fixed_ctx(p_speaking):
    x = make_test_signal(3.0, SR, kind="speechlike", seed=2)
    assert_streaming_equivalent(Gate, x, ctx=make_ctx(p_speaking, suppression=0.9))


@pytest.mark.parametrize("hop", [1, 37, 160, 1000])
def test_streaming_equivalent_odd_hops(hop):
    x = make_test_signal(0.6, SR, kind="speechlike", seed=3)
    assert_streaming_equivalent(Gate, x, hop=hop, ctx=make_ctx(0.1, suppression=0.7))


# --------------------------------------------------------------- (b) duck depth
@pytest.mark.parametrize("s", [0.9, 0.5])
def test_silent_second_ducked_by_slider(s):
    x = make_test_signal(3.0, SR, kind="noise", seed=4)  # Stationary so RMS is comparable.
    out, _ = run_scripted(Gate(), x, speak_silent_speak, suppression=s)
    # Measure after hold (400 ms) + release (300 ms) have fully settled.
    settled = slice(int(1.75 * SR), int(2.0 * SR))
    expected_db = 20.0 * np.log10(duck_gain(s))  # the T-093 slider curve (-27 dB / -15 dB here)
    assert abs((rms_db(out[settled]) - rms_db(x[settled])) - expected_db) < 1.0
    # Speaking seconds are untouched once the attack has completed.
    assert np.allclose(out[: SR], x[: SR])
    assert np.allclose(out[int(2.1 * SR) :], x[int(2.1 * SR) :])


def test_slider_zero_leaves_silence_untouched():
    x = make_test_signal(3.0, SR, kind="noise", seed=4)
    out, gains = run_scripted(Gate(), x, speak_silent_speak, suppression=0.0)
    assert np.array_equal(out, x)
    assert np.all(gains == 1.0)


def test_hold_bridges_short_pauses():
    x = make_test_signal(1.0, SR, kind="noise", seed=5)
    short_pause = lambda t: 0.1 if 0.40 <= t < 0.50 else 0.9  # 100 ms < 150 ms hold
    out, gains = run_scripted(Gate(), x, short_pause, suppression=0.9)
    assert np.array_equal(out, x)
    assert np.all(gains == 1.0)


def test_hysteresis_keeps_gate_open_between_thresholds():
    stage = Gate(hold_ms=0, on_threshold=0.5, off_threshold=0.35, partial_lo=0.35)  # no partial duck
    x = np.ones(HOP, dtype=np.float32)
    stage.process(AudioChunk(x, SR, 0, 0.0), make_ctx(0.9, 0.9))  # on
    ctx = make_ctx(0.4, 0.9)  # between off (0.35) and on (0.5): stays open
    assert np.array_equal(stage.process(AudioChunk(x, SR, 1, 0.01), ctx).samples, x)
    ctx = make_ctx(0.3, 0.9)  # below off: closes
    assert stage.process(AudioChunk(x, SR, 2, 0.02), ctx).samples[-1] < 1.0
    ctx = make_ctx(0.4, 0.9)  # between thresholds while closed: stays closed
    stage.process(AudioChunk(x, SR, 3, 0.03), ctx)
    assert ctx.telemetry.extras["gate"]["gain"] < 1.0


# ------------------------------------------------------- (c) bounded envelope slope
def test_envelope_slope_bounded_no_clicks():
    stage = Gate()
    x = np.ones(3 * SR, dtype=np.float32)  # DC so the output IS the gain envelope.
    env, _ = run_scripted(stage, x, speak_silent_speak, suppression=0.9)
    d = np.diff(env.astype(np.float64))
    tol = 1e-6
    assert d.max() <= stage.attack_step + tol
    assert d.min() >= -stage.release_step - tol
    floor = duck_gain(0.9)  # room_gain(0.9) = -27 dB, above the -30 dB floor
    assert env.min() >= floor - tol and env.max() <= 1.0 + tol
    # Slopes are defined for a full 0 <-> 1 swing, so a (1 - floor) swing takes that share of
    # release_ms (300) and attack_ms (30), within a chunk of slack.
    swing = 1.0 - floor
    falling = np.flatnonzero(d < -tol)
    rising = np.flatnonzero(d > tol)
    assert abs((falling[-1] - falling[0]) / SR - swing * 0.300) < 0.012
    assert abs((rising[-1] - rising[0]) / SR - swing * 0.030) < 0.012
    # Release begins only after the 400 ms hold.
    assert abs(falling[0] / SR - 1.400) < 0.012


# ------------------------------------------------------------------ (d) bypass
@pytest.mark.parametrize("ctx", [make_ctx(0.1, 0.9, target_id=None), make_ctx(None, 0.9)])
def test_bypass_without_target(ctx):
    stage = Gate()
    x = make_test_signal(0.5, SR, kind="speechlike", seed=6)
    out = stage.process(AudioChunk(x, SR, 3, 1.5), ctx)
    assert np.array_equal(out.samples, x)
    assert (out.seq, out.capture_ts, out.sample_rate) == (3, 1.5, SR)
    assert ctx.telemetry.extras["gate"] == {"gain": 1.0, "goal": 1.0, "active": False, "stale": False}


def test_bypass_releases_duck_smoothly_and_reset_clears():
    stage = Gate()
    x = np.ones(SR, dtype=np.float32)
    stage.process(AudioChunk(x, SR, 0, 0.0), make_ctx(0.0, 0.9))  # fully ducked after 1 s
    assert stage.gain == pytest.approx(room_gain(0.9))
    out = stage.process(AudioChunk(x[:HOP], SR, 1, 1.0), make_ctx(0.0, 0.9, target_id=None))
    assert out.samples[0] < 1.0 and out.samples[-1] == pytest.approx(room_gain(0.9) + 160 * stage.attack_step)
    stage.reset()
    assert stage.gain == 1.0
    ctx = make_ctx(0.0, 0.9)
    stage.process(AudioChunk(x[:HOP], SR, 2, 1.01), ctx)
    assert ctx.telemetry.extras["gate"]["gain"] == pytest.approx(1.0 - 160 * stage.release_step)


@pytest.mark.parametrize("suppression", [0.75, 1.0])
def test_no_target_holds_unity_gain_at_any_slider(suppression):
    """T-102 / Q-233: Natural (dfn_ll>gate>mixer) with no camera lock must not cut the talker.

    A fresh session context (no face, nothing selected) streamed for 2 s at s=75/100 keeps
    the gate at exactly 1.0: it only ducks when a target exists and ASD says it is silent."""
    stage = Gate()
    ctx = StageContext.default()
    ctx.suppression = suppression
    assert ctx.target.track_id is None and not ctx.tracks
    x = make_test_signal(2.0, SR, kind="speechlike", seed=9)
    gains = []
    for i in range(0, len(x), HOP):
        out = stage.process(AudioChunk(x[i : i + HOP], SR, i // HOP, i / SR), ctx)
        assert np.array_equal(out.samples, x[i : i + HOP])
        gains.append(ctx.telemetry.extras["gate"]["gain"])
    assert set(gains) == {1.0}


# ------------------------------------------------------------------- compute cost
def measure_compute_ms(n_chunks: int = 2000) -> dict[str, float]:
    """Wall time per 10 ms chunk while ramping (the slowest path) and while open."""
    stage = Gate()
    x = make_test_signal(1.0, SR, kind="speechlike", seed=8)[:HOP]
    results = {}
    for label, p in (("ramping", None), ("open", 0.9)):
        ctx = make_ctx(0.9, 0.9)
        times = []
        for i in range(n_chunks):
            if p is None:
                ctx.tracks[0].p_speaking = 0.9 if (i // 20) % 2 else 0.1  # toggle every 200 ms
            else:
                ctx.tracks[0].p_speaking = p
            t0 = time.perf_counter()
            stage.process(AudioChunk(x, SR, i, i * 0.01), ctx)
            times.append(1000.0 * (time.perf_counter() - t0))
        results[label] = {"p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
    return results


def test_compute_per_chunk_under_budget():
    results = measure_compute_ms(500)
    print(f"\ngate compute per 10 ms chunk: {results}")
    assert results["ramping"]["p95_ms"] < 0.5


if __name__ == "__main__":
    from hearing import device

    print({"device": device.describe()["machine"], **measure_compute_ms()})


def test_max_duck_db_bounds_the_duck():
    """T-095: at suppression 100 the gate ducks to its -30 dB floor, with max_duck_db=20 only 20 dB."""
    x = np.full(3 * SR, 0.1, np.float32)
    _, free = run_scripted(Gate(hold_ms=0), x, speak_silent_speak, 1.0)
    _, bounded = run_scripted(Gate(hold_ms=0, max_duck_db=20), x, speak_silent_speak, 1.0)
    assert free.min() == pytest.approx(duck_gain(1.0), abs=1e-6)
    assert bounded.min() == pytest.approx(0.1, abs=1e-6)
    assert bounded[-1] == pytest.approx(1.0)
    # a slider below the cap is unchanged (room_gain(0.5) = -15 dB)
    _, mild = run_scripted(Gate(hold_ms=0, max_duck_db=20), x, speak_silent_speak, 0.5)
    assert mild.min() == pytest.approx(room_gain(0.5), abs=1e-6)
    assert registry.build_pipeline("gate(max_duck_db=20)").stages[0].max_duck_db == 20.0
    with pytest.raises(ValueError):
        Gate(max_duck_db=-1)


# ---------------------------------------------------------- T-096: floor, partial duck, stale ASD
def two_face_ctx(p_target: float, p_other: float, suppression: float = 1.0, seen: float = 0.0) -> StageContext:
    ctx = make_ctx(p_target, suppression)
    ctx.tracks[0].last_seen_ts = seen
    ctx.tracks.append(FaceTrack(9, (0.7, 0.3, 0.2, 0.3), None, p_other, seen))
    return ctx


def test_duck_gain_curve_floors_at_minus_30_db():
    floor = 10 ** (DUCK_FLOOR_DB / 20)
    assert duck_gain(0.0) == 1.0 and duck_gain(0.5) == pytest.approx(room_gain(0.5))  # -15 dB
    assert duck_gain(0.9) == pytest.approx(room_gain(0.9))  # -27 dB: the T-093 curve above the floor
    assert duck_gain(1.0) == pytest.approx(floor) and duck_gain(0.99) == pytest.approx(floor)
    assert duck_gain(float("nan")) == 1.0  # fail open
    gains = [duck_gain(s) for s in np.linspace(0, 1, 101)]
    assert np.all(np.diff(gains) <= 0)  # monotone


def test_suppression_100_is_not_digital_silence():
    stage = Gate()
    x = np.ones(SR, dtype=np.float32)
    out = stage.process(AudioChunk(x, SR, 0, 0.0), make_ctx(0.0, 1.0)).samples
    assert out[-1] == pytest.approx(10 ** (-30 / 20), rel=1e-4)


@pytest.mark.parametrize(("p_t", "p_o", "lo_db", "hi_db"), [
    (0.9, 0.0, 0.0, 0.0),  # target alone: open
    (0.9, 0.9, -6.5, -5.5),  # both talking, equal ASD: halfway (-6 dB of -12)
    (0.9, 0.5, -1.0, 0.0),  # target clearly ahead (margin 0.4 >= span): open
    (0.35, 0.9, -12.1, -11.9),  # target barely on, other ahead: overlap floor
])
def test_both_talking_partial_gain_from_margin(p_t, p_o, lo_db, hi_db):
    stage = Gate()
    x = np.ones(SR, dtype=np.float32)
    ctx = two_face_ctx(p_t, p_o)
    out = stage.process(AudioChunk(x, SR, 0, 0.0), ctx).samples
    g_db = 20 * np.log10(out[-1])
    assert lo_db <= g_db <= hi_db
    assert ctx.telemetry.extras["gate"]["goal"] == pytest.approx(out[-1], rel=1e-4)


def test_overlap_gain_never_below_slider_duck():
    stage = Gate()
    duck = duck_gain(0.1)  # -3 dB
    assert stage.open_gain(0.3, 0.9, duck) == pytest.approx(duck)  # -12 dB < duck


def test_lip_confidence_partial_duck():
    stage = Gate()
    duck = duck_gain(1.0)
    assert stage.closed_gain(0.0, duck) == duck  # sure it is silent: full duck
    assert stage.closed_gain(stage.partial_lo, duck) == pytest.approx(duck)
    mid = stage.closed_gain((stage.partial_lo + stage.off_threshold) / 2, duck)
    assert 20 * np.log10(mid) == pytest.approx((-30 + -6) / 2, abs=0.01)  # linear in dB
    # just under off_threshold: -6 dB
    assert 20 * np.log10(stage.closed_gain(stage.off_threshold - 1e-9, duck)) == pytest.approx(-6, abs=0.01)
    # a gentle slider (duck above -6 dB) is never deepened by the partial duck
    assert stage.closed_gain(0.15, duck_gain(0.1)) == pytest.approx(duck_gain(0.1))  # -3 dB


def test_stale_asd_holds_last_gain():
    stage = Gate()
    x = np.ones(SR // 2, dtype=np.float32)
    stage.process(AudioChunk(x, SR, 0, 10.0), make_ctx(0.9, 1.0))  # open
    ctx = two_face_ctx(0.0, 0.0, seen=10.0)  # says "silent" but the frame is 300 ms old
    out = stage.process(AudioChunk(x, SR, 1, 10.3), ctx).samples
    assert np.all(out == 1.0) and ctx.telemetry.extras["gate"]["stale"] is True
    ctx = two_face_ctx(0.0, 0.0, seen=10.4)  # fresh frame (100 ms old): hold, then ducks
    stage.process(AudioChunk(np.ones(SR, dtype=np.float32), SR, 2, 10.5), ctx)
    assert stage.gain < 0.1 and ctx.telemetry.extras["gate"]["stale"] is False
    ctx = two_face_ctx(0.9, 0.0, seen=11.0)  # stale while ducked: stays ducked, no jump
    out = stage.process(AudioChunk(x, SR, 3, 11.5), ctx).samples
    assert np.allclose(out, stage.gain)


def test_unrelated_clocks_do_not_freeze_the_gate():
    stage = Gate()
    x = np.ones(SR // 2, dtype=np.float32)
    ctx = two_face_ctx(0.0, 0.0, seen=5.0)
    stage.process(AudioChunk(x, SR, 0, 1_000_000.0), ctx)  # ages > 10 s: clocks differ, trust p
    assert ctx.telemetry.extras["gate"]["stale"] is False and stage.gain < 1.0


def test_click_free_under_random_two_face_telemetry():
    """Random p for both faces every 40 ms (vision rate) and a moving slider: the envelope never
    steps faster than one attack/release step per sample, so no clicks."""
    stage = Gate()
    x = np.ones(6 * SR, dtype=np.float32)
    outs = []
    for i in range(0, len(x), HOP):
        k = i // 640
        r = np.random.default_rng(k)
        ctx = two_face_ctx(float(r.random()), float(r.random()), suppression=float(r.choice([0.5, 0.8, 1.0])))
        outs.append(stage.process(AudioChunk(x[i : i + HOP], SR, i // HOP, i / SR), ctx).samples)
    env = np.concatenate(outs).astype(np.float64)
    d = np.diff(env)
    assert d.max() <= stage.attack_step + 1e-6 and d.min() >= -stage.release_step - 1e-6
    assert env.min() >= 10 ** (-30 / 20) - 1e-6


def test_streaming_equivalent_two_faces():
    x = make_test_signal(2.0, SR, kind="speechlike", seed=9)
    assert_streaming_equivalent(Gate, x, ctx=two_face_ctx(0.8, 0.7, suppression=0.9))
