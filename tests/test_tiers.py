"""The fallback ladder: eligibility, hysteresis, per-tier delays and crossfaded switches."""

import itertools

import numpy as np
import pytest

from hearing import registry
from hearing.audio import (
    basic as _basic,  # noqa: F401  (registers passthrough, gain, delay)
)
from hearing.fusion import tiers
from hearing.fusion.tiers import TIERS, LadderConfig, TierLadder, TierPolicy
from hearing.types import AudioChunk, FaceTrack, StageContext, TargetState

SR = 16000
HOP = 160
HEALTH = {}  # key -> (ok, reason): what the "flag" test stage reports


class Flag:
    """Pass-through stage whose health the test sets (stands in for tse / remote)."""

    sample_rate = SR
    latency_ms = 0.0

    def __init__(self, key: str = "hq", crash: bool = False):
        self.key, self.crash = key, crash
        self.name = f"t019_flag({key})"

    def process(self, chunk, ctx):
        if self.crash and HEALTH.get(self.key, (True, ""))[0] == "crash":
            raise RuntimeError("model exploded")
        return chunk

    def reset(self):
        pass

    def health(self):
        ok, reason = HEALTH.get(self.key, (True, ""))
        return {"ok": ok is True, "reason": reason}


if "t019_flag" not in registry.names():
    registry.register("t019_flag")(Flag)


def ladder(**overrides) -> TierLadder:
    config = {
        "crossfade_ms": 50,
        "upgrade_after_s": 0.5,
        "tiers": [
            {"name": "focus_hq", "pipeline": 't019_flag(key="hq", crash=true)>delay(ms=800)',
             "needs": ["camera", "target", "healthy"]},
            {"name": "enrolled", "pipeline": 't019_flag(key="enr")>gain(db=-6)>delay(ms=800)',
             "needs": ["healthy"]},
            {"name": "gate", "pipeline": "gain(db=-12)", "needs": ["camera", "target"]},
            {"name": "audio-only", "pipeline": ["no_such_stage", "gain(db=-20)>delay(ms=400)"]},
        ],
        **overrides,
    }
    return TierLadder(config)


def camera(ctx: StageContext, on: bool, confidence: float = 0.9) -> None:
    if on:
        ctx.tracks = [FaceTrack(3, (0.4, 0.3, 0.2, 0.4), None, 0.9, 0.0)]
        ctx.target = TargetState(3, confidence, False, 0.0)
    else:
        ctx.tracks, ctx.target = [], TargetState(None, 0.0, False, 0.0)


GAINS = {"focus_hq": 1.0, "enrolled": 10 ** (-6 / 20), "gate": 10 ** (-12 / 20), "audio-only": 0.1}
DELAYS = {"focus_hq": 800, "enrolled": 800, "gate": 0, "audio-only": 400}  # ms


def delayed(x: np.ndarray, ms: float) -> np.ndarray:
    d = round(ms * SR / 1000)
    return np.concatenate([np.zeros(d, np.float32), x[: len(x) - d]])


def longest_run(mask: np.ndarray) -> int:
    best = run_ = 0
    for m in mask:
        run_ = run_ + 1 if m else 0
        best = max(best, run_)
    return best


def run(lad: TierLadder, ctx: StageContext, seconds: float, events=None, t0: float = 0.0):
    """Feed a 1 kHz-free DC-offset tone; returns (input, output, tier per hop)."""
    n = round(seconds * SR)
    t = (t0 * SR + np.arange(n)) / SR
    x = (0.3 + 0.2 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)  # slow, never zero
    ys, tiers_ = [], []
    for i in range(0, n, HOP):
        if events and i in events:
            events[i](ctx)
        ys.append(lad.process(AudioChunk(x[i : i + HOP], SR, i // HOP, (t0 * SR + i) / SR), ctx).samples)
        tiers_.append(ctx.telemetry.tier)
    return x, np.concatenate(ys), tiers_


@pytest.fixture(autouse=True)
def _health():
    HEALTH.clear()
    yield
    HEALTH.clear()


def _tse_args(spec: str) -> dict:
    """kwargs of the ``tse(...)`` stage in a pipeline spec."""
    stage = next(part for part in registry._split(spec, ">") if part.strip().startswith("tse("))
    name, kwargs = registry._parse_stage(stage)
    assert name == "tse"
    return kwargs


def test_default_config_is_the_bakeoff_v2_ladder():
    """docs/decisions/focus-hq-bakeoff.md: enrolled (1 s / 0.5 s local, 0.5 s / 0.25 s remote)
    -> AV-MossFormer2 at 2 s (cold start, before enrollment) -> gate -> audio-only."""
    cfg = LadderConfig.load()
    assert [t.name for t in cfg.tiers] == ["enrolled", "focus_hq", "gate", "audio-only"]
    enrolled, av, gate, audio_only = cfg.tiers
    assert enrolled.needs == {"healthy"}  # health is "enrolling" until 3 s of ASD-gated speech
    assert enrolled.avoid == {"same_bucket"}  # T-070: same-gender pairs skip voice matching
    assert not (av.avoid or gate.avoid or audio_only.avoid)
    assert av.needs == {"camera", "target", "healthy"}
    assert gate.needs == {"camera", "target"}
    assert audio_only.needs == frozenset()

    local_enr, local_av = _tse_args(enrolled.pipelines[0]), _tse_args(av.pipelines[0])
    assert local_enr == {"candidate": "enrolled", "window_s": 1.0, "hop_s": 0.5, "threaded": True,
                         "require_gpu": True}  # T-089: threaded explicitly
    assert local_av == {"candidate": "av_mossformer2", "window_s": 2.0, "hop_s": 1.0, "require_gpu": True}
    assert enrolled.pipelines[0].endswith(">mixer(max_duck_db=20)")  # T-095: bounded duck, no holes
    assert all(p.endswith(">mixer") for p in av.pipelines)

    import yaml

    remote = yaml.safe_load(tiers.DEFAULT_CONFIG.read_text())["remote"]
    remote_enr, remote_av = _tse_args(remote["enrolled"]), _tse_args(remote["focus_hq"])
    assert (remote_enr["candidate"], remote_enr["window_s"], remote_enr["hop_s"]) == ("enrolled", 0.5, 0.125)  # T-086
    assert (remote_av["candidate"], remote_av["window_s"], remote_av["hop_s"]) == ("av_mossformer2", 2.0, 1.0)
    for spec in remote.values():
        assert _tse_args(spec)["require_gpu"] is True and spec.endswith(">mixer")


def test_default_ladder_builds_and_each_tier_keeps_its_own_delay(monkeypatch):
    """Models load lazily, so the default ladder builds without weights. Each tier plays at its
    own delay: enrolled 1.5 s (1 s window + 0.5 s hop), AV-MossFormer2 3 s (2 s + 1 s). Before
    the first chunk the ladder declares the slowest tier's delay, the safe bound."""
    from hearing.tse import stage as _tse  # noqa: F401  (registers tse)

    monkeypatch.setenv("HEARING_DEVICE", "cuda")  # the AV tier is built on CUDA hosts only
    lad = TierLadder(only="enrolled,focus_hq,audio-only")
    try:
        assert lad.order == ("enrolled", "focus_hq", "audio-only")
        assert lad.built["enrolled"].delay_ms == 1500.0
        assert lad.built["focus_hq"].delay_ms == 3000.0
        assert lad.latency_ms == 3000.0  # nothing plays yet
        status = lad.status()
        assert {t["name"]: t["delay_ms"] for t in status["tiers"]}["enrolled"] == 1500.0
    finally:
        lad.close()


@pytest.mark.parametrize(("host", "built"), [("cuda", True), ("mps", False), ("cpu", False)])
def test_av_tier_is_built_only_on_cuda(monkeypatch, host, built):
    """T-072: on the Mac (MPS) AV-MossFormer2 overran its 1 s hop and starved vision while never
    becoming eligible, so the default ladder does not build it there (``devices: [cuda]``)."""
    from hearing.tse import stage as _tse  # noqa: F401  (registers tse)

    monkeypatch.setenv("HEARING_DEVICE", host)
    spec = next(t for t in LadderConfig.load().tiers if t.name == "focus_hq")
    assert spec.devices == frozenset({"cuda"})
    lad = TierLadder(only="enrolled,focus_hq,gate,audio-only")
    try:
        assert ("focus_hq" in lad.built) is built
        assert lad.latency_ms == (3000.0 if built else 1500.0)  # slowest BUILT tier before chunk 1
        if not built:
            assert lad.unavailable["focus_hq"] == f"needs cuda, this host is {host}"
            row = next(t for t in lad.status()["tiers"] if t["name"] == "focus_hq")
            assert not row["eligible"] and row["reason"].startswith("unavailable: needs cuda")
    finally:
        lad.close()


def test_config_errors_are_reported():
    with pytest.raises(ValueError, match="unknown devices"):
        LadderConfig.from_dict({"tiers": [{"name": "a", "pipeline": "passthrough", "devices": ["tpu"]}]})
    with pytest.raises(ValueError, match="unknown needs"):
        LadderConfig.from_dict({"tiers": [{"name": "a", "pipeline": "passthrough", "needs": ["gpu"]}]})
    with pytest.raises(ValueError, match="unknown avoid_when"):
        LadderConfig.from_dict({"tiers": [{"name": "a", "pipeline": "passthrough", "avoid_when": "rain"}]})
    assert LadderConfig.from_dict(
        {"tiers": [{"name": "a", "pipeline": "passthrough", "avoid_when": "same_bucket"}]}
    ).tiers[0].avoid == {"same_bucket"}
    with pytest.raises(ValueError, match="at least one"):
        LadderConfig.from_dict({"tiers": []})
    with pytest.raises(ValueError, match="no tier could be built"):
        TierLadder({"tiers": [{"name": "a", "pipeline": "no_such_stage"}]})


def test_policy_downgrades_at_once_and_upgrades_after_the_hold():
    p = TierPolicy(TIERS, upgrade_after_s=1.0)
    assert p.choose([False, False, True, True], 0.0) == 2  # start: best available now
    assert p.choose([True, False, True, True], 0.1) == 2  # better tier, not long enough
    assert p.choose([True, False, True, True], 1.05) == 2
    assert p.choose([True, False, True, True], 1.1) == 0  # eligible for 1.0 s: move up
    assert p.choose([False, False, True, True], 1.2) == 2  # lost: down at once
    assert p.choose([True, False, True, True], 1.3) == 2
    assert p.choose([False, False, True, True], 1.4) == 2  # flapping restarts the hold
    assert p.choose([True, False, True, True], 2.0) == 2
    assert p.choose([False, False, False, False], 2.1) == 3  # nothing eligible: the last tier


def test_each_tier_has_its_own_delay_and_unbuildable_specs_fall_through():
    lad = ladder()
    assert {k: b.delay_ms for k, b in lad.built.items()} == DELAYS
    assert lad.latency_ms == 800.0  # before the first chunk: the slowest tier
    assert lad.built["audio-only"].pipeline.name == "gain(db=-20)>delay(ms=400)"


def test_fade_kind_follows_the_jump_in_delay():
    assert tiers.fade_kind(0.0) == "aligned"
    assert tiers.fade_kind(-0.5) == "aligned"
    assert tiers.fade_kind(13.0) == "splice"
    assert tiers.fade_kind(-500.0) == "splice"
    assert tiers.fade_kind(1500.0) == "dip" and tiers.fade_kind(-1487.0) == "dip"
    n = 1000
    pos = np.arange(n)
    old, new = tiers.fade_gains("aligned", pos, n)
    np.testing.assert_allclose(old + new, 1.0, atol=1e-12)
    old, new = tiers.fade_gains("splice", pos, n)
    np.testing.assert_allclose(old**2 + new**2, 1.0, atol=1e-12)
    old, new = tiers.fade_gains("dip", pos, n)
    assert np.all(old[n // 2 :] < 1e-12) and np.all(new[: n // 2] < 1e-12)  # never both at once
    for g in (old, 1 - new):
        assert g[0] > 0.99 and g[-1] < 0.01 and np.all(np.diff(g) <= 0)
    with pytest.raises(ValueError, match="unknown fade"):
        tiers.fade_gains("nope", pos, n)


def test_tier_switches_follow_camera_and_health_with_crossfades():
    """The acceptance scenario: gate -> focus_hq (after the hold) -> enrolled (camera lost)
    -> audio-only (enrollment unhealthy). Each tier plays at its own delay; the switches are a
    dip (0 -> 800 ms), an aligned crossfade (800 -> 800 ms) and a splice (800 -> 400 ms)."""
    lad, ctx = ladder(), StageContext.default()
    camera(ctx, True)
    HEALTH["hq"] = (False, "loading")
    HEALTH["enr"] = (False, "enrolling")
    second = SR  # samples
    events = {
        0: lambda c: None,
        1 * second: lambda c: HEALTH.update(hq=(True, "")),  # model loaded
        3 * second: lambda c: HEALTH.update(enr=(True, "")),  # 3 s of enrollment collected
        4 * second: lambda c: camera(c, False),  # user looks away: no face
        5 * second: lambda c: HEALTH.update(enr=(False, "overrun")),  # GPU too slow
    }
    x, y, seen = run(lad, ctx, 6.0, events)
    hop_s = HOP / SR
    at = {round(i * hop_s, 2): t for i, t in enumerate(seen)}
    assert at[0.5] == "gate"  # HQ not loaded yet; camera + target: gate
    assert at[1.4] == "gate"  # loaded but inside the 0.5 s upgrade hold
    assert at[1.6] == "focus_hq"
    assert at[3.9] == "focus_hq"  # enrolled became ready, but HQ is better
    assert at[4.05] == "enrolled"  # camera gone: down at once to the enrolled tier
    assert at[5.05] == "audio-only"
    assert [h[1] for h in lad.history] == ["gate", "focus_hq", "enrolled", "audio-only"]
    assert lad.switches == 3
    assert "no camera" in lad.history[2][2] and "overrun" in lad.history[3][2]

    # Away from switches, output = the input delayed by the PLAYING tier's own delay x its gain.
    for t_s, tier in ((1.0, "gate"), (2.5, "focus_hq"), (4.5, "enrolled"), (5.8, "audio-only")):
        i = round(t_s * SR)
        ref = GAINS[tier] * delayed(x, DELAYS[tier])
        np.testing.assert_allclose(y[i : i + 800], ref[i : i + 800], rtol=1e-5)
    fade = round(0.05 * SR)
    # Same delay (focus_hq -> enrolled): the gain glides between two versions of one moment.
    i = round(4.0 * SR)
    ratio = y[i : i + fade] / delayed(x, 800)[i : i + fade]
    assert GAINS["enrolled"] - 1e-4 <= ratio.min() and ratio.max() <= GAINS["focus_hq"] + 1e-4
    assert np.all(np.diff(ratio) <= 1e-6)  # monotone glide down
    assert abs(ratio[fade // 2] - (GAINS["focus_hq"] + GAINS["enrolled"]) / 2) < 0.05
    # Every switch is over within the crossfade, and none clicks or leaves a gap.
    for switch_s in (1.5, 4.0, 5.0):
        i = round(switch_s * SR)
        tier = seen[round(switch_s * SR) // HOP + 1]
        after = GAINS[tier] * delayed(x, DELAYS[tier])
        np.testing.assert_allclose(y[i + fade : i + 2 * fade], after[i + fade : i + 2 * fade], rtol=1e-5)
    assert np.max(np.abs(np.diff(y))) < 0.01  # no step anywhere in the stream
    assert longest_run(np.abs(y) < 0.005) < fade  # the dip is shorter than the crossfade

    status = ctx.telemetry.extras["tiers"]
    assert status["active"] == "audio-only" and status["delay_ms"] == 400.0  # its own, not 800
    assert lad.latency_ms == 400.0
    assert [t["name"] for t in status["tiers"]] == list(TIERS)
    assert [t["delay_ms"] for t in status["tiers"]] == [DELAYS[t] for t in TIERS]
    assert status["tiers"][0]["reason"] == "no camera"


@pytest.mark.parametrize(("fast_ms", "slow_ms", "kind"), [
    (100.0, 100.0, "aligned"), (13.0, 250.0, "splice"), (13.0, 1500.0, "dip")])
def test_a_tone_through_two_tiers_switches_without_click_or_gap(fast_ms, slow_ms, kind):
    """A 440 Hz tone through two tiers of different delay, switched up (fast -> slow, after the
    hold) and down (slow -> fast, at once): no step larger than the tone's own, and no stretch
    quieter than -20 dB longer than the crossfade."""
    cfg = {"crossfade_ms": 150, "upgrade_after_s": 0.0, "tiers": [
        {"name": "slow", "pipeline": f't019_flag(key="slow")>delay(ms={slow_ms})', "needs": ["healthy"]},
        {"name": "fast", "pipeline": f"gain(db=-3)>delay(ms={fast_ms})"}]}
    lad, ctx = TierLadder(cfg), StageContext.default()
    HEALTH["slow"] = (False, "warming up")
    n, amp = round(4.0 * SR), 0.5
    x = (amp * np.sin(2 * np.pi * 440 * np.arange(n) / SR)).astype(np.float32)
    ys, delays = [], []
    for i in range(0, n, HOP):
        if i == round(1.8 * SR):
            HEALTH["slow"] = (True, "")  # up: fast -> slow
        if i == round(3.0 * SR):
            HEALTH["slow"] = (False, "overrun")  # down: slow -> fast
        ys.append(lad.process(AudioChunk(x[i : i + HOP], SR, i // HOP, i / SR), ctx).samples)
        delays.append(ctx.telemetry.extras["tiers"]["delay_ms"])
        if i == round(1.8 * SR):
            assert ctx.telemetry.extras["tiers"]["fade_kind"] == kind
    y = np.concatenate(ys)
    assert lad.switches == 2 and [h[1] for h in lad.history] == ["fast", "slow", "fast"]
    assert delays[round(1.0 * SR) // HOP] == fast_ms and delays[round(2.5 * SR) // HOP] == slow_ms

    start = round((max(fast_ms, slow_ms) + 50) * SR / 1000)  # past the tiers' own start-up silence
    tone_step = amp * 2 * np.pi * 440 / SR
    assert np.max(np.abs(np.diff(y[start:]))) < 1.5 * tone_step  # no click
    rms = np.sqrt(np.convolve(y[start:] ** 2, np.ones(80) / 80, mode="same"))  # 5 ms envelope
    quiet = rms < 0.1 * amp / np.sqrt(2) * 10 ** (-3 / 20)  # -20 dB below the quieter tier
    assert longest_run(quiet) < round(0.15 * SR)  # no gap longer than the crossfade


def test_low_target_confidence_blocks_av_tiers():
    lad, ctx = ladder(), StageContext.default()
    camera(ctx, True, confidence=0.1)
    HEALTH["enr"] = (False, "enrolling")
    run(lad, ctx, 0.2)
    assert ctx.telemetry.tier == "audio-only"
    assert "confidence" in ctx.telemetry.extras["tiers"]["tiers"][0]["reason"]


def test_a_crashing_tier_is_disabled_and_the_ladder_plays_on():
    lad, ctx = ladder(upgrade_after_s=0.0), StageContext.default()
    camera(ctx, True)
    run(lad, ctx, 0.3)
    assert ctx.telemetry.tier == "focus_hq"
    HEALTH["hq"] = ("crash", "")
    _, y, seen = run(lad, ctx, 0.5, t0=0.3)
    assert seen[-1] == "enrolled" and np.isfinite(y).all()  # the next healthy tier
    assert ctx.telemetry.extras["tiers"]["tiers"][0]["reason"].startswith("crashed")


def test_only_and_registered_spec():
    lad = registry.build_pipeline('tiers(only="gate,audio-only", crossfade_ms=20)').stages[0]
    assert lad.order == ("gate", "audio-only")
    assert lad._fade_n == 320
    with pytest.raises(ValueError, match="unknown tiers"):
        TierLadder(only="gate,nope")


def test_reset_starts_over():
    lad, ctx = ladder(upgrade_after_s=0.0), StageContext.default()
    camera(ctx, True)
    run(lad, ctx, 0.3)
    assert lad.switches == 0 and lad.active == "focus_hq"
    lad.reset()
    assert lad.active is None and lad.switches == 0
    assert lad.latency_ms == 800.0 and lad.status()["fade_kind"] is None


def test_remote_stage_health_is_its_connection():
    class FakeRemote:
        name = "remote"

        def __init__(self, up):
            self.up = up

        def remote_up(self):
            return self.up

    ok, reason, _ = tiers.stage_health(FakeRemote(False))
    assert not ok and "remote" in reason
    assert tiers.stage_health(FakeRemote(True))[0]


# ------------------------------------------------------------------ same-gender guard (T-070)


def guarded_ladder() -> TierLadder:
    """The default ladder's order and rules with test stages: enrolled (avoids same_bucket) ->
    focus_hq (AV, needs lips: health key "hq") -> gate -> audio-only."""
    return TierLadder({
        "crossfade_ms": 50,
        "upgrade_after_s": 1.0,
        "tiers": [
            {"name": "enrolled", "pipeline": 't019_flag(key="enr")>gain(db=-6)>delay(ms=800)',
             "needs": ["healthy"], "avoid_when": ["same_bucket"]},
            {"name": "focus_hq", "pipeline": 't019_flag(key="hq")>delay(ms=1600)',
             "needs": ["camera", "target", "healthy"]},
            {"name": "gate", "pipeline": "gain(db=-12)", "needs": ["camera", "target"]},
            {"name": "audio-only", "pipeline": "gain(db=-20)"},
        ],
    })


def two_faces(ctx: StageContext, same: bool | None) -> None:
    """Target 3 + another face 5; ``same`` is what SameBucket would publish (None: no flag at all)."""
    ctx.tracks = [FaceTrack(3, (0.1, 0.3, 0.2, 0.4), None, 0.9, 0.0),
                  FaceTrack(5, (0.6, 0.3, 0.2, 0.4), None, 0.1, 0.0)]
    ctx.target = TargetState(3, 1.0, True, 0.0)
    if same is None:
        ctx.telemetry.extras.pop(tiers.SAME_BUCKET_KEY, None)
    else:
        ctx.telemetry.extras[tiers.SAME_BUCKET_KEY] = same


@pytest.mark.parametrize("lips", [True, False])
def test_same_bucket_pair_never_plays_enrolled(lips):
    """Two same-bucket faces: enrolled is healthy (enrolled) but never plays; the ladder prefers AV
    Focus HQ when lips are available, else the gate, and says why."""
    lad, ctx = guarded_ladder(), StageContext.default()
    two_faces(ctx, True)
    HEALTH["hq"] = (True, "") if lips else (False, "no lips")
    _, _, seen = run(lad, ctx, 4.0)
    assert "enrolled" not in seen
    assert set(seen) == {"focus_hq" if lips else "gate"}
    status = ctx.telemetry.extras["tiers"]
    enrolled = next(t for t in status["tiers"] if t["name"] == "enrolled")
    assert enrolled == {**enrolled, "eligible": False, "reason": "same-gender pair: enrolled skipped"}
    assert status["same_bucket"] is True


def test_same_bucket_appearing_drops_enrolled_at_once_and_its_return_waits_for_the_hold():
    lad, ctx = guarded_ladder(), StageContext.default()
    HEALTH["hq"] = (False, "no lips")
    second = SR
    events = {
        0: lambda c: two_faces(c, False),  # M/F: enrolled plays
        2 * second: lambda c: two_faces(c, True),  # the other face turns out to be the same gender
        4 * second: lambda c: two_faces(c, False),  # the other face left > 3 s ago
    }
    _, _, seen = run(lad, ctx, 6.0, events)
    at = {round(i * HOP / SR, 2): t for i, t in enumerate(seen)}
    assert at[1.9] == "enrolled"
    assert at[2.0] == "gate"  # down at once
    assert at[3.99] == "gate"
    assert at[4.9] == "gate"  # back to eligible, inside the 1 s upgrade hold
    assert at[5.05] == "enrolled"
    assert [h[1:] for h in lad.history] == [
        ("enrolled", "start"), ("gate", "same-gender pair: enrolled skipped"), ("enrolled", "enrolled ready")]
    assert lad.switches == 2


def test_no_flag_or_false_flag_keeps_the_old_ladder():
    lad, ctx = guarded_ladder(), StageContext.default()
    two_faces(ctx, None)  # sessions/evals that never publish the flag
    _, _, seen = run(lad, ctx, 1.0)
    assert set(seen) == {"enrolled"} and ctx.telemetry.extras["tiers"]["same_bucket"] is False


def test_same_bucket_on_a_mac_plays_the_gate_even_with_lips(monkeypatch):
    """T-072 builds AV on CUDA only: on the Mac a same-gender pair lands on the gate."""
    monkeypatch.setenv("HEARING_DEVICE", "mps")
    lad = TierLadder({
        "crossfade_ms": 50,
        "upgrade_after_s": 1.0,
        "tiers": [
            {"name": "enrolled", "pipeline": 't019_flag(key="enr")>gain(db=-6)>delay(ms=800)',
             "needs": ["healthy"], "avoid_when": ["same_bucket"]},
            {"name": "focus_hq", "pipeline": 't019_flag(key="hq")>delay(ms=1600)',
             "needs": ["camera", "target", "healthy"], "devices": ["cuda"]},
            {"name": "gate", "pipeline": "gain(db=-12)", "needs": ["camera", "target"]},
            {"name": "audio-only", "pipeline": "gain(db=-20)"},
        ],
    })
    ctx = StageContext.default()
    two_faces(ctx, True)
    HEALTH["hq"] = (True, "")
    _, _, seen = run(lad, ctx, 3.0)
    assert set(seen) == {"gate"} and "focus_hq" not in lad.built
    enrolled = next(t for t in ctx.telemetry.extras["tiers"]["tiers"] if t["name"] == "enrolled")
    assert enrolled["reason"] == "same-gender pair: enrolled skipped"


# ------------------------------------------------------------------ T-095 stability policy

def _run(p, rows, dt=0.1, hard=None, overrun=None):
    """Feed eligibility rows (one per dt) and return the tier index picked for each."""
    out = []
    for k, ok in enumerate(rows):
        out.append(p.choose(list(ok), k * dt, hard, overrun))
    return out


def test_policy_promotes_only_after_upgrade_after_s_and_rate_limits():
    from hearing.fusion.tiers import TierPolicy
    p = TierPolicy(("enrolled", "gate", "audio-only"), upgrade_after_s=(3.0, 1.0, 0.0), min_switch_s=10.0)
    picks = _run(p, [(False, False, True)] * 10 + [(False, True, True)] * 10 + [(True, True, True)] * 120)
    assert picks[0] == 2 and picks.index(1) == 20  # gate 1 s after it became eligible
    assert picks.index(0) == 120  # eligible at 2 s, ready at 5 s, but the last switch was at 2 s -> 12 s
    assert sum(a != b for a, b in itertools.pairwise(picks)) == 2


def test_policy_soft_failure_grace_and_hard_failure():
    from hearing.fusion.tiers import TierPolicy
    p = TierPolicy(("enrolled", "gate"), upgrade_after_s=0.0, downgrade_after_s=1.5)
    p.choose([True, True], 0.0)
    assert p.current == 0
    soft = [False, False]
    for k in range(1, 15):  # overrun for 1.4 s: keeps playing
        assert p.choose([False, True], k * 0.1, soft, [True, False]) == 0
    assert p.choose([False, True], 1.6, soft, [True, False]) == 1  # 1.5 s unhealthy -> demote
    q = TierPolicy(("enrolled", "gate"), upgrade_after_s=0.0, downgrade_after_s=1.5)
    q.choose([True, True], 0.0)
    assert q.choose([False, True], 0.1, [True, False]) == 1  # hard failure: at once


def test_policy_cooling_blocks_promotion_after_overrun():
    from hearing.fusion.tiers import TierPolicy
    p = TierPolicy(("enrolled", "gate"), upgrade_after_s=3.0, downgrade_after_s=0.0, cooldown_s=10.0)
    p.choose([False, True], 0.0)
    p.choose([False, True], 1.0, [False, False], [True, False])  # enrolled overran at 1 s
    assert p.cooling(0, 1.0) == pytest.approx(10.0)
    picks = [p.choose([True, True], t / 10) for t in range(11, 140)]
    t_up = (11 + picks.index(0)) / 10
    assert t_up == pytest.approx(11.0)  # 10 s after the overrun (eligible 3 s long well before)
    assert p.cooling(0, t_up) == 0.0


def test_policy_flapping_tier_switches_at_most_twice():
    """The Oct 6 trace: enrolled alternates ok / overrun every ~2.5 s for 30 s."""
    from hearing.fusion.tiers import TierPolicy
    p = TierPolicy(("enrolled", "gate", "audio-only"), upgrade_after_s=(3.0, 1.0, 0.0),
                   downgrade_after_s=1.5, min_switch_s=10.0, cooldown_s=10.0)
    rows, ov = [], []
    for k in range(600):
        t = k * 0.1
        enrolled_ok = t > 25 and int((t - 25) / 2.5) % 2 == 0
        rows.append((enrolled_ok, t > 1.7, True))
        ov.append([not enrolled_ok and t > 25, False, False])
    picks = [p.choose(list(r), k * 0.1, [False, False, False], o) for k, (r, o) in enumerate(zip(rows, ov, strict=True))]
    assert sum(a != b for a, b in itertools.pairwise(picks)) <= 2


def test_ladder_reads_policy_times_from_config():
    cfg = LadderConfig.from_dict({"upgrade_after_s": 3, "downgrade_after_s": 1.5, "min_switch_s": 10,
                                  "cooldown_s": 10,
                                  "tiers": [{"name": "gate", "pipeline": "passthrough", "upgrade_after_s": 1},
                                            {"name": "audio-only", "pipeline": "passthrough"}]})
    lad = TierLadder(cfg)
    assert lad.policy.upgrade_after_s == (1.0, 3.0)
    assert (lad.policy.downgrade_after_s, lad.policy.min_switch_s, lad.policy.cooldown_s) == (1.5, 10.0, 10.0)
    ctx = StageContext.default()
    lad.process(AudioChunk(np.zeros(160, np.float32), 16000, 0, 0.0), ctx)
    row = lad.status()["tiers"][0]
    assert "cooling_s" in row and "quality_db" not in row
    with pytest.raises(ValueError):
        LadderConfig.from_dict({"cooldown_s": -1, "tiers": [{"name": "a", "pipeline": "passthrough"}]})


# ------------------------------------------------------------------ T-116: idle tiers suspended


CALLS: dict[str, int] = {}  # key -> chunks processed by the "count" test stage


class Count:
    """Pass-through stage that counts the chunks it processes (stands in for dfn_ll)."""

    sample_rate = SR
    latency_ms = 0.0

    def __init__(self, key: str = "c"):
        self.key = key
        self.name = f"t116_count({key})"

    def process(self, chunk, ctx):
        CALLS[self.key] = CALLS.get(self.key, 0) + 1
        return chunk

    def reset(self):
        pass


if "t116_count" not in registry.names():
    registry.register("t116_count")(Count)


def idle_ladder(idle: str, gate_ms: float = 800.0, low_ms: float = 400.0) -> TierLadder:
    """enrolled (needs healthy, always runs) > gate (camera) > audio-only, the last two counted."""
    return TierLadder({"crossfade_ms": 50, "upgrade_after_s": 0.5, "idle": idle, "prime_s": 0.5, "tiers": [
        {"name": "enrolled", "pipeline": 't019_flag(key="enr")>gain(db=-6)>delay(ms=800)', "needs": ["healthy"]},
        {"name": "gate", "pipeline": f't116_count(key="gate")>gain(db=-12)>delay(ms={gate_ms})',
         "needs": ["camera", "target"]},
        {"name": "audio-only", "pipeline": f't116_count(key="low")>gain(db=-20)>delay(ms={low_ms})'}]})


def _idle_scenario(idle: str, gate_ms: float = 800.0):
    """enrolled plays with the camera on; camera lost at 2 s (audio-only becomes the fallback),
    enrolled fails hard at 3 s (audio-only plays), camera back at 4 s (gate promoted at 4.5 s),
    enrolled healthy again at 5 s (promoted after the 0.5 s hold)."""
    CALLS.clear()
    HEALTH.clear()
    lad, ctx = idle_ladder(idle, gate_ms), StageContext.default()
    camera(ctx, True)
    events = {
        2 * SR: lambda c: camera(c, False),
        3 * SR: lambda c: HEALTH.update(enr=(False, "model missing")),
        4 * SR: lambda c: camera(c, True),
        5 * SR: lambda c: HEALTH.update(enr=(True, "")),
    }
    x, y, seen = run(lad, ctx, 6.5, events)
    return lad, ctx, x, y, seen, dict(CALLS)


@pytest.mark.parametrize("gate_ms", [800.0, 0.0])  # aligned (800 -> 800) and dip promotions
def test_suspended_tiers_are_primed_and_play_exactly_what_a_running_tier_would(gate_ms):
    lad_s, _, _, y_s, seen_s, calls_s = _idle_scenario("suspend", gate_ms)
    _, _, _, y_r, seen_r, calls_r = _idle_scenario("run", gate_ms)
    assert seen_s == seen_r
    assert [h[1] for h in lad_s.history] == ["enrolled", "audio-only", "gate", "enrolled"]
    # Every sample, crossfades included, is the same as with every tier running all the time:
    # priming replays enough input that the suspended tier's delay line holds the right moment.
    np.testing.assert_allclose(y_s, y_r, atol=1e-6)
    hops = len(seen_s)
    assert calls_r == {"gate": hops, "low": hops}
    # audio-only is idle while enrolled plays with the gate as its fallback (0-2 s, after 5.6 s);
    # the gate is idle between the camera loss and the enrolled failure (2-3 s, not the fallback).
    # Counts include the replayed priming chunks.
    assert calls_s["low"] < 0.7 * hops and calls_s["gate"] < hops
    assert lad_s.built["audio-only"].primes >= 1 and lad_s.built["gate"].primes >= 1


def test_a_suspended_tier_picked_in_the_same_chunk_is_primed_before_it_plays():
    """The fallback seen at the last refresh (the gate) goes away in the very chunk the playing
    tier fails hard: the ladder primes the suspended audio-only tier before it plays its first sample."""
    def scenario(idle):
        CALLS.clear()
        HEALTH.clear()
        lad, ctx = idle_ladder(idle), StageContext.default()
        camera(ctx, True)
        both = {2 * SR: lambda c: (camera(c, False), HEALTH.update(enr=(False, "model missing")))}
        return lad, ctx, run(lad, ctx, 3.0, both)

    lad, _, (x, y, seen) = scenario("suspend")
    _, _, (_, y_r, _) = scenario("run")
    i = 2 * SR // HOP
    assert seen[i - 1] == "enrolled" and seen[i] == "audio-only"
    assert lad.built["audio-only"].primes == 1
    np.testing.assert_allclose(y, y_r, atol=1e-6)
    np.testing.assert_allclose(y[2 * SR + 800:], (0.1 * delayed(x, 400))[2 * SR + 800:], rtol=1e-5)


def test_idle_status_config_and_env(monkeypatch):
    CALLS.clear()
    lad, ctx = idle_ladder("suspend"), StageContext.default()
    camera(ctx, True)
    run(lad, ctx, 1.0)
    status = ctx.telemetry.extras["tiers"]
    assert status["suspended"] == ["audio-only"]
    assert [t["running"] for t in status["tiers"]] == [True, True, False]
    assert status["hop_ms"]["p50"] >= 0.0 and status["hop_ms"]["p95"] >= status["hop_ms"]["p50"]
    assert lad.built["enrolled"].always and not lad.built["gate"].always  # needs healthy: never idle
    with pytest.raises(ValueError, match="idle"):
        LadderConfig.from_dict({"idle": "sleep", "tiers": [{"name": "a", "pipeline": "passthrough"}]})
    with pytest.raises(ValueError, match="idle"):
        LadderConfig.from_dict({"tiers": [{"name": "a", "pipeline": "passthrough", "idle": "x"}]})
    per_tier = TierLadder({"tiers": [{"name": "a", "pipeline": "passthrough"},
                                     {"name": "b", "pipeline": "passthrough", "idle": "run"}]})
    assert per_tier.built["b"].always and not per_tier.built["a"].always
    monkeypatch.setenv(tiers.IDLE_ENV, "run")
    lad = idle_ladder("suspend")
    camera(ctx, True)
    run(lad, ctx, 0.5)
    assert ctx.telemetry.extras["tiers"]["suspended"] == []
    monkeypatch.setenv(tiers.IDLE_ENV, "bogus")
    with pytest.raises(ValueError, match=tiers.IDLE_ENV):
        idle_ladder("suspend")
