import logging
import sys
import types

import numpy as np
import pytest

from hearing.audio.io import PIPELINE_SR
from hearing.synth import voice_match
from hearing.synth.tts import bucket_names
from hearing.synth.voice_match import (
    AGES,
    ApparentEstimate,
    FaceEstimator,
    MockFaceEstimator,
    MockVoiceEstimator,
    VoiceChoice,
    VoiceEstimator,
    VoiceMatcher,
    fuse,
    get_face_estimator,
    get_voice_estimator,
    to_telemetry,
)

FRAME = np.zeros((360, 640, 3), np.uint8)
BOX = (0.4, 0.3, 0.2, 0.3)
ONE_SECOND = np.zeros(PIPELINE_SR, np.float32)


def face(bucket="male_adult", confidence=0.9):
    return MockFaceEstimator(bucket, confidence).estimate(FRAME, BOX)


def voice(bucket="male_adult", confidence=0.9):
    return MockVoiceEstimator(bucket, confidence).estimate(ONE_SECOND)


def test_mocks_implement_protocols_and_are_deterministic():
    assert isinstance(MockFaceEstimator(), FaceEstimator)
    assert isinstance(MockVoiceEstimator(), VoiceEstimator)
    assert face() == face()
    assert face().source == "face" and voice().source == "voice"
    assert MockVoiceEstimator().estimate(ONE_SECOND[: PIPELINE_SR - 1]) is None


def test_estimate_is_frozen():
    est = face()
    with pytest.raises(AttributeError):
        est.p_female = 0.5


@pytest.mark.parametrize("bucket", bucket_names())
def test_mocks_map_back_to_their_bucket(bucket):
    assert fuse(face(bucket), None).bucket == bucket
    assert fuse(None, voice(bucket)).bucket == bucket


def test_mock_rejects_unknown_bucket():
    with pytest.raises(ValueError, match="neutral"):
        MockFaceEstimator("robot_adult")


def test_override_wins_over_confident_estimates():
    m = VoiceMatcher()
    m.update(1, face=face("male_senior", 1.0), voice=voice("male_senior", 1.0))
    c = m.choice(1, override="female_adult")
    assert (c.bucket, c.source) == ("female_adult", "override")
    assert m.choice(None, override="neutral").source == "override"


def test_unknown_override_raises_listing_buckets():
    with pytest.raises(ValueError, match="female_adult"):
        VoiceMatcher().choice(1, override="robot")


def test_no_estimates_is_neutral_default():
    assert fuse(None, None) == VoiceChoice("neutral", "default", 0.0)
    m = VoiceMatcher()
    assert (m.choice(7).bucket, m.choice(7).source) == ("neutral", "default")
    assert m.choice(None).source == "default"


def test_low_confidence_is_neutral_default():
    c = fuse(face("female_young", 0.3), voice("female_young", 0.5))
    assert (c.bucket, c.source) == ("neutral", "default")
    # one confident estimate still counts on its own
    assert fuse(face("female_young", 0.3), voice("female_young", 0.8)).source == "voice"


def test_sources():
    assert fuse(face("female_senior"), voice("female_senior")) == VoiceChoice(
        "female_senior", "face+voice", pytest.approx(0.9)
    )
    assert fuse(face("male_young"), None).source == "face"
    assert fuse(None, voice("male_young")).source == "voice"
    m = VoiceMatcher()
    m.update(3, face=face("male_young"), voice=voice("male_young"))
    assert (m.choice(3).bucket, m.choice(3).source) == ("male_young", "face+voice")


def test_disagreeing_gender_is_neutral():
    f = ApparentEstimate(0.9, (0.1, 0.8, 0.1), 0.8, "face")
    v = ApparentEstimate(0.1, (0.1, 0.8, 0.1), 0.8, "voice")
    assert fuse(f, v).bucket == "neutral"


def test_confidence_weighting_breaks_ties():
    f = ApparentEstimate(0.9, (0.1, 0.8, 0.1), 0.95, "face")
    v = ApparentEstimate(0.1, (0.1, 0.1, 0.8), 0.65, "voice")
    assert fuse(f, v).bucket == "neutral"  # p = (0.855 + 0.065) / 1.6 = 0.575
    v = ApparentEstimate(0.4, (0.1, 0.1, 0.8), 0.65, "voice")
    assert fuse(f, v).bucket == "female_adult"  # p = 0.697; age adult 0.52 > senior 0.38


def test_every_fused_bucket_is_a_voice():
    rng = np.random.default_rng(0)
    names, seen = set(bucket_names()), set()
    for _ in range(2000):
        ests = []
        for src in ("face", "voice"):
            if rng.random() < 0.8:
                p_age = tuple(rng.dirichlet(np.ones(len(AGES))))
                ests.append(ApparentEstimate(rng.random(), p_age, rng.random(), src))
            else:
                ests.append(None)
        c = fuse(*ests)
        assert c.bucket in names
        seen.add(c.bucket)
    assert seen == names


def test_ema_smooths_per_track():
    m = VoiceMatcher(alpha=0.3)
    m.update(1, face=face("female_adult"))
    m.update(2, face=face("male_adult"))
    m.update(1, face=face("male_adult"))  # one contrary frame: 0.7 * 0.9 + 0.3 * 0.1 = 0.66
    assert m.choice(1).bucket == "female_adult"
    assert m.choice(2).bucket == "male_adult"
    for _ in range(5):
        m.update(1, face=face("male_adult"))
    assert m.choice(1).bucket == "male_adult"


def test_to_telemetry_shape():
    m = VoiceMatcher()
    m.update(1, face=face(), voice=voice())
    t = to_telemetry(m.choice(1), None)
    assert t == {"bucket": "male_adult", "source": "face+voice", "override": None}
    t = to_telemetry(m.choice(1, "neutral"), "neutral")
    assert set(t) == {"bucket", "source", "override"}
    assert t == {"bucket": "neutral", "source": "override", "override": "neutral"}


def test_forget_and_reset():
    m = VoiceMatcher()
    m.update(1, face=face("female_adult"))
    m.update(2, voice=voice("female_adult"))
    m.forget(1)
    assert 1 not in m and m.choice(1).source == "default"
    assert m.choice(2).bucket == "female_adult"
    m.forget(99)  # unknown track is a no-op
    m.reset()
    assert 2 not in m and m.choice(2).source == "default"


def test_factories_honour_mock_env(monkeypatch):
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    assert isinstance(get_face_estimator(), MockFaceEstimator)
    assert isinstance(get_voice_estimator(), MockVoiceEstimator)


def test_factories_return_none_without_adapters(monkeypatch):
    monkeypatch.delenv("HEARING_VOICE_MATCH", raising=False)
    # None in sys.modules makes the import raise ImportError even once the adapters exist
    monkeypatch.setitem(sys.modules, "hearing.synth.apparent_face", None)
    monkeypatch.setitem(sys.modules, "hearing.synth.apparent_voice", None)
    assert get_face_estimator() is None
    assert get_voice_estimator() is None


def test_factories_use_adapters_and_respect_unavailable_weights(monkeypatch):
    monkeypatch.delenv("HEARING_VOICE_MATCH", raising=False)
    face_mod = types.ModuleType("hearing.synth.apparent_face")
    face_mod.FaceEstimator = MockFaceEstimator
    face_mod.available = lambda: True
    voice_mod = types.ModuleType("hearing.synth.apparent_voice")
    voice_mod.VoiceEstimator = MockVoiceEstimator
    voice_mod.available = lambda: False
    monkeypatch.setitem(sys.modules, "hearing.synth.apparent_face", face_mod)
    monkeypatch.setitem(sys.modules, "hearing.synth.apparent_voice", voice_mod)
    assert isinstance(get_face_estimator(), MockFaceEstimator)
    assert get_voice_estimator() is None

    def missing_weights():
        raise FileNotFoundError("weights")

    voice_mod.available = lambda: True
    voice_mod.VoiceEstimator = missing_weights
    assert get_voice_estimator() is None


def test_no_log_records(caplog):
    caplog.set_level(logging.DEBUG)
    m = VoiceMatcher()
    for i in range(3):
        m.update(i, face=face("female_young"), voice=voice("female_young"))
        m.choice(i)
        m.choice(i, override="neutral")
        to_telemetry(m.choice(i), None)
    fuse(face(), voice())
    m.forget(0)
    m.reset()
    assert caplog.records == []
    assert not hasattr(voice_match, "log")


# --- same-bucket signal (T-070) ---------------------------------------------


def _faces(*ids):
    from hearing.types import FaceTrack

    return [FaceTrack(i, (0.1 * i, 0.3, 0.2, 0.4), None, 0.5, 0.0) for i in ids]


def test_voice_group_is_the_gender_half_of_the_bucket():
    assert voice_match.voice_group("male_adult") == "male"
    assert voice_match.voice_group("female_senior") == "female"
    assert voice_match.voice_group("neutral") is None and voice_match.voice_group(None) is None


def test_same_bucket_needs_a_known_matching_group_on_another_recent_face():
    from hearing.types import StageContext, TargetState

    buckets = {1: "male_adult", 2: "male_senior", 3: "female_adult", 4: "neutral"}
    sb = voice_match.SameBucket(recent_s=3.0)
    sb.update(_faces(1, 3), buckets.get, now=0.0)
    assert not sb.same(1) and not sb.same(3)  # M/F
    sb.update(_faces(1, 2), buckets.get, now=1.0)
    assert sb.same(1) and sb.same(2)  # male_adult + male_senior: same voice group
    assert not sb.same(None) and not sb.same(9)
    sb.update(_faces(1), buckets.get, now=3.9)
    assert sb.same(1)  # face 2 seen 2.9 s ago still counts
    sb.update(_faces(1), buckets.get, now=4.1)
    assert not sb.same(1) and 2 not in sb.buckets()  # forgotten after 3 s

    ctx = StageContext.default()
    ctx.target = TargetState(4, 1.0, True, 0.0)
    sb.update(_faces(4, 1), buckets.get, now=5.0)
    assert sb.publish(ctx) is False  # unknown target bucket -> False
    assert ctx.telemetry.extras[voice_match.SAME_BUCKET_KEY] is False
    assert ctx.telemetry.extras[voice_match.VOICE_BUCKETS_KEY] == {1: "male_adult", 4: "neutral"}
    ctx.target = TargetState(1, 1.0, True, 0.0)
    sb.update(_faces(1, 2), buckets.get, now=5.0)
    assert sb.publish(ctx) is False  # latch: the match must persist 1 s first
    sb.update(_faces(1, 2), buckets.get, now=6.1)
    assert sb.publish(ctx) is True and ctx.telemetry.extras[voice_match.SAME_BUCKET_KEY] is True
    sb.reset()
    assert sb.buckets() == {} and sb.publish(ctx) is False


def test_same_bucket_key_matches_the_tier_ladder():
    from hearing.fusion import tiers

    assert tiers.SAME_BUCKET_KEY == voice_match.SAME_BUCKET_KEY


def test_same_bucket_flapping_latches_once():
    """T-091: buckets flipping every 0.5 s for 10 s give at most one flag change."""
    sb = voice_match.SameBucket(recent_s=3.0)
    tracks = _faces(1, 2)
    flips, last = 0, False
    for i in range(201):
        now = i * 0.05
        bucket = "neutral" if int(now / 0.5) % 2 == 1 else "male_adult"
        sb.update(tracks, lambda tid, b=bucket: b, now)
        cur = sb.latched(1)
        flips += cur != last
        last = cur
    assert flips <= 1
