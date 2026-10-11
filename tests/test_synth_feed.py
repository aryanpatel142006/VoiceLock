"""``control.synth_feed``: which stream feeds Synthetic/Text ASR (mixture | natural | focus | auto)."""

import json

import numpy as np
import pytest

from hearing.audio.io import float_to_pcm16
from hearing.server.protocol import Control, ProtocolError, parse_message
from hearing.server.session import AUTO_FOCUS_ATTACK_S, ServerConfig, Session
from hearing.synth.pipeline import PhraseRecord


@pytest.fixture(autouse=True)
def _mock_engines(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.setenv("HEARING_ASR", "mock")
    monkeypatch.setenv("HEARING_TTS", "mock")
    monkeypatch.setenv("HEARING_VOICE_MATCH", "mock")
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)


def _session(**kw) -> Session:
    return Session(ServerConfig(vision_threaded=False, **kw), "cpu")


def _control(session: Session, **fields) -> dict:
    events = session.handle_text(json.dumps({"type": "control", **fields}))
    return events[-1]


def test_control_accepts_synth_feed_and_rejects_unknown():
    assert Control(type="control", synth_feed="focus").synth_feed == "focus"
    with pytest.raises(ProtocolError):
        parse_message({"type": "control", "synth_feed": "bogus"})


def test_default_is_auto_resolving_to_mixture_with_one_talker():
    s = _session()
    t = _control(s, mode="synthetic")
    assert (t["synth_feed"], t["synth_feed_request"]) == ("mixture", "auto")
    assert t["pipeline"].startswith("passthrough>synth(")
    s.close()


def test_mixture_is_the_raw_room_even_with_a_configured_natural_chain():
    s = _session(pipelines={"natural": "gain(db=-6)"})
    assert _control(s, mode="text")["pipeline"].startswith("passthrough>synth(")
    assert _control(s, synth_feed="natural")["pipeline"].startswith("gain(db=-6)>synth(")
    s.close()


@pytest.mark.parametrize(("feed", "head"), [("mixture", "passthrough>"), ("natural", "dfn_ll>"),
                                            ("focus", "tiers>")])
def test_explicit_feed_selects_the_chain_and_shows_in_telemetry(feed, head):
    s = _session()
    t = _control(s, mode="text", synth_feed=feed)
    assert t["synth_feed"] == feed
    assert t["pipeline"].startswith(head) and t["pipeline"].endswith("synth(speak=false)")
    s.close()


def test_focus_feed_follows_the_latency_preset():
    s = _session()
    _control(s, mode="synthetic", synth_feed="focus")
    t = _control(s, latency_preset="speed")
    assert "preset" in t["pipeline"] and t["synth_feed"] == "focus"
    s.close()


def _two_talkers(s: Session) -> None:
    for _ in range(AUTO_FOCUS_ATTACK_S):
        s._update_auto_feed(2)


def test_auto_goes_focus_with_two_talkers_and_back_after_release():
    s = _session()
    _control(s, mode="synthetic")
    _two_talkers(s)
    assert _control(s)["synth_feed"] == "focus"
    for _ in range(4):
        s._update_auto_feed(1)
    assert s.synth_feed == "focus"  # hysteresis: still holding
    s._update_auto_feed(1)
    assert s.synth_feed == "mixture"
    s.close()


def test_a_one_second_blip_of_two_talkers_does_not_flip_the_feed():
    s = _session()
    _control(s, mode="text")
    for speakers in (2, 1, 2, 1, 0, 2, 1):
        s._update_auto_feed(speakers)
        assert s.synth_feed == "mixture"
    s.close()


def test_auto_falls_back_to_mixture_when_the_ladder_cannot_build():
    s = _session()
    _control(s, mode="synthetic")
    _two_talkers(s)
    s._substitute[s._synth_spec("synthetic", "focus")] = "silence"  # what a failed build records
    assert s.synth_feed == "mixture"
    s.close()


def _feed_audio(s: Session, seconds: float) -> list:
    out = []
    for _ in range(int(seconds * 50)):  # 20 ms frames
        out += s._process_audio(0, 0.0, float_to_pcm16(0.1 * np.ones(320, np.float32)))
    return out


def test_a_feed_flip_keeps_the_synth_stage_its_queue_and_captions():
    s = _session(pipelines={"natural": "gain(db=-6)"})
    _control(s, mode="text")
    _feed_audio(s, 0.1)
    synth = s._synth()
    synth._captions.append({"text": "see you at noon", "final": True, "t0": 0.0, "t1": 1.0})
    with synth._cond:
        synth._queue.append(PhraseRecord("see you at noon", None, 0.0, 1.0, 1.0, 0.0))
        gen = synth._gen  # reset() bumps it and drops the queue / the phrase being spoken
    t = _control(s, synth_feed="natural")
    assert t["pipeline"].startswith("gain(db=-6)>synth(")
    out = _feed_audio(s, 0.1)
    assert s._synth() is synth  # the same ASR/TTS instance, only the upstream changed
    assert synth._gen == gen
    assert [e["text"] for e in out if isinstance(e, dict) and e["type"] == "caption"] == ["see you at noon"]
    s.close()


def test_a_mode_change_still_resets_the_synth_stage():
    s = _session()
    _control(s, mode="text")
    _feed_audio(s, 0.1)
    synth = s._synth()
    synth._captions.append({"text": "stale", "final": True, "t0": 0.0, "t1": 1.0})
    _control(s, mode="synthetic")
    out = _feed_audio(s, 0.1)
    assert s._synth() is not synth
    assert not [e for e in out if isinstance(e, dict) and e["type"] == "caption"]
    s.close()


def test_explicit_synthetic_pipeline_is_not_overridden():
    s = _session(pipelines={"synthetic": "gain(db=-6)>synth(speak=true)"})
    t = _control(s, mode="synthetic", synth_feed="focus")
    assert t["pipeline"] == "gain(db=-6)>synth(speak=true)"
    s.close()


@pytest.mark.parametrize("mode", ["natural", "focus_hq", "quiet"])
def test_focus_feed_in_a_non_synth_mode_does_not_break_telemetry(mode):
    s = _session()
    _control(s, mode=mode)
    _two_talkers(s)
    assert s.telemetry()["synth_feed"] == "focus"
    t = _control(s, synth_feed="focus")
    assert t["synth_feed"] == "focus"
    s.close()


def test_judge_profile_pins_auto_feed_by_device():
    """T-136: under the judge profile ``auto`` is the mixture on the Mac, the ladder on a CUDA server."""
    s = _session()
    try:
        s.profile, s._auto_feed = "judge", "focus"
        assert s.synth_feed == "mixture"
        s.device = "cuda"
        assert s.synth_feed == "focus"
        s.synth_feed_request = "natural"  # an explicit feed still wins
        assert s.synth_feed == "natural"
    finally:
        s.close()
