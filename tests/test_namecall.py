"""Name Call mode (T-120): name matching, the listener's enroll / re-target / release flow, session wiring."""
from __future__ import annotations

import json

import numpy as np
import pytest

from hearing.server import namecall as nc
from hearing.server.namecall import NameCaller, Word, find_name, phonetic, word_matches
from hearing.tse.enroll import AutoEnroller

SR = 16000


@pytest.mark.parametrize("word", ["Aryan", "aryan,", "ARYAN!", "Arian", "Aryan's", "Aryn", "Arion"])
def test_name_variants_match(word):
    assert word_matches(word, "Aryan")


@pytest.mark.parametrize("word", ["iron", "Orion", "and", "around", "Brian", "a", "", "ocean", "arena"])
def test_other_words_do_not_match(word):
    assert not word_matches(word, "Aryan")


def test_phonetic_key_treats_initial_vowels_alike():
    assert phonetic("Aryan") == phonetic("Arian") == phonetic("Erian")
    assert phonetic("Joseph") != phonetic("Aryan")


def test_find_name_with_aliases_and_two_word_names():
    words = [Word(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(["hey", "mister", "patel", "over", "here"])]
    assert find_name(words, ["Aryan"]) is None
    assert find_name(words, ["Aryan", "Patel"]) == (2, 3)
    assert find_name(words, ["Mister Patel"]) == (1, 3)


def _tone(seconds: float, hz: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def _embed(x: np.ndarray) -> np.ndarray:
    """Dominant frequency band as a one-hot "voice": 200 Hz and 600 Hz tones are different people."""
    spec = np.abs(np.fft.rfft(x, n=4096))
    v = np.array([spec[(f - 50) * 4096 // SR:(f + 50) * 4096 // SR].sum() for f in (200, 600)])
    return v / max(np.linalg.norm(v), 1e-9)


def _scripted(hz_says: dict[float, str]):
    """Transcriber: the phrase is "hello <word>" where the word depends on the dominant tone."""
    def transcribe(x: np.ndarray) -> list[Word]:
        e = _embed(x)
        hz = (200, 600)[int(np.argmax(e))]
        d = len(x) / SR
        return [Word("hello", 0.0, d / 2), Word(hz_says.get(hz, "nothing"), d / 2, d)]
    return transcribe


def _feed(caller: NameCaller, x: np.ndarray) -> list:
    events = []
    for i in range(0, len(x) - 159, 160):
        caller.feed(x[i : i + 160])
        events += caller.poll()
    return events


def _caller(says: dict) -> NameCaller:
    return NameCaller("Aryan", transcriber=_scripted(says), embedder=_embed, cleaner=None, threaded=False)


def test_name_heard_enrolls_that_voice():
    caller = _caller({200: "Aryan"})
    quiet = np.zeros(SR, np.float32)
    events = _feed(caller, np.concatenate([quiet, _tone(1.5, 200), quiet, quiet]))
    enrolls = [e for e in events if e.kind == "enroll"]
    assert enrolls and enrolls[0].detection is not None and enrolls[0].detection.word == "Aryan"
    assert len(enrolls[0].clip) >= nc.MIN_CLIP_S * SR * 0.9
    assert np.argmax(_embed(enrolls[0].clip)) == 0
    assert caller.status()["target"] and caller.status()["detections"] == 1


def test_speech_without_the_name_never_enrolls():
    caller = _caller({200: "hello", 600: "Ryanair"})
    quiet = np.zeros(SR // 2, np.float32)
    events = _feed(caller, np.concatenate([_tone(1.5, 200), quiet, _tone(1.5, 600), quiet] * 3))
    assert events == [] and caller.status()["target"] is False


def test_new_voice_saying_the_name_retargets_with_old_as_negative():
    caller = _caller({200: "Aryan", 600: "Aryan"})
    quiet = np.zeros(SR, np.float32)
    events = _feed(caller, np.concatenate([_tone(1.5, 200), quiet, quiet, quiet, _tone(1.5, 600), quiet, quiet]))
    enrolls = [e for e in events if e.kind == "enroll" and e.detection is not None]
    assert len(enrolls) == 2
    assert not enrolls[0].retarget and enrolls[1].retarget
    assert np.argmax(_embed(enrolls[1].clip)) == 1  # the 600 Hz voice
    assert enrolls[1].other is not None and np.argmax(_embed(enrolls[1].other)) == 0


def test_same_voice_saying_the_name_again_does_not_retarget():
    caller = _caller({200: "Aryan"})
    quiet = np.zeros(SR, np.float32)
    events = _feed(caller, np.concatenate([_tone(1.5, 200), quiet, quiet, quiet, _tone(1.5, 200), quiet, quiet]))
    assert not any(e.retarget for e in events)
    assert caller.ignored == 1


def test_release_and_timeout(monkeypatch):
    caller = _caller({200: "Aryan"})
    _feed(caller, np.concatenate([_tone(1.5, 200), np.zeros(SR, np.float32)]))
    caller.release("by hand")
    assert [e.kind for e in caller.poll()] == ["release"] and caller.status()["target"] is False
    monkeypatch.setattr(nc, "TIMEOUT_S", 3.0)
    _feed(caller, np.concatenate([_tone(1.5, 200), np.zeros(SR, np.float32)]))
    assert caller.status()["target"]
    events = _feed(caller, np.zeros(4 * SR, np.float32))
    assert [e.kind for e in events] == ["release"]


def test_threaded_caller_matches_inline():
    caller = NameCaller("Aryan", transcriber=_scripted({200: "Aryan"}), embedder=_embed, cleaner=None, threaded=True)
    try:
        x = np.concatenate([_tone(1.5, 200), np.zeros(2 * SR, np.float32)])
        for i in range(0, len(x) - 159, 160):
            caller.feed(x[i : i + 160])
        caller.drain()
        events = caller.poll()
        assert [e.kind for e in events][:1] == ["enroll"]
        assert events[0].detection.latency_ms >= 0
    finally:
        caller.close()


def test_enroller_takes_a_clip_as_calibrated():
    e = AutoEnroller(1.75, SR)
    e.enroll_clip(np.ones(SR, np.float32) * 0.1, other=np.ones(SR // 2, np.float32))
    assert e.ready and e.calibrated and len(e.clip()) == SR
    assert e.other_clip() is not None and len(e.other_clip()) == SR // 2
    e.clear()
    assert not e.ready


def test_session_name_call_mode(monkeypatch):
    from hearing.audio.io import float_to_pcm16
    from hearing.server.protocol import KIND_AUDIO_IN, pack_frame
    from hearing.server.session import ServerConfig, Session, load_stage_modules

    load_stage_modules()
    made = {}

    def fake_caller(name="", aliases=None, **_):
        made["c"] = NameCaller(name, aliases, transcriber=_scripted({200: "Aryan"}), embedder=_embed,
                               cleaner=None, threaded=False)
        return made["c"]

    monkeypatch.setattr("hearing.server.session.NameCaller", fake_caller)
    sess = Session(ServerConfig(prefs_log=None, faces="none", vision_threaded=False,
                                pipelines={"name_call": 'tiers(only="audio-only")'}), "cpu")
    try:
        sess.handle_text(json.dumps({"type": "hello", "sample_rate": SR, "frame_ms": 20, "video": False}))
        out = sess.handle_text(json.dumps({"type": "control", "mode": "name_call",
                                           "name": {"text": "Aryan", "aliases": ["Ary"]}}))
        assert out[-1]["name_call"]["state"] == "listening"
        assert out[-1]["name_call"]["aliases"] == ["Ary"]
        x = np.concatenate([np.zeros(SR // 2, np.float32), _tone(1.5, 200), np.zeros(SR, np.float32)])
        for k, i in enumerate(range(0, len(x) - 319, 320)):
            sess.handle_binary(pack_frame(KIND_AUDIO_IN, k, i / SR, float_to_pcm16(x[i : i + 320])))
        assert sess.enroller.calibrated and sess.enroller.ready
        t = sess.telemetry()["name_call"]
        assert t["target"] and t["state"] == "locking"  # audio-only ladder here: never "isolated"
        sess.handle_text(json.dumps({"type": "control", "name_call": "release"}))
        sess.handle_binary(pack_frame(KIND_AUDIO_IN, 999, 9.0, float_to_pcm16(np.zeros(320, np.float32))))
        assert not sess.enroller.ready
        assert sess.telemetry()["name_call"]["state"] == "listening"
    finally:
        sess.close()


def test_protocol_rejects_bad_name_call():
    from hearing.server.protocol import ProtocolError, parse_message

    assert parse_message({"type": "control", "mode": "name_call", "name": {"text": "Aryan"}}).name.text == "Aryan"
    with pytest.raises(ProtocolError):
        parse_message({"type": "control", "name": {"text": ""}})
    with pytest.raises(ProtocolError):
        parse_message({"type": "control", "name_call": "grab"})


def test_pages_have_the_name_call_controls():
    from pathlib import Path

    web = Path(__file__).resolve().parents[1] / "web" / "console"
    simple, judge = (web / "simple.html").read_text(), (web / "judge.html").read_text()
    for el in ('data-mode="name_call"', 'id="namecall-name"', 'id="namecall-pill"', 'id="namecall-release"'):
        assert el in simple
    for el in ('id="namecall-on"', 'id="namecall-name"', 'id="namecall-release"'):
        assert el in judge
    js = (web / "name-call.js").read_text()
    for text in ("Listening for", "from a new voice — locking", "Isolated", "Target quiet"):
        assert text in js


def test_target_heard_during_the_name_blocks_a_retarget():
    caller = _caller({200: "Aryan", 600: "Aryan"})
    quiet = np.zeros(SR, np.float32)
    _feed(caller, np.concatenate([_tone(1.5, 200), quiet, quiet, quiet]))
    caller.poll()
    x = np.concatenate([_tone(1.5, 600), quiet, quiet, quiet])
    events = []
    for i in range(0, len(x) - 159, 160):
        caller.feed(x[i : i + 160])
        caller.heard()  # the enrolled tier keeps choosing the current target's stream
        events += caller.poll()
    assert not any(e.retarget for e in events) and caller.ignored == 1
    assert np.argmax(_embed(caller.target.clip)) == 0


# ------------------------------------------------------------------ T-123 false-trigger guards

def _words(*texts: str) -> list[Word]:
    return [Word(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(texts)]


@pytest.mark.parametrize(("word", "strict", "normal", "loose"), [
    ("Aryan", True, True, True), ("aryan's", True, True, True),
    ("Ariane", False, True, True), ("Arian", False, True, True),
    ("Ryan", False, False, True), ("Brian", False, False, False), ("table", False, False, False),
])
def test_match_levels(word, strict, normal, loose):
    words = _words("hey", word, "there")
    got = [find_name(words, ["Aryan"], m) is not None for m in ("strict", "normal", "loose")]
    assert got == [strict, normal, loose]


def test_strict_still_matches_aliases_exactly():
    assert find_name(_words("hey", "Ary"), ["Aryan", "Ary"], "strict") == (1, 2)
    assert find_name(_words("hey", "Arie"), ["Aryan", "Ary"], "strict") is None


def test_exclude_list_is_checked_before_the_fuzzy_match():
    assert find_name(_words("Arian", "said"), ["Aryan"], "loose", ["Arian"]) is None
    assert find_name(_words("Ryan", "said"), ["Aryan"], "loose", ["ryan"]) is None
    assert find_name(_words("Ryan's", "bag"), ["Aryan"], "loose", ["Ryan"]) is None
    # the name itself (or an alias) is never excluded, and a later real hit still counts
    assert find_name(_words("Ryan", "and", "Aryan"), ["Aryan"], "loose", ["Ryan", "Aryan"]) == (2, 3)


def test_normal_lets_an_alias_through_its_confusable_list():
    assert find_name(_words("hey", "Ryan"), ["Aryan", "Ryan"], "normal") == (1, 2)


def test_caller_match_and_exclude():
    caller = NameCaller("Aryan", transcriber=lambda x: _words("Ryan", "hi"), embedder=_embed, cleaner=None,
                        threaded=False)
    assert caller.match == nc.DEFAULT_MATCH == "normal" and caller.status()["match"] == "normal"
    caller.set_match("loose", ["Arian"])
    assert caller.match == "loose" and caller.status()["exclude"] == ["Arian"]
    caller.set_match(None, None)  # partial update keeps both
    assert caller.match == "loose" and caller.exclude == ["Arian"]
    with pytest.raises(ValueError):
        caller.set_match("fuzzy")


def test_whisper_prompt_spells_the_names():
    assert nc.whisper_prompt(["Aryan", "Ary"]) == "Aryan, Ary."
    assert nc.whisper_prompt([]) == ""


def test_whisper_transcriber_passes_prompt_and_hotwords(monkeypatch):
    from hearing import models
    from hearing.synth import asr

    seen = {}

    class FakeModel:
        def transcribe(self, x, language=None, hotwords=None, **kw):
            seen.update(kw, hotwords=hotwords)
            return [], None

    monkeypatch.setattr(asr, "local_available", lambda: True)
    monkeypatch.setattr(asr, "_whisper_loader", lambda name: "fake")
    monkeypatch.setattr(models, "load", lambda key: FakeModel())
    nc.whisper_transcriber(lambda: ["Aryan"])(np.zeros(SR, np.float32))
    assert seen["initial_prompt"] == "Aryan." and seen["hotwords"] == "Aryan."
    seen.clear()
    nc.whisper_transcriber(None)(np.zeros(SR, np.float32))
    assert "initial_prompt" not in seen and seen["hotwords"] is None


def test_session_forwards_match_and_exclude(monkeypatch):
    from hearing.server.session import ServerConfig, Session, load_stage_modules

    load_stage_modules()
    made = {}

    def fake_caller(name="", aliases=None, **kw):
        made["kw"] = kw
        made["c"] = NameCaller(name, aliases, transcriber=_scripted({}), embedder=_embed, cleaner=None,
                               threaded=False, **kw)
        return made["c"]

    monkeypatch.setattr("hearing.server.session.NameCaller", fake_caller)
    sess = Session(ServerConfig(prefs_log=None, faces="none", vision_threaded=False,
                                pipelines={"name_call": 'tiers(only="audio-only")'}), "cpu")
    try:
        sess.handle_text(json.dumps({"type": "hello", "sample_rate": SR, "frame_ms": 20, "video": False}))
        sess.handle_text(json.dumps({"type": "control", "mode": "name_call",
                                     "name": {"text": "Aryan", "match": "strict", "exclude": ["Ryan"]}}))
        assert made["kw"] == {"match": "strict", "exclude": ["Ryan"]}
        sess.handle_text(json.dumps({"type": "control", "name": {"text": "Aryan", "match": "loose"}}))
        assert made["c"].match == "loose" and made["c"].exclude == ["Ryan"]
        assert sess.telemetry()["name_call"]["match"] == "loose"
    finally:
        sess.close()


def test_protocol_rejects_unknown_match():
    from hearing.server.protocol import ProtocolError, parse_message

    with pytest.raises(ProtocolError):
        parse_message({"type": "control", "name": {"text": "Aryan", "match": "fuzzy"}})


# ------------------------------------------------------------------ T-131 prompt-mode guard

def _scored(texts: list[str], p: float = 0.9, logprob: float = -0.3, no_speech: float = 0.05) -> list[Word]:
    return [Word(w, i * 0.3, i * 0.3 + 0.25, p, logprob, no_speech) for i, w in enumerate(texts)]


@pytest.mark.parametrize(("words", "why"), [
    (_scored(["hey", "Aryan", "there"]), None),
    (_scored(["Aryan."]), "bare name"),
    (_scored(["Aryan", ".", "-"]), "bare name"),  # punctuation is not a word
    (_scored(["hey", "Aryan", "there"], p=0.05), "word p"),
    (_scored(["hey", "Aryan", "there"], logprob=-1.2), "avg_logprob"),
    (_scored(["hey", "Aryan", "there"], no_speech=0.7), "no_speech"),
])
def test_guard_reject(words, why):
    hit = find_name(words, ["Aryan"], "strict")
    got = nc.guard_reject(words, hit)
    assert (got is None) if why is None else got.startswith(why)


def test_guard_is_on_with_the_prompt_only():
    def mk(**kw):
        return NameCaller("Aryan", transcriber=lambda x: [], embedder=_embed, cleaner=None, threaded=False, **kw)
    assert nc.DEFAULT_PROMPT is False and mk().guard is False and mk(prompt=True).guard is True
    assert mk(prompt=True, guard=False).guard is False and mk().status()["guard"] is False


def test_guarded_caller_drops_a_bare_name_and_keeps_a_real_one():
    quiet = np.zeros(SR, np.float32)
    x = np.concatenate([quiet, _tone(1.5, 200), quiet, quiet])
    bare = NameCaller("Aryan", transcriber=lambda x: _scored(["Aryan."]), embedder=_embed, cleaner=None,
                      threaded=False, guard=True)
    assert _feed(bare, x) == [] and bare.guarded >= 1 and bare.guard_reasons[-1] == "bare name"
    real = NameCaller("Aryan", transcriber=lambda x: _scored(["hey", "Aryan", "over", "here"]), embedder=_embed,
                      cleaner=None, threaded=False, guard=True)
    assert [e.kind for e in _feed(real, x)] == ["enroll"] and real.guarded == 0


def test_profile_gate_needs_a_known_voice():
    quiet = np.zeros(SR, np.float32)
    x = np.concatenate([quiet, _tone(1.5, 200), quiet, quiet])  # the 200 Hz "voice" says the name

    def caller(profile_hz):
        return NameCaller("Aryan", transcriber=lambda x: _scored(["hey", "Aryan", "there"]), embedder=_embed,
                          cleaner=None, threaded=False, guard=True, profiles=[_tone(2.0, profile_hz)])
    stranger = caller(600)
    assert _feed(stranger, x) == [] and stranger.guard_reasons[-1].startswith("profile cos")
    known = caller(200)
    assert [e.kind for e in _feed(known, x)] == ["enroll"] and known.status()["profiles"] == 1


def test_whisper_transcriber_keeps_word_scores(monkeypatch):
    from types import SimpleNamespace

    from hearing import models
    from hearing.synth import asr

    seen = {}

    class FakeModel:
        def transcribe(self, x, **kw):
            seen.update(kw)
            w = SimpleNamespace(word=" Aryan,", start=0.1, end=0.5, probability=0.42)
            return [SimpleNamespace(words=[w], avg_logprob=-0.8, no_speech_prob=0.3)], None

    monkeypatch.setattr(asr, "local_available", lambda: True)
    monkeypatch.setattr(asr, "_whisper_loader", lambda name: "fake")
    monkeypatch.setattr(models, "load", lambda key: FakeModel())
    (w,) = nc.whisper_transcriber(None)(np.zeros(SR, np.float32))
    assert (w.text, w.probability, w.avg_logprob, w.no_speech_prob) == ("Aryan,", 0.42, -0.8, 0.3)
    assert seen["temperature"] == 0.0 and seen["beam_size"] == 1 and seen["condition_on_previous_text"] is False


def test_caption_asr_decodes_at_temperature_zero(monkeypatch):
    from types import SimpleNamespace

    from hearing import models
    from hearing.synth import asr

    seen = {}

    class FakeModel:
        def transcribe(self, x, **kw):
            seen.update(kw)
            return [SimpleNamespace(text=" hi")], None

    monkeypatch.setattr(models, "load", lambda key: FakeModel())
    eng = asr.WhisperASR.__new__(asr.WhisperASR)
    eng._key = "fake"
    eng.vocab, eng._floor, eng.dropped = [], [], 0
    eng._frames, eng._voiced, eng._rms, eng._t0, eng._t1 = [np.zeros(SR, np.float32)], True, [], 0.0, 1.0
    assert [t.text for t in eng._transcribe()] == ["hi"]
    assert seen["temperature"] == 0.0 and seen["beam_size"] == 1 and seen["condition_on_previous_text"] is False


def test_session_prompt_env_passes_profiles(monkeypatch, tmp_path):
    from hearing.audio import io as aio
    from hearing.server.session import ServerConfig, Session, load_stage_modules

    load_stage_modules()
    (tmp_path / "joseph").mkdir()
    aio.save(tmp_path / "joseph" / "a.wav", _tone(3.0, 200), SR)
    monkeypatch.setenv("HEARING_PROFILES_DIR", str(tmp_path))
    monkeypatch.setenv(nc.NAME_PROMPT_ENV, "1")
    made = {}

    def fake_caller(name="", aliases=None, **kw):
        made["kw"] = kw
        return NameCaller(name, aliases, transcriber=_scripted({}), embedder=_embed, cleaner=None, threaded=False,
                          **kw)

    monkeypatch.setattr("hearing.server.session.NameCaller", fake_caller)
    sess = Session(ServerConfig(prefs_log=None, faces="none", vision_threaded=False,
                                pipelines={"name_call": 'tiers(only="audio-only")'}), "cpu")
    try:
        sess._ensure_namecall()
        assert made["kw"]["prompt"] is True and len(made["kw"]["profiles"]) == 1
        assert sess.namecall.guard is True
    finally:
        sess.close()


# ------------------------------------------------------------------ T-136: snap to a saved profile

def _saved() -> dict:
    return {"aryan": [_tone(2.5, 200), _tone(2.5, 200, 0.1)], "joseph": [_tone(2.5, 600)]}


def _enroll_at(caller: NameCaller, x: np.ndarray) -> tuple[list, int | None]:
    events, at = [], None
    for i in range(0, len(x) - 159, 160):
        caller.feed(x[i : i + 160])
        got = caller.poll()
        if at is None and any(e.kind == "enroll" for e in got):
            at = i + 160
        events += got
    return events, at


def test_name_from_a_saved_voice_snaps_to_its_profile_sooner():
    quiet = np.zeros(SR, np.float32)
    x = np.concatenate([quiet, _tone(0.8, 200), quiet, quiet, quiet])  # the name alone: < MIN_CLIP_S voiced
    plain_events, plain_at = _enroll_at(_caller({200: "Aryan"}), x)
    caller = NameCaller("Aryan", transcriber=_scripted({200: "Aryan"}), embedder=_embed, cleaner=None,
                        threaded=False, saved=_saved())
    events, at = _enroll_at(caller, x)
    enroll = next(e for e in events if e.kind == "enroll")
    assert enroll.profile == "aryan" and len(enroll.pieces) == 2 and len(enroll.other_pieces) == 1
    assert enroll.detection.profile == "aryan" and enroll.detection.profile_sim > 0.9
    assert at <= plain_at - 2 * SR  # no wait for MIN_CLIP_S (here: EXTEND_S) of the mixed utterance
    st = caller.status()
    assert st["target_profile"] == "aryan" and st["saved_profiles"] == ["aryan", "joseph"]
    assert st["last"]["profile"] == "aryan"
    assert plain_events[0].profile is None


def test_saved_profiles_retarget_and_same_profile_is_ignored():
    caller = NameCaller("Aryan", transcriber=_scripted({200: "Aryan", 600: "Aryan"}), embedder=_embed,
                        cleaner=None, threaded=False, saved=_saved())
    quiet = np.zeros(SR, np.float32)
    events = _feed(caller, np.concatenate([_tone(1.5, 200), quiet, quiet, _tone(1.5, 200), quiet, quiet,
                                           _tone(1.5, 600), quiet, quiet]))
    enrolls = [e for e in events if e.kind == "enroll"]
    assert [e.profile for e in enrolls] == ["aryan", "joseph"]
    assert enrolls[1].retarget and caller.ignored == 1
    assert caller.status()["target_profile"] == "joseph"


def test_unknown_voice_keeps_the_clip_path():
    def embed3(x):  # three voices: 200, 400 and 600 Hz
        spec = np.abs(np.fft.rfft(x, n=4096))
        v = np.array([spec[(f - 50) * 4096 // SR:(f + 50) * 4096 // SR].sum() for f in (200, 400, 600)])
        return v / max(np.linalg.norm(v), 1e-9)

    def transcribe(x):
        d = len(x) / SR
        return [Word("hello", 0.0, d / 2), Word("Aryan", d / 2, d)]
    caller = NameCaller("Aryan", transcriber=transcribe, embedder=embed3, cleaner=None, threaded=False,
                        saved=_saved())
    events = _feed(caller, np.concatenate([_tone(1.5, 400), np.zeros(2 * SR, np.float32)]))
    enroll = next(e for e in events if e.kind == "enroll")
    assert enroll.profile is None and caller.status()["target_profile"] is None
