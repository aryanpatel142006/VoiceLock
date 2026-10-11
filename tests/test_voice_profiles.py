"""T-125: voice profiles ($HEARING_PROFILES_DIR/<name>/*.wav) on the calibrated enrollment path."""

import json

import numpy as np
import pytest

from hearing.audio.io import save
from hearing.server.remote import RemoteStage
from hearing.server.session import ServerConfig, Session, load_stage_modules
from hearing.tse.enroll import (
    PROFILE_TARGET_ENV,
    PROFILES_ENV,
    AutoEnroller,
    list_profiles,
    load_profile,
    profile_clip,
    profile_pieces,
    split_pieces,
)
from hearing.tse.enrolled import EnrolledCandidate
from hearing.types import FaceTrack, StageContext, TargetState

SR = 16000
HOP = 160


def _voice(seconds: float, seed: int) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(round(seconds * SR))).astype(np.float32)


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    for name, seed in (("aryan", 1), ("joseph", 2)):
        (tmp_path / name).mkdir()
        save(tmp_path / name / "a.wav", _voice(6.0, seed), SR)
    monkeypatch.setenv(PROFILES_ENV, str(tmp_path))
    monkeypatch.delenv(PROFILE_TARGET_ENV, raising=False)
    return tmp_path


def _ctx(target, p_target=0.0, p_other=0.0):
    ctx = StageContext.default()
    ctx.tracks = [FaceTrack(1, (0.1, 0.3, 0.2, 0.4), None, p_target, 0.0),
                  FaceTrack(2, (0.6, 0.3, 0.2, 0.4), None, p_other, 0.0)]
    ctx.target = TargetState(target, 1.0, False, 0.0, 0)
    return ctx


def test_split_pieces_are_2_to_3_s_and_a_short_tail_joins_the_last():
    assert [len(p) / SR for p in split_pieces(_voice(6.0, 0))] == [2.5, 3.5]
    assert [len(p) / SR for p in split_pieces(_voice(7.5, 0))] == [2.5, 2.5, 2.5]
    assert [len(p) / SR for p in split_pieces(_voice(1.2, 0))] == [1.2]


def test_load_and_list(profiles):
    assert list_profiles() == ["aryan", "joseph"]
    assert sum(len(p) for p in load_profile("aryan")) == 6 * SR
    with pytest.raises(ValueError):
        load_profile("nobody")
    with pytest.raises(ValueError):
        load_profile("../aryan")


def test_profile_clip_round_trips_its_pieces_and_other_clips_are_not_profiles():
    pieces = [_voice(2.5, 3), _voice(2.0, 4)]
    clip = profile_clip(pieces)
    back = profile_pieces(clip.copy())  # a copy (the tse stage re-concatenates) is still recognised
    assert back is not None and [len(p) for p in back] == [len(p) for p in pieces]
    assert np.array_equal(back[1], pieces[1])
    assert profile_pieces(_voice(4.5, 5)) is None


def test_enroll_profile_is_ready_at_once_and_keeps_its_known_negative(profiles):
    e = AutoEnroller(1.75, SR)
    target, other = load_profile("aryan"), load_profile("joseph")
    e.enroll_profile("aryan", target, other)
    assert e.ready and e.calibrated and e.source == "profile" and e.profile == "aryan"
    negative = e.other_clip().copy()
    ctx = _ctx(1, p_target=0.0, p_other=0.9)  # the camera sees the other face talking alone
    x = _voice(3.0, 9)
    for i in range(0, len(x), HOP):
        e.update(x[i : i + HOP], ctx)
    assert e.track_id == 1  # the profile follows the selected face
    assert np.array_equal(e.other_clip(), negative)  # the profile negative is not replaced
    assert profile_pieces(e.clip()) is not None
    e.clear()
    assert e.source == "auto" and e.profile is None and not e.ready


def test_failed_calibration_window_keeps_the_profile(profiles):
    e = AutoEnroller(1.75, SR)
    e.enroll_profile("aryan", load_profile("aryan"))
    e.calibrate(1.0)
    silence = np.zeros(HOP, np.float32)
    for _ in range(100):
        e.update(silence, _ctx(1))
    assert e.calibration_failed and e.source == "profile" and e.ready


class _Embedder:
    def __init__(self):
        self.calls = []

    def embed(self, x):
        self.calls.append(len(x))
        v = np.zeros(4)
        v[len(self.calls) % 4] = 1.0 + len(x) / SR  # a different direction per piece
        return v


class _Separator:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def load(self):
        pass

    def separate(self, x, sr):
        self.calls += 1
        return [x, x * 0.1]


def test_candidate_embeds_a_profile_as_the_mean_of_unit_piece_embeddings_without_separating():
    sep, emb = _Separator(), _Embedder()
    cand = EnrolledCandidate(separator=sep, embedder=emb)
    clip = profile_clip([_voice(2.5, 1), _voice(2.5, 2)])
    e = cand.target_embedding(clip)
    assert sep.calls == 0 and emb.calls == [2.5 * SR, 2.5 * SR]
    assert np.allclose(e, np.array([0, 1, 1, 0]) / np.sqrt(2))
    cand.target_embedding(_voice(3.0, 3))  # a room clip still goes through the separator
    assert sep.calls == 1


def _session():
    load_stage_modules()
    return Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": "passthrough"}), device="cpu")


def test_off_without_the_env_var(monkeypatch):
    monkeypatch.delenv(PROFILES_ENV, raising=False)
    s = _session()
    try:
        out = s.handle_text(json.dumps({"type": "control", "enroll": {"action": "profile", "name": "aryan"}}))
        assert any(m["type"] == "error" and m["stage"] == "enroll" for m in out)
        s.handle_text('{"type":"control","profile":"judge"}')  # the judge profile does not invent one
        assert s.enroller.source == "auto" and "voice_profile" not in s.ctx.telemetry.extras
    finally:
        s.close()


def test_control_enroll_profile_and_clear(profiles):
    s = _session()
    try:
        out = s.handle_text(json.dumps({"type": "control", "enroll": {"action": "profile", "name": "joseph"}}))
        assert not [m for m in out if m["type"] == "error"]
        tel = next(m for m in out if m["type"] == "telemetry")
        assert tel["enrollment"]["source"] == "profile"
        assert s.enroller.source == "profile" and s.enroller.profile == "joseph"
        assert s.enroller.other_clip() is not None  # every other profile is the known negative
        status = s._enrollment_status()
        assert status["source"] == "profile" and status["profile"] == "joseph" and status["state"] == "calibrated"
        assert s.ctx.telemetry.extras["voice_profile"] == {"name": "joseph"}
        out = s.handle_text(json.dumps({"type": "control", "enroll": {"action": "profile", "name": "nobody"}}))
        assert any(m["type"] == "error" for m in out) and s.enroller.profile == "joseph"
        s.handle_text('{"type":"control","enroll":{"action":"clear"}}')
        assert s.enroller.source == "auto" and "voice_profile" not in s.ctx.telemetry.extras
        assert s._enrollment_status()["source"] is None
    finally:
        s.close()


@pytest.mark.parametrize(("env", "expected"), [(None, "aryan"), ("joseph", "joseph")])
def test_judge_profile_applies_a_profile(profiles, monkeypatch, env, expected):
    if env:
        monkeypatch.setenv(PROFILE_TARGET_ENV, env)
    s = _session()
    try:
        s.handle_text('{"type":"control","profile":"judge"}')
        assert s.enroller.source == "profile" and s.enroller.profile == expected
    finally:
        s.close()


def test_remote_stage_forwards_the_profile_on_its_own_and_then_the_clear():
    stage = RemoteStage(url="ws://127.0.0.1:9/ws/session", delay_ms=50)
    sent = []
    stage.client.send_control = lambda **fields: sent.append(fields) or True
    ctx = StageContext.default()
    try:
        ctx.telemetry.extras["voice_profile"] = {"name": "aryan", "other": "joseph"}
        stage._forward_control(ctx)
        assert {"enroll": {"action": "profile", "name": "aryan", "other": "joseph"}} in sent
        assert all("enroll" not in f or len(f) == 1 for f in sent)  # never bundled with other fields
        n = len(sent)
        stage._forward_control(ctx)
        assert len(sent) == n  # unchanged: not re-sent
        del ctx.telemetry.extras["voice_profile"]
        stage._forward_control(ctx)
        assert sent[-1] == {"enroll": {"action": "clear"}}
        stage._forward_control(ctx)
        assert sent[-1] == {"enroll": {"action": "clear"}} and len(sent) == n + 1
    finally:
        stage.close(wait=True)


def test_make_profile_trims_silence_and_writes_the_dir(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_profile.py"
    spec = importlib.util.spec_from_file_location("make_profile", path)
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    x = np.concatenate([np.zeros(2 * SR, np.float32), _voice(3.0, 1), np.zeros(2 * SR, np.float32)])
    save(tmp_path / "rec.wav", x, SR)
    assert mp.main(["aryan", str(tmp_path / "rec.wav"), "--out", str(tmp_path / "profiles")]) == 0
    pieces = load_profile("aryan", tmp_path / "profiles")
    total = sum(len(p) for p in pieces) / SR
    assert 3.0 <= total <= 3.3  # the 4 s of silence are gone, 100 ms of padding on each side kept


# ------------------------------------------------------------------ T-136: preload and swap

def test_hello_preloads_every_profile_pair_and_telemetry_lists_them(profiles):
    s = _session()
    try:
        s.handle_text(json.dumps({"type": "hello", "sample_rate": SR, "frame_ms": 20}))
        assert len(s.enroller.preload) == 2
        (t0, o0), (t1, o1) = s.enroller.preload
        assert profile_pieces(t0) is not None and profile_pieces(o0) is not None  # pieces path, no separator
        assert len(t0) == len(o1) and len(t1) == len(o0)
        assert s._enrollment_status()["profiles"] == ["aryan", "joseph"]
        s.handle_text(json.dumps({"type": "control", "enroll": {"action": "profile", "name": "joseph"}}))
        assert s.enroller.profile == "joseph" and np.array_equal(s.enroller.clip(), t1)
    finally:
        s.close()


def test_candidate_cache_keeps_both_profiles_so_a_swap_embeds_nothing():
    sep, emb = _Separator(), _Embedder()
    cand = EnrolledCandidate(separator=sep, embedder=emb)
    a, b = profile_clip([_voice(2.5, 1)]), profile_clip([_voice(2.5, 2)])
    cand.prepare(a, b)
    cand.prepare(b, a)
    n = len(emb.calls)
    cand.prepare(a, b)
    cand.prepare(b, a)
    assert len(emb.calls) == n == 4  # a and b once per slot, then cache hits
