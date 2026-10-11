"""Context features (1 Hz room vector), the preference model, toasts, Auto, and the wire events."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hearing.audio.io import float_to_pcm16, pcm16_to_float
from hearing.context.features import FEATURES, ContextTracker, ContextVector, hop_db
from hearing.context.prefs import (
    AUTO_MODES,
    LABELS,
    Advisor,
    AdvisorConfig,
    PrefLog,
    PrefModel,
    PrefRow,
    Recommender,
    evaluate,
    simulate,
)
from hearing.server.app import create_app
from hearing.server.protocol import KIND_AUDIO_IN, pack_frame
from hearing.server.session import ServerConfig, Session
from hearing.testing import make_test_signal
from hearing.types import FaceTrack, StageContext, TargetState

SR = 16000


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.delenv("HEARING_PREFS_LOG", raising=False)


def _room(seconds: float, speech: float, noise: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = round(seconds * SR)
    x = speech * make_test_signal(seconds, SR, seed=seed)[:n] + rng.normal(0, noise, n)
    return np.clip(x, -1, 1).astype(np.float32)


def _vectors(x: np.ndarray, ctx: StageContext | None = None) -> list[ContextVector]:
    tracker, ctx = ContextTracker(), ctx or StageContext.default()
    out = [tracker.push(x[i : i + 160], ctx, "natural") for i in range(0, len(x) - 159, 160)]
    return [v for v in out if v is not None]


def _track(i: int, p: float) -> FaceTrack:
    return FaceTrack(i, (0.1 * i, 0.2, 0.1, 0.2), None, p, 0.0)


# ---------------------------------------------------------------- features


def test_hop_db_of_full_scale_sine_and_silence():
    t = np.arange(160) / SR
    assert hop_db(np.sin(2 * np.pi * 1000 * t)) == pytest.approx(-3.0, abs=0.1)
    assert hop_db(np.zeros(160)) == -100.0


def test_one_vector_per_second_and_noise_and_snr_track_the_room():
    quiet = _vectors(_room(3.0, speech=0.3, noise=0.001))
    noisy = _vectors(_room(3.0, speech=0.3, noise=0.05))
    assert len(quiet) == len(noisy) == 3
    q, n = quiet[-1], noisy[-1]
    assert q.noise_db == pytest.approx(-60, abs=3) and n.noise_db == pytest.approx(-26, abs=3)
    assert q.snr_db > n.snr_db + 20
    assert q.speakers == n.speakers == 1  # audio-only fallback: speech present
    assert _vectors(_room(2.0, speech=0.0, noise=0.05))[-1].speakers == 0


def test_tracks_give_speakers_overlap_and_target_confidence():
    ctx = StageContext.default()
    ctx.tracks = [_track(1, 0.9), _track(2, 0.7), _track(3, 0.1)]
    ctx.target = TargetState(1, 0.8, False, 0.0)
    v = _vectors(_room(1.0, 0.3, 0.01), ctx)[0]
    assert (v.speakers, v.overlap, v.target_conf) == (2, 1.0, 0.8)
    ctx.target = TargetState(None, 0.8, False, 0.0)  # confidence without a target counts as 0
    assert _vectors(_room(1.0, 0.3, 0.01), ctx)[0].target_conf == 0.0


def test_context_message_matches_contract_fields():
    msg = ContextVector(-28.5, 6.1, 2, 0.35, 0.9, "natural").message(auto=False)
    assert msg["type"] == "context"
    assert set(msg) >= {"noise_db", "snr_db", "speakers", "overlap", "target_conf", "mode"}
    json.dumps(msg)


# ---------------------------------------------------------------- preference model


def test_prefs_heldout_accuracy_beats_baseline_and_personalises():
    r = evaluate(seed=1)
    print("\nprefs held-out:", json.dumps(r))
    assert r["prior_heldout_acc"] >= 0.75 > r["majority_baseline_acc"] + 0.3
    assert r["revoicer_acc_after_60_choices"] > r["revoicer_acc_prior_only"]
    assert r["recommend_ms"] < 1.0


def test_model_fits_separable_toy_data():
    x, y = simulate(400, seed=3, label_noise=0.0)
    m = PrefModel().fit(x, y)
    assert m.accuracy(x, y) > 0.85
    p = m.proba(x[:5])
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p.shape == (5, len(LABELS))


def test_pref_log_round_trips_jsonl_and_skips_bad_lines(tmp_path):
    path = tmp_path / "sub" / "prefs.jsonl"
    log = PrefLog(path)
    log.append(PrefRow([-30.0, 5.0, 3.0, 0.4, 0.8], "synthetic", "natural", "explicit", 1.0))
    with path.open("a") as f:
        f.write("{torn\n" + json.dumps({"chosen": "natural"}) + "\n")
    rows = PrefLog(path).rows
    assert len(rows) == 1 and rows[0].chosen == "synthetic" and rows[0].features[2] == 3.0
    record = json.loads(path.read_text().splitlines()[0])
    assert set(record) == {*FEATURES, "chosen", "previous", "source", "ts"}  # nothing about the person


def test_logged_choices_move_the_recommendation(tmp_path):
    v = ContextVector(-60.0, 40.0, 1, 0.0, 0.0, "natural")
    rec = Recommender(PrefLog(tmp_path / "p.jsonl"))
    assert rec.recommend(v).mode == "natural"
    for _ in range(40):
        rec.log.append(PrefRow(v.values(), "text", "natural", "explicit"))
    rec.refit()
    assert rec.recommend(v).mode == "text"
    assert Recommender(PrefLog(tmp_path / "p.jsonl")).recommend(v).mode == "text"  # persisted


# ---------------------------------------------------------------- toasts and Auto

QUIET_ROOM = ContextVector(-60.0, 40.0, 1, 0.0, 0.0, "text")  # the seed model says natural
PARTY = ContextVector(-20.0, 1.0, 4, 0.6, 0.2, "natural")


def test_toast_needs_stable_seconds_then_cools_down():
    adv = Advisor(config=AdvisorConfig(stable_s=3, toast_cooldown_s=20, repeat_s=60, dwell_s=1e9))
    events = [adv.on_context(QUIET_ROOM, "text", float(t)) for t in range(1, 70)]
    times = [t for t, e in enumerate(events, start=1) if e]
    assert times == [3, 63]  # stable for 3 s, then the same mode only after repeat_s
    toast = events[2][0]
    assert toast["type"] == "recommendation" and toast["mode"] == "natural" and not toast["auto"]
    assert toast["confidence"] >= 0.6 and "quiet room" in toast["reason"]
    assert not any(adv.on_context(QUIET_ROOM, "natural", float(t)) for t in range(70, 80))


def test_user_change_is_logged_as_accepted_and_silences_toasts():
    adv = Advisor(config=AdvisorConfig(stable_s=1, user_quiet_s=10, dwell_s=1e9))
    assert adv.on_context(QUIET_ROOM, "text", 1.0)
    adv.on_user_mode("natural", "text")
    row = adv.rec.log.rows[-1]
    assert (row.chosen, row.previous, row.source) == ("natural", "text", "accepted")
    adv.on_user_mode("quiet", "natural")
    assert adv.rec.log.rows[-1].source == "explicit"
    assert not adv.on_context(QUIET_ROOM, "quiet", 5.0)  # within user_quiet_s


def test_dwell_logs_the_kept_mode_and_ignored_toasts_fade():
    adv = Advisor(config=AdvisorConfig(dwell_s=5, repeat_s=0, toast_cooldown_s=0))
    toasts = [bool(adv.on_context(QUIET_ROOM, "text", float(t))) for t in range(1, 101)]
    assert {r.source for r in adv.rec.log.rows} == {"dwell"} and len(adv.rec.log.rows) == 20
    assert toasts[2] and not toasts[-1]  # staying in text here taught the model


def test_auto_switches_with_hysteresis_and_cooldown():
    adv = Advisor(config=AdvisorConfig(stable_s=3, auto_cooldown_s=10))
    switches = []
    for t in range(1, 31):
        room = PARTY if t <= 15 else QUIET_ROOM
        for e in adv.on_context(room, "auto", float(t)):
            switches.append((t, e["mode"], e["auto"]))
    # t=1 is the "entering Auto" note (the model is confident, so it is in charge)
    assert switches == [(1, "natural", True), (3, "synthetic", True), (18, "natural", True)]
    assert adv.auto_mode == "natural"


# measured on grid-two-face.mp4 (T-076): the seeded model is unsure (< 0.6) in both rooms
TWO_TALKERS = ContextVector(-36.2, 8.9, 2, 0.16, 0.41, "natural")
ONE_TALKER = ContextVector(-36.4, 18.6, 1, 0.0, 0.41, "natural")
QUIET_PAIR = ContextVector(-60.0, 40.0, 2, 0.0, 0.9, "natural")  # model: natural, ~1.0


def test_unsure_model_falls_back_to_the_two_talker_rule():
    adv = Advisor(config=AdvisorConfig(stable_s=3, auto_cooldown_s=10))
    assert adv.rec.recommend(TWO_TALKERS, AUTO_MODES).confidence < adv.cfg.min_conf
    events = {}
    script = [ONE_TALKER] + [TWO_TALKERS] * 3 + [ONE_TALKER] * 8
    for t, v in enumerate(script, 1):
        events[t] = adv.on_context(v, "auto", float(t))
    assert [(e["mode"], e["source"]) for e in events[1]] == [("natural", "rule")]  # entering Auto
    assert events[2][0]["mode"] == "focus_hq" and events[2][0]["reason"] == "rule: 2 talkers"
    assert events[2][0]["auto"] is True
    assert not any(events[t] for t in range(3, 9))  # held while below 2 talkers for t=5..8
    assert events[9][0]["mode"] == "natural" and events[9][0]["source"] == "rule"  # 5th second
    assert adv.auto_mode == "natural" and not any(events[t] for t in (10, 11, 12))


def test_confident_model_stays_in_charge_over_the_rule():
    adv = Advisor(config=AdvisorConfig(stable_s=3))
    for t in range(1, 6):
        adv.on_context(QUIET_PAIR, "auto", float(t))
    assert adv.auto_mode == "natural"  # 2 talkers, but the model is sure


def test_auto_never_picks_a_silent_mode():
    adv = Advisor(config=AdvisorConfig(stable_s=1))
    nobody = ContextVector(-20.0, 1.0, 0, 0.0, 0.0, "natural")  # the seed model says quiet
    assert adv.rec.recommend(nobody).mode == "quiet"
    for t in range(1, 5):
        adv.on_context(nobody, "auto", float(t))
    assert adv.auto_mode in ("natural", "focus_hq", "synthetic")


# ---------------------------------------------------------------- session and wire


def _frames(x: np.ndarray, n: int = 320):
    return [pack_frame(KIND_AUDIO_IN, i, i * n / SR, float_to_pcm16(x[i * n : (i + 1) * n]))
            for i in range(len(x) // n)]


def test_session_auto_mode_drives_the_pipeline():
    config = ServerConfig(pipelines={"synthetic": "silence"}, advisor=AdvisorConfig(stable_s=2))
    s = Session(config, device="cpu")
    s.handle_text(json.dumps({"type": "control", "mode": "auto"}))
    assert s.effective_mode == "natural"
    s.advisor.on_context(PARTY, "auto", 1.0)
    s.advisor.on_context(PARTY, "auto", 2.0)
    assert s.effective_mode == "synthetic"
    x = pcm16_to_float(float_to_pcm16(_room(0.02, 0.3, 0.01)))
    s.handle_binary(_frames(x)[0])
    t = s.telemetry()
    assert (t["mode"], t["effective_mode"], t["pipeline"]) == ("auto", "synthetic", "silence")


class _ScriptedContext:
    """Stands in for the session's ContextTracker: one scripted vector per second of audio."""

    def __init__(self, vectors):
        self.vectors, self.n = list(vectors), 0

    def push(self, samples, ctx, mode):
        self.n += len(samples)
        if self.n < SR or not self.vectors:
            return None
        self.n -= SR
        return self.vectors.pop(0)


def test_session_auto_reaches_focus_hq_on_two_talkers_and_says_why():
    config = ServerConfig(pipelines={"focus_hq": "silence"}, advisor=AdvisorConfig())
    s = Session(config, device="cpu")
    s.context = _ScriptedContext([ONE_TALKER] + [TWO_TALKERS] * 3 + [ONE_TALKER] * 6)
    s.handle_text(json.dumps({"type": "control", "mode": "auto"}))
    x = pcm16_to_float(float_to_pcm16(_room(10.0, 0.3, 0.01)))
    modes, recs = [], []
    for i, frame in enumerate(_frames(x)):
        for e in s.handle_binary(frame):
            if isinstance(e, dict) and e.get("type") == "recommendation":
                recs.append((i * 320 / SR, e))
        modes.append(s.effective_mode)
    second = lambda t: modes[round(t * SR / 320) - 1]
    assert second(1.5) == "natural" and second(2.5) == "focus_hq" and second(6.5) == "focus_hq"
    assert second(9.5) == "natural"
    assert [(round(t), e["mode"], e["reason"]) for t, e in recs] == [
        (1, "natural", "rule: fewer than 2 talkers"),
        (2, "focus_hq", "rule: 2 talkers"),
        (9, "natural", "rule: fewer than 2 talkers"),
    ]
    assert s.telemetry()["mode"] == "auto"


@pytest.mark.slow  # > 2 s (T-098: keeps verify_fast under 60 s)
def test_fake_client_receives_context_and_a_recommendation(tmp_path):
    log = tmp_path / "prefs.jsonl"
    config = ServerConfig(console_dir=tmp_path, prefs_log=log, advisor=AdvisorConfig(user_quiet_s=2))
    client = TestClient(create_app(config))
    x = _room(4.2, speech=0.3, noise=0.001)
    msgs = []
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "hello", "sample_rate": SR, "frame_ms": 20}))
        ws.send_text(json.dumps({"type": "control", "mode": "text"}))
        for frame in _frames(x):
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "control", "mode": "natural"}))  # accept the toast
        for _ in range(2000):
            raw = ws.receive()
            if raw.get("text") is None:
                continue
            msgs.append(json.loads(raw["text"]))
            m = msgs[-1]
            if m["type"] == "telemetry" and m["mode"] == "natural" and len(msgs) > 10:
                break  # (the server starts in natural: skip telemetry sent before "text")
    contexts = [m for m in msgs if m["type"] == "context"]
    recs = [m for m in msgs if m["type"] == "recommendation"]
    assert len(contexts) == 4
    c = contexts[-1]
    assert set(c) >= {"noise_db", "snr_db", "speakers", "overlap", "target_conf", "mode"}
    assert c["mode"] == "text" and c["speakers"] == 1 and c["noise_db"] < -50
    assert recs and recs[0]["mode"] == "natural" and recs[0]["confidence"] >= 0.6
    rows = PrefLog(log).rows
    assert [(r.chosen, r.previous, r.source) for r in rows] == [("natural", "text", "accepted")]
