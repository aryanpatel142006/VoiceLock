import json

import pytest

from hearing.context.prefs import PrefLog
from hearing.server.protocol import ProtocolError, parse_message
from hearing.server.session import ServerConfig, Session
from hearing.types import FaceTrack
from hearing.vision.target import ATTENTION_DEFAULTS, ATTENTION_PRESETS, TargetSelector


def face(i, cx, p=1.0, ts=0.0):
    return FaceTrack(i, (cx - 0.06, 0.4, 0.12, 0.2), None, p, ts)


def test_defaults_documented_and_hold_is_two_seconds():
    sel = TargetSelector()
    assert sel.attention == ATTENTION_DEFAULTS == {"dwell_s": 0.5, "hold_s": 2.0, "switch_margin": 0.04,
                                                    "tie_margin": 0.15, "hold_lock_s": 10.0}
    assert ATTENTION_PRESETS["normal"] == ATTENTION_DEFAULTS


@pytest.mark.parametrize("bad", [{"dwell_s": 0.0}, {"hold_s": 11}, {"switch_margin": -0.1},
                                 {"hold_s": float("nan")}, {"nope": 1}, {"dwell_s": True}])
def test_set_attention_rejects_invalid_and_keeps_old(bad):
    sel = TargetSelector()
    with pytest.raises(ValueError):
        sel.set_attention(**bad)
    assert sel.attention == ATTENTION_DEFAULTS


def test_protocol_validates_ranges():
    assert parse_message({"type": "control", "attention": {"hold_s": 4}}).attention.hold_s == 4
    for bad in ({"hold_s": 99}, {"dwell_s": 0}, {"switch_margin": 1}, {"preset": "wild"}):
        with pytest.raises(ProtocolError):
            parse_message({"type": "control", "attention": bad})


def test_live_hold_change_keeps_target_through_a_gap():
    def run(hold):
        sel = TargetSelector(hold_s=hold)
        sel.update([face(1, 0.5), face(2, 0.8)], 0.0)
        sel.update([face(1, 0.5), face(2, 0.8)], 0.1)
        sel.set_attention(hold_s=hold)
        return sel.update([face(2, 0.8)], 1.6).track_id  # A gone 1.5 s
    assert run(1.0) != 1
    assert run(4.0) == 1


def test_session_applies_attention_live_logs_it_and_echoes_telemetry(tmp_path):
    log = tmp_path / "prefs.jsonl"
    session = Session(ServerConfig(prefs_log=log), device="cpu")
    assert session.telemetry()["attention"]["hold_s"] == 2.0
    session.handle_text(json.dumps({"type": "control",
                                    "attention": {"dwell_s": 1.0, "hold_s": 4.0, "switch_margin": 0.1,
                                                  "preset": "sticky"}}))
    assert session.vision.selector.attention == {"dwell_s": 1.0, "hold_s": 4.0, "switch_margin": 0.1,
                                                         "tie_margin": 0.15, "hold_lock_s": 10.0}
    assert session.telemetry()["attention"]["hold_s"] == 4.0
    session.close()  # close flushes deferred preference writes
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["kind"] == "attention" and rec["preset"] == "sticky" and rec["hold_s"] == 4.0
    assert PrefLog(log).rows == []  # attention lines are not mode labels
    session.handle_text(json.dumps({"type": "control", "attention": {"hold_s": 50}}))
    assert session.vision.selector.hold_s == 4.0  # invalid update rejected


@pytest.mark.asyncio
async def test_attention_write_does_not_block_loop_and_failure_is_reported(tmp_path):
    import asyncio
    import threading

    session = Session(ServerConfig(prefs_log=tmp_path / 'prefs' / 'p.jsonl', faces='none'), 'cpu')
    # A file where the log's parent directory should be forces real filesystem failure.
    (tmp_path / 'prefs').write_text('blocked')
    loop_thread = threading.get_ident()
    original = session.advisor.rec.log.append_attention
    writer_threads = []

    def append(*args):
        writer_threads.append(threading.get_ident())
        return original(*args)

    session.advisor.rec.log.append_attention = append
    try:
        replies = session.handle_text(json.dumps({'type': 'control', 'attention': {'hold_s': 4}}))
        errors = [event for event in replies if event['type'] == 'error']
        for _ in range(100):
            await asyncio.sleep(.01)
            errors.extend(session.poll_pref_errors())
            if errors:
                break
        assert writer_threads and loop_thread not in writer_threads
        assert any(e['type'] == 'error' and e['stage'] == 'prefs' for e in errors)
        assert session.handle_text('{"type":"control","suppression":20}')[-1]['attention']['hold_s'] == 4
    finally:
        await asyncio.to_thread(session.close)


def test_deferred_attention_writes_preserve_order_and_flush_on_close(tmp_path):
    path = tmp_path / 'prefs.jsonl'
    session = Session(ServerConfig(prefs_log=path, faces='none'), 'cpu')
    for hold in (3, 4, 5):
        session.handle_text(json.dumps({'type': 'control', 'attention': {'hold_s': hold}}))
    session.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record['hold_s'] for record in records] == [3, 4, 5]


def test_tiebreak_prefers_the_face_speaking_alone_when_equally_central():
    sel = TargetSelector(speech_memory_s=None)
    for i in range(50):  # both off-centre at the same distance; only 2 talks
        ts = i / 25
        st = sel.update([face(1, 0.2, 0.0, ts), face(2, 0.8, 0.9, ts)], ts)
    assert st.track_id == 2


def test_tiebreak_off_beyond_margin_and_for_lock():
    sel = TargetSelector(speech_memory_s=None, tie_margin=0.0)
    st = sel.update([face(1, 0.45, 0.0, 0.0), face(2, 0.8, 0.9, 0.0)], 0.0)
    st = sel.update([face(1, 0.45, 0.0, 0.2), face(2, 0.8, 0.9, 0.2)], 0.2)
    assert st.track_id == 1  # central wins when margin is 0
    sel = TargetSelector(speech_memory_s=None)
    sel.lock(1, 0.0)
    for i in range(50):
        ts = i / 25
        st = sel.update([face(1, 0.2, 0.0, ts), face(2, 0.8, 0.9, ts)], ts)
    assert st.track_id == 1 and st.locked


def test_tiebreak_defers_to_centre_when_speaking_lead_is_below_gap():
    sel = TargetSelector(speech_memory_s=None)
    for i in range(50):  # lead 0.15 < 0.2: no evidence, the more central face 1 wins
        ts = i / 25
        st = sel.update([face(1, 0.21, 0.5, ts), face(2, 0.8, 0.65, ts)], ts)
    assert st.track_id == 1


def test_tiebreak_waits_for_evidence_before_a_one_frame_speaker_wins():
    sel = TargetSelector()
    st = sel.update([face(1, 0.2, 0.2, 0.0), face(2, 0.8, 0.55, 0.0)], 0.0)
    st = sel.update([face(1, 0.2, 0.2, 0.2), face(2, 0.8, 0.1, 0.2)], 0.2)
    assert st.track_id is None  # tied faces, only 0.2 s of evidence
