"""Offline replay decoding, wire framing, clock scheduling and same-session routing."""
import json
from itertools import pairwise

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eval.fixtures import video
from hearing.audio.io import save
from hearing.server.app import create_app
from hearing.server.protocol import unpack_frame
from hearing.server.session import ServerConfig


def clip(tmp_path, seconds=0.12):
    frames = np.full((round(seconds * 25), 90, 160, 3), 128, dtype=np.uint8)
    frames[:, 20:65, 62:96] = (0, 255, 0)
    frames[:, 25:65, 8:35] = (0, 255, 0)
    for i in range(len(frames)):
        if i % 6 < 3:
            frames[i, 49:58, 70:87] = 0
    t = np.arange(round(seconds * 16000)) / 16000
    save(tmp_path / 'mix.wav', (np.where((t * 25).astype(int) % 6 < 3, .2, .001) * np.sin(t * 440 * 2 * np.pi)).astype(np.float32), 16000)
    return video.write_video(tmp_path / 'tiny.mp4', frames, 25, audio_wav=tmp_path / 'mix.wav')


@pytest.mark.asyncio
async def test_feeder_schedules_audio_and_video_with_shared_clock(tmp_path):
    from hearing.replay import Clip, feed

    decoded = Clip.load(clip(tmp_path))
    clock = [5.0]
    sent = []

    async def sleep(delay):
        clock[0] += delay

    async def send(message):
        sent.append((clock[0], message))

    await feed(decoded, send, clock=lambda: clock[0], sleep=sleep, timestamp=100.0)
    assert sent[0][1]['type'] == 'hello'
    assert sent[1][1]['type'] == 'control'
    frames = [(at, unpack_frame(msg)) for at, msg in sent if isinstance(msg, bytes)]
    audio = [(at, f) for at, f in frames if f.kind == 1]
    images = [(at, f) for at, f in frames if f.kind == 2]
    assert len(audio) == 6
    assert len(images) == 3
    assert all(len(f.payload) == 640 for _, f in audio)
    assert [f.seq for _, f in audio] == list(range(6))  # one counter per kind (T-072)
    assert [f.seq for _, f in images] == list(range(3))
    for at, frame in frames:
        assert at - 5 == pytest.approx(frame.capture_ts - 100)
    assert [at - 5 for at, _ in images] == pytest.approx([0, .04, .08])
    assert clock[0] == pytest.approx(5.12)


def test_replay_routes_processed_output_to_existing_socket(tmp_path, monkeypatch):
    monkeypatch.setenv('HEARING_DEVICE', 'cpu')
    clip(tmp_path)
    config = ServerConfig(replay_dir=tmp_path, pipelines={'natural': 'passthrough'}, faces='mock',
                          asd='lipmotion', vision_threaded=False)
    with TestClient(create_app(config)) as client, client.websocket_connect('/ws/session?replay=1') as ws:
        session_id = ws.receive_json()['session_id']
        response = client.post('/replay', params={'session_id': session_id, 'clip': 'tiny.mp4'})
        assert response.status_code == 200
        output = []
        while True:
            event = ws.receive()
            if event.get('bytes'):
                output.append(unpack_frame(event['bytes']))
            else:
                msg = json.loads(event['text'])
                if msg.get('type') == 'replay' and not msg['active']:
                    break
        assert any(f.kind == 3 and len(f.payload) == 640 for f in output)
        assert any(f.kind == 2 for f in output)
        assert client.post('/replay', params={'session_id': session_id, 'clip': '../mix.wav'}).status_code == 400
        assert client.post('/replay', params={'session_id': session_id, 'clip': 'missing.mp4'}).status_code == 404
        assert client.post('/replay', params={'session_id': session_id, 'action': 'stop'}).status_code == 200
    assert client.post('/replay', params={'session_id': session_id}).status_code == 404


def test_demo_builder_creates_twelve_second_local_clip(tmp_path):
    from eval import make_fixtures
    source = tmp_path / 'generated/av_full/008_two_face_snr-5_sir0_babble'
    source.mkdir(parents=True)
    clip(source)
    (source / 'video.mp4').write_bytes((source / 'tiny.mp4').read_bytes())
    (source / 'manifest.json').write_text(json.dumps({'layout': 'two_face', 'source': {'dataset': 'GRID'}}))
    path = make_fixtures.make_demo(tmp_path / 'generated', tmp_path / 'demo')
    info = video.probe(path)
    assert info.has_audio
    from hearing.replay import Clip
    assert Clip.load(path).duration == pytest.approx(12, abs=.001)
    assert info.duration_s == pytest.approx(12, abs=.1)
    assert json.loads((path.parent / 'manifest.json').read_text())['source_fixture'] == source.name


def test_replay_tap_contains_faces_target_and_audio_out(tmp_path, monkeypatch):
    monkeypatch.setenv('HEARING_DEVICE', 'cpu')
    monkeypatch.setenv('HEARING_SESSION_LOG', str(tmp_path / 'tap'))
    clip(tmp_path, seconds=1.4)
    config = ServerConfig(replay_dir=tmp_path, pipelines={'natural': 'passthrough>gate'},
                          faces='mock', asd='lipmotion', vision_threaded=False)
    with TestClient(create_app(config)) as client, client.websocket_connect('/ws/session?replay=1') as ws:
        session_id = ws.receive_json()['session_id']
        assert client.post('/replay', params={'session_id': session_id, 'clip': 'tiny.mp4', 'loop': True}).status_code == 200
        for _ in range(180):
            event = ws.receive()
            if event.get('text'):
                msg = json.loads(event['text'])
                if msg.get('faces') and msg.get('target', {}).get('track_id') is not None:
                    assert msg['recording'] is True  # T-346: the tap is on, telemetry says so
                    break
        else:
            pytest.fail('replay never selected a target')
        assert client.post('/replay', params={'session_id': session_id, 'action': 'stop'}).status_code == 200
    tap_dir, = (tmp_path / 'tap').iterdir()
    records = [json.loads(line)['event'] for line in (tap_dir / 'events.jsonl').read_text().splitlines()]
    import wave
    with wave.open(str(tap_dir / 'in.wav')) as raw, wave.open(str(tap_dir / 'out.wav')) as processed:
        assert raw.getnframes() == processed.getnframes() > 0
        assert raw.getframerate() == processed.getframerate() == 16000
    assert list((tap_dir / 'frames').glob('*.jpg'))
    assert any(r.get('kind') == 3 for r in records)
    assert any(r.get('faces') and r.get('target', {}).get('track_id') is not None for r in records)
    assert all('voice' not in r for r in records)


def test_replay_cli_imports_in_fresh_process():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, 'scripts/replay.py', '--help'], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert '--loop' in result.stdout


@pytest.mark.asyncio
async def test_loop_keeps_sequences_and_capture_clock_monotonic(tmp_path):
    import asyncio

    from hearing.replay import Clip, feed
    clock = [0.0]
    frames = []

    async def sleep(delay):
        clock[0] += delay

    async def send(message):
        if isinstance(message, bytes):
            frames.append(unpack_frame(message))
            if len(frames) == 19:
                raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await feed(Clip.load(clip(tmp_path)), send, loop=True, clock=lambda: clock[0],
                   sleep=sleep, timestamp=100)
    for kind in (1, 2):  # per-kind counters keep counting across the loop
        seqs = [f.seq for f in frames if f.kind == kind]
        assert seqs == list(range(len(seqs)))
    assert len([f for f in frames if f.kind == 1]) > 6  # crossed the loop boundary
    assert frames[9].capture_ts == pytest.approx(100.12)
    assert frames[18].capture_ts == pytest.approx(100.24)
    assert all(b.capture_ts >= a.capture_ts for a, b in pairwise(frames))


def test_replay_does_not_feed_a_second_page_session(tmp_path, monkeypatch):
    monkeypatch.setenv('HEARING_DEVICE', 'cpu')
    clip(tmp_path)
    config = ServerConfig(replay_dir=tmp_path, pipelines={'natural': 'passthrough'}, faces='none')
    with TestClient(create_app(config)) as client, client.websocket_connect('/ws/session?replay=1') as a, client.websocket_connect('/ws/session?replay=1') as b:
        aid = a.receive_json()['session_id']
        bid = b.receive_json()['session_id']
        assert aid != bid
        assert client.post('/replay', params={'session_id': aid, 'clip': 'tiny.mp4'}).status_code == 200
        # A second page keeps its own input/controls, never receives replay media.
        b.send_json({'type': 'hello', 'sample_rate': 16000})
        response = b.receive_json()
        assert response['type'] == 'telemetry'
        assert response['seq'] is None
        assert client.post('/replay', params={'session_id': aid, 'action': 'stop'}).status_code == 200


@pytest.mark.parametrize('failure', [ValueError('bad feed'), TypeError('bad handler')])
def test_replay_reports_unexpected_failures_and_stops(tmp_path, monkeypatch, failure):
    from hearing import replay

    monkeypatch.setenv('HEARING_DEVICE', 'cpu')
    (tmp_path / 'tiny.mp4').touch()
    monkeypatch.setattr(replay.Clip, 'load', lambda path: None)

    async def broken_feed(*args, **kwargs):
        raise failure

    monkeypatch.setattr(replay, 'feed', broken_feed)
    config = ServerConfig(replay_dir=tmp_path, pipelines={'natural': 'passthrough'}, faces='none')
    with TestClient(create_app(config)) as client, client.websocket_connect('/ws/session?replay=1') as ws:
        sid = ws.receive_json()['session_id']
        assert client.post('/replay', params={'session_id': sid, 'clip': 'tiny.mp4'}).status_code == 200
        events = [ws.receive_json() for _ in range(3)]
        assert events[1]['type'] == 'error' and events[1]['stage'] == 'replay'
        assert events[2]['type'] == 'replay' and events[2]['active'] is False
        assert client.post('/replay', params={'session_id': sid, 'action': 'stop'}).status_code == 200


def test_replay_import_without_eval_package():
    import subprocess
    import sys

    code = '''
import sys
class NoEval:
    def find_spec(self, fullname, *args):
        if fullname == "eval" or fullname.startswith("eval."):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoEval())
from hearing.replay import Clip, feed
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_telemetry_says_not_recording_without_the_session_log(monkeypatch):
    """T-346: the app's truth strip reads telemetry ``recording``; False when HEARING_SESSION_LOG is unset."""
    monkeypatch.setenv('HEARING_DEVICE', 'cpu')
    monkeypatch.delenv('HEARING_SESSION_LOG', raising=False)
    config = ServerConfig(pipelines={'natural': 'passthrough'}, faces='mock', asd='lipmotion', vision_threaded=False)
    with TestClient(create_app(config)) as client, client.websocket_connect('/ws/session') as ws:
        for _ in range(50):
            msg = ws.receive_json()
            if msg.get('type') == 'telemetry':
                assert msg['recording'] is False
                break
        else:
            pytest.fail('no telemetry')
