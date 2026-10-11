"""Companion routes use real sessions and taps, without hardware or model weights."""
import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hearing.server.app import create_app
from hearing.server.session import ServerConfig
from hearing.server.tls import ConnectInfo


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('HEARING_DEVICE', 'cpu')
    monkeypatch.setenv('HEARING_SESSION_LOG', str(tmp_path / 'logs'))
    config = ServerConfig(pipelines={'natural': 'passthrough'}, faces='mock',
                          asd='lipmotion', vision_threaded=False)
    with TestClient(create_app(config)) as client:
        yield client


def until(ws, kind, **fields):
    for _ in range(100):
        msg = ws.receive_json()
        if msg['type'] == kind and all(msg.get(k) == v for k, v in fields.items()):
            return msg
    pytest.fail(f'No {kind} with {fields}')


def test_lists_only_live_sessions_and_watcher_does_not_close_owner(client):
    assert client.get('/sessions').json() == []
    with client.websocket_connect('/ws/session?replay=1') as owner:
        sid = owner.receive_json()['session_id']
        owner.send_json({'type': 'hello', 'client': 'app', 'video': False})
        until(owner, 'telemetry')
        row, = client.get('/sessions').json()
        assert row['id'] == sid and row['client'] == 'app'
        assert datetime.fromisoformat(row['started']).utcoffset().total_seconds() == 0
        with client.websocket_connect(f'/ws/watch?session={sid}') as watcher:
            until(watcher, 'telemetry')
            assert client.get('/health').json()['sessions_active'] == 1
        owner.send_json({'type': 'control', 'ab': 'raw'})
        until(owner, 'telemetry', ab='raw')
    assert client.get('/sessions').json() == []


def test_two_watchers_forward_and_record_control_once(client, tmp_path):
    with client.websocket_connect('/ws/session?replay=1') as owner:
        sid = owner.receive_json()['session_id']
        with (client.websocket_connect(f'/ws/watch?session={sid}') as first,
              client.websocket_connect(f'/ws/watch?session={sid}') as second):
            control = {'type': 'control', 'ab': 'raw'}
            first.send_json(control)
            a = until(first, 'telemetry', ab='raw')
            b = until(second, 'telemetry', ab='raw')
            assert a == b
            until(owner, 'telemetry', ab='raw')
            first.send_json({'type': 'control', 'mode': 'invalid'})
            assert until(first, 'error') == until(second, 'error') == until(owner, 'error')
    records = [json.loads(line) for path in (tmp_path / 'logs').glob('*/events.jsonl')
               for line in path.read_text().splitlines()]
    assert sum(r['dir'] == 'in' and r['event'] == control for r in records) == 1


@pytest.mark.parametrize('query', ['', '?session=missing'])
def test_unknown_session(client, query):
    with client.websocket_connect('/ws/watch' + query) as ws:
        assert ws.receive_json()['stage'] == 'watch'
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008


def test_connect_page_preserves_origin_and_encodes_query(client):
    client.app.state.connect = ConnectInfo(['https://hearing.local:8000/simple'], True)
    response = client.get('/connect', params={'page': '/app/?watch=abc&x=1'})
    assert response.status_code == 200
    assert 'href="https://hearing.local:8000/app/?watch=abc&amp;x=1"' in response.text
    for page in ['https://evil.test/app', '//evil.test/app', '/\\evil.test/app']:
        assert client.get('/connect', params={'page': page}).status_code == 400


@pytest.mark.asyncio
async def test_mirror_filters_media_keeps_events_and_bounds_slow_watchers():
    import asyncio

    from hearing.server.watch import QUEUE_SIZE, SessionWatch

    hub = SessionWatch()
    hub.open('one', {})
    hub.open('other', {})
    queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    hub.sessions['one'].watchers.add(queue)
    events = [
        {'type': 'telemetry', 'seq': 1},
        {'type': 'caption', 'text': 'hello', 'final': True, 't0': 0, 't1': 1},
        {'type': 'error', 'stage': 'test', 'message': 'failure'},
        {'type': 'replay', 'active': True, 'clip': 'test.mp4'},
    ]
    for event in events:
        hub.publish('one', event)
    hub.publish('one', b'no audio or video')
    hub.publish('one', {'type': 'context'})
    hub.publish('other', {'type': 'caption', 'text': 'different session'})
    assert [json.loads(queue.get_nowait()) for _ in events] == events
    assert queue.empty()
    for seq in range(QUEUE_SIZE + 1):
        hub.publish('one', {'type': 'telemetry', 'seq': seq})
    assert queue.get_nowait() == 1013
    assert not hub.sessions['one'].watchers
    assert 'one' in hub.sessions


def test_owner_close_disconnects_watchers(client):
    with client.websocket_connect('/ws/session?replay=1') as owner:
        sid = owner.receive_json()['session_id']
        # Keep the companion open while closing its owner.
        with client.websocket_connect(f'/ws/watch?session={sid}') as watcher:
            until(watcher, 'telemetry')
            owner.close()
            with pytest.raises(WebSocketDisconnect) as exc:
                while True:
                    watcher.receive_json()
            assert exc.value.code == 1000
            assert client.get('/sessions').json() == []


@pytest.mark.parametrize('payload', ['[]', '{', '{"type":"hello","client":"phone"}'])
def test_non_control_does_not_reach_tap(client, tmp_path, payload):
    with client.websocket_connect('/ws/session?replay=1') as owner:
        sid = owner.receive_json()['session_id']
        with client.websocket_connect(f'/ws/watch?session={sid}') as watcher:
            watcher.send_text(payload)
            assert until(watcher, 'error')['stage'] == 'watch'
            watcher.send_json({'type': 'control', 'ab': 'raw'})
            until(watcher, 'telemetry', ab='raw')
    records = [json.loads(line) for path in (tmp_path / 'logs').glob('*/events.jsonl')
               for line in path.read_text().splitlines()]
    incoming = [r['event'] for r in records if r['dir'] == 'in']
    assert incoming == [{'type': 'control', 'ab': 'raw'}]


@pytest.mark.asyncio
async def test_invalid_message_burst_closes_overloaded_watcher():
    """A buffered receive burst can fill the queue before its sender gets CPU."""
    import asyncio

    from fastapi import FastAPI

    from hearing.server.watch import QUEUE_SIZE, SessionWatch

    class BufferedSocket:
        def __init__(self):
            self.query_params = {'session': 'one'}
            self.received = 0
            self.close_codes = []

        async def accept(self):
            pass

        async def receive(self):
            self.received += 1
            if self.received > QUEUE_SIZE + 1:
                await asyncio.Future()
            return {'type': 'websocket.receive', 'text': '[]'}

        async def send_text(self, text):
            await asyncio.Future()  # Simulate a companion that has stopped reading.

        async def close(self, code):
            self.close_codes.append(code)

    hub = SessionWatch()
    hub.open('one', {})
    app = FastAPI()
    hub.mount(app)
    endpoint = next(route.endpoint for route in app.routes if route.path == '/ws/watch')
    ws = BufferedSocket()
    await asyncio.wait_for(endpoint(ws), timeout=1)
    assert ws.close_codes == [1013]
    assert 'one' in hub.sessions
    assert not hub.sessions['one'].watchers


def test_websocket_mirrors_cached_state_and_live_caption_replay_without_media():
    from fastapi import FastAPI

    from hearing.server.watch import SessionWatch

    hub = SessionWatch()
    hub.open('one', {})
    telemetry = {'type': 'telemetry', 'seq': 7}
    replay = {'type': 'replay', 'active': True, 'clip': 'fixture.mp4'}
    hub.publish('one', telemetry)
    hub.publish('one', replay)
    app = FastAPI()
    hub.mount(app)
    with (TestClient(app) as client,
          client.websocket_connect('/ws/watch?session=one') as first,
          client.websocket_connect('/ws/watch?session=one') as second):
        for ws in (first, second):
            assert ws.receive_json() == telemetry
            assert ws.receive_json() == replay
        client.portal.call(hub.publish, 'one', b'audio must not be sent')
        client.portal.call(hub.publish, 'one', {'type': 'context'})
        for event in [
            {'type': 'caption', 'text': 'real event', 'final': True, 't0': 0, 't1': 1},
            {'type': 'replay', 'active': False, 'clip': 'fixture.mp4'},
        ]:
            client.portal.call(hub.publish, 'one', event)
            assert first.receive_json() == second.receive_json() == event
