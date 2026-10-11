"""JSON-only companions to existing sessions; no engine or media ownership."""
from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

MIRRORED = frozenset({'telemetry', 'caption', 'error', 'replay'})
QUEUE_SIZE = 128  # Disconnect lagging companions instead of stalling the owner.


@dataclass
class WatchedSession:
    connection: dict
    started: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    watchers: set = field(default_factory=set)
    latest: dict = field(default_factory=dict)


class SessionWatch:
    def __init__(self):
        self.sessions: dict[str, WatchedSession] = {}

    def open(self, session_id: str, connection: dict) -> None:
        self.sessions[session_id] = WatchedSession(connection)

    def publish(self, session_id: str, message: bytes | dict) -> None:
        state = self.sessions.get(session_id)
        if state is None or not isinstance(message, dict) or message.get('type') not in MIRRORED:
            return
        # Serialize now: queued events must not change with the session's mutable state.
        text = json.dumps(message)
        if message['type'] in {'telemetry', 'replay'}:
            state.latest[message['type']] = text
        for queue in tuple(state.watchers):
            if queue.full():
                state.watchers.discard(queue)
                self._end(queue, 1013)
            else:
                queue.put_nowait(text)

    @staticmethod
    def _end(queue: asyncio.Queue, code: int) -> None:
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(code)

    def close(self, session_id: str) -> None:
        state = self.sessions.pop(session_id, None)
        if state:
            for queue in state.watchers:
                self._end(queue, 1000)

    def mount(self, app: FastAPI) -> None:
        @app.get('/sessions')
        async def sessions() -> list[dict]:
            return [{'id': sid, 'client': state.connection['session'].hello.client,
                     'started': state.started} for sid, state in self.sessions.items()]

        @app.websocket('/ws/watch')
        async def watch(ws: WebSocket) -> None:
            await ws.accept()
            sid = ws.query_params.get('session')
            state = self.sessions.get(sid)
            if state is None:
                await ws.send_json({'type': 'error', 'stage': 'watch',
                                    'message': 'session disconnected'})
                await ws.close(code=1008)
                return
            queue = asyncio.Queue(maxsize=QUEUE_SIZE)
            state.watchers.add(queue)
            for text in state.latest.values():
                queue.put_nowait(text)

            async def send() -> None:
                while True:
                    item = await queue.get()
                    if isinstance(item, int):
                        await ws.close(code=item)
                        return
                    await ws.send_text(item)

            async def receive() -> None:
                while True:
                    message = await ws.receive()
                    if message['type'] == 'websocket.disconnect':
                        return
                    try:
                        event = json.loads(message.get('text') or '')
                    except ValueError:
                        event = None
                    if not isinstance(event, dict) or event.get('type') != 'control':
                        if queue.full():
                            await ws.close(code=1013)
                            return
                        queue.put_nowait(json.dumps({'type': 'error', 'stage': 'watch',
                                                    'message': 'only control JSON is accepted'}))
                        continue
                    if self.sessions.get(sid) is not state:
                        return
                    connection = state.connection
                    text = json.dumps(event)
                    if connection['tap'] is not None:
                        connection['tap'].incoming(text)
                    # No await between recording and handling: same event-loop ordering as owner.
                    replies = connection['session'].handle_text(text)
                    for reply in replies:
                        await connection['send'](reply)

            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                state.watchers.discard(queue)
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect,
                                             RuntimeError):
                        await task
