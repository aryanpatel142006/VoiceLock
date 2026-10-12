"""``/ws/who``: the Who's speaking page's socket (hearing.who). Audio in, labels out, never audio out.

The page sends the same frames as ``/ws/session`` (``hello`` with its ``sample_rate``, then ``KIND_AUDIO_IN`` PCM16
frames from the browser mic: the demo mic the judge page uses, chosen by the same saved deviceId), so the ID model
hears exactly what the live engine hears. The server resamples to 16 kHz, keeps the newest 1.5 s, scores it with
:class:`hearing.who.WhoModel` in a worker thread (about 0.4 s per analysis on the Mac, under TORCH_LOCK so it never
runs beside a live window) and sends::

    {"type": "who", "speaker": "aryan" | "joseph" | null, "state": "speaking" | "hold" | "unsure" | "listening",
     "confidence": 0..1, "t": seconds, "speech": bool, "level_db": .., "noise_db": .., "raw": {"slot", "ecapa", "cos"},
     "match": {"aryan": 0..1, "joseph": 0..1} | null}

``match`` is each saved voice profile's ECAPA cosine mapped linearly 0.2 -> 0, 0.8 -> 1 (:func:`hearing.who.match_of`),
what the page's per-person meters show; null until ECAPA has scored a window.

at :data:`EMIT_HZ` (the smoother's clock keeps running between analyses, so a hold expires on time). The first message
is ``who_status`` (``loading`` while the models load on the first connection) then ``who_ready`` with the model,
method and ``sessions_active``: the page warns when a Listen session is still open, because both would share the
Mac's GPU. Nothing here touches the live session's state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from hearing.audio.io import pcm16_to_float
from hearing.server.protocol import KIND_AUDIO_IN, Hello, ProtocolError, unpack_frame
from hearing.server.resample import StreamResampler
from hearing.who import METHOD, SR, WhoModel, WhoTracker, who_checkpoint

log = logging.getLogger(__name__)

EMIT_HZ = 5.0
_LOAD_LOCK = threading.Lock()


def get_model(app: FastAPI) -> WhoModel:
    """The process-wide :class:`WhoModel` (``app.state.who_model``; tests inject one), loaded once."""
    model = getattr(app.state, "who_model", None)
    if model is None:
        model = WhoModel()
        app.state.who_model = model
    with _LOAD_LOCK:
        model.load()
    return model


def mount_who(app: FastAPI, sessions: dict | None = None) -> None:
    sessions = sessions if sessions is not None else {}

    @app.get("/who/status")
    def who_status() -> dict:
        model = getattr(app.state, "who_model", None)
        ckpt = None
        with contextlib.suppress(RuntimeError):
            ckpt = who_checkpoint()
        return {"loaded": bool(model and model.loaded), "checkpoint": str(model.checkpoint if model and model.checkpoint else ckpt),
                "names": list(model.names if model else ()), "method": METHOD, "sessions_active": len(sessions)}

    @app.websocket("/ws/who")
    async def ws_who(ws: WebSocket) -> None:
        await ws.accept()
        send_lock = asyncio.Lock()

        async def send(msg: dict) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(msg))

        try:
            await send({"type": "who_status", "state": "loading"})
            try:
                model = await asyncio.to_thread(get_model, app)
            except Exception as exc:  # noqa: BLE001
                log.warning("who: model load failed: %s", exc)
                await send({"type": "who_status", "state": "error", "message": str(exc)})
                return
            await send({"type": "who_ready", "names": list(model.names), "checkpoint": str(model.checkpoint),
                        "device": model.device, "method": METHOD, "sessions_active": len(sessions)})
        except (WebSocketDisconnect, RuntimeError):
            return
        tracker = WhoTracker(model)
        t0 = time.monotonic()
        clock = lambda: time.monotonic() - t0
        resampler: StreamResampler | None = None
        audio_lock = threading.Lock()
        stop = asyncio.Event()

        async def analyse() -> None:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                while not stop.is_set():
                    with audio_lock:
                        due = tracker.due()
                    if not due:
                        await asyncio.sleep(0.02)
                        continue
                    started = clock()
                    try:
                        result = await asyncio.to_thread(_analyse_locked, tracker, audio_lock, started)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("who: analysis failed: %s", exc)
                        await asyncio.sleep(0.2)
                        continue
                    await send({"type": "who", **result, "compute_ms": round(1000 * (clock() - started))})

        async def tick() -> None:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                while not stop.is_set():
                    await asyncio.sleep(1.0 / EMIT_HZ)
                    with audio_lock:
                        result = tracker.tick(clock())
                    await send({"type": "who", **result})

        tasks = [asyncio.create_task(analyse()), asyncio.create_task(tick())]
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    try:
                        frame = unpack_frame(msg["bytes"])
                    except ProtocolError as exc:
                        await send({"type": "error", "stage": "who", "message": str(exc)})
                        continue
                    if frame.kind != KIND_AUDIO_IN:
                        continue
                    x = pcm16_to_float(frame.payload)
                    if resampler is not None:
                        x = resampler.process(x)
                    with audio_lock:
                        tracker.push(np.asarray(x, np.float32))
                elif msg.get("text") is not None:
                    try:
                        data = json.loads(msg["text"])
                    except ValueError:
                        await send({"type": "error", "stage": "who", "message": "not JSON"})
                        continue
                    if isinstance(data, dict) and data.get("type") == "hello":
                        try:
                            hello = Hello.model_validate(data)
                        except ValueError as exc:
                            await send({"type": "error", "stage": "who", "message": str(exc)})
                            continue
                        resampler = None if hello.sample_rate == SR else StreamResampler(hello.sample_rate, SR)
                        with audio_lock:
                            tracker.smoother.reset()
                    elif isinstance(data, dict) and data.get("type") == "reset":
                        with audio_lock:
                            tracker.smoother.reset()
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


def _analyse_locked(tracker: WhoTracker, audio_lock: threading.Lock, t: float) -> dict:
    """Snapshot the buffer under the lock, score outside it (the model call takes ~0.4 s)."""
    with audio_lock:
        window = tracker.buffer.copy()
        tracker.analysed_at = tracker.audio_t
    score = tracker.model.score(window, use_slot=tracker.use_slot, use_ecapa=tracker.use_ecapa)
    with audio_lock:
        return tracker.apply(t, score)
