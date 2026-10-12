"""FastAPI session server: ``/health``, ``/ws/session``, the debug console at ``/`` and the simple page at ``/simple``.

Run with ``python -m hearing serve`` (see ``hearing.cli``) or
``uvicorn --factory hearing.server.app:create_app``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from hearing import __version__, registry
from hearing.context.prefs import prior_model
from hearing.device import pick
from hearing.server import rehearsal, team, warm
from hearing.server.comfort import mount_comfort
from hearing.server.connect import mount_connect
from hearing.server.hostload import host_load
from hearing.server.session import (
    LAST_CLIENT_DEVICES,
    ServerConfig,
    Session,
    load_stage_modules,
)
from hearing.server.static import mount_console
from hearing.server.tap import SessionTap
from hearing.server.team import read_git_rev
from hearing.server.watch import SessionWatch
from hearing.server.who import mount_who
from hearing.tse.enroll import list_profiles, profile_versions
from hearing.tse.stage import tse_window

TEAM_REPLAY_CLIP = "demo-team-easy/clip.mp4"  # T-173

log = logging.getLogger(__name__)


def remote_identity(remote_url: str | None) -> dict:
    """T-170: the GPU server's ``git_rev`` (last ``/health`` read by a probe or monitor; None before
    one) and the stale-rev warning against this checkout."""
    rev = team.remote_rev(remote_url)
    return {"remote_git_rev": rev, "remote_rev_warning": team.rev_warning(team.local_rev(), rev)}


def config_hashes(root: Path) -> dict[str, str | None]:
    """Short sha256 of each tiers config the ladder may load (``config/tiers*.yaml``), keyed by file
    name; ``scripts/demo_regress.py`` compares the remote's with the local file (T-118)."""
    out: dict[str, str | None] = {}
    for p in sorted((root / "config").glob("tiers*.yaml")):
        try:
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        except OSError:
            out[p.name] = None
    return out


def create_app(config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig()
    # Snapshot once: later pulls must never relabel a running process as new code.
    started_at = datetime.now(UTC).isoformat()
    git_rev = read_git_rev(Path(__file__).resolve().parents[2])
    tiers_config = config_hashes(Path(__file__).resolve().parents[2])  # as loaded at start (T-118)
    identity = {"version": __version__, "git_rev": git_rev,
                "started_at": started_at, "pid": os.getpid()}
    load_stage_modules()
    device = pick()  # once: probing torch per request would be slow
    prior_model()  # fit the seed preference model now, not inside the first session (~0.3 s)
    app = FastAPI(title="hearing", version=__version__)
    app.state.config = config
    app.state.device = device
    mount_comfort(app)

    sessions = {}
    watchers = SessionWatch()
    watchers.mount(app)

    async def stop_replay(connection):
        task = connection.get("task")
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            connection["task"] = None

    @app.post("/replay")
    async def replay(session_id: str, clip: str = "demo-mf-easy",
                     action: str = "start", loop: bool = False) -> dict:
        from hearing.replay import Clip, feed

        connection = sessions.get(session_id)
        if connection is None:
            raise HTTPException(404, "session disconnected")
        if action not in {"start", "stop"}:
            raise HTTPException(400, "action must be start or stop")
        async with connection["lock"]:
            if action == "stop":
                await stop_replay(connection)
                return {"active": False}
            root = Path(config.replay_dir).resolve()
            path = (root / clip).resolve()
            if path.is_dir() and path.parent == root:  # an ingested clip: <name>/clip.mp4 (T-090)
                path = path / "clip.mp4"
            if path.suffix.lower() != ".mp4" or root not in (path.parent, path.parent.parent):
                raise HTTPException(400, "clip must name an MP4 under the demo directory (<name>.mp4 or <name>/clip.mp4)")
            if not path.is_file():
                raise HTTPException(404, "clip missing; run python -m eval.make_fixtures --demo before the demo")
            try:
                decoded = await asyncio.to_thread(Clip.load, path)
            except (RuntimeError, ValueError, OSError) as exc:
                raise HTTPException(400, f"cannot decode clip: {exc}") from exc
            if sessions.get(session_id) is not connection:
                raise HTTPException(404, "session disconnected")
            await stop_replay(connection)

            async def run_replay():
                connection["active"] = True
                try:
                    await connection["send"]({"type": "replay", "active": True, "clip": clip})

                    async def ingest(message):
                        session = connection["session"]
                        tap = connection["tap"]
                        if tap is not None:
                            tap.incoming(json.dumps(message) if isinstance(message, dict) else message)
                        if isinstance(message, dict):
                            replies = session.handle_text(json.dumps(message))
                        else:
                            replies = session.handle_binary(message)
                            # Echo replay inputs only: the UI displays the exact JPEG being analysed.
                            await connection["send"](message, record=False)
                        for reply in replies:
                            await connection["send"](reply)

                    await feed(decoded, ingest, loop=loop)
                except Exception as exc:  # noqa: BLE001  (report replay worker failures to the client)
                    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                        await connection["send"]({"type": "error", "stage": "replay", "message": str(exc)})
                finally:
                    connection["active"] = False
                    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                        await connection["send"]({"type": "replay", "active": False, "clip": clip})

            connection["task"] = asyncio.create_task(run_replay())
            return {"active": True, "clip": clip}

    @app.get("/replay/clips")
    def replay_clips() -> dict:
        """Replayable clips: ``<name>.mp4`` and ingested ``<name>/clip.mp4`` (scripts/ingest_clip.py)."""
        from hearing.replay import DEFAULT_CLIP, list_clips
        from hearing.server.team import active_preset

        clips = list_clips(Path(config.replay_dir))
        default = f"{DEFAULT_CLIP}/clip.mp4"
        # T-173: under the team preset R / Replay opens the two-teammates mix (two voices recorded separately,
        # mixed digitally); it is only ever played behind the REPLAY badge.
        if active_preset() == "team" and any(c["clip"] == TEAM_REPLAY_CLIP for c in clips):
            default = TEAM_REPLAY_CLIP
        return {"default": default, "clips": clips}

    @app.get("/health")
    def health() -> dict:
        stages = {name: "ready" for name in registry.names()}
        for name in sorted(config.stage_names() - set(stages)):
            stages[name] = "unavailable"
        return {"ok": True, "device": device, "stages": stages, "warm": warm.status(),
                "tiers_config": tiers_config, "client_devices": dict(LAST_CLIENT_DEVICES) or None,
                "profiles": list_profiles(),  # T-172: the voice profiles a session would load
                "profile_versions": profile_versions(),  # T-172 (Q-467): their active v<N>, null when flat
                "sessions_active": len(sessions),  # T-182: open /ws/session connections (shared-GPU contamination check)
                "tse_window": tse_window(),  # T-419: the balanced enrolled tier's window/hop and expected delay
                "host_load": host_load(),  # T-485: load average per core, the app's "laptop is busy" warning
                **remote_identity(config.remote_url), **identity}

    @app.websocket("/ws/session")
    async def ws_session(ws: WebSocket) -> None:
        await ws.accept()
        session = Session(config, device)
        send_lock = asyncio.Lock()
        session_id = uuid.uuid4().hex
        connection = {"session": session, "active": False, "lock": asyncio.Lock()}
        sessions[session_id] = connection
        tap = SessionTap.from_env()
        connection["tap"] = tap
        if tap is not None:
            session.pre_agc_sink = tap.pre_agc

        async def send(msg: bytes | dict, *, record: bool = True) -> None:
            if isinstance(msg, dict) and msg.get("type") == "telemetry":
                # T-346: `recording` = this session is written to HEARING_SESSION_LOG (the app's truth strip)
                msg = {**msg, **identity, "recording": tap is not None, "host_load": host_load()}  # T-485
            async with send_lock:
                if tap is not None and record:
                    # Preserve the replay tap's privacy rule for voice estimates.
                    tap.outgoing({k: v for k, v in msg.items() if k != "voice"}
                                 if isinstance(msg, dict) else msg)
                if isinstance(msg, dict):
                    await ws.send_text(json.dumps(msg))
                    watchers.publish(session_id, msg)
                else:
                    await ws.send_bytes(msg)

        connection["send"] = send
        watchers.open(session_id, connection)
        if ws.query_params.get("replay") == "1":
            await send({"type": "session", "session_id": session_id})

        async def tick() -> None:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                while True:
                    await asyncio.sleep(1.0 / config.telemetry_hz)
                    for event in session.poll_pref_errors():
                        await send(event)
                    await send(session.telemetry())

        ticker = asyncio.create_task(tick())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                # Processing runs inline: stages are stateful and must see frames in order.
                if msg.get("bytes") is not None:
                    if connection["active"]:
                        continue  # exclusive input source during replay; controls still work
                    if tap is not None:
                        tap.incoming(msg["bytes"])
                    replies = session.handle_binary(msg["bytes"])
                elif msg.get("text") is not None:
                    if tap is not None:
                        tap.incoming(msg["text"])
                    replies = session.handle_text(msg["text"])
                else:
                    continue
                for reply in replies:
                    if isinstance(reply, dict) and reply["type"] == "error":
                        log.warning("session error: %s", reply)
                    await send(reply)
        except WebSocketDisconnect:
            pass
        finally:
            sessions.pop(session_id, None)
            watchers.close(session_id)
            await stop_replay(connection)
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            await asyncio.to_thread(session.close)  # joins synth TTS workers (< 1 s)
            if tap is not None:
                tap.close()

    rehearsal.mount(app)  # T-285: /api/rehearsal/*
    mount_who(app, sessions)  # /ws/who + /who/status: the Who's speaking page (hearing.who), no audio out
    mount_connect(app, config.console_dir)
    mount_console(app, config.console_dir)
    return app
