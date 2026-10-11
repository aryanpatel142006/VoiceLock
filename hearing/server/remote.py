"""Remote GPU path: run a pipeline (Focus HQ) on another machine over the same ``/ws/session``.

The GPU box runs the ordinary session server (``python -m hearing serve`` inside the
Dockerfile, or ``scripts/gpu_up.sh``). The laptop keeps the fast path local and uses the
``remote`` stage for the HQ mode::

    python -m hearing serve --pipelines \\
        'focus_hq=remote(url="ws://127.0.0.1:18765/ws/session", fallback="passthrough")'

The stage is a *client* of the remote server's wire contract (docs/CONTRACTS.md#events):
``hello`` (16 kHz, 10 ms frames), ``control`` (mode, suppression, target lock), one
``audio in`` frame per 10 ms hop, and the ``audio out`` frame the remote returns with the
same ``seq``. The round trip of each ``seq`` is the RTT.

Playout. ``process`` never blocks on the network: it queues the hop for sending and plays
the remote output of the hop ``delay_ms`` earlier (a fixed jitter buffer). A hop that has
not come back in time is filled with the local fallback output of that same (delayed)
hop. When the remote is down (unreachable, disconnected, or silent for ``stall_ms``) the
stage plays the local fallback undelayed and reconnects in the background every
``retry_s``. Every source switch is crossfaded over one hop.

Camera frames reach the remote through :meth:`RemoteClient.send_video` (the session forwards
every JPEG while a ``remote`` stage plays), so the remote runs its own vision and AV Focus HQ.
Track ids are per server: a locked local target is forwarded as the remote track whose box
overlaps it most (the remote's ``faces`` telemetry), or as the local id when there is no camera.

Where Focus HQ runs (``control.hq_backend``, docs/CONTRACTS.md#events) is the session's choice;
this module gives it :class:`HealthMonitor` (``/health`` + RTT in the background) and the
``auto`` rule (:func:`auto_rule`, :class:`AutoBackend`): remote when ``/health`` is green and
the RTT is below :data:`AUTO_MAX_RTT_MS`.

Probe from the command line (prints JSON: health, RTT p50/p95, remote latency)::

    python -m hearing.server.remote --url ws://127.0.0.1:18765/ws/session --seconds 5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, replace
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from hearing import registry
from hearing.audio import (
    basic as _basic,  # noqa: F401  (registers the default fallback "passthrough")
)
from hearing.audio.io import HOP, PIPELINE_SR, float_to_pcm16, pcm16_to_float
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    KIND_VIDEO_IN,
    pack_frame,
    unpack_frame,
)
from hearing.server.team import local_rev, note_rev, rev_warning
from hearing.types import AudioChunk, StageContext

if __name__ == "__main__":
    # ``python -m hearing.server.remote`` runs a second copy of this module after the package
    # import already registered the ``remote`` stage; run the probe from the imported copy.
    from hearing.server.remote import main as _main

    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(_main())

log = logging.getLogger(__name__)

ENV_URL = "HEARING_REMOTE_URL"
# T-129: "0" keeps camera frames on the laptop (hello.video false, nothing forwarded): with a voice
# profile the remote's Enrolled tier needs no camera, and its vision costs the remote CPU.
ENV_VIDEO = "HEARING_REMOTE_VIDEO"


def video_enabled() -> bool:
    """Forward camera frames to the remote (default) unless ``HEARING_REMOTE_VIDEO=0``."""
    return os.environ.get(ENV_VIDEO, "1").strip().lower() not in {"0", "false", "no", "off"}
HQ_BACKENDS = ("local", "remote", "auto")
AUTO_MAX_RTT_MS = 150.0  # auto: remote only while the round trip is below this
AUTO_UPGRADE_S = 2.0  # auto: the remote must stay good this long before Focus HQ moves to it
HEALTH_EVERY_S = 1.0  # /health poll period
HEALTH_WINDOW = 5  # health RTTs kept (median)
HEALTH_STALE_S = 3.0  # no successful check for this long: not green
MIN_TARGET_IOU = 0.3  # local target box vs remote face box to forward the lock
RTT_WINDOW = 500  # RTT samples kept for p50/p95 (5 s of hops)
MAX_PENDING = 1000  # sent-but-unanswered / answered-but-unplayed hops kept (10 s)
LOG_EVERY_S = 10.0
# The local latency preset -> the one the GPU session runs (T-108). Since T-086 the GPU box's ladder
# (config/tiers-remote.yaml) carries its own balanced enrolled windows, 0.5 s / 0.125 s (0.625 s
# delay, +10.68 dB on the A100; ``speed`` would be 0.5 s / 0.25 s, +9.73 dB at 0.75 s), so every
# preset is forwarded as is.
REMOTE_PRESETS: dict[str, str] = {}


def health_url(ws_url: str) -> str:
    """``ws://host:port/ws/session`` -> ``http://host:port/health`` (``wss`` -> ``https``)."""
    parts = urlsplit(ws_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "/health", "", ""))


def _percentile(values, q: float) -> float | None:
    return round(float(np.percentile(np.fromiter(values, dtype=np.float64), q)), 3) if values else None


class RemoteClient:
    """Background WebSocket client of a remote ``/ws/session``.

    Thread-safe and non-blocking for callers: :meth:`send_audio` / :meth:`send_video` /
    :meth:`send_control` only enqueue; one thread per connection sends, another receives.
    Reconnects every ``retry_s`` until :meth:`close`.
    """

    def __init__(
        self,
        url: str,
        *,
        mode: str = "focus_hq",
        sample_rate: int = PIPELINE_SR,
        frame_ms: float = 1000.0 * HOP / PIPELINE_SR,
        connect_timeout: float = 2.0,
        retry_s: float = 2.0,
        max_queue: int = 200,
        client: str = "remote-stage",
        video: bool = False,
    ):
        self.url = url
        self.sample_rate = sample_rate
        self.video = video
        self.connect_timeout = connect_timeout
        self.retry_s = retry_s
        self._hello = {"type": "hello", "sample_rate": sample_rate, "frame_ms": frame_ms,
                       "video": video, "client": client}
        self._control: dict = {"type": "control", "mode": mode}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._connected = threading.Event()
        self._closed = threading.Event()
        self._attempted = threading.Event()  # set after the first connection attempt ends
        self._thread: threading.Thread | None = None
        self._ws = None
        self._sent_at: dict[int, float] = {}
        self._results: dict[int, np.ndarray] = {}
        self._rtt: deque[float] = deque(maxlen=RTT_WINDOW)
        self.last_rx = 0.0  # perf_counter of the last audio-out frame
        self.sent = self.received = self.dropped = self.connects = 0
        self.remote_telemetry: dict | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------ lifecycle

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> RemoteClient:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="hearing-remote", daemon=True)
            self._thread.start()
        return self

    def wait_connected(self, timeout: float) -> bool:
        """Wait until connected; returns early (False) if the first attempt fails."""
        deadline = time.perf_counter() + timeout
        while not self._connected.wait(0.01):
            if self._attempted.is_set() or time.perf_counter() > deadline:
                return False
        return True

    def close(self, wait: bool = False) -> None:
        """Stop reconnecting and close the socket; ``wait`` joins the thread (tests, probe)."""
        self._closed.set()
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                ws.close()
        if wait and self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.connect_timeout + 1.0)

    def _run(self) -> None:
        from websockets.sync.client import connect

        while not self._closed.is_set():
            try:
                with connect(self.url, open_timeout=self.connect_timeout, close_timeout=1.0,
                             compression=None, max_size=2**22) as ws:
                    self._serve(ws)
            except Exception as e:  # noqa: BLE001  (refused, DNS, timeout, TLS, HTTP 404)
                self._note_error(f"connect {self.url}: {type(e).__name__}: {e}")
            finally:
                self._attempted.set()
            self._closed.wait(self.retry_s)

    def _serve(self, ws) -> None:
        with self._lock:
            self._sent_at.clear()
            self._results.clear()
            control = dict(self._control)
        _drain(self._queue)
        try:
            ws.send(json.dumps(self._hello))
            ws.send(json.dumps(control))
        except Exception as e:  # noqa: BLE001
            self._note_error(f"handshake: {type(e).__name__}: {e}")
            ws.close()
            return
        self._ws = ws
        self.connects += 1
        self.last_rx = time.perf_counter()
        self._connected.set()
        log.info("remote: connected to %s", self.url)
        sender = threading.Thread(target=self._send_loop, args=(ws,), name="hearing-remote-tx",
                                  daemon=True)
        sender.start()
        try:
            for msg in ws:
                self._on_message(msg)
        except Exception as e:  # noqa: BLE001  (connection lost)
            self._note_error(f"receive: {type(e).__name__}: {e}")
        finally:
            self._connected.clear()
            self._ws = None
            ws.close()
            sender.join(timeout=1.0)
            if not self._closed.is_set():
                log.warning("remote: disconnected from %s; local fallback until it is back", self.url)

    def _send_loop(self, ws) -> None:
        while self._connected.is_set() and not self._closed.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                ws.send(item)
            except Exception as e:  # noqa: BLE001
                self._note_error(f"send: {type(e).__name__}: {e}")
                ws.close()  # wakes the receive loop, which reconnects
                return

    def _note_error(self, message: str) -> None:
        if message != self.last_error:
            log.info("remote: %s", message)
        self.last_error = message

    # ------------------------------------------------------------ messages

    def _on_message(self, msg: bytes | str) -> None:
        now = time.perf_counter()
        if isinstance(msg, str):
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                return
            if data.get("type") == "telemetry":
                self.remote_telemetry = data
            elif data.get("type") == "error":
                self._note_error(f"remote error: {data}")
            return
        frame = unpack_frame(msg)
        if frame.kind != KIND_AUDIO_OUT:
            return
        samples = pcm16_to_float(frame.payload)
        with self._lock:
            sent = self._sent_at.pop(frame.seq, None)
            if sent is not None:
                self._rtt.append(1000.0 * (now - sent))
            self._results[frame.seq] = samples
            if len(self._results) > MAX_PENDING:
                del self._results[next(iter(self._results))]
            self.received += 1
            self.last_rx = now

    def _enqueue(self, data: bytes | str) -> bool:
        if not self._connected.is_set():
            return False
        try:
            self._queue.put_nowait(data)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def send_audio(self, seq: int, capture_ts: float, samples: np.ndarray) -> bool:
        """Queue one PCM16 ``audio in`` frame; False when disconnected or the queue is full."""
        seq &= 0xFFFFFFFF
        with self._lock:
            self._sent_at[seq] = time.perf_counter()
            if len(self._sent_at) > MAX_PENDING:
                del self._sent_at[next(iter(self._sent_at))]
        ok = self._enqueue(pack_frame(KIND_AUDIO_IN, seq, capture_ts, float_to_pcm16(samples)))
        if ok:
            self.sent += 1
        else:
            with self._lock:
                self._sent_at.pop(seq, None)
        return ok

    def send_video(self, seq: int, capture_ts: float, jpeg: bytes) -> bool:
        """Forward one JPEG ``video in`` frame (for remote AV-TSE); dropped when the hello said no video."""
        if not self.video:
            return False
        return self._enqueue(pack_frame(KIND_VIDEO_IN, seq, capture_ts, jpeg))

    def send_control(self, **fields) -> bool:
        """Merge ``fields`` into the control state (re-sent on reconnect) and send it."""
        with self._lock:
            self._control.update(fields)
        return self._enqueue(json.dumps({"type": "control", **fields}))

    def pop(self, seq: int) -> np.ndarray | None:
        with self._lock:
            return self._results.pop(seq & 0xFFFFFFFF, None)

    def stats(self) -> dict:
        with self._lock:
            rtt = list(self._rtt)
        remote_latency = (self.remote_telemetry or {}).get("latency_ms")
        return {
            "url": self.url,
            "connected": self.connected,
            "rtt_ms": {"p50": _percentile(rtt, 50), "p95": _percentile(rtt, 95),
                       "max": round(max(rtt), 3) if rtt else None, "n": len(rtt)},
            "sent": self.sent,
            "received": self.received,
            "dropped": self.dropped,
            "connects": self.connects,
            "remote_latency_ms": remote_latency,
            "remote_device": (self.remote_telemetry or {}).get("device"),
            # the remote ladder's playing tier (enrolled / focus_hq / gate ...), None without a ladder
            "remote_tier": ((self.remote_telemetry or {}).get("tiers") or {}).get("active"),
            "last_error": self.last_error,
        }


def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


@registry.register("remote")
class RemoteStage:
    """Runs ``mode``'s pipeline on a remote session server; local ``fallback`` when it is down.

    ``latency_ms`` follows the active path: while remote, the playout buffer ``delay_ms`` plus the
    remote pipeline's own delay (``latency_ms.algorithmic`` of the remote session's telemetry,
    :attr:`remote_algorithmic_ms`: 625 ms for the GPU box's 0.5 s / 0.125 s enrolled spec; 0 until
    the remote has reported it); the fallback's latency when playing locally or without a url.
    The remote returns each hop's output under the same ``seq`` as soon as it is computed, so its
    window delay adds to ``delay_ms``, which only has to cover the round trip. Each call sets
    ``ctx.telemetry.tier`` to the remote ladder's active tier (``stats()["remote_tier"]``; ``tier``
    when the remote runs a single pipeline) while remote audio plays and ``fallback_tier``
    otherwise (``fallback_tier=""``: the tier the fallback itself reported, e.g. a ``tiers``
    ladder's, else ``audio-only``), ``ctx.telemetry.latency_ms["remote_rtt"]`` (p50) and
    ``ctx.telemetry.extras["remote"]`` (:meth:`stats`).
    """

    name = "remote"
    sample_rate = PIPELINE_SR

    def __init__(
        self,
        url: str | None = None,
        fallback: str = "passthrough",
        delay_ms: float = 200.0,
        mode: str = "focus_hq",
        tier: str = "focus_hq",
        fallback_tier: str = "audio-only",
        stall_ms: float = 1000.0,
        connect_timeout: float = 2.0,
        retry_s: float = 2.0,
    ):
        if delay_ms < 0 or stall_ms <= 0:
            raise ValueError("delay_ms must be >= 0 and stall_ms > 0")
        self.url = url or os.environ.get(ENV_URL) or None
        self.fallback = registry.build_pipeline(fallback) if fallback else None
        if self.fallback is not None and self.fallback.sample_rate != PIPELINE_SR:
            raise ValueError(f"fallback {fallback!r} runs at {self.fallback.sample_rate} Hz, not 16 kHz")
        self.delay_hops = round(delay_ms * PIPELINE_SR / 1000.0 / HOP)
        self.remote_latency_ms = 1000.0 * self.delay_hops * HOP / PIPELINE_SR
        self.tier, self.fallback_tier = tier, fallback_tier
        self.stall_s = stall_ms / 1000.0
        self.client = (
            RemoteClient(self.url, mode=mode, connect_timeout=connect_timeout, retry_s=retry_s,
                         video=video_enabled())
            if self.url else None
        )
        if self.client is not None:
            # The client's threads reference only the client, so a dropped session's stage is
            # collected and its remote connection closed.
            weakref.finalize(self, self.client.close)
        else:
            log.warning("remote: no url (set %s); playing the local fallback", ENV_URL)
        self._seq = 0  # never reset: in-flight replies from before reset() must not collide
        self._history: deque[np.ndarray] = deque(maxlen=self.delay_hops + 1)
        self._source = "local"
        self._started = False  # until the first hop a url'd stage declares the remote path
        self._fallback_reported = ""  # the tier the fallback set on the last hop
        self._forwarded: dict = {}
        self.misses = self.remote_hops = 0
        self._last_log = time.perf_counter()

    # ------------------------------------------------------------ helpers

    def _local(self, chunk: AudioChunk, ctx: StageContext) -> np.ndarray:
        ctx.telemetry.tier = ""
        out = self.fallback.process(chunk, ctx) if self.fallback is not None else chunk
        self._fallback_reported = ctx.telemetry.tier
        return np.asarray(out.samples, dtype=np.float32)

    def _remote_target(self, ctx: StageContext) -> dict | None:
        """The ``control.target`` for the remote, or ``_PENDING`` while the lock cannot be mapped.

        A locked local target with a box is matched to the remote face it overlaps most (the
        two servers number tracks independently); without a local box the id goes as is."""
        target = ctx.target
        if not target.locked or target.track_id is None:
            return None
        box = next((t.box for t in ctx.tracks if t.track_id == target.track_id), None)
        if box is None:
            return {"lock_track_id": target.track_id}
        faces = (self.client.remote_telemetry or {}).get("faces") or []
        scored = [(_iou(box, f.get("box") or (0, 0, 0, 0)), f.get("track_id")) for f in faces]
        scored = [(iou, tid) for iou, tid in scored if tid is not None and iou >= MIN_TARGET_IOU]
        if not scored:
            return _PENDING
        return {"lock_track_id": int(max(scored)[1])}

    def _forward_control(self, ctx: StageContext) -> None:
        state = {"suppression": round(100.0 * float(ctx.suppression), 2)}
        preset = ctx.telemetry.extras.get("latency_preset") or "balanced"
        state["latency_preset"] = REMOTE_PRESETS.get(preset, preset)
        state["agc"] = False  # T-169: the laptop session levels the output once; a remote AGC doubled it
        # T-110: the remote session plays its own ladder, so it needs the same strict isolation and
        # judge profile (auto-calibration, sticky attention) the local session runs, or it leaks the
        # other talker (-10 dB vs the Mac's -21 dB on demo-mf-easy).
        for key in ("profile", "isolation"):
            if ctx.telemetry.extras.get(key):
                state[key] = ctx.telemetry.extras[key]
        profile = ctx.telemetry.extras.get("voice_profile")  # T-125: the remote enrolls its own copy
        if profile:
            state["enroll"] = {"action": "profile", **profile}
        elif self._forwarded.get("enroll"):
            state["enroll"] = {"action": "clear"}
        target = self._remote_target(ctx)
        if target is not _PENDING:
            state["target"] = target
        changed = {k: v for k, v in state.items() if self._forwarded.get(k, object()) != v}
        if "profile" in changed:  # the remote applies a profile's defaults first: re-send every field after it
            changed = state
        enroll = changed.pop("enroll", None)
        if changed and self.client.send_control(**changed):
            self._forwarded.update(changed)
        # T-125: on its own, so a remote without voice profiles rejects only this message
        if enroll is not None and self.client.send_control(enroll=enroll):
            self._forwarded["enroll"] = enroll

    @property
    def latency_ms(self) -> float:
        """Latency of the active path: the playout buffer plus the remote pipeline's own delay when
        remote audio plays (or is expected), the fallback's own latency (0 for passthrough) when
        the stage plays locally."""
        if self.client is None or (self._started and self._source == "local"):
            return float(self.fallback.latency_ms) if self.fallback is not None else 0.0
        return self.remote_latency_ms + self.remote_algorithmic_ms

    @property
    def remote_algorithmic_ms(self) -> float:
        """The remote pipeline's own delay from its last ``telemetry`` (0 before the first)."""
        telemetry = self.client.remote_telemetry if self.client is not None else None
        value = ((telemetry or {}).get("latency_ms") or {}).get("algorithmic")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) and value > 0 else 0.0

    def remote_tiers(self) -> dict | None:
        """The ``tiers`` block for telemetry while remote audio plays, else ``None``: the remote
        ladder's status (its playing tier, e.g. ``enrolled``) with ``delay_ms`` set to what this
        stage puts into the stream (playout buffer + the remote's own delay), ``remote_delay_ms``
        the remote's own. The RTT is inside the fixed buffer, not added to it."""
        if self.client is None or self._source == "local":
            return None
        telemetry = self.client.remote_telemetry or {}
        tiers = telemetry.get("tiers") if isinstance(telemetry.get("tiers"), dict) else {}
        active = tiers.get("active") or telemetry.get("tier") or self.tier
        return {**tiers, "active": active, "delay_ms": round(self.latency_ms, 3), "remote": True,
                "remote_delay_ms": round(self.remote_algorithmic_ms, 3),
                "buffer_ms": self.remote_latency_ms}

    @property
    def source(self) -> str:
        """What played on the last hop: ``remote``, ``delayed`` (a late hop filled locally) or
        ``local`` (the fallback, undelayed)."""
        return self._source

    def rtt_ms(self) -> float | None:
        """Stream RTT p50 over the last 5 s while connected, else ``None``."""
        if self.client is None or not self.client.connected:
            return None
        return self.client.stats()["rtt_ms"]["p50"]

    def remote_up(self) -> bool:
        c = self.client
        return c is not None and c.connected and time.perf_counter() - c.last_rx < self.stall_s

    def stats(self) -> dict:
        base = self.client.stats() if self.client is not None else {"url": None, "connected": False}
        return {**base, "source": self._source, "misses": self.misses,
                "remote_hops": self.remote_hops, "delay_ms": self.remote_latency_ms,
                "remote_algorithmic_ms": self.remote_algorithmic_ms, "latency_ms": self.latency_ms}

    # ------------------------------------------------------------ Stage

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        seq = self._seq
        self._seq += 1
        local = self._local(chunk, ctx)
        self._history.append(local)
        filled = len(self._history) > self.delay_hops
        delayed = self._history[0] if filled else np.zeros_like(local)
        remote = None
        if self.client is not None:
            self.client.start()
            if self.client.connected:
                self._forward_control(ctx)
                self.client.send_audio(seq, chunk.capture_ts, chunk.samples)
            else:
                self._forwarded.clear()  # re-forward on reconnect
            if seq >= self.delay_hops:
                remote = self.client.pop(seq - self.delay_hops)
            if remote is not None and len(remote) != len(local):
                remote = None
        up = self.remote_up()
        if remote is not None:
            source, out = "remote", remote
            self.remote_hops += 1
        elif up and filled:
            source, out = "delayed", delayed  # a late hop: keep the timeline, fill locally
            self.misses += 1
        else:
            source, out = "local", local
        candidates = {"remote": remote, "delayed": delayed, "local": local}
        if source != self._source:
            before = candidates[self._source]
            if before is None:
                before = delayed
            ramp = np.linspace(0.0, 1.0, len(out), endpoint=False, dtype=np.float32)
            out = before * (1.0 - ramp) + out * ramp
            if (source == "local") != (self._source == "local"):
                log.info("remote: playing %s", source)
            self._source = source
        self._started = True
        self._report(ctx)
        return replace(chunk, samples=np.asarray(out, dtype=np.float32))

    def _report(self, ctx: StageContext) -> None:
        if self._source == "remote" and self.client is not None:
            # a remote ladder names what it plays (T-075); a single remote pipeline is ``tier``
            ladder = ((self.client.remote_telemetry or {}).get("tiers") or {}).get("active")
            ctx.telemetry.tier = ladder or self.tier
        elif self._source != "local":
            ctx.telemetry.tier = self.tier
        else:
            ctx.telemetry.tier = self.fallback_tier or self._fallback_reported or "audio-only"
        ctx.telemetry.latency_ms["remote_stage"] = self.latency_ms
        if self.client is None:
            return
        stats = self.stats()
        ctx.telemetry.extras["remote"] = stats
        if stats["rtt_ms"]["p50"] is not None:
            ctx.telemetry.latency_ms["remote_rtt"] = stats["rtt_ms"]["p50"]
        now = time.perf_counter()
        if now - self._last_log >= LOG_EVERY_S:
            self._last_log = now
            log.info("remote: %s rtt p50=%s p95=%s ms, misses=%d, source=%s", stats["url"],
                     stats["rtt_ms"]["p50"], stats["rtt_ms"]["p95"], self.misses, self._source)

    def reset(self) -> None:
        if self.fallback is not None:
            self.fallback.reset()
        self._history.clear()
        self.misses = self.remote_hops = 0

    def close(self, wait: bool = False) -> None:
        if self.client is not None:
            self.client.close(wait)
        for stage in getattr(self.fallback, "stages", ()):
            close = getattr(stage, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as e:  # noqa: BLE001  (closing must not raise)
                    log.warning("closing fallback %s failed: %s", stage.name, e)


_PENDING = object()  # a target lock that cannot be mapped to a remote track yet


def _iou(a, b) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    inter = max(0.0, w) * max(0.0, h)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------ backend choice (hq_backend)


def auto_rule(health_ok: bool | None, rtt_ms: float | None,
              max_rtt_ms: float = AUTO_MAX_RTT_MS) -> tuple[bool, str]:
    """``hq_backend="auto"``: ``(use remote, why)``. Remote only when ``/health`` is green and
    the round trip is measured and below ``max_rtt_ms``."""
    if not health_ok:
        return False, "remote /health not green"
    if rtt_ms is None:
        return False, "remote RTT not measured yet"
    if rtt_ms >= max_rtt_ms:
        return False, f"remote RTT {rtt_ms:.0f} ms >= {max_rtt_ms:g} ms"
    return True, f"remote healthy, RTT {rtt_ms:.0f} ms < {max_rtt_ms:g} ms"


@dataclass
class AutoBackend:
    """Hysteresis for ``auto``: drop to local at the first bad check; move to remote only after
    it has been good for ``upgrade_after_s`` (so a flapping link cannot pump the crossfade)."""

    upgrade_after_s: float = AUTO_UPGRADE_S
    remote: bool = False
    _good_since: float | None = None

    def update(self, good: bool, now: float) -> bool:
        if not good:
            self._good_since, self.remote = None, False
        else:
            if self._good_since is None:
                self._good_since = now
            if now - self._good_since >= self.upgrade_after_s:
                self.remote = True
        return self.remote


class HealthMonitor:
    """Polls ``GET /health`` of a remote session server every ``every_s`` on a daemon thread.

    :meth:`snapshot` is ``{"ok", "rtt_ms", "error"}``: ``ok`` is ``None`` before the first
    check, False when the last check failed or no check succeeded for ``stale_s``; ``rtt_ms``
    (``None`` unless ``ok``) is the median HTTP round trip of the last :data:`HEALTH_WINDOW`
    successful checks (keep-alive connection, so it is close to the network RTT).
    """

    def __init__(self, ws_url: str, every_s: float = HEALTH_EVERY_S, timeout_s: float = 1.0,
                 stale_s: float = HEALTH_STALE_S):
        self.url = health_url(ws_url)
        self.every_s, self.timeout_s, self.stale_s = every_s, timeout_s, stale_s
        self._lock = threading.Lock()
        self._ok: bool | None = None
        self._last_ok = 0.0
        self._rtts: deque[float] = deque(maxlen=HEALTH_WINDOW)
        self._error: str | None = None
        self._closed = threading.Event()
        self._checked = threading.Event()
        self._thread: threading.Thread | None = None
        self.checks = 0

    def start(self) -> HealthMonitor:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="hearing-remote-health",
                                            daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        import httpx

        with httpx.Client(timeout=self.timeout_s) as http:
            while not self._closed.is_set():
                t0 = time.perf_counter()
                try:
                    r = http.get(self.url)
                    ok = r.status_code == 200 and bool(r.json().get("ok"))
                    note_rev(self.url, r.json())  # T-170: telemetry.remote.git_rev
                    error = None if ok else f"/health {r.status_code}: {r.text[:80]}"
                except Exception as e:  # noqa: BLE001  (refused, DNS, timeout, bad JSON)
                    ok, error = False, f"{type(e).__name__}: {e}"
                rtt = 1000.0 * (time.perf_counter() - t0)
                with self._lock:
                    self._ok, self._error = ok, error
                    if ok:
                        self._last_ok = time.monotonic()
                        self._rtts.append(rtt)
                    self.checks += 1
                self._checked.set()
                self._closed.wait(self.every_s)

    def wait_checked(self, timeout: float) -> bool:
        """Wait for the first check (tests)."""
        return self._checked.wait(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            ok = self._ok
            if ok and time.monotonic() - self._last_ok > self.stale_s:
                ok = False
            rtt = _percentile(self._rtts, 50) if ok else None  # no stale RTT for a dead remote
            return {"ok": ok, "rtt_ms": rtt, "error": self._error}

    def close(self) -> None:
        self._closed.set()


# ------------------------------------------------------------ probe


def probe(url: str, seconds: float = 5.0, connect_timeout: float = 5.0,
          health_timeout: float | None = None) -> dict:
    """Stream a 440 Hz tone in real time to ``url`` and report health and round-trip times."""
    import httpx

    result: dict = {"url": url, "health": None}

    def fetch_health() -> None:  # in parallel with the stream (T-109): a slow /health costs no time
        try:
            result["health"] = httpx.get(health_url(url), timeout=health_timeout or connect_timeout).json()
            note_rev(health_url(url), result["health"])
        except Exception as e:  # noqa: BLE001
            result["health_error"] = f"{type(e).__name__}: {e}"

    health = threading.Thread(target=fetch_health, name="hearing-probe-health", daemon=True)
    health.start()
    client = RemoteClient(url, connect_timeout=connect_timeout, client="remote-probe").start()
    try:
        if not client.wait_connected(connect_timeout + 1.0):
            health.join(health_timeout or connect_timeout)
            return {**result, "ok": False, **client.stats()}
        hop_s = HOP / PIPELINE_SR
        t = np.arange(HOP, dtype=np.float32) / PIPELINE_SR
        n = max(1, round(seconds / hop_s))
        start = time.perf_counter()
        for i in range(n):
            tone = (0.1 * np.sin(2 * np.pi * 440.0 * (t + i * hop_s))).astype(np.float32)
            client.send_audio(i, time.time(), tone)
            client.pop(i - 50)  # keep the result map small
            time.sleep(max(0.0, start + (i + 1) * hop_s - time.perf_counter()))
        deadline = time.perf_counter() + 1.0
        while client.received < client.sent and time.perf_counter() < deadline:
            time.sleep(0.01)
        stats = client.stats()
        health.join(health_timeout or connect_timeout)
        stats["lost"] = stats["sent"] - stats["received"]
        return {**result, "ok": stats["received"] > 0, **stats}
    finally:
        client.close(wait=True)


def judge_probe(url: str, attempts: int = 3, budget_s: float = 6.0, seconds: float = 2.0,
                max_rtt_ms: float = AUTO_MAX_RTT_MS) -> tuple[bool, str]:
    """``demo.sh --judge``'s GPU check (T-109): up to ``attempts`` probes of ``budget_s`` each.

    The GPU passes on the first attempt whose ``/health`` is ok and whose stream round trip p50 is
    below ``max_rtt_ms``: the same rule the session's ``hq_backend=auto`` applies at run time (a
    p95 gate rejected a healthy A100 on its first-packet spikes). Returns ``(ok, text)``: the RTT
    when it passes, else the last attempt's reason.
    """
    reason, remote_git_rev, ok = "not probed", None, False
    for _ in range(max(1, attempts)):
        box: dict = {}

        def run(box: dict = box) -> None:
            try:
                box["r"] = probe(url, seconds, connect_timeout=max(1.0, budget_s - seconds - 1.0),
                                 health_timeout=budget_s - 0.5)
            except Exception as e:  # noqa: BLE001
                box["r"] = {"ok": False, "last_error": f"{type(e).__name__}: {e}"}

        t = threading.Thread(target=run, name="hearing-judge-probe", daemon=True)
        t.start()
        t.join(budget_s)
        r = box.get("r")
        if r is None:
            reason = f"no answer in {budget_s:g} s"
            continue
        health, rtt = r.get("health") or {}, r.get("rtt_ms") or {}
        remote_git_rev = health.get("git_rev") or remote_git_rev
        p50 = rtt.get("p50")
        if not health.get("ok"):
            reason = f"/health not ok ({r.get('health_error') or 'ok=false'})"
        elif not r.get("ok") or p50 is None:
            reason = f"no audio back ({r.get('last_error') or 'stream failed'})"
        elif p50 >= max_rtt_ms:
            reason = f"RTT p50 {p50:.0f} ms >= {max_rtt_ms:g} ms"
        else:
            ok, reason = True, f"RTT p50 {p50:.0f} ms, p95 {rtt.get('p95', p50):.0f} ms"
            break
    # T-170: the JUDGE line says when the GPU serves older code than this checkout
    stale = rev_warning(local_rev(), remote_git_rev)
    return ok, f"{reason} · {stale}" if stale else reason


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m hearing.server.remote",
                                description="Probe a remote session server: health and RTT.")
    p.add_argument("--url", default=os.environ.get(ENV_URL), help=f"ws(s)://host:port/ws/session (env {ENV_URL})")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--judge", action="store_true",
                   help="demo.sh --judge: 3 attempts of 6 s, print one line 'GPU <rtt>' or 'FAIL <reason>'")
    args = p.parse_args(argv)
    if not args.url:
        p.error(f"--url or {ENV_URL} is required")
    if args.judge:
        ok, text = judge_probe(args.url)
        print(f"{'GPU' if ok else 'FAIL'} {text}")
        return 0 if ok else 1
    result = probe(args.url, args.seconds, args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1

