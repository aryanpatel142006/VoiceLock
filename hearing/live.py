"""Local live loop (mic -> pipeline -> headphones) and the mic-to-ear latency probe.

Everything that can be tested without hardware lives in :class:`LiveEngine`: it takes one
device block of float32 samples and returns one block of the same length, running the
pipeline of the current mode in 10 ms hops at 16 kHz (resampling on its own edges when
the device runs at another rate), applying the suppression slider through
:class:`StageContext`, and keeping compute/underrun statistics. :func:`run_live` wraps
it in a ``sounddevice`` duplex callback stream; :func:`run_latency_test` runs the same
stream with the monitor muted, injects clicks into the output and cross-correlates the
input to measure the real round trip (output buffer -> DAC -> air -> ADC -> input buffer),
which with headphones held to the mic is the latency the wearer hears, plus the
pipeline's algorithmic delay.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy.signal import correlate

from hearing import registry
from hearing.audio.io import HOP, PIPELINE_SR
from hearing.audio.stream import Chunker, RingBuffer
from hearing.server.resample import StreamResampler
from hearing.server.session import PIPELINE_MODES
from hearing.types import AudioChunk, StageContext

MODES = tuple(PIPELINE_MODES)  # natural, focus_hq, synthetic, text, quiet
# Natural/Focus HQ default: the first registered candidate; missing weights degrade at run
# time to the fallback. "dfn" is T-010's DeepFilterNet3 stage (registry name agreed with the
# orchestrator); until it merges, GTCRN (T-011) is the real-time enhancer.
NATURAL_CANDIDATES = ("dfn", "gtcrn", "passthrough")
STAGE_MODULES = ("hearing.audio.basic", "hearing.audio.agc", "hearing.server.stages", "hearing.enhance.dfn",
                 "hearing.enhance.gtcrn", "hearing.fusion")
_RAW = ""


def load_stage_modules() -> None:
    for module in STAGE_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            if e.name != module and not e.name.startswith(module + "."):
                raise


def default_pipeline(mode: str) -> str:
    """The pipeline ``hearing live`` uses for ``mode`` when none is given explicitly."""
    load_stage_modules()
    if mode in ("natural", "focus_hq"):
        override = os.environ.get("HEARING_LIVE_NATURAL")
        if override:
            return override
        known = set(registry.names())
        return next((c for c in NATURAL_CANDIDATES if c in known), "passthrough")
    return "silence"


@dataclass
class LiveConfig:
    """Settings shared by ``hearing live`` and ``hearing latency-test``.

    ``pipeline`` ``None`` means "the default for the current mode"; a spec string is used
    for every mode. ``sample_rate`` and ``block_ms`` describe the *device* stream; the
    pipeline always runs at 16 kHz / 10 ms. ``latency`` is sounddevice's hint
    (``"low"``, ``"high"`` or seconds; ``0.0`` asks for the minimum). On laptop-a (M4 Pro,
    built-in mic -> speakers) the measured passthrough round trip was 64 ms at 16 kHz /
    5 ms / 0.0, 76 ms at 10 ms blocks and 99-109 ms with the ``"low"`` hint, so the
    defaults are the first. ``monitor`` False mutes the pipeline output
    (used by the latency probe so a mic -> speaker loop cannot howl).
    """

    pipeline: str | None = None
    mode: str = "natural"
    suppression: float = 50.0  # percent, like the wire protocol
    block_ms: float = 5.0  # measured on laptop-a: 5 ms blocks + latency 0 give the lowest round trip
    sample_rate: int = PIPELINE_SR
    input: str | int | None = None
    output: str | int | None = None
    latency: str | float = 0.0  # PortAudio hint; 0 = as small as the host allows
    monitor: bool = True
    fallback: str = "passthrough"

    def __post_init__(self):
        if self.mode not in MODES and self.mode != "auto":
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")
        if self.block_ms <= 0 or self.sample_rate <= 0:
            raise ValueError("block_ms and sample_rate must be positive")
        if abs(self.sample_rate * self.block_ms / 1000 - round(self.sample_rate * self.block_ms / 1000)) > 1e-6:
            raise ValueError(f"block_ms={self.block_ms} is not a whole number of samples at {self.sample_rate} Hz")


@dataclass
class LatencyResult:
    delays_ms: list[float]
    clicks_sent: int
    clicks_found: int
    sample_rate: int
    block_ms: float
    stream_latency_ms: float | None = None
    algorithmic_ms: float = 0.0
    compute_ms_p95: float = 0.0
    pipeline: str = ""
    device: dict = field(default_factory=dict)

    @property
    def round_trip_ms(self) -> float:
        return float(np.median(self.delays_ms)) if self.delays_ms else float("nan")

    def to_dict(self) -> dict:
        return {
            "round_trip_ms": None if not self.delays_ms else round(self.round_trip_ms, 2),
            "delays_ms": [round(d, 2) for d in self.delays_ms],
            "clicks_sent": self.clicks_sent,
            "clicks_found": self.clicks_found,
            "sample_rate": self.sample_rate,
            "block_ms": self.block_ms,
            "stream_latency_ms": self.stream_latency_ms,
            "algorithmic_ms": self.algorithmic_ms,
            "compute_ms_p95": round(self.compute_ms_p95, 3),
            "pipeline": self.pipeline,
            "device": self.device,
        }

    def format(self) -> str:
        if not self.delays_ms:
            return (
                f"latency-test: no click detected in {self.clicks_sent} tries. Put the headphones "
                "against the mic (or turn the speakers up) and retry."
            )
        spread = f"{min(self.delays_ms):.1f}-{max(self.delays_ms):.1f}"
        reported = (
            f", PortAudio-reported in+out {self.stream_latency_ms:.1f} ms"
            if self.stream_latency_ms is not None else ""
        )
        lines = [
            (f"round trip (mic-to-ear, passthrough): {self.round_trip_ms:.1f} ms  "
             f"(median of {self.clicks_found}/{self.clicks_sent} clicks, range {spread} ms)"),
            f"device stream: {self.sample_rate} Hz, {self.block_ms:g} ms blocks{reported}",
        ]
        if self.pipeline:
            total = self.round_trip_ms + self.algorithmic_ms
            lines.append(
                f"pipeline {self.pipeline!r}: +{self.algorithmic_ms:g} ms algorithmic, compute p95 "
                f"{self.compute_ms_p95:.2f} ms per hop -> expected mic-to-ear {total:.1f} ms"
            )
        if self.device:
            lines.append("devices: " + ", ".join(f"{k}={v}" for k, v in self.device.items()))
        return "\n".join(lines)


# ----------------------------------------------------------------------------- probe


def make_click(sr: int, ms: float = 5.0, freq: float = 2000.0, amplitude: float = 0.8) -> np.ndarray:
    """A Hann-windowed tone burst: band-limited (no speaker clipping), sharp in correlation."""
    n = max(2, round(sr * ms / 1000.0))
    t = np.arange(n) / sr
    window = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)
    return (amplitude * window * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def find_delay(
    recorded: np.ndarray, reference: np.ndarray, sr: int, min_ms: float = 0.0,
    max_ms: float = 1000.0, threshold: float = 8.0,
) -> float | None:
    """Lag (ms) of ``reference`` inside ``recorded`` by cross-correlation, or ``None``.

    A peak must stand ``threshold`` robust standard deviations above the rest of the
    correlation in the search window, so noise alone never yields a delay. Among the
    peaks that qualify the *earliest* wins: the direct path always arrives before echoes
    and room noise, even when those are louder.
    """
    recorded = np.asarray(recorded, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if len(recorded) < len(reference) or not len(reference):
        return None
    full = correlate(recorded, reference, mode="full")
    corr = full[len(reference) - 1 :]  # corr[k] = sum_i rec[k+i] ref[i] for lags k >= 0
    lo = max(0, round(min_ms * sr / 1000.0))
    hi = min(len(corr) - 1, round(max_ms * sr / 1000.0))
    if hi < lo:
        return None
    window = np.abs(corr[lo : hi + 1])
    peak = int(np.argmax(window))
    value = window[peak]
    if value <= 0:
        return None
    rest = np.delete(window, peak)
    scale = 1.4826 * float(np.median(np.abs(rest - np.median(rest)))) if len(rest) > 1 else 0.0
    if scale <= 0:  # digital silence around the click: fall back to the mean magnitude
        scale = float(np.mean(rest)) if len(rest) else 0.0
    if value < threshold * scale:
        return None
    first = int(np.argmax(window >= threshold * scale))  # earliest qualifying lag
    span = window[first : first + len(reference)]  # the true peak sits within one click length
    return 1000.0 * (lo + first + int(np.argmax(span))) / sr


# ----------------------------------------------------------------------------- engine


class LiveEngine:
    """Block in, block out; everything stateful about the live loop except the device."""

    def __init__(self, config: LiveConfig):
        self.config = config
        self.sample_rate = int(config.sample_rate)
        self.block_samples = round(self.sample_rate * config.block_ms / 1000.0)
        self.mode = "natural" if config.mode == "auto" else config.mode
        self.ctx = StageContext.default()
        self.ctx.mode = self.mode
        self.ctx.telemetry.tier = "audio-only"
        self._lock = threading.Lock()
        self._suppression = 0.0
        self.set_suppression(config.suppression)
        self._chunker = Chunker(HOP)
        if self.sample_rate == PIPELINE_SR:
            self._to_16k = self._from_16k = None
            self.resample_ms = 0.0
        else:
            self._to_16k = StreamResampler(self.sample_rate, PIPELINE_SR)
            self._from_16k = StreamResampler(PIPELINE_SR, self.sample_rate)
            self.resample_ms = self._to_16k.delay_ms + self._from_16k.delay_ms
        self._out = RingBuffer(max(8 * self.block_samples, 4 * self.sample_rate), "drop_oldest")
        self._specs: dict[str, str] = {}  # mode -> spec in use (after degrades)
        self._built: dict[str, registry.Pipeline] = {}
        self._active_spec: str | None = None
        self.errors: list[str] = []
        self._hop_seq = 0
        self._blocks = 0
        self._hops = 0
        self._fills = 0
        self._underruns = 0
        self._overflows = 0
        self._underrun_at: list[float] = []  # stream seconds of the first 50 flags
        self._overflow_at: list[float] = []
        self._compute = deque(maxlen=60_000)  # per-hop ms, ~10 min
        self._out_pos = 0
        # probe state
        self._probe_click: np.ndarray | None = None
        self._probe_schedule: list[int] = []
        self._probe_sent: list[int] = []
        self._probe_inject = np.zeros(0, dtype=np.float32)
        self._probe_window = 0
        self._recording: list[np.ndarray] = []
        self._recording_on = False
        load_stage_modules()
        self.resolve_pipeline()

    # ----------------------------------------------------------------- controls

    @property
    def suppression(self) -> float:
        return self._suppression

    def set_suppression(self, percent: float) -> None:
        value = float(min(100.0, max(0.0, float(percent))))
        with self._lock:
            self._suppression = value
            self.ctx.suppression = value / 100.0

    def set_mode(self, mode: str) -> None:
        if mode == "auto":
            mode = "natural"
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        with self._lock:
            self.mode = mode
            self.ctx.mode = mode

    def spec_for(self, mode: str) -> str:
        if mode not in self._specs:
            self._specs[mode] = self.config.pipeline if self.config.pipeline is not None else default_pipeline(mode)
        return self._specs[mode]

    def resolve_pipeline(self) -> registry.Pipeline | None:
        """Build (once) the pipeline for the current mode, degrading if it cannot be built."""
        while True:
            spec = self.spec_for(self.mode)
            if spec == _RAW:
                return None
            pipeline = self._built.get(spec)
            if pipeline is None:
                try:
                    pipeline = self._built[spec] = registry.build_pipeline(spec)
                except Exception as e:  # noqa: BLE001  (unknown stage, bad args, no runtime)
                    self._degrade(spec, f"cannot build {spec!r}: {type(e).__name__}: {e}")
                    continue
            if spec != self._active_spec:
                pipeline.reset()
                self._active_spec = spec
            return pipeline

    def _degrade(self, spec: str, message: str) -> None:
        target = self.config.fallback if spec != self.config.fallback else _RAW
        for mode, s in self._specs.items():
            if s == spec:
                self._specs[mode] = target
        self.errors.append(f"{message} -> degraded to {target or 'raw'!r}")

    @property
    def pipeline_name(self) -> str:
        return self.spec_for(self.mode) or "raw"

    @property
    def algorithmic_ms(self) -> float:
        pipeline = self._built.get(self.spec_for(self.mode))
        return float(pipeline.latency_ms) if pipeline is not None else 0.0

    # ----------------------------------------------------------------- audio

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """One device block in (any length), one block of the same length out, clipped."""
        x = np.asarray(x, dtype=np.float32).ravel()
        n = len(x)
        if self._recording_on:
            self._recording.append(x.copy())
        y_in = self._to_16k.process(x) if self._to_16k is not None else x
        hops = [self._process_hop(h) for h in self._chunker.push(y_in)]
        if hops:
            y = np.concatenate(hops)
            if self._from_16k is not None:
                y = self._from_16k.process(y)
            self._out.push(y)
        available = min(n, self._out.available)
        if available < n:
            self._fills += 1
        out = np.zeros(n, dtype=np.float32)
        if available:
            out[n - available :] = self._out.pull(available)
        if not self.config.monitor:
            out[:] = 0.0
        if self._probe_click is not None:
            self._probe_add(out)
        self._out_pos += n
        self._blocks += 1
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def _process_hop(self, samples: np.ndarray) -> np.ndarray:
        chunk = AudioChunk(samples, PIPELINE_SR, self._hop_seq, time.time())
        self._hop_seq += 1
        self._hops += 1
        self.ctx.telemetry.seq = chunk.seq
        while True:
            pipeline = self.resolve_pipeline()
            if pipeline is None:
                return samples
            t0 = time.perf_counter()
            try:
                out = pipeline.process(chunk, self.ctx)
            except Exception as e:  # noqa: BLE001  (a stage crashed: degrade, keep going)
                self._degrade(self._active_spec, f"{pipeline.name!r} failed: {type(e).__name__}: {e}")
                continue
            self._compute.append(1000.0 * (time.perf_counter() - t0))
            return np.asarray(out.samples, dtype=np.float32)

    def warm_up(self) -> float:
        """Run one silent hop through the pipeline (loading weights, JIT, caches) and reset.

        Call before the device stream starts: a first-hop weight load inside the audio
        callback stalls it (GTCRN: 724 ms measured) and shows up as input overflows.
        Returns the wall time in ms; nothing is counted in :meth:`stats`.
        """
        t0 = time.perf_counter()
        pipeline = self.resolve_pipeline()
        if pipeline is not None:
            chunk = AudioChunk(np.zeros(HOP, dtype=np.float32), PIPELINE_SR, -1, time.time())
            try:
                pipeline.process(chunk, self.ctx)
            except Exception as e:  # noqa: BLE001  (missing weights etc.: degrade now, not live)
                self._degrade(self._active_spec, f"{pipeline.name!r} failed: {type(e).__name__}: {e}")
                self.resolve_pipeline()
            else:
                pipeline.reset()
        return 1000.0 * (time.perf_counter() - t0)

    def note_status(self, input_overflow: bool = False, output_underflow: bool = False) -> None:
        """Record PortAudio status flags, stamped with the stream time of the current block."""
        at = round(self._out_pos / self.sample_rate, 3)
        if output_underflow:
            self._underruns += 1
            if len(self._underrun_at) < 50:
                self._underrun_at.append(at)
        if input_overflow:
            self._overflows += 1
            if len(self._overflow_at) < 50:
                self._overflow_at.append(at)

    # ----------------------------------------------------------------- probe

    def start_probe(self, clicks: int = 5, interval_s: float = 1.0, lead_s: float = 0.2) -> None:
        """Schedule ``clicks`` clicks into the output and record the input for the probe."""
        if clicks < 1 or interval_s <= 0:
            raise ValueError("clicks must be >= 1 and interval_s positive")
        self._probe_click = make_click(self.sample_rate)
        self._probe_window = round(interval_s * self.sample_rate)
        first = self._out_pos + round(lead_s * self.sample_rate)
        self._probe_schedule = [first + i * self._probe_window for i in range(clicks)]
        self._probe_sent = []
        self._probe_inject = np.zeros(0, dtype=np.float32)
        self._recording = []
        self._recording_start = self._out_pos
        self._recording_on = True

    def _probe_add(self, out: np.ndarray) -> None:
        n = len(out)
        while self._probe_schedule and self._probe_schedule[0] < self._out_pos + n:
            start = self._probe_schedule.pop(0)
            offset = start - self._out_pos
            need = offset + len(self._probe_click)
            if len(self._probe_inject) < need:
                self._probe_inject = np.concatenate(
                    [self._probe_inject, np.zeros(need - len(self._probe_inject), dtype=np.float32)]
                )
            self._probe_inject[offset:need] += self._probe_click
            self._probe_sent.append(start)
        take = min(n, len(self._probe_inject))
        if take:
            out[:take] += self._probe_inject[:take]
            self._probe_inject = self._probe_inject[take:]

    @property
    def probe_active(self) -> bool:
        if self._probe_click is None:
            return False
        if self._probe_schedule:
            return True
        last = self._probe_sent[-1] if self._probe_sent else self._out_pos
        return self._out_pos < last + self._probe_window

    def probe_result(self) -> LatencyResult:
        clicks = len(self._probe_sent) + len(self._probe_schedule)
        rec = np.concatenate(self._recording) if self._recording else np.zeros(0, dtype=np.float32)
        delays = []
        for start in self._probe_sent:
            i = start - self._recording_start
            segment = rec[i : i + self._probe_window]
            delay = find_delay(segment, self._probe_click, self.sample_rate,
                               max_ms=1000.0 * self._probe_window / self.sample_rate)
            if delay is not None:
                delays.append(delay)
        stats = self.stats()
        return LatencyResult(
            delays_ms=delays, clicks_sent=clicks, clicks_found=len(delays),
            sample_rate=self.sample_rate, block_ms=self.config.block_ms,
            algorithmic_ms=self.algorithmic_ms, compute_ms_p95=stats["compute_ms_p95"],
            pipeline=self.pipeline_name,
        )

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict:
        compute = np.fromiter(self._compute, dtype=np.float64) if self._compute else np.zeros(0)
        p50, p95, mx = (
            (float(np.percentile(compute, 50)), float(np.percentile(compute, 95)), float(compute.max()))
            if len(compute) else (0.0, 0.0, 0.0)
        )
        return {
            "mode": self.mode,
            "pipeline": self.pipeline_name,
            "suppression": self.suppression,
            "sample_rate": self.sample_rate,
            "block_ms": float(self.config.block_ms),
            "blocks": self._blocks,
            "hops": self._hops,
            "fills": self._fills,
            "underruns": self._underruns,
            "overflows": self._overflows,
            "underrun_at_s": list(self._underrun_at),
            "overflow_at_s": list(self._overflow_at),
            "compute_ms_p50": p50,
            "compute_ms_p95": p95,
            "compute_ms_max": mx,
            "algorithmic_ms": self.algorithmic_ms,
            "resample_ms": float(self.resample_ms),
            "errors": list(self.errors),
        }


# ----------------------------------------------------------------------------- controls

_KEY_HELP = "keys: +/- or arrows = slider ±5 · 0-9 = 0-90 % · m = next mode · q = quit"


def handle_key(engine: LiveEngine, key: str) -> bool:
    """Apply one key press; returns True when the user asked to quit."""
    if key in ("q", "Q", "\x03", "\x04"):
        return True
    if key in ("+", "=", "up"):
        engine.set_suppression(engine.suppression + 5)
    elif key in ("-", "_", "down"):
        engine.set_suppression(engine.suppression - 5)
    elif key.isdigit() and len(key) == 1:
        engine.set_suppression(int(key) * 10)
    elif key in ("m", "M"):
        engine.set_mode(MODES[(MODES.index(engine.mode) + 1) % len(MODES)])
    return False


def _keyboard_thread(engine: LiveEngine, quit_event: threading.Event, on_change: Callable[[], None]) -> threading.Thread:
    import select
    import termios
    import tty

    def run():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not quit_event.is_set():
                if not select.select([sys.stdin], [], [], 0.2)[0]:
                    continue
                key = os.read(fd, 1).decode(errors="ignore")
                if key == "\x1b":  # arrow keys: ESC [ A/B
                    seq = os.read(fd, 2).decode(errors="ignore") if select.select([sys.stdin], [], [], 0.05)[0] else ""
                    key = {"[A": "up", "[B": "down", "[C": "up", "[D": "down"}.get(seq, "")
                if handle_key(engine, key):
                    quit_event.set()
                else:
                    on_change()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    thread = threading.Thread(target=run, name="hearing-keys", daemon=True)
    thread.start()
    return thread


_CONTROL_PAGE = """<!doctype html><meta charset="utf-8"><title>hearing live</title>
<body style="font:16px system-ui;max-width:32em;margin:2em auto">
<h2>hearing live</h2>
<p>suppression <output id="v"></output> %<br>
<input id="s" type="range" min="0" max="100" step="1" style="width:100%"></p>
<p>mode <select id="m">{options}</select></p>
<pre id="stats"></pre>
<script>
const s=document.getElementById('s'),v=document.getElementById('v'),m=document.getElementById('m');
async function send(body){{const r=await fetch('/control',{{method:'POST',body:JSON.stringify(body)}});show(await r.json());}}
function show(st){{v.value=Math.round(st.suppression);s.value=st.suppression;m.value=st.mode;
 document.getElementById('stats').textContent=JSON.stringify(st.stats,null,1);}}
s.oninput=()=>{{v.value=s.value;}};s.onchange=()=>send({{suppression:+s.value}});
m.onchange=()=>send({{mode:m.value}});
setInterval(async()=>show(await (await fetch('/state')).json()),1000);
fetch('/state').then(r=>r.json()).then(show);
</script>"""


class ControlServer:
    """Tiny HTTP control: ``GET /`` slider page, ``GET /state``, ``POST /control`` JSON
    (``{"suppression": 0-100, "mode": "natural"}``) or ``GET /control?suppression=70``.
    Binds ``port`` 0 by default (ephemeral); read :attr:`port` after :meth:`start`.
    """

    def __init__(self, engine: LiveEngine, host: str = "127.0.0.1", port: int = 0,
                 on_change: Callable[[], None] | None = None):
        self.engine = engine
        self.host = host
        self._port = port
        self.on_change = on_change
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self._port

    def state(self) -> dict:
        return {
            "suppression": self.engine.suppression,
            "mode": self.engine.mode,
            "pipeline": self.engine.pipeline_name,
            "stats": self.engine.stats(),
        }

    def apply(self, body: dict) -> None:
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")  # noqa: TRY004  (a 400, not a bug)
        if "suppression" in body:
            try:
                self.engine.set_suppression(float(body["suppression"]))
            except (TypeError, ValueError):
                raise ValueError(f"suppression must be a number 0-100, got {body['suppression']!r}") from None
        if "mode" in body:
            self.engine.set_mode(str(body["mode"]))
        if self.on_change:
            self.on_change()

    def start(self) -> None:
        control = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):  # quiet
                pass

            def _json(self, code: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _control(self, body: dict) -> None:
                try:
                    control.apply(body)
                except ValueError as e:
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, control.state())

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/":
                    options = "".join(f'<option value="{m}">{m}</option>' for m in MODES)
                    data = _CONTROL_PAGE.format(options=options).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif url.path == "/state":
                    self._json(200, control.state())
                elif url.path == "/control":
                    query = {k: v[-1] for k, v in parse_qs(url.query).items()}
                    self._control(query)
                else:
                    self._json(404, {"error": f"no route {url.path}"})

            def do_POST(self):
                url = urlparse(self.path)
                if url.path != "/control":
                    self._json(404, {"error": f"no route {url.path}"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError as e:
                    self._json(400, {"error": f"invalid JSON: {e}"})
                    return
                self._control(body)

        self._server = ThreadingHTTPServer((self.host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="hearing-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


# ----------------------------------------------------------------------------- streams


def describe_devices(config: LiveConfig) -> dict:
    """Resolved input/output device indices and names (needs sounddevice)."""
    from hearing.audio.devices import default_device, list_devices, resolve_device

    devices = list_devices()
    by_index = {d.index: d for d in devices}
    result = {}
    for kind, spec in (("input", config.input), ("output", config.output)):
        index = resolve_device(spec, kind, devices)
        if index is None:
            index = default_device(kind)
        result[kind] = f"{index}: {by_index[index].name}" if index in by_index else str(index)
        result[kind + "_index"] = index
    return result


def parse_latency(value: str | float) -> str | float:
    """``"min"``/``0`` -> 0.0 (smallest the host allows), ``"low"``/``"high"`` as is, else seconds."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("low", "high"):
            return text
        if text in ("min", "minimum", ""):
            return 0.0
        return float(text)
    return float(value)


def open_stream(config: LiveConfig, callback):
    """A started-later ``sounddevice.Stream`` (duplex, float32, mono in, mono-copied out)."""
    from hearing.audio.devices import sounddevice

    sd = sounddevice()

    devices = describe_devices(config)
    out_info = sd.query_devices(devices["output_index"], "output") if devices["output_index"] is not None else sd.query_devices(kind="output")
    channels_out = 1 if int(out_info["max_output_channels"]) < 2 else 2
    blocksize = round(config.sample_rate * config.block_ms / 1000.0)
    latency = parse_latency(config.latency)
    return sd.Stream(
        samplerate=config.sample_rate,
        blocksize=blocksize,
        device=(devices["input_index"], devices["output_index"]),
        channels=(1, channels_out),
        dtype="float32",
        latency=latency,
        callback=callback,
    )


def _make_callback(engine: LiveEngine):
    def callback(indata, outdata, frames, time_info, status):
        if status:
            engine.note_status(
                input_overflow=bool(getattr(status, "input_overflow", False)),
                output_underflow=bool(getattr(status, "output_underflow", False)),
            )
        out = engine.process_block(indata[:, 0])
        outdata[:] = out[:, None]

    return callback


def _stream_latency_ms(stream) -> float | None:
    latency = getattr(stream, "latency", None)
    try:
        return 1000.0 * float(sum(latency))
    except TypeError:
        return None


def _log(text: str) -> None:
    print(text, flush=True)  # the control URL must appear even when stdout is a file/pipe


def run_live(
    config: LiveConfig,
    duration_s: float | None = None,
    stream_factory: Callable = open_stream,
    control_port: int | None = 0,
    keyboard: bool | None = None,
    on_control_started: Callable[[ControlServer], None] | None = None,
    log: Callable[[str], None] = _log,
    status_every_s: float = 2.0,
) -> dict:
    """Run the live loop until ``duration_s`` elapses or the user quits; return the summary.

    ``control_port`` None disables the HTTP control; 0 picks an ephemeral port. ``keyboard``
    defaults to "stdin is a TTY".
    """
    engine = LiveEngine(config)
    quit_event = threading.Event()
    last_state = {"text": ""}

    def show_state():
        text = f"mode={engine.mode} pipeline={engine.pipeline_name} suppression={engine.suppression:.0f}%"
        if text != last_state["text"]:
            last_state["text"] = text
            log(text)

    stream = stream_factory(config, _make_callback(engine))
    control = None
    if control_port is not None:
        control = ControlServer(engine, port=control_port, on_change=show_state)
        control.start()
        log(f"control: http://127.0.0.1:{control.port}/  (GET /state, POST /control)")
        if on_control_started:
            on_control_started(control)
    if keyboard is None:
        keyboard = sys.stdin.isatty()
    keys = None
    if keyboard:
        keys = _keyboard_thread(engine, quit_event, show_state)
        log(_KEY_HELP)
    show_state()
    warm_up_ms = engine.warm_up()
    if warm_up_ms > 50:
        log(f"warm-up: {warm_up_ms:.0f} ms (weights loaded before the stream started)")
    started = time.perf_counter()
    stream.start()
    try:
        log(f"live: {config.sample_rate} Hz, {config.block_ms:g} ms blocks, "
            f"PortAudio in+out {_stream_latency_ms(stream) or 0:.1f} ms"
            + (f", stopping after {duration_s:g} s" if duration_s else ", q to quit"))
        next_status = started + status_every_s
        while not quit_event.is_set():
            now = time.perf_counter()
            if duration_s is not None and now - started >= duration_s:
                break
            if now >= next_status:
                next_status = now + status_every_s
                s = engine.stats()
                log(f"  {now - started:6.1f} s  blocks={s['blocks']} underruns={s['underruns']} "
                    f"overflows={s['overflows']} fills={s['fills']} compute p50/p95 "
                    f"{s['compute_ms_p50']:.2f}/{s['compute_ms_p95']:.2f} ms")
                for err in engine.errors[len(last_state.get("errors", [])):]:
                    log(f"  ! {err}")
                last_state["errors"] = list(engine.errors)
            quit_event.wait(min(0.1, status_every_s))
    except KeyboardInterrupt:
        pass
    finally:
        quit_event.set()
        try:
            stream.stop()
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
            if control is not None:
                control.stop()
            if keys is not None:
                keys.join(timeout=1)
    summary = engine.stats()
    summary["duration_s"] = time.perf_counter() - started
    summary["warm_up_ms"] = warm_up_ms
    summary["stream_latency_ms"] = _stream_latency_ms(stream)
    # PortAudio's reported figure is an upper bound (170 ms reported vs 64 ms measured on
    # laptop-a); ``hearing latency-test`` measures the real round trip.
    summary["estimated_total_ms"] = (
        (summary["stream_latency_ms"] or 0.0) + summary["algorithmic_ms"] + summary["resample_ms"]
        + summary["compute_ms_p95"]
    )
    return summary


def run_latency_test(
    config: LiveConfig,
    clicks: int = 5,
    interval_s: float = 1.0,
    stream_factory: Callable = open_stream,
    log: Callable[[str], None] = _log,
) -> LatencyResult:
    """Play clicks through the live duplex stream (monitor muted) and measure the round trip."""
    config = replace(config, monitor=False)
    engine = LiveEngine(config)
    engine.warm_up()
    engine.start_probe(clicks=clicks, interval_s=interval_s)
    stream = stream_factory(config, _make_callback(engine))
    stream.start()
    deadline = time.perf_counter() + 0.5 + clicks * interval_s + 2.0
    try:
        while engine.probe_active and time.perf_counter() < deadline:
            time.sleep(0.05)
    finally:
        stream.stop()
        close = getattr(stream, "close", None)
        if close:
            close()
    result = engine.probe_result()
    result.stream_latency_ms = _stream_latency_ms(stream)
    if stream_factory is open_stream:
        try:
            devices = describe_devices(config)
        except Exception as e:  # noqa: BLE001  (no sounddevice: the result is still valid)
            log(f"devices: {type(e).__name__}: {e}")
        else:
            result.device = {k: v for k, v in devices.items() if not k.endswith("_index")}
    return result
