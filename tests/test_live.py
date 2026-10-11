"""Local live loop, latency test and device resolution (no audio hardware needed)."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from typing import ClassVar

import numpy as np
import pytest

from hearing import registry
from hearing.audio import devices as devices_mod
from hearing.audio.devices import DeviceInfo, format_devices, resolve_device
from hearing.cli import build_parser
from hearing.live import (
    ControlServer,
    LatencyResult,
    LiveConfig,
    LiveEngine,
    find_delay,
    handle_key,
    make_click,
    run_latency_test,
    run_live,
)
from hearing.testing import make_test_signal
from hearing.types import AudioChunk, StageContext

SR = 16000
HOP = 160

DEVICES = [
    DeviceInfo(0, "AirPods Pro", 1, 0, 24000.0, "Core Audio"),
    DeviceInfo(1, "AirPods Pro", 0, 2, 48000.0, "Core Audio"),
    DeviceInfo(2, "MacBook Pro Microphone", 1, 0, 48000.0, "Core Audio"),
    DeviceInfo(3, "MacBook Pro Speakers", 0, 2, 44100.0, "Core Audio"),
    DeviceInfo(4, "ZoomAudioDevice", 2, 2, 48000.0, "Core Audio"),
]


@registry.register("test_live_scale")
class _Scale:
    """Multiplies by (1 - suppression) so tests can see the slider reach the pipeline."""

    name = "test_live_scale"
    sample_rate = SR
    latency_ms = 0.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        return AudioChunk(
            np.asarray(chunk.samples, dtype=np.float32) * np.float32(1.0 - ctx.suppression),
            chunk.sample_rate,
            chunk.seq,
            chunk.capture_ts,
        )

    def reset(self) -> None:
        pass


@registry.register("test_live_boom")
class _Boom:
    """Raises on every chunk so the loop must degrade instead of dying."""

    name = "test_live_boom"
    sample_rate = SR
    latency_ms = 0.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        raise RuntimeError("boom")

    def reset(self) -> None:
        pass


# --------------------------------------------------------------------------- devices


@pytest.fixture(autouse=True)
def _no_portaudio(monkeypatch):
    """Importing sounddevice initialises PortAudio/CoreAudio, which aborts headless runs on macOS."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # any import raises ImportError
    monkeypatch.delenv("HEARING_HW_TESTS", raising=False)


def test_device_access_is_refused_under_pytest_without_hw_flag(monkeypatch):
    monkeypatch.delitem(sys.modules, "sounddevice")  # even if it were importable
    for call in (devices_mod.list_devices, lambda: devices_mod.default_device("input")):
        with pytest.raises(RuntimeError, match="HEARING_HW_TESTS"):
            call()
    assert "sounddevice" not in sys.modules


def test_resolve_device_none_means_system_default():
    assert resolve_device(None, "input", DEVICES) is None


def test_resolve_device_by_index_and_numeric_string():
    assert resolve_device(2, "input", DEVICES) == 2
    assert resolve_device("3", "output", DEVICES) == 3


def test_resolve_device_by_case_insensitive_substring():
    assert resolve_device("macbook pro micro", "input", DEVICES) == 2
    assert resolve_device("speakers", "output", DEVICES) == 3


def test_resolve_device_prefers_devices_of_the_requested_kind():
    # Both AirPods entries share a name; the kind picks the right one.
    assert resolve_device("airpods", "input", DEVICES) == 0
    assert resolve_device("airpods", "output", DEVICES) == 1


def test_resolve_device_rejects_wrong_kind_unknown_and_ambiguous():
    with pytest.raises(ValueError, match="no input"):
        resolve_device(3, "input", DEVICES)  # speakers have no inputs
    with pytest.raises(ValueError, match="No input device"):
        resolve_device("usb", "input", DEVICES)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_device("o", "output", DEVICES)  # AirPods, Speakers, Zoom all match
    with pytest.raises(ValueError, match="out of range"):
        resolve_device(99, "output", DEVICES)


def test_format_devices_lists_every_device_with_direction_marks():
    text = format_devices(DEVICES)
    lines = text.strip().splitlines()
    assert len(lines) == len(DEVICES)
    assert "MacBook Pro Microphone" in text and "48000" in text
    assert any(line.startswith("4") and "in=2" in line and "out=2" in line for line in lines)


# --------------------------------------------------------------------------- engine


def test_engine_passthrough_block_equals_input_when_block_is_a_hop_multiple():
    engine = LiveEngine(LiveConfig(pipeline="passthrough", block_ms=20.0))
    x = make_test_signal(0.5)
    out = np.concatenate([engine.process_block(b) for b in x.reshape(-1, 2 * HOP)])
    np.testing.assert_allclose(out, x, atol=1e-6)
    assert out.dtype == np.float32
    stats = engine.stats()
    assert stats["blocks"] == 25 and stats["hops"] == 50
    assert stats["underruns"] == 0 and stats["fills"] == 0


def test_engine_keeps_block_length_when_block_is_not_a_hop_multiple():
    engine = LiveEngine(LiveConfig(pipeline="passthrough", block_ms=7.5))  # 120 samples
    x = make_test_signal(0.3)
    blocks = [x[i : i + 120] for i in range(0, len(x) - 119, 120)]
    outs = [engine.process_block(b) for b in blocks]
    assert all(len(o) == 120 for o in outs)
    out = np.concatenate(outs)
    # The first hop needs two blocks, so the engine pads one block of silence once and
    # from then on the output is the input delayed by exactly that block (7.5 ms).
    np.testing.assert_array_equal(out[:120], 0.0)
    np.testing.assert_allclose(out[120:], x[: len(out) - 120], atol=1e-6)
    assert engine.stats()["fills"] == 1


def test_engine_suppression_and_mode_reach_the_stage_context():
    engine = LiveEngine(LiveConfig(pipeline="test_live_scale", suppression=0))
    ones = np.ones(HOP, dtype=np.float32) * 0.5
    np.testing.assert_allclose(engine.process_block(ones), ones)
    engine.set_suppression(100)
    assert engine.suppression == 100
    np.testing.assert_allclose(engine.process_block(ones), 0.0)
    engine.set_suppression(250)  # clamped
    assert engine.suppression == 100
    engine.set_suppression(-3)
    assert engine.suppression == 0
    engine.set_mode("quiet")
    assert engine.mode == "quiet" and engine.ctx.mode == "quiet"
    with pytest.raises(ValueError):
        engine.set_mode("loud")


def test_engine_clips_output_to_unit_range():
    engine = LiveEngine(LiveConfig(pipeline="gain(db=20)"))
    out = engine.process_block(np.full(HOP, 0.9, dtype=np.float32))
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_engine_resamples_when_device_rate_differs():
    engine = LiveEngine(LiveConfig(pipeline="passthrough", sample_rate=48000, block_ms=10.0))
    assert engine.block_samples == 480
    t = np.arange(48000) / 48000.0
    x = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    out = np.concatenate([engine.process_block(b) for b in x.reshape(-1, 480)])
    assert len(out) == len(x)
    steady = out[24000:]
    assert 0.45 < np.abs(steady).max() < 0.55  # the tone survives the 48k -> 16k -> 48k trip
    assert engine.stats()["resample_ms"] > 0


def test_engine_stats_report_compute_percentiles_and_latency_terms():
    engine = LiveEngine(LiveConfig(pipeline="delay(ms=30)", block_ms=10.0))
    for _ in range(20):
        engine.process_block(np.zeros(HOP, dtype=np.float32))
    stats = engine.stats()
    assert stats["algorithmic_ms"] == 30.0
    assert stats["block_ms"] == 10.0
    assert 0 <= stats["compute_ms_p50"] <= stats["compute_ms_p95"] <= stats["compute_ms_max"]
    assert stats["compute_ms_max"] < 10.0
    engine.note_status(input_overflow=True, output_underflow=True)
    assert engine.stats()["underruns"] == 1 and engine.stats()["overflows"] == 1
    # Flags are stamped with the stream time of the block they arrived in (20 blocks x 10 ms).
    assert engine.stats()["overflow_at_s"] == [0.2] and engine.stats()["underrun_at_s"] == [0.2]


def test_engine_degrades_to_passthrough_when_a_stage_raises():
    engine = LiveEngine(LiveConfig(pipeline="test_live_boom"))
    x = np.full(HOP, 0.25, dtype=np.float32)
    np.testing.assert_allclose(engine.process_block(x), x)
    assert engine.stats()["pipeline"] == "passthrough"
    assert engine.stats()["errors"]


@registry.register("test_live_counter")
class _Counter:
    """Counts process/reset calls so the warm-up can be observed."""

    calls: ClassVar[list[str]] = []
    name = "test_live_counter"
    sample_rate = SR
    latency_ms = 0.0

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        _Counter.calls.append("process")
        return chunk

    def reset(self) -> None:
        _Counter.calls.append("reset")


def test_engine_warm_up_runs_one_silent_hop_then_resets_without_counting_it():
    _Counter.calls.clear()
    engine = LiveEngine(LiveConfig(pipeline="test_live_counter"))
    assert _Counter.calls == ["reset"]  # build-time reset only
    ms = engine.warm_up()
    assert ms >= 0.0
    assert _Counter.calls == ["reset", "process", "reset"]
    stats = engine.stats()
    assert stats["hops"] == 0 and stats["blocks"] == 0 and stats["compute_ms_max"] == 0.0


# --------------------------------------------------------------------------- probe


def test_make_click_is_short_and_in_range():
    click = make_click(SR)
    assert click.dtype == np.float32
    assert 0.002 * SR <= len(click) <= 0.02 * SR
    assert np.abs(click).max() <= 1.0
    assert np.abs(click).max() > 0.5


def test_find_delay_recovers_a_known_lag_in_noise():
    rng = np.random.default_rng(1)
    click = make_click(SR)
    lag = int(0.0834 * SR)  # 83.4 ms
    rec = rng.normal(0, 0.01, SR).astype(np.float32)
    rec[lag : lag + len(click)] += 0.3 * click
    delay = find_delay(rec, click, SR, max_ms=500)
    assert delay is not None
    assert abs(delay - 1000.0 * lag / SR) < 0.2


def test_find_delay_takes_the_earliest_peak_not_a_louder_later_transient():
    rng = np.random.default_rng(3)
    click = make_click(SR)
    rec = rng.normal(0, 0.01, SR).astype(np.float32)
    rec[640 : 640 + len(click)] += 0.2 * click  # the direct path at 40 ms
    rec[4800 : 4800 + len(click)] += 0.9 * click  # a louder echo / room noise at 300 ms
    assert find_delay(rec, click, SR, max_ms=500) == pytest.approx(40.0, abs=0.2)


def test_find_delay_returns_none_without_a_click():
    rng = np.random.default_rng(2)
    rec = rng.normal(0, 0.01, SR).astype(np.float32)
    assert find_delay(rec, make_click(SR), SR, max_ms=500) is None


def test_engine_probe_measures_its_own_round_trip():
    """Simulate the duplex path: output -> 37 ms of delay -> input."""
    engine = LiveEngine(LiveConfig(pipeline="passthrough", block_ms=10.0, monitor=False))
    engine.start_probe(clicks=3, interval_s=0.3)
    loop_delay = int(0.037 * SR)
    fifo = np.zeros(loop_delay, dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(150):  # 1.5 s
        x = fifo[:HOP] + rng.normal(0, 0.002, HOP).astype(np.float32)
        fifo = fifo[HOP:]
        out = engine.process_block(x)
        assert np.all(out == 0) or engine.probe_active  # monitor off: only clicks leave
        fifo = np.concatenate([fifo, out])
    result = engine.probe_result()
    assert isinstance(result, LatencyResult)
    assert result.clicks_sent == 3 and result.clicks_found == 3
    assert abs(result.round_trip_ms - 37.0) < 0.5
    assert all(abs(d - 37.0) < 0.5 for d in result.delays_ms)


# --------------------------------------------------------------------------- control


def test_handle_key_adjusts_slider_mode_and_quit():
    engine = LiveEngine(LiveConfig(suppression=50))
    assert handle_key(engine, "+") is False and engine.suppression == 55
    assert handle_key(engine, "-") is False and engine.suppression == 50
    handle_key(engine, "9")
    assert engine.suppression == 90
    handle_key(engine, "0")
    assert engine.suppression == 0
    handle_key(engine, "m")
    assert engine.mode == "focus_hq"
    assert handle_key(engine, "q") is True


def _http(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_control_server_reads_and_writes_the_engine_state():
    engine = LiveEngine(LiveConfig(suppression=40))
    server = ControlServer(engine, port=0)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        status, state = _http("GET", f"{base}/state")
        assert status == 200 and state["suppression"] == 40 and state["mode"] == "natural"
        assert "stats" in state
        status, state = _http("POST", f"{base}/control", {"suppression": 75, "mode": "quiet"})
        assert status == 200 and engine.suppression == 75 and engine.mode == "quiet"
        status, _ = _http("GET", f"{base}/control?suppression=20")
        assert status == 200 and engine.suppression == 20
        status, err = _http("POST", f"{base}/control", {"mode": "loud"})
        assert status == 400 and "error" in err
        with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
            assert resp.status == 200 and b"<input" in resp.read()
    finally:
        server.stop()


# --------------------------------------------------------------------------- run loop


class FakeStream:
    """Drives the duplex callback from a thread at roughly real time."""

    created: ClassVar[list[FakeStream]] = []

    def __init__(self, config: LiveConfig, callback, loop_delay_s: float = 0.0):
        self.config = config
        self.callback = callback
        self.samplerate = config.sample_rate
        self.blocksize = round(config.sample_rate * config.block_ms / 1000)
        self.latency = (0.012, 0.008)
        self._fifo = np.zeros(round(loop_delay_s * self.samplerate), dtype=np.float32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.calls = 0
        FakeStream.created.append(self)

    def _run(self):
        period = self.blocksize / self.samplerate
        signal = make_test_signal(2.0, self.samplerate)
        pos = 0
        while not self._stop.is_set():
            x = signal[pos : pos + self.blocksize]
            pos = (pos + self.blocksize) % (len(signal) - self.blocksize)
            if len(self._fifo) >= self.blocksize:
                x = x + self._fifo[: self.blocksize]
                self._fifo = self._fifo[self.blocksize :]
            indata = x.reshape(-1, 1).astype(np.float32)
            outdata = np.zeros((self.blocksize, 2), dtype=np.float32)
            self.callback(indata, outdata, self.blocksize, None, 0)
            self._fifo = np.concatenate([self._fifo, outdata[:, 0]])
            self.calls += 1
            time.sleep(period)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def close(self):
        pass


def test_run_live_runs_for_the_duration_and_reports_stats():
    FakeStream.created.clear()
    summary = run_live(
        LiveConfig(pipeline="passthrough", suppression=30, block_ms=10.0),
        duration_s=0.5,
        stream_factory=FakeStream,
        control_port=None,
        keyboard=False,
    )
    assert summary["blocks"] >= 20
    assert summary["underruns"] == 0
    assert summary["suppression"] == 30
    assert summary["stream_latency_ms"] == pytest.approx(20.0)
    assert summary["estimated_total_ms"] == pytest.approx(20.0 + summary["compute_ms_p95"], abs=1e-6)
    assert summary["duration_s"] >= 0.45
    assert summary["warm_up_ms"] >= 0.0
    assert FakeStream.created and FakeStream.created[-1].calls >= 20


def test_run_live_control_server_changes_the_slider_mid_run():
    results: dict = {}
    port_box: list[int] = []

    def on_start(server: ControlServer):
        port_box.append(server.port)
        _http("POST", f"http://127.0.0.1:{server.port}/control", {"suppression": 85})

    def runner():
        results["summary"] = run_live(
            LiveConfig(pipeline="passthrough"),
            duration_s=0.4,
            stream_factory=FakeStream,
            control_port=0,
            keyboard=False,
            on_control_started=on_start,
        )

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert port_box and results["summary"]["suppression"] == 85


def test_run_latency_test_measures_round_trip_through_a_simulated_loop():
    def factory(config, callback):
        return FakeStream(config, callback, loop_delay_s=0.045)

    result = run_latency_test(
        LiveConfig(pipeline="passthrough"), clicks=3, interval_s=0.25, stream_factory=factory
    )
    assert result.clicks_found == 3
    assert abs(result.round_trip_ms - 45.0) < 1.0
    assert result.sample_rate == SR and result.block_ms == 5.0  # default: lowest measured
    text = result.format()
    assert "round trip" in text and "45." in text


# --------------------------------------------------------------------------- cli


def test_cli_has_live_latency_test_and_devices_commands():
    parser = build_parser()
    args = parser.parse_args(["live", "--mode", "natural", "--in", "3", "--out", "speakers"])
    assert args.mode == "natural" and args.input == "3" and args.output == "speakers"
    assert args.pipeline is None and args.suppression == 50
    assert args.block_ms == 5.0 and args.latency == "min"
    args = parser.parse_args(["latency-test", "--clicks", "4", "--sr", "48000"])
    assert args.clicks == 4 and args.sr == 48000
    args = parser.parse_args(["devices"])
    assert callable(args.func)


def test_default_config_uses_five_ms_blocks_and_minimum_latency_hint():
    config = LiveConfig()
    assert config.block_ms == 5.0 and config.latency == 0.0
    assert LiveEngine(LiveConfig(pipeline="passthrough")).block_samples == 80


def test_cli_live_dry_run_prints_the_pipeline_without_opening_audio(capsys):
    from hearing.cli import main

    assert main(["live", "--mode", "quiet", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "silence" in out and "quiet" in out
    assert sys.modules.get("sounddevice") is None
