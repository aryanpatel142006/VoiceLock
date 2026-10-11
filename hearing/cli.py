"""Command line: ``python -m hearing <command>``. Commands: serve, qr, live, latency-test, devices, synth-test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hearing.live import MODES
from hearing.server import tls


def _add_serve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("serve", help="run the session server and debug console")
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to accept phones on the LAN (pair with --tls)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tls", action="store_true",
                   help="serve https + wss with a cached self-signed certificate for the LAN IPs and "
                   "hearing.local (~/.hearing/tls or $HEARING_TLS_DIR); phones need it for the mic")
    p.add_argument("--cert", type=Path, default=None, help="PEM certificate to use instead (e.g. mkcert)")
    p.add_argument("--key", type=Path, default=None, help="PEM private key for --cert")
    p.add_argument(
        "--pipelines",
        default=os.environ.get("HEARING_PIPELINES", ""),
        help='per-mode pipelines, e.g. "natural=dfn>gate>mixer; quiet=silence" '
        "(env HEARING_PIPELINES; unspecified modes keep their defaults)",
    )
    p.add_argument("--fallback", default="passthrough", help="pipeline a failing one degrades to")
    p.add_argument("--hq-backend", choices=["auto", "local", "remote"], default=None,
                   help="where Focus HQ runs (default: $HEARING_HQ_BACKEND or auto)")
    p.add_argument("--remote-url", default=None,
                   help="GPU session server for Focus HQ (default: $HEARING_REMOTE_URL)")
    p.add_argument("--telemetry-hz", type=float, default=10.0)
    p.add_argument("--console-dir", type=Path, default=None, help="default: web/console")
    p.add_argument("--log-level", default="info")
    p.add_argument("--no-warm", dest="warm", action="store_false",
                   help="do not preload the live models at start (hearing.server.warm)")
    p.set_defaults(func=_serve)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from hearing.server.app import create_app
    from hearing.server.session import ServerConfig, parse_pipelines
    from hearing.server.team import apply_env

    apply_env()  # HEARING_DEMO_PRESET=team: the caption ASR env (T-130); a no-op when the preset is off

    extra = {}
    if args.hq_backend:
        extra["hq_backend"] = args.hq_backend
    if args.remote_url:
        extra["remote_url"] = args.remote_url
    config = ServerConfig(
        **extra,
        pipelines=parse_pipelines(args.pipelines),
        fallback=args.fallback,
        telemetry_hz=args.telemetry_hz,
        console_dir=args.console_dir,
    )
    app = create_app(config)
    ssl = _tls_files(args)
    if ssl is None:
        return 2
    sock = None
    if args.port == 0:  # bind here so the port is known (and printed) before uvicorn serves
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, 0))
        args.port = sock.getsockname()[1]
        scheme = "https" if ssl else "http"
        print(f"hearing serve: bound port {args.port}; health: {scheme}://127.0.0.1:{args.port}/health", flush=True)
    info = tls.connect_info(args.host, args.port, bool(ssl))
    app.state.connect = info
    app.state.cert_path = ssl["ssl_certfile"] if ssl else None
    if ssl or args.host not in ("127.0.0.1", "localhost"):
        print(tls.banner(info, app.state.cert_path), flush=True)
    if args.warm:
        from hearing.server import warm

        warm.start()  # first session's vision must not drop frames while models load (T-072)
    if sock is None:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, **ssl)
    else:
        uvicorn.Server(uvicorn.Config(app, log_level=args.log_level, **ssl)).run(sockets=[sock])
    return 0


def _tls_files(args: argparse.Namespace) -> dict | None:
    """uvicorn ssl kwargs ({} without TLS); None (after a message) when --cert/--key are half given."""
    if args.cert or args.key:
        if not (args.cert and args.key and args.cert.is_file() and args.key.is_file()):
            print("hearing serve: --cert and --key must both name existing PEM files")
            return None
        return {"ssl_certfile": str(args.cert), "ssl_keyfile": str(args.key)}
    if not args.tls:
        return {}
    extra = [] if args.host in ("0.0.0.0", "127.0.0.1", "localhost") else [args.host]
    cert, key = tls.ensure_cert(tls.cert_hosts(extra))
    return {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}


def _add_qr(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("qr", help="print the phone URLs and a QR code for a running hearing serve")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--http", dest="tls", action="store_false", help="plain http URLs (no --tls)")
    p.add_argument("--page", default="/simple")
    p.set_defaults(func=_qr)


def _qr(args: argparse.Namespace) -> int:
    info = tls.connect_info(args.host, args.port, args.tls, args.page)
    cert = tls.tls_dir() / "cert.pem" if args.tls else None
    print(tls.banner(info, cert), flush=True)
    return 0


def _add_live_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--in", dest="input", default=None,
                   help="input device index or name substring (default: system default)")
    p.add_argument("--out", dest="output", default=None,
                   help="output device index or name substring (default: system default)")
    p.add_argument("--sr", type=int, default=16000,
                   help="device sample rate; the pipeline always runs at 16 kHz (default 16000)")
    p.add_argument("--block-ms", type=float, default=5.0,
                   help="device block size in ms (default 5: lowest measured round trip on laptop-a)")
    p.add_argument("--latency", default="min",
                   help="PortAudio latency hint: min (default), low, high or seconds")
    p.add_argument("--pipeline", default=None,
                   help="stage spec for every mode, e.g. 'gtcrn' or 'dfn>gate>mixer' "
                        "(default: the mode's default; HEARING_LIVE_NATURAL overrides natural)")
    p.add_argument("--json", action="store_true", help="print the final summary as JSON")


def _add_live(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("live", help="mic -> pipeline -> headphones with a slider (keys or HTTP)")
    p.add_argument("--mode", choices=[*MODES, "auto"], default="natural")
    p.add_argument("--suppression", type=float, default=50, help="slider 0-100 (default 50)")
    _add_live_common(p)
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: run until q)")
    p.add_argument("--control-port", type=int, default=0,
                   help="HTTP slider port (0 = ephemeral, printed at start; -1 = off)")
    p.add_argument("--no-keys", action="store_true", help="do not read the keyboard")
    p.add_argument("--dry-run", action="store_true", help="print the resolved pipeline and devices, do not open audio")
    p.set_defaults(func=_live)


def _add_latency_test(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("latency-test",
                       help="play clicks through the live stream and measure the mic-to-ear round trip")
    p.add_argument("--clicks", type=int, default=5)
    p.add_argument("--interval", type=float, default=1.0, help="seconds between clicks")
    _add_live_common(p)
    p.set_defaults(func=_latency_test)


def _add_devices(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("devices", help="list sound devices")
    p.set_defaults(func=_devices)


def _live_config(args: argparse.Namespace, **extra):
    from hearing.live import LiveConfig

    return LiveConfig(
        pipeline=args.pipeline, sample_rate=args.sr, block_ms=args.block_ms, input=args.input,
        output=args.output, latency=args.latency, **extra,
    )


def _live(args: argparse.Namespace) -> int:
    from hearing.live import LiveEngine, describe_devices, run_live

    config = _live_config(args, mode=args.mode, suppression=args.suppression)
    if args.dry_run:
        engine = LiveEngine(config)
        print(f"mode={engine.mode} pipeline={engine.pipeline_name} suppression={engine.suppression:.0f}% "
              f"algorithmic={engine.algorithmic_ms:g} ms sr={config.sample_rate} block={config.block_ms:g} ms")
        for err in engine.errors:
            print(f"! {err}")
        try:
            devices = describe_devices(config)
            print(f"input={devices['input']} output={devices['output']}")
        except Exception as e:  # noqa: BLE001  (no sounddevice / no such device)
            print(f"devices: {type(e).__name__}: {e}")
        return 0
    summary = run_live(
        config, duration_s=args.duration, control_port=None if args.control_port < 0 else args.control_port,
        keyboard=False if args.no_keys else None,
    )
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        print(
            f"live: {summary['duration_s']:.1f} s, {summary['blocks']} blocks, {summary['hops']} hops, "
            f"underruns={summary['underruns']} overflows={summary['overflows']} fills={summary['fills']}\n"
            f"compute per hop p50/p95/max {summary['compute_ms_p50']:.2f}/{summary['compute_ms_p95']:.2f}/"
            f"{summary['compute_ms_max']:.2f} ms; algorithmic {summary['algorithmic_ms']:g} ms; "
            f"PortAudio-reported in+out {summary['stream_latency_ms'] or 0:.1f} ms (an upper bound: "
            f"run 'hearing latency-test' for the measured mic-to-ear round trip)"
        )
        for key in ("underrun_at_s", "overflow_at_s"):
            if summary[key]:
                print(f"{key[:-5]}s at stream seconds: {summary[key]}")
        for err in summary["errors"]:
            print(f"! {err}")
    return 0 if summary["underruns"] == 0 else 1


def _latency_test(args: argparse.Namespace) -> int:
    from hearing.live import run_latency_test

    result = run_latency_test(_live_config(args), clicks=args.clicks, interval_s=args.interval)
    print(json.dumps(result.to_dict(), indent=1) if args.json else result.format())
    return 0 if result.clicks_found else 1


def _devices(args: argparse.Namespace) -> int:
    from hearing.audio.devices import default_device, format_devices

    print(format_devices(), end="")
    print(f"default input={default_device('input')} output={default_device('output')}")
    return 0


def _add_synth_test(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("synth-test",
                       help="render one sentence per voice bucket with the configured TTS (no speakers)")
    p.add_argument("--tts", default=None, help="auto|mock|local|elevenlabs|elevenlabs-mock (default HEARING_TTS)")
    p.add_argument("--text", default=None)
    p.add_argument("--out", type=Path, default=None, help="directory for one wav per bucket")
    p.set_defaults(func=_synth_test)


def _synth_test(args: argparse.Namespace) -> int:
    from hearing.synth import tts

    engine = tts.get_tts(args.tts)
    text = args.text or tts.SAMPLE_TEXT
    try:
        tts.synth_test(engine, text, buckets=["neutral"])  # connect / load
        rows = tts.synth_test(engine, text, out_dir=args.out)
    except tts.TtsError as e:
        print(f"synth-test FAILED ({getattr(engine, 'name', '?')}): {e}")
        return 1
    print(f"engine={engine.name}  text={text!r}")
    for r in rows:
        print(f"{r['bucket']:<14} {r['voice']:<24} first={r['first_audio_ms']:6.0f} ms  "
              f"total={r['total_ms']:6.0f} ms  rtf={r['rtf']:.2f}")
    return 0


def load_dotenv(*dirs: Path) -> list[Path]:
    """Export ``KEY=value`` lines of the first ``.env`` found (cwd, then the repo root); the shell wins.

    docs/TESTING_GUIDE.md and docs/DEMO.md tell the human to put ``ELEVENLABS_API_KEY`` in ``.env``.
    Only ``python -m hearing`` calls this, so tests never pick up a real key."""
    for d in dirs or (Path.cwd(), Path(__file__).resolve().parents[1]):
        path = d / ".env"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            key, eq, value = line.partition("=")
            if line.startswith("#") or not eq or not key.strip():
                continue
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return [path]
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hearing", description="Selective hearing voice backend")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_serve(sub)
    _add_qr(sub)
    _add_live(sub)
    _add_latency_test(sub)
    _add_devices(sub)
    _add_synth_test(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
