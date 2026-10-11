"""Demo audio path round trip (T-144): MacBook Pro Microphone -> hearing -> AirPods Pro 3 over A2DP.

    bash scripts/bt_latency.sh --dry-run      # resolve the devices, print the procedure, open nothing
    bash scripts/bt_latency.sh                # 3 runs x 5 clicks, writes results/bt_latency_<stamp>.json

Wraps ``python -m hearing latency-test --json --in "MacBook Pro Microphone" --out "AirPods"`` (each run in
its own process). Refuses unless the output resolves to AirPods, and refuses an AirPods input: the AirPods mic
switches the buds to HFP (16 kHz SCO, more delay), which the demo never uses. Pools every detected click and
judges the demo bar: median <= 300 ms and p95 <= 350 ms. Prints a ``docs/BENCHMARKS.md`` "Demo audio path"
row for the human to paste. Numbers come only from a human run with the buds: tests never open a device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hearing.audio.devices import DeviceInfo, resolve_device

DEFAULT_IN = "MacBook Pro Microphone"
DEFAULT_OUT = "AirPods"
RUNS = 3
CLICKS = 5
MEDIAN_BAR_MS = 300.0
P95_BAR_MS = 350.0

PROCEDURE = """\
Procedure (demo audio path: MacBook mic in, AirPods Pro 3 out over A2DP):
  1. AirPods Settings: ear detection OFF (a bud out of the ear must keep playing), Mic Mode Standard,
     Transparency / Adaptive / Conversation Awareness / spatial audio OFF.
  2. ANC OFF for this measurement only (the demo runs with ANC ON): Noise Control > Off.
  3. Input is the 'MacBook Pro Microphone', never the AirPods mic (that forces HFP).
  4. Hold one bud ~2 cm from the mic grille (beside the keyboard, left of the F-keys), speaker facing it;
     quiet room, volume at about 70 %.
  5. Keep still while the clicks play: {runs} runs x {clicks} clicks."""

NO_CLICKS_HINT = ("no click detected: move the bud closer (~2 cm, speaker toward the mic grille), raise the volume, "
                  "check ear detection is OFF (a bud out of the ear pauses) and that the AirPods are the output.")


def is_airpods(name: str | None) -> bool:
    return "airpods" in (name or "").casefold()


def _child_devices() -> list[DeviceInfo]:
    """Every device, from a child process (PortAudio stays out of this one, like judge_check)."""
    code = ("import json; from hearing.audio.devices import list_devices; "
            "print(json.dumps([d.__dict__ for d in list_devices()]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         check=False, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"could not list devices: {out.stderr.strip()[-300:]}")
    return [DeviceInfo(**d) for d in json.loads(out.stdout.strip().splitlines()[-1])]


def check_devices(in_spec: str, out_spec: str, devices: list[DeviceInfo]) -> dict:
    """Resolve both specs; raise ``ValueError`` unless output is AirPods and input is not."""
    by_index = {d.index: d for d in devices}
    i = resolve_device(in_spec, "input", devices)
    o = resolve_device(out_spec, "output", devices)
    if i is None or o is None:
        raise ValueError("pass explicit --in and --out devices; the system default is not checked")
    inp, out = by_index[i].name, by_index[o].name
    if not is_airpods(out):
        raise ValueError(f"output {out!r} is not the AirPods: this measures the AirPods (A2DP) demo path; "
                         "connect them and pass --out AirPods")
    if is_airpods(inp):
        raise ValueError(f"input {inp!r} is the AirPods mic: that switches the buds to HFP; use "
                         f"--in {DEFAULT_IN!r}")
    return {"input": inp, "output": out, "input_index": i, "output_index": o}


def latency_cmd(inp: str, out: str, clicks: int, interval: float) -> list[str]:
    return [sys.executable, "-m", "hearing", "latency-test", "--json", "--in", inp, "--out", out,
            "--clicks", str(clicks), "--interval", f"{interval:g}"]


def parse_json(stdout: str) -> dict:
    """The latency-test JSON object (the CLI may log lines before it)."""
    dec = json.JSONDecoder()
    for pos in [0] + [i + 1 for i, c in enumerate(stdout) if c == "\n"]:
        if stdout[pos:pos + 1] == "{":
            try:
                return dec.raw_decode(stdout, pos)[0]
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in latency-test output: {stdout.strip()[-200:]!r}")


def summarize(runs: list[dict]) -> dict:
    """Pool every detected click over the runs: median / range / p95 and the bar verdicts."""
    delays = [float(d) for r in runs for d in (r.get("delays_ms") or [])]
    sent = sum(int(r.get("clicks_sent") or 0) for r in runs)
    found = sum(int(r.get("clicks_found") or 0) for r in runs)
    if delays:
        median, p95 = float(np.median(delays)), float(np.percentile(delays, 95))
        rng = [min(delays), max(delays)]
    else:
        median = p95 = None
        rng = None
    return {
        "median_ms": None if median is None else round(median, 1),
        "p95_ms": None if p95 is None else round(p95, 1),
        "range_ms": None if rng is None else [round(x, 1) for x in rng],
        "run_medians_ms": [r.get("round_trip_ms") for r in runs],
        "clicks_sent": sent,
        "clicks_found": found,
        "bars": {"median_ms": MEDIAN_BAR_MS, "p95_ms": P95_BAR_MS},
        "median_pass": median is not None and median <= MEDIAN_BAR_MS,
        "p95_pass": p95 is not None and p95 <= P95_BAR_MS,
        "pass": median is not None and median <= MEDIAN_BAR_MS and p95 <= P95_BAR_MS,
    }


def verdict_lines(s: dict) -> list[str]:
    def fmt(v):
        return "n/a" if v is None else f"{v:.1f} ms"

    return [
        f"{'PASS' if s['median_pass'] else 'FAIL'}  median {fmt(s['median_ms'])} (bar <= {MEDIAN_BAR_MS:g} ms)",
        f"{'PASS' if s['p95_pass'] else 'FAIL'}  p95 {fmt(s['p95_ms'])} (bar <= {P95_BAR_MS:g} ms)",
    ]


def bench_row(s: dict, devices: dict, date: str, result_name: str) -> str:
    """One ``docs/BENCHMARKS.md`` "Demo audio path" table row (header in :data:`BENCH_HEADER`)."""
    rng = "-" if s["range_ms"] is None else f"{s['range_ms'][0]:.0f}-{s['range_ms'][1]:.0f}"
    med = "-" if s["median_ms"] is None else f"{s['median_ms']:.0f}"
    p95 = "-" if s["p95_ms"] is None else f"{s['p95_ms']:.0f}"
    return (f"| {date} | {devices['input']} | {devices['output']} (A2DP, ANC off) | {s['clicks_found']}/{s['clicks_sent']} "
            f"| {med} | {rng} | {p95} | {'PASS' if s['pass'] else 'FAIL'} | `{result_name}` |")


BENCH_HEADER = ("| date | input | output | clicks | median ms | range ms | p95 ms | bar (<= 300 / <= 350) | result |\n"
                "|---|---|---|---|---|---|---|---|---|")


def results_dir() -> Path:
    env = os.environ.get("HEARING_RESULTS_DIR")
    return Path(env) if env else ROOT / "results"


def main(argv: list[str] | None = None, *, devices: Callable[[], list[DeviceInfo]] = _child_devices,
         run: Callable = subprocess.run, out_dir: Path | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="input", default=DEFAULT_IN)
    p.add_argument("--out", dest="output", default=DEFAULT_OUT)
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--clicks", type=int, default=CLICKS)
    p.add_argument("--interval", type=float, default=1.0, help="seconds between clicks (default 1)")
    p.add_argument("--dry-run", action="store_true", help="resolve devices and print the procedure; no audio")
    args = p.parse_args(argv)

    try:
        dev = check_devices(args.input, args.output, devices())
    except (ValueError, RuntimeError) as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"input  = {dev['input']} (#{dev['input_index']})\noutput = {dev['output']} (#{dev['output_index']})")
    print(PROCEDURE.format(runs=args.runs, clicks=args.clicks))
    cmd = latency_cmd(dev["input"], dev["output"], args.clicks, args.interval)
    if args.dry_run:
        print("dry run: would run " + " ".join(cmd[1:]) + f"  x{args.runs}")
        return 0

    runs = []
    for k in range(args.runs):
        print(f"run {k + 1}/{args.runs} ...", flush=True)
        out = run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        try:
            r = parse_json(out.stdout)
        except ValueError:
            print(f"REFUSED: latency-test did not run (exit {out.returncode}): {out.stderr.strip()[-300:]}")
            return 2
        runs.append(r)
        print(f"  {r.get('clicks_found', 0)}/{r.get('clicks_sent', 0)} clicks, median {r.get('round_trip_ms')} ms "
              f"{r.get('delays_ms')}")
        if not r.get("clicks_found"):
            print(f"  hint: {NO_CLICKS_HINT}")

    s = summarize(runs)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = out_dir or results_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"bt_latency_{stamp}.json"
    payload = {"task": "T-144", "stamp": stamp, "devices": dev, "runs_requested": args.runs,
               "clicks_per_run": args.clicks, "summary": s, "runs": runs}
    path.write_text(json.dumps(payload, indent=1))
    if not s["clicks_found"]:
        print(f"hint: {NO_CLICKS_HINT}")
    print("\n".join(verdict_lines(s)))
    print(f"wrote {path}")
    print("\ndocs/BENCHMARKS.md 'Demo audio path' row (paste under the header if the table is new):")
    print(BENCH_HEADER)
    print(bench_row(s, dev, time.strftime("%Y-%m-%d"), path.name))
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
