"""iPhone-over-USB mic vs MacBook mic A/B (T-171): latency, level, overlap, target-absent, dropouts, listening.

    bash scripts/mic_ab.sh                                   # the human run (opens the mic for the clicks)
    bash scripts/mic_ab.sh --dry-run --no-hw --fake-devices "Aryan patel's iPhone Microphone,MacBook Pro Microphone,AirPods Pro"
    bash scripts/mic_ab.sh --from results/mic_ab_<stamp>.json   # rebuild the sheet from a run's marks

Decision Oct 9: a mic on the target is the biggest isolation lever. Joseph holds an iPhone cabled to the Mac by
USB (Continuity: macOS lists it as ``<name> iPhone Microphone``; never Bluetooth = HFP). For each mic (iPhone
first, then ``MacBook Pro Microphone``):

* Part A: ``scripts/bt_latency.py --in <mic> --out AirPods`` (3 runs x 5 clicks through the AirPods, one row per run).
* Part B: 3 judge takes of 25 s on ``/judge`` (served by ``demo.sh --judge --team --mic <mic>``, recorded under
  ``HEARING_SESSION_LOG``): target alone 5 s, the other talker over them 10 s, target alone 10 s. Read from the
  session files: mic peak/rms (``in.wav`` while the target talks alone), wrong-speaker share (``rehearse``'s
  ECAPA 1 s windows of ``out.wav`` in the overlap) and leak dB (output level of the windows closest to the
  other voice minus that of the target's windows, same overlap).
* Target-absent take: 25 s, target silent, the other talker speaks 5-15 s: output level relative to input and
  the wrong-speaker share (voiced other-voice windows / all 1 s windows) of that 10 s, its own row.
* iPhone only: 2 minutes hand-held: input frames per second and the longest gap (frame arrival, and digital
  silence in ``in.wav``); PASS with no gap > 200 ms.
* One Shift+C live calibration per person (report-only): Isolated within 8 s, or rejected.
* The three-way listening comparison on ONE unchanged cafe take (A ANC off raw room, B ANC on no app,
  C ANC on + Focus locked) and an optional blind Raw / Natural / Focus row.

Writes ``results/mic_ab_<stamp>.md`` (+ ``.json`` with the marks), ``results/listening_<stamp>.md`` (one line per
take, four yes/no columns for the listener) and ``results/listening_<stamp>_key.md`` (the blind key: open it after
listening). Gate: round trip median <= 300 ms and p95 <= 350 ms, peak -48..-2 dBFS, and a lower wrong-speaker share
than the MacBook mic in the same session set; else the fallback is the MacBook mic with the target at 40 cm.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hearing.audio.devices import DeviceInfo
from scripts import rehearse as rh
from scripts.bt_latency import is_airpods

MACBOOK = "MacBook Pro Microphone"
OUTPUT = "AirPods"
SESSION_LOG = "results/sessions"  # the Part B restart line, as in CLAUDE.md (run from the main checkout)
PROFILES_DIR = "$PWD/eval/fixtures/generated/profiles"
TAKES = 3
CLICKS = 5
MEDIAN_BAR_MS = 300.0
P95_BAR_MS = 350.0
PEAK_MIN_DB, PEAK_MAX_DB = -48.0, -2.0
TAKE_S = 25.0
# judge take: (start s, end s, what happens); the overlap is the stretch scored for wrong speaker and leak
JUDGE_CUES = ((0.0, "TARGET talks alone"), (5.0, "OTHER talker joins, over the target"), (15.0, "OTHER stops: TARGET alone"))
ABSENT_CUES = ((0.0, "nobody talks (target stays silent the whole take)"), (5.0, "OTHER talker speaks"),
               (15.0, "OTHER stops: silence"))
OVERLAP = (5.0, 15.0)
SOLO = ((0.0, 5.0), (15.0, 25.0))
DROPOUT_S = 120.0
GAP_MAX_MS = 200.0
CALIBRATE_S = 8.0
HFP_HINT = "it switches the AirPods to HFP (16 kHz, no ANC); use the iPhone over USB or the MacBook mic"
SILENT_AMP = 1e-4  # |x| below this in in.wav = digital silence (a Continuity pause)
NOT_MEASURED = rh.NOT_MEASURED

SHEET_NOTES = """\
Gate (T-171): iPhone wins when, over its 3 latency runs, round trip median <= 300 ms and p95 <= 350 ms, every take's
mic peak is within -48..-2 dBFS, and its wrong-speaker share is lower than the MacBook mic's in this same session set.
Fallback: MacBook Pro Microphone with the target at 40 cm.
iPhone (Apple's rules for iPhone as a Mac mic): cabled by USB (never Bluetooth = HFP), held still with the screen OFF
(unlocking it pauses the audio), Mic Mode Standard (Control Centre; not readable from a script).
AirPods: output only, never the input (HFP). ANC off for the clicks; ANC on for the takes unless a row says otherwise."""


# ---------------------------------------------------------------- devices


def _child_devices() -> list[DeviceInfo]:
    from scripts.bt_latency import _child_devices as child

    return child()


def fake_devices(spec: str) -> list[DeviceInfo]:
    """``"A,B,C"``: every name is both an input and an output (tests and dry runs; nothing is opened)."""
    names = [n.strip() for n in spec.split(",") if n.strip()]
    return [DeviceInfo(index=i, name=n, inputs=1, outputs=2, default_samplerate=48000.0)
            for i, n in enumerate(names)]


def pick_mics(devices: list[DeviceInfo], iphone: str | None, macbook: str) -> dict[str, str | None]:
    """The two inputs by exact label (``iphone`` None: the first input whose name contains "iPhone")."""
    inputs = [d.name for d in devices if d.inputs > 0]

    def find(want: str | None, word: str) -> str | None:
        if want:
            return next((n for n in inputs if n == want), None) or next((n for n in inputs if want.casefold() in n.casefold()), None)
        return next((n for n in inputs if word in n.casefold()), None)

    return {"iphone": find(iphone, "iphone"), "macbook": find(macbook, "macbook")}


def refuse_bluetooth(label: str | None) -> str | None:
    """Why ``label`` cannot be a demo mic (the AirPods / any Bluetooth mic switches the buds to HFP), else None."""
    from scripts.judge_check import BLUETOOTH

    if label and (is_airpods(label) or BLUETOOTH.search(label)):
        return f"{label!r} is a Bluetooth mic: {HFP_HINT}"
    return None


# ---------------------------------------------------------------- numbers


def run_stats(delays: list[float]) -> dict:
    """Median / nearest-rank p95 of one latency run's click delays (None when no click was heard)."""
    from scripts.judge_check import p95_nearest_rank

    if not delays:
        return {"median_ms": None, "p95_ms": None}
    return {"median_ms": float(np.median(delays)), "p95_ms": p95_nearest_rank([float(d) for d in delays])}


def db(x: np.ndarray | None, peak: bool = False) -> float | None:
    if x is None or not len(x):
        return None
    x = np.asarray(x, np.float64)
    v = np.abs(x).max() if peak else np.sqrt(np.mean(x ** 2))
    return float(20 * np.log10(v + 1e-9))


def _cat(parts: list) -> np.ndarray | None:
    parts = [p for p in parts if p is not None and len(p)]
    return np.concatenate(parts) if parts else None


def take_numbers(sessions: list[Path], t0: float, target: str | None, *, embed: Callable | None = None,
                 profiles_root: Path | None = None) -> dict:
    """One 25 s judge take started at wall-clock ``t0``: mic peak/rms while the target talks alone, wrong-speaker
    share and leak dB over the overlap. Every value None when the recording does not have it."""
    solo = _cat([rh.stretch(d, t0 + a, t0 + b, "in.wav", "in", 1) for d in sessions for a, b in SOLO])
    over = [y for d in sessions if (y := rh.out_stretch(d, t0 + OVERLAP[0], t0 + OVERLAP[1])) is not None and len(y)]
    wins = rh.speaker_windows(over, target, embed=embed, profiles_root=profiles_root) if over else None
    wrong = leak = None
    if wins:
        wrong = sum(n != target for n, _ in wins) / len(wins)
        mine = [d for n, d in wins if n == target]
        other = [d for n, d in wins if n != target]
        if mine and other:
            leak = float(np.mean(other) - np.mean(mine))
    return {"peak_db": db(solo, peak=True), "rms_db": db(solo), "wrong": wrong, "leak_db": leak}


def absent_numbers(sessions: list[Path], t0: float, target: str | None, *, embed: Callable | None = None,
                   profiles_root: Path | None = None) -> dict:
    """The target-absent take: output level relative to input while only the other talker speaks, and the share
    of that stretch's 1 s windows where the wearer heard the other voice (silent windows count as right)."""
    a, b = t0 + OVERLAP[0], t0 + OVERLAP[1]
    x = _cat([rh.stretch(d, a, b, "in.wav", "in", 1) for d in sessions])
    y = _cat([rh.out_stretch(d, a, b) for d in sessions])
    rel = None if db(x) is None or db(y) is None else db(y) - db(x)
    wrong = None
    if y is not None:
        wins = rh.speaker_windows([y], target, embed=embed, profiles_root=profiles_root)
        n_all = len(y) // int(16000 * rh.WRONG_WIN_S)
        if wins is not None and n_all:
            wrong = sum(n != target for n, _ in wins) / n_all
    return {"in_db": db(x), "out_db": db(y), "out_rel_in_db": rel, "wrong": wrong}


def dropout_numbers(sessions: list[Path], t0: float, t1: float) -> dict:
    """Input frames per second (min / median over 1 s bins), the longest gap between frame arrivals and the
    longest digital-silence run in ``in.wav`` over [t0, t1]."""
    ws = sorted(r["w"] for d in sessions for r in rh.load_session(d)
                if r["dir"] == "in" and r["event"].get("type") == "binary" and r["event"].get("kind") == 1
                and t0 <= r["w"] <= t1)
    if len(ws) < 2:
        return {"fps_min": None, "fps_median": None, "gap_ms": None, "silence_ms": None}
    bins = np.bincount(np.floor(np.asarray(ws) - ws[0]).astype(int))[:-1]  # the last bin is partial
    x = _cat([rh.stretch(d, t0, t1, "in.wav", "in", 1) for d in sessions])
    silence = None
    if x is not None:
        quiet = np.abs(x) < SILENT_AMP
        longest = run = 0
        for q in quiet:
            run = run + 1 if q else 0
            longest = max(longest, run)
        silence = 1000.0 * longest / 16000
    return {"fps_min": float(bins.min()) if len(bins) else None, "fps_median": float(np.median(bins)) if len(bins) else None,
            "gap_ms": float(1000 * np.diff(ws).max()), "silence_ms": silence}


def calibrate_result(sessions: list[Path], t0: float) -> str | None:
    """Shift+C at ``t0``: ``Isolated in X s`` when the acquisition reads ``locked`` within 8 s, ``rejected`` when the
    enrollment reads ``failed``, else ``not Isolated in 8 s``; None without ticks."""
    ticks = [t for t in rh.ticks_of([r for d in sessions for r in rh.load_session(d)]) if t0 <= t["w"] <= t0 + CALIBRATE_S]
    if not ticks:
        return None
    hit = next((t["w"] for t in ticks if (t.get("acquisition") or {}).get("state") == "locked"), None)
    if hit is not None:
        return f"Isolated in {hit - t0:.1f} s"
    if any((t.get("enrollment") or {}).get("state") == "failed" for t in ticks):
        return "rejected (calibration failed)"
    return f"not Isolated in {CALIBRATE_S:g} s"


def target_of(sessions: list[Path], t0: float, t1: float, default: str | None) -> str | None:
    ticks = [t for t in rh.ticks_of([r for d in sessions for r in rh.load_session(d)]) if t0 <= t["w"] <= t1]
    return next((p for t in reversed(ticks) if (p := rh.profile_of(t))), default)


def sessions_at(sessions: list[Path], t0: float, t1: float) -> list[str]:
    """Session ids (dir names) with an event in [t0, t1]."""
    return [d.name for d in sessions if any(t0 <= r["w"] <= t1 for r in rh.load_session(d))]


# ---------------------------------------------------------------- gate


def gate(rows: dict[str, list[dict]]) -> tuple[bool, str]:
    """(iPhone passes, one-line verdict). ``rows[mic]`` = per-take dicts with ``delays`` (that run's click
    delays), ``peak_db`` and ``wrong``; latency is pooled over the runs."""
    from scripts.judge_check import p95_nearest_rank

    def pooled(mic):
        d = [float(v) for r in rows.get(mic, []) for v in (r.get("delays") or [])]
        return (float(np.median(d)), p95_nearest_rank(d)) if d else (None, None)

    def mean_wrong(mic):
        w = [r["wrong"] for r in rows.get(mic, []) if r.get("wrong") is not None]
        return float(np.mean(w)) if w else None

    why = []
    med, p95 = pooled("iphone")
    if med is None:
        why.append("iPhone round trip not measured")
    else:
        if med > MEDIAN_BAR_MS:
            why.append(f"iPhone round trip median {med:.0f} > {MEDIAN_BAR_MS:g} ms")
        if p95 > P95_BAR_MS:
            why.append(f"iPhone round trip p95 {p95:.0f} > {P95_BAR_MS:g} ms")
    peaks = [r.get("peak_db") for r in rows.get("iphone", [])]
    if not peaks or any(p is None for p in peaks):
        why.append("iPhone peak not measured on every take")
    elif any(not PEAK_MIN_DB <= p <= PEAK_MAX_DB for p in peaks):
        why.append(f"iPhone peak outside {PEAK_MIN_DB:g}..{PEAK_MAX_DB:g} dBFS ({', '.join(f'{p:.1f}' for p in peaks)})")
    wi, wm = mean_wrong("iphone"), mean_wrong("macbook")
    if wi is None or wm is None:
        why.append("wrong-speaker share not measured on both mics")
    elif not wi < wm:
        why.append(f"iPhone wrong speaker {100 * wi:.1f} % is not lower than MacBook {100 * wm:.1f} %")
    if why:
        return False, "VERDICT: FALLBACK = MacBook Pro Microphone, target at 40 cm (" + "; ".join(why) + ")"
    return True, (f"VERDICT: iPhone over USB PASSES the gate (median {med:.0f} ms, p95 {p95:.0f} ms, peaks in range, "
                  f"wrong speaker {100 * wi:.1f} % < MacBook {100 * wm:.1f} %)")


# ---------------------------------------------------------------- sheets


def _f(v, spec: str = ".1f", unit: str = "") -> str:
    return NOT_MEASURED if v is None else f"{format(v, spec)}{unit}"


def render(m: dict, stamp: str) -> str:
    """``results/mic_ab_<stamp>.md``: 3 rows per mic, the target-absent rows, dropouts, Shift+C, the verdict."""
    line = gate(m["rows"])[1]
    labels = m["labels"]
    out = [f"# Mic A/B {stamp} (T-171)", "",
           (f"iPhone: {labels.get('iphone') or 'not found'} | MacBook: {labels.get('macbook') or 'not found'} | "
            f"output: {labels.get('output')}"), "", SHEET_NOTES, "",
           "| mic | take | session | round trip median ms | p95 ms | clicks | peak dBFS | rms dBFS | wrong speaker % | leak dB |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for mic in ("iphone", "macbook"):
        for k, r in enumerate(m["rows"].get(mic, []), 1):
            st = run_stats(r.get("delays") or [])
            out.append(f"| {mic} | {k} | {', '.join(r.get('sessions') or []) or '-'} | {_f(st['median_ms'], '.0f')} | "
                       f"{_f(st['p95_ms'], '.0f')} | {len(r.get('delays') or [])} | {_f(r.get('peak_db'))} | "
                       f"{_f(r.get('rms_db'))} | {_f(None if r.get('wrong') is None else 100 * r['wrong'])} | "
                       f"{_f(r.get('leak_db'))} |")
    out += ["", ("Leak dB = output level of the overlap's 1 s windows closest to the other voice minus that of the "
                 "target's windows (ECAPA, report-only; NOT MEASURED when one side has no window)."), "",
            "## Target absent (target silent, other talker 5-15 s)", "",
            "| mic | session | input dBFS | output dBFS | output re input dB | wrong speaker % of the 10 s |", "|---|---|---|---|---|---|"]
    for mic, a in (m.get("absent") or {}).items():
        out.append(f"| {mic} | {', '.join(a.get('sessions') or []) or '-'} | {_f(a.get('in_db'))} | {_f(a.get('out_db'))} | "
                   f"{_f(a.get('out_rel_in_db'))} | {_f(None if a.get('wrong') is None else 100 * a['wrong'])} |")
    d = m.get("dropout") or {}
    gaps = [g for g in (d.get("gap_ms"), d.get("silence_ms")) if g is not None]
    dv = NOT_MEASURED if len(gaps) < 2 else ("PASS" if max(gaps) <= GAP_MAX_MS else "FAIL")
    out += ["", f"## iPhone hand-held dropout check ({DROPOUT_S / 60:g} min, no gap > {GAP_MAX_MS:g} ms)", "",
            "| frames/s min | frames/s median | longest arrival gap ms | longest digital silence ms | verdict |",
            "|---|---|---|---|---|",
            (f"| {_f(d.get('fps_min'), '.0f')} | {_f(d.get('fps_median'), '.0f')} | {_f(d.get('gap_ms'), '.0f')} | "
             f"{_f(d.get('silence_ms'), '.0f')} | {dv} |"), "",
            "## Shift+C live calibration (report-only; keys 1/2 stay the stage path)", "",
            "| person | result |", "|---|---|"]
    for who, res in (m.get("calibrate") or {}).items():
        out.append(f"| {who} | {res or NOT_MEASURED} |")
    out += ["", line, ""]
    return "\n".join(out)


LISTEN_COLS = "| take | mic | session | wrong person (y/n) | leak (y/n) | robotic (y/n) | delay (y/n) |\n|---|---|---|---|---|---|---|"
THREE_WAY = (("A", "AirPods noise control OFF, raw room (app Raw)", "live"),
             ("B", "ANC on, no app", "live (listener only: ANC is in the earbuds, the app cannot record it)"),
             ("C", "ANC on + Focus locked on the target (judge profile, team preset)", "live"))
BLIND = ("Raw", "Natural (noise reduction, mode natural)", "Focus (locked profile)")


def render_listening(m: dict, stamp: str, seed: int | None = None) -> tuple[str, str]:
    """(``listening_<stamp>.md`` for the listener, ``listening_<stamp>_key.md``: the blind order, read after)."""
    rows = [f"# Listening log {stamp} (T-171)", "", "One line per take; the listener writes y or n in each column.", "",
            LISTEN_COLS]
    for mic in ("iphone", "macbook"):
        for k, r in enumerate(m["rows"].get(mic, []), 1):
            rows.append(f"| judge {k} | {mic} | {', '.join(r.get('sessions') or []) or '-'} |  |  |  |  |")
    for mic, a in (m.get("absent") or {}).items():
        rows.append(f"| target absent | {mic} | {', '.join(a.get('sessions') or []) or '-'} |  |  |  |  |")
    cafe = m.get("cafe") or {}
    rows += ["", "## Three-way on ONE unchanged cafe take (same mic, same AirPods, matched output loudness)", "",
             "| row | settings | run id | live/replay | apparent target loudness | side speaker audible (Y/N + a word) | delay |",
             "|---|---|---|---|---|---|---|"]
    for key, settings, label in THREE_WAY:
        sid = "-" if key == "B" else (", ".join(cafe.get(key) or []) or "-")
        rows.append(f"| {key} | {settings} | {sid} | {label} |  |  |  |")
    order = list(BLIND)
    random.Random(seed).shuffle(order)
    rows += ["", "## Optional blind preference: Raw / Natural / Focus on the same take, played as A / B / C", "",
             "Replay the take's in.wav three ways (e.g. scripts/replay.py) at matched loudness; the order is in the key file.",
             "", "| clip | preferred (rank 1-3) | note |", "|---|---|---|"]
    rows += [f"| {c} |  |  |" for c in "ABC"]
    key = [f"# Blind key {stamp} (open after listening)", ""] + [f"- {c} = {o}" for c, o in zip("ABC", order, strict=True)]
    return "\n".join(rows) + "\n", "\n".join(key) + "\n"


# ---------------------------------------------------------------- run


def measure(marks: dict, sessions: list[Path], latency: dict[str, list[list[float]]], *, target: str | None,
            takes: int = TAKES, embed: Callable | None = None, profiles_root: Path | None = None) -> dict:
    """Every number of the sheet from the marks (wall-clock starts) and the recorded sessions."""
    m: dict = {"labels": marks.get("labels", {}), "rows": {}, "absent": {}, "calibrate": {}, "cafe": {}}
    for mic in ("iphone", "macbook"):
        runs = latency.get(mic) or []
        starts = (marks.get("takes") or {}).get(mic) or []
        rows = []
        for k in range(takes):
            r: dict = {"delays": runs[k] if k < len(runs) else [], "sessions": []}
            r.update({"peak_db": None, "rms_db": None, "wrong": None, "leak_db": None})
            if k < len(starts):
                t0 = starts[k]
                who = target_of(sessions, t0, t0 + TAKE_S, target)
                r.update(take_numbers(sessions, t0, who, embed=embed, profiles_root=profiles_root))
                r["sessions"] = sessions_at(sessions, t0, t0 + TAKE_S)
            rows.append(r)
        m["rows"][mic] = rows
        t0 = (marks.get("absent") or {}).get(mic)
        a = {"in_db": None, "out_db": None, "out_rel_in_db": None, "wrong": None, "sessions": []}
        if t0 is not None:
            a.update(absent_numbers(sessions, t0, target_of(sessions, t0, t0 + TAKE_S, target), embed=embed,
                                    profiles_root=profiles_root))
            a["sessions"] = sessions_at(sessions, t0, t0 + TAKE_S)
        m["absent"][mic] = a
    t0 = marks.get("dropout")
    m["dropout"] = dropout_numbers(sessions, t0, t0 + DROPOUT_S) if t0 is not None else {}
    for who, t in (marks.get("calibrate") or {}).items():
        m["calibrate"][who] = calibrate_result(sessions, t) if t is not None else None
    for key, (a, b) in (marks.get("cafe") or {}).items():
        m["cafe"][key] = sessions_at(sessions, a, b)
    return m


def _say(msg: str) -> None:
    print(msg, flush=True)


def cue(cues: tuple, seconds: float, sleep: Callable[[float], None]) -> None:
    t = 0.0
    for at, what in cues:
        sleep(at - t)
        t = at
        _say(f"  [{at:4.0f} s] {what}")
    sleep(seconds - t)
    _say(f"  [{seconds:4.0f} s] STOP")


def latency_runs(mic: str, output: str, runs: int, clicks: int, run=subprocess.run) -> list[list[float]]:
    """``scripts/bt_latency.py --in <mic>``: one list of click delays per run (empty when it could not run)."""
    cmd = [sys.executable, "scripts/bt_latency.py", "--in", mic, "--out", output, "--runs", str(runs), "--clicks", str(clicks)]
    res = run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    _say(res.stdout.rstrip())
    path = next((Path(ln[len("wrote "):].strip()) for ln in res.stdout.splitlines() if ln.startswith("wrote ")), None)
    if path is None or not path.exists():
        _say(f"  !! latency on {mic} wrote no result (exit {res.returncode}): {res.stderr.strip()[-200:]}")
        return []
    return [list(r.get("delays_ms") or []) for r in json.loads(path.read_text()).get("runs", [])]


def demo_line(label: str) -> str:
    """The full server restart for Part B: without HEARING_SESSION_LOG the tap records nothing (every row NOT
    MEASURED) and without HEARING_PROFILES_DIR keys 1/2 do nothing."""
    log = os.environ.get("HEARING_SESSION_LOG") or SESSION_LOG  # where this script looks for the sessions
    profiles = os.environ.get("HEARING_PROFILES_DIR") or PROFILES_DIR
    return (f"HEARING_SESSION_LOG={log} HEARING_PROFILES_DIR={profiles} "
            f"bash scripts/demo.sh --judge --team --mic \"{label}\" --restart")


def missing_sessions(sessions: list[Path], log_root: Path) -> str | None:
    """Why there is nothing to score (no session dir, or one without events.jsonl), None when every dir is there."""
    if not sessions:
        return (f"no session dir under {log_root} since the run began; the server was not recording. Restart it with "
                f"{demo_line('<mic label>')} and redo the takes (or pass --session <dir>).")
    bad = [str(d) for d in sessions if not (Path(d) / "events.jsonl").is_file()]
    return f"session dir missing or without events.jsonl: {', '.join(bad)}" if bad else None


def results_dir() -> Path:
    return Path(os.environ.get("HEARING_RESULTS_DIR") or ROOT / "results")


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = input, sleep: Callable[[float], None] = time.sleep,
         devices: Callable[[], list[DeviceInfo]] | None = None, run=subprocess.run, embed: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iphone", default=os.environ.get("HEARING_IPHONE_MIC"),
                   help="exact iPhone input label (default: the first input containing 'iPhone')")
    p.add_argument("--macbook", default=MACBOOK)
    p.add_argument("--out", dest="output", default=OUTPUT, help="output device for the clicks (AirPods)")
    p.add_argument("--takes", type=int, default=TAKES)
    p.add_argument("--dropout-s", type=float, default=DROPOUT_S)
    p.add_argument("--target", default=None, help="target voice profile (default: the session's, else pair.target.profile)")
    p.add_argument("--dry-run", action="store_true", help="print the procedure, write the sheets with NOT MEASURED, wait for nothing")
    p.add_argument("--no-hw", action="store_true", help="never open a device (skips the clicks)")
    p.add_argument("--fake-devices", default=None, help="comma-separated device names instead of the real list")
    p.add_argument("--from", dest="from_json", type=Path, default=None, help="rebuild the sheets from a mic_ab_<stamp>.json")
    p.add_argument("--session", type=Path, action="append", default=None, help="session dir(s) (default: recorded since start)")
    p.add_argument("--results", type=Path, default=None, help="where the sheets go (default $HEARING_RESULTS_DIR or results/)")
    args = p.parse_args(argv)

    from hearing.server.team import load_preset

    preset = load_preset("team")
    target = args.target or ((preset.get("pair") or {}).get("target") or {}).get("profile")
    log_root = Path(os.environ.get("HEARING_SESSION_LOG") or ROOT / "results" / "sessions").expanduser()
    folder = args.results or results_dir()
    began = time.time()

    if args.from_json:
        saved = json.loads(args.from_json.read_text())
        marks, latency, began = saved["marks"], saved["latency"], saved["began"]
        stamp = saved["stamp"]
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        listing = devices or ((lambda: fake_devices(args.fake_devices)) if args.fake_devices else _child_devices)
        found = pick_mics(listing(), args.iphone, args.macbook)
        for mic, label in found.items():
            if why := refuse_bluetooth(label):
                _say(f"REFUSED: {why}")
                return 2
            _say(f"{mic:8s}= {label or 'NOT FOUND (connect it: USB cable, Continuity; then rerun)'}")
        marks = {"labels": {**found, "output": args.output}, "takes": {}, "absent": {}, "calibrate": {}, "cafe": {}}
        latency: dict[str, list[list[float]]] = {}
        _say("\n" + SHEET_NOTES)
        wait = (lambda msg: None) if args.dry_run else ask
        tick = (lambda s: None) if args.dry_run else sleep
        if not args.dry_run:
            os.environ["HEARING_SESSION_LOG"] = str(log_root)
            rh.profiles_env()
        for mic in ("iphone", "macbook"):
            label = found[mic]
            _say(f"\n== {mic}: {label or 'NOT FOUND'} ==")
            if label is None:
                continue
            _say(f"  Part A: {args.takes} runs x {CLICKS} clicks, {label} -> {args.output} (ANC off for the clicks)")
            if args.dry_run or args.no_hw:
                _say(f"  {'dry run' if args.dry_run else 'SKIP (--no-hw)'}: scripts/bt_latency.py --in {label!r} "
                     f"--out {args.output} --runs {args.takes} --clicks {CLICKS}")
            else:
                wait("  Enter to play the clicks > ")
                latency[mic] = latency_runs(label, args.output, args.takes, CLICKS, run)
            _say(f"  Part B: restart the server on this mic (recorded, with the profiles):\n    {demo_line(label)}\n"
                 "  on /judge pick it under D > Microphone, AirPods out, ANC on, Start, lock the target.")
            for k in range(args.takes):
                wait(f"  judge take {k + 1}/{args.takes}: Enter to start the 25 s > ")
                marks["takes"].setdefault(mic, []).append(time.time())
                cue(JUDGE_CUES, TAKE_S, tick)
            wait("  target-absent take: Enter to start the 25 s > ")
            marks["absent"][mic] = time.time()
            cue(ABSENT_CUES, TAKE_S, tick)
            if mic == "iphone":
                wait(f"  dropout check: Joseph holds the iPhone (cabled, screen off) for {args.dropout_s:g} s; Enter > ")
                marks["dropout"] = time.time()
                cue(((0.0, "hold still, talk now and then"),), args.dropout_s, tick)
        _say("\n== Shift+C live calibration, once per person (report-only) ==")
        for who in ("target", "other"):
            wait(f"  {who} talks alone; press Shift+C on /judge and Enter here at the same moment > ")
            marks["calibrate"][who] = time.time()
            tick(CALIBRATE_S)
        _say("\n== Three-way listening on ONE unchanged cafe take (same mic, same AirPods, matched loudness) ==")
        for key, settings, _label in THREE_WAY:
            wait(f"  {key}: {settings}; Enter to start 25 s > ")
            t = time.time()
            tick(TAKE_S)
            marks["cafe"][key] = [t, time.time()]

    sessions = args.session or ([] if args.dry_run else rh.find_sessions(log_root, began))
    if not args.dry_run and (why := missing_sessions(sessions, log_root)):
        if not args.from_json:  # keep the marks and clicks: rerun with --from once the sessions are found
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"mic_ab_{stamp}.json").write_text(json.dumps(
                {"task": "T-171", "stamp": stamp, "began": began, "marks": marks, "latency": latency, "sessions": []},
                indent=1))
        _say(f"REFUSED to score: {why}")
        return 3
    m = measure(marks, sessions, latency, target=target, takes=args.takes, embed=embed,
                profiles_root=Path(os.environ["HEARING_PROFILES_DIR"]) if os.environ.get("HEARING_PROFILES_DIR") else None)
    folder.mkdir(parents=True, exist_ok=True)
    sheet = render(m, stamp)
    listen, key = render_listening(m, stamp)
    (folder / f"mic_ab_{stamp}.md").write_text(sheet)
    (folder / f"listening_{stamp}.md").write_text(listen)
    (folder / f"listening_{stamp}_key.md").write_text(key)
    if not args.from_json:
        (folder / f"mic_ab_{stamp}.json").write_text(json.dumps(
            {"task": "T-171", "stamp": stamp, "began": began, "marks": marks, "latency": latency,
             "sessions": [str(s) for s in sessions]}, indent=1))
    _say("\n" + sheet)
    _say(f"wrote {folder / f'mic_ab_{stamp}.md'}\nwrote {folder / f'listening_{stamp}.md'} (key: listening_{stamp}_key.md)")
    return 0 if gate(m["rows"])[0] else 1


if __name__ == "__main__":
    sys.exit(main())
