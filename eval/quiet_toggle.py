"""T-149: Focus -> Quiet -> Focus toggles on a replayed clip, scored from what the client received.

    .venv/bin/python -m eval.quiet_toggle [<clip dir>] [--toggles 10] [--quiet-s 3] [--focus-s 3]

Starts ``hearing serve`` (session log on, ``HEARING_DEMO_PRESET=team`` and the fixture voice profiles
unless already set), replays the clip in a loop with the manifest's target locked (oracle) under the
judge profile, waits for Isolated, then toggles. Per toggle: the target id, the output RMS from
``SILENT_AFTER_S`` after the Quiet press to the Focus press, the first final caption after the Quiet
press, Focus -> Isolated, and each Quiet final's attribution (its time span vs the manifest's speaking
intervals, shifted back by the ladder delay: words spoken while only the other talker spoke count as
theirs). Prints a table and writes ``$HEARING_RESULTS_DIR/quiet_toggle_<stamp>.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hearing.audio.io import pcm16_to_float
from hearing.server.protocol import unpack_frame

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(os.environ.get("HEARING_FIXTURES_DIR", ROOT / "eval/fixtures/generated"))
DEFAULT_CLIP = FIXTURES / "t141/aryan/demo-team-easy/clip.mp4"
SILENT_AFTER_S = 0.5  # acceptance: <= SILENT_DBFS from this long after the Quiet press
SILENT_DBFS = -60.0
ISOLATED_BACK_S = 2.0  # acceptance: back in Focus, Isolated within this
FIRST_CAPTION_P50_S = 3.0
OTHER_WORDS_MAX = 0.10
ISOLATING = ("enrolled", "focus_hq")  # hearing.server.session.ISOLATING_TIERS
ISOLATED_WAIT_S = 25.0  # first Isolated before the toggles start


def _dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 1e-10)) if len(x) else -200.0


def _overlap(a0: float, a1: float, spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(a1, b1) - max(a0, b0)) for b0, b1 in spans)


def attribute(t0: float, t1: float, target: list, other: list, duration: float) -> str:
    """``target`` | ``other`` | ``none`` for a caption spanning clip time [t0, t1] (looped)."""
    a0, a1 = t0 % duration, t0 % duration + max(0.0, t1 - t0)
    spans_t = [(s, e) for s, e in target] + [(s + duration, e + duration) for s, e in target]
    spans_o = [(s, e) for s, e in other] + [(s + duration, e + duration) for s, e in other]
    only_o = []  # other talker speaking while the target is silent
    for s, e in spans_o:
        cuts = sorted((max(s, ts), min(e, te)) for ts, te in spans_t if te > s and ts < e)
        cur = s
        for c0, c1 in cuts:
            if c0 > cur:
                only_o.append((cur, c0))
            cur = max(cur, c1)
        if cur < e:
            only_o.append((cur, e))
    ot, oo = _overlap(a0, a1, spans_t), _overlap(a0, a1, only_o)
    if ot == 0 and oo == 0:
        return "none"
    return "other" if oo > ot else "target"


async def session(url: str, clip, manifest: dict, args) -> dict:
    import websockets

    from hearing.replay import feed
    from scripts.replay import OracleLock

    oracle = OracleLock({"target": manifest.get("target")})
    log: dict = {"audio": [], "telemetry": [], "captions": [], "presses": [], "errors": []}
    t_start = time.monotonic()

    def now() -> float:
        return time.monotonic() - t_start

    async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
        async def send(message):
            await ws.send(json.dumps(message) if isinstance(message, dict) else message)

        async def drain():
            async for message in ws:
                t = now()
                if isinstance(message, bytes):
                    f = unpack_frame(message)
                    if f.kind == 3:
                        log["audio"].append((t, pcm16_to_float(f.payload)))
                    continue
                m = json.loads(message)
                if m.get("type") == "telemetry":
                    log["telemetry"].append((t, {k: m.get(k) for k in (
                        "effective_mode", "target", "acquisition", "quiet", "synth_feed", "tiers", "hq_active",
                        "enrollment")}))
                elif m.get("type") == "caption":
                    log["captions"].append((t, m))
                elif m.get("type") == "error":
                    log["errors"].append((t, m))
                control = oracle.on_message(m)
                if control:
                    await send(control)

        def isolated(since: float) -> float | None:
            return next((t for t, m in log["telemetry"] if t >= since and is_isolated(m)), None)

        receiver = asyncio.create_task(drain())
        control = {"mode": "focus_hq", "suppression": 100, "profile": "judge"}
        player = asyncio.create_task(feed(clip, send, loop=True, control=control))
        try:
            deadline = now() + ISOLATED_WAIT_S
            while isolated(0.0) is None and now() < deadline:
                await asyncio.sleep(0.1)
            log["first_isolated_s"] = isolated(0.0)
            for _ in range(args.toggles):
                log["presses"].append(("quiet", now()))
                await send({"type": "control", "mode": "quiet"})
                await asyncio.sleep(args.quiet_s)
                log["presses"].append(("focus_hq", now()))
                await send({"type": "control", "mode": "focus_hq"})
                await asyncio.sleep(args.focus_s)
        finally:
            player.cancel()
            await asyncio.sleep(0.5)
            receiver.cancel()
    log["epoch_offset"] = None
    return log


def is_isolated(m: dict) -> bool:
    """Focus playing the target: the pill's Isolated, or (camera-free, a voice-profile target) the
    playing tier is an isolating one (the clip's faces may not be detected)."""
    if m.get("effective_mode") != "focus_hq":
        return False
    return ((m.get("acquisition") or {}).get("state") == "isolated"
            or (m.get("tiers") or {}).get("active") in ISOLATING)


def identity(m: dict) -> tuple:
    """Who the target is: the locked face and the enrolled voice (profile name, or a learned clip)."""
    e = m.get("enrollment") or {}
    return ((m.get("target") or {}).get("track_id"), e.get("profile") or e.get("source"), e.get("state"))


def score(log: dict, manifest: dict) -> dict:
    duration = float(manifest["duration_s"])
    faces = {f["role"]: [tuple(s) for s in f["speaking"]] for f in manifest["faces"]}
    presses = log["presses"]
    tele = log["telemetry"]
    rows = []
    first_cap_epoch = None
    for i in range(0, len(presses) - 1, 2):
        tq, tf = presses[i][1], presses[i + 1][1]
        t_next = presses[i + 2][1] if i + 2 < len(presses) else tf + 3.0
        silent = [y for t, y in log["audio"] if tq + SILENT_AFTER_S <= t < tf]
        quiet_ticks = [m for t, m in tele if tq <= t < tf]
        targets = {identity(m) for t, m in tele if tq - 0.5 <= t < t_next}
        back = next((t for t, m in tele if t >= tf and is_isolated(m)), None)
        finals = [(t, c) for t, c in log["captions"] if tq <= t < tf + 1.0 and c.get("final")]
        delay = statistics.median([float((m.get("tiers") or {}).get("delay_ms") or 0.0) for m in quiet_ticks] or [0.0])
        words = {"target": 0, "other": 0, "none": 0}
        for t, c in finals:
            n = len(c.get("text", "").split())
            if first_cap_epoch is None and c["t0"] > 1e6:
                first_cap_epoch = True
            t0, t1 = float(c["t0"]), float(c["t1"])
            base = log.get("epoch") or 0.0
            who = attribute(t0 - base - delay / 1000.0, t1 - base - delay / 1000.0, faces.get("target", []),
                            faces.get("other", []), duration)
            words[who] += n
        feeds = [((m.get("quiet") or {}).get("feed")) for m in quiet_ticks]
        rows.append({
            "toggle": i // 2 + 1,
            "targets": sorted(map(str, targets)),
            "quiet_out_dbfs": round(_dbfs(np.concatenate(silent)), 1) if silent else None,
            "quiet_out_peak_frame_dbfs": round(max((_dbfs(y) for y in silent), default=-200.0), 1),
            "first_final_s": round(finals[0][0] - tq, 2) if finals else None,
            "back_isolated_s": round(back - tf, 2) if back is not None else None,
            "words": words,
            "feed_focus_ticks": feeds.count("focus"),
            "feed_paused_ticks": feeds.count("paused"),
            "delay_ms": delay,
            "finals": [c["text"] for _, c in finals],
        })
    ids = {str(identity(m)) for t, m in tele if presses and t >= presses[0][1]}
    total = sum(sum(r["words"].values()) for r in rows)
    other = sum(r["words"]["other"] for r in rows)
    firsts = [r["first_final_s"] for r in rows if r["first_final_s"] is not None]
    return {
        "first_isolated_s": log.get("first_isolated_s"),
        "toggles": len(rows),
        "targets_after_first_press": sorted(ids),
        "target_unchanged": sum(1 for r in rows if len(r["targets"]) == 1),
        "quiet_silent": sum(1 for r in rows if r["quiet_out_dbfs"] is not None and r["quiet_out_peak_frame_dbfs"] <= SILENT_DBFS),
        "back_isolated": sum(1 for r in rows if r["back_isolated_s"] is not None and r["back_isolated_s"] <= ISOLATED_BACK_S),
        "caption_words": total,
        "other_words": other,
        "other_share": round(other / total, 3) if total else None,
        "first_final_p50_s": round(statistics.median(firsts), 2) if firsts else None,
        "toggles_with_caption": len(firsts),
        "errors": [m for _, m in log["errors"]],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    from hearing.replay import Clip
    from scripts import demo_regress as dr

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", nargs="?", type=Path, default=DEFAULT_CLIP)
    ap.add_argument("--toggles", type=int, default=10)
    ap.add_argument("--quiet-s", type=float, default=3.0)
    ap.add_argument("--focus-s", type=float, default=3.0)
    ap.add_argument("--url", help="an already running server (ws://...); default: start one")
    args = ap.parse_args(argv)
    manifest = json.loads((args.clip.parent / "manifest.json").read_text())
    clip = Clip.load(args.clip)
    os.environ.setdefault("HEARING_DEMO_PRESET", "team")
    if (FIXTURES / "profiles").is_dir():
        os.environ.setdefault("HEARING_PROFILES_DIR", str(FIXTURES / "profiles"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"quiet-toggle-{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = None
    url = args.url
    if url is None:
        proc, port = dr.start_server(log_dir, log_dir / "server.log")
        url = f"ws://127.0.0.1:{port}/ws/session"
    try:
        epoch = time.time()  # feed() stamps capture_ts from time.time() at its start (+ clip time)
        log = asyncio.run(session(url, clip, manifest, args))
        log["epoch"] = epoch
    finally:
        if proc is not None:
            dr.stop_server(proc)
    result = score(log, manifest)
    result.update(clip=str(args.clip), log_dir=str(log_dir), host=os.uname().nodename,
                  preset=os.environ.get("HEARING_DEMO_PRESET"), load=os.getloadavg())
    for r in result["rows"]:
        print(f"#{r['toggle']:2d} ids {r['targets']} quiet out {r['quiet_out_dbfs']} dBFS (peak frame "
              f"{r['quiet_out_peak_frame_dbfs']}) first final {r['first_final_s']} s back->Isolated "
              f"{r['back_isolated_s']} s words {r['words']} feed focus/paused {r['feed_focus_ticks']}/"
              f"{r['feed_paused_ticks']} {r['finals']}")
    summary = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(summary, indent=1))
    out = Path(os.environ.get("HEARING_RESULTS_DIR", ROOT / "eval/results")) / f"quiet_toggle_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
