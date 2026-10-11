#!/usr/bin/env python3
"""Summarize a recorded live session (see hearing/server/tap.py).

    python scripts/session_report.py [results/sessions/<dir>]   # default: newest

Prints what the engine did while a person tested: modes and suppression over time, tier
ladder usage, latency, target lock, faces, captions, errors, and an in-vs-out audio check
(level change, dropouts, clipping, length match). Meant for a human or an agent to turn
into fixes — it never changes anything.
"""
from __future__ import annotations

import itertools
import json
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np


def load(path: Path):
    rows = [json.loads(l) for l in (path / "events.jsonl").read_text().splitlines() if l.strip()]
    return rows


def wav(path: Path):
    if not path.exists():
        return None, 0
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return x, sr


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def hello_segments(rows, sr):
    """[(client, start_sample, end_sample)] of in.wav, cut at each hello (a replay or a page that
    reconnects sends a new one) by counting the audio-in frames that follow it."""
    segs, pos = [], 0
    for r in rows:
        ev = r["event"]
        if r["dir"] != "in":
            continue
        if ev.get("type") == "hello":
            segs.append([ev.get("client", "unknown"), pos, pos, int(sr * float(ev.get("frame_ms", 20)) / 1000)])
        elif ev.get("type") == "binary" and ev.get("kind") == 1 and segs:
            segs[-1][2] += segs[-1][3]
            pos = segs[-1][2]
    return [(c, a, b) for c, a, b, _ in segs if b > a]


def level_stats(a, sr):
    """Level of a signal, with exact digital zeros (every sample 0: a muted or dead stream) counted
    apart from audio that is merely quiet (20 ms frame RMS below -80 dBFS but not all zero)."""
    frame = max(1, sr // 50)
    n = len(a) // frame
    if not n:
        return {"seconds": len(a) / sr, "frames": 0, "rms_db": None, "peak_db": None, "zero_pct": 0.0, "below_80_pct": 0.0}
    f = a[: n * frame].reshape(n, frame)
    zero = ~f.any(axis=1)
    rms = np.sqrt(np.mean(f.astype(np.float64) ** 2, axis=1))
    below = (rms < 1e-4) & ~zero  # 1e-4 = -80 dBFS
    return {"seconds": len(a) / sr, "frames": n,
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-9)),
            "peak_db": float(20 * np.log10(np.abs(a).max() + 1e-9)),
            "zero_pct": float(zero.mean() * 100), "below_80_pct": float(below.mean() * 100)}


def format_level(name, st):
    if not st["frames"]:
        return f"  {name}: {st['seconds']:.2f} s (shorter than 20 ms)"
    return (f"  {name}: {st['seconds']:6.2f} s  rms {st['rms_db']:6.1f} dBFS  peak {st['peak_db']:6.1f} dBFS  "
            f"exact digital zero {st['zero_pct']:.1f}%  below -80 dBFS (not zero) {st['below_80_pct']:.1f}%")


def main(argv):
    root = Path("results/sessions")
    if len(argv) > 1:
        d = Path(argv[1])
    else:
        sessions = [p for p in root.glob("*/") if (p / "events.jsonl").exists()]
        if not sessions:
            sys.exit(f"no recorded session under {root}/: start the server with HEARING_SESSION_LOG={root} (docs/TESTING_GUIDE.md section 4)")
        d = max(sessions, key=lambda p: p.stat().st_mtime)
    rows = load(d)
    print(f"# session {d.name}  ({rows[-1]['t']:.1f} s, {len(rows)} wire events)")
    producer = next((r["event"] for r in rows if r["dir"] == "out"
                     and r["event"].get("git_rev")), None)
    if producer:
        print(f"  server git_rev {producer['git_rev']} | "
              f"started_at {producer.get('started_at', 'unknown')} | "
              f"version {producer.get('version', 'unknown')}")

    ins = [r for r in rows if r["dir"] == "in" and r["event"].get("type") != "binary"]
    outs = [r["event"] for r in rows if r["dir"] == "out" and r["event"].get("type") != "binary"]
    bins = [r for r in rows if r["event"].get("type") == "binary"]
    tele = [dict(r["event"], t=r["t"]) for r in rows
            if r["dir"] == "out" and r["event"].get("type") == "telemetry"]
    caps = [e for e in outs if e.get("type") == "caption"]
    errs = [e for e in outs if e.get("type") in ("error", "degraded")]
    other = Counter(e.get("type") for e in outs if e.get("type") not in ("telemetry", "caption", "error", "degraded"))

    print("\n## client")
    # T-121: the page's hello carries the mic/output it used (the last non-null one wins; Stop sends null)
    devs = {k: next((r["event"][k] for r in reversed(ins) if r["event"].get("type") == "hello" and r["event"].get(k)), None)
            for k in ("input_device", "output_device")}
    print(f"  mic: {devs['input_device'] or 'not reported'} | output: {devs['output_device'] or 'not reported'}")
    for r in ins:
        print(f"  t={r['t']:7.2f}  {json.dumps(r['event'])[:160]}")

    print("\n## traffic")
    kinds = Counter(r["event"]["kind"] for r in bins)
    print(f"  audio in frames {kinds.get(1,0)} · audio out frames {kinds.get(3,0)} · video frames {kinds.get(2,0)}")
    if not kinds.get(1) and rows[-1]["t"] > 10:  # T-158: a page that sat connected without Start
        clients = sorted({str(r["event"].get("client", "unknown")) for r in ins if r["event"].get("type") == "hello"})
        print(f"  WARNING: 0 audio frames in over {rows[-1]['t']:.0f} s (client {', '.join(clients) or 'none'}): "
              "the page never streamed its mic. Press Start, allow the mic, pick MacBook Pro Microphone, "
              "and check the level meter moves")
    if kinds.get(1) and kinds.get(3) is not None:
        print(f"  out/in frame ratio {kinds.get(3,0)/kinds[1]:.3f}  (≈1.0 expected once the window fills)")
    # seq counts per media kind (CONTRACTS.md#events, browser clients and hearing.replay since
    # T-072). Older replays shared one counter across audio and video: video seqs then fill the
    # audio stream's "gaps", which are not lost frames.
    in_seqs = {k: [r["event"]["seq"] for r in bins if r["event"]["kind"] == k] for k in (1, 2)}
    for kind, other_kind, label in ((1, 2, "audio-in"), (2, 1, "video-in")):
        seqs, others = in_seqs[kind], set(in_seqs[other_kind])
        gaps = [(a, b) for a, b in itertools.pairwise(seqs) if b - a != 1]
        if gaps and all(set(range(a + 1, b)) <= others for a, b in gaps):
            print(f"  {label} seq gaps: {len(gaps)}, all filled by the other kind's seqs "
                  "(shared counter, pre-T-072 replay): no frames lost")
        elif gaps:
            lost = sum(b - a - 1 for a, b in gaps)
            print(f"  !! {label} seq gaps: {len(gaps)} (total {lost} frames lost before the server)")
    if other:
        print(f"  other events: {dict(other)}")

    print("\n## telemetry")
    if tele:
        keys = sorted({k for e in tele for k in e})
        print(f"  fields: {keys}")
        for key in ("mode", "effective_mode", "pipeline", "tier", "transition", "hq_backend", "hq_active", "hq_reason", "device"):
            vals = Counter(str(e.get(key)) for e in tele if key in e)
            if vals:
                print(f"  {key}: {dict(vals.most_common(6))}")
        lat_keys = sorted({k for e in tele for k in (e.get("latency_ms") or {}) if isinstance(e.get("latency_ms"), dict)})
        for key in lat_keys:
            v = np.array([e["latency_ms"][key] for e in tele if isinstance(e.get("latency_ms"), dict) and isinstance(e["latency_ms"].get(key), (int, float))], dtype=float)
            if len(v):
                print(f"  latency {key}: p50 {pct(v,50):.1f}  p90 {pct(v,90):.1f}  max {v.max():.1f} ms")
        for key in ("remote_rtt_ms",):
            v = np.array([e[key] for e in tele if isinstance(e.get(key), (int, float))], dtype=float)
            if len(v):
                print(f"  {key}: p50 {pct(v,50):.1f}  p90 {pct(v,90):.1f}  max {v.max():.1f}")
        vis = [e.get("vision") for e in tele if e.get("vision")]
        if vis:
            last = vis[-1]
            frames, dropped = last.get("frames") or 0, last.get("dropped") or 0
            live_fps = [v["fps"] for v in vis if v.get("status") == "live" and v.get("fps")]
            print(f"  vision (last): {last.get('status')} {last.get('detector')}/{last.get('asd')} · "
                  f"fps {last.get('fps')} (live p50 {pct(live_fps, 50):.1f}) · frames {frames} · "
                  f"dropped {dropped} ({100.0 * dropped / max(frames, 1):.1f} %) · "
                  f"lip_only {last.get('lip_only')} · asd_skipped {last.get('asd_skipped')}")
        if all(e.get("seq") is None for e in tele):
            print("  !! no audio frame ever reached the engine (seq null in every tick): the client never "
                  "streamed its microphone (Start not pressed?), so no pipeline ran")
        gpu = [e for e in tele if isinstance(e.get("remote"), dict)]
        if gpu:  # T-156: what the GPU played, per tick
            src = Counter(e["remote"].get("source") for e in gpu)
            played = Counter(e["remote"].get("remote_tier") for e in gpu if e["remote"].get("source") == "remote")
            rtt = [e["remote"]["rtt_ms"].get("p50") for e in gpu if (e["remote"].get("rtt_ms") or {}).get("p50") is not None]
            last = gpu[-1]["remote"]
            print(f"  GPU: ticks with remote stats {len(gpu)}/{len(tele)} · source {dict(src)} · GPU tier {dict(played)} · "
                  f"rtt p50 {pct(rtt, 50):.1f} ms · hops {last.get('remote_hops')} · misses {last.get('misses')} · "
                  f"device {last.get('remote_device')} · remote latency {last.get('latency_ms')} ms")
        tiers = [e.get("tiers") for e in tele if e.get("tiers")]
        if tiers:
            print(f"  tiers (last): {json.dumps(tiers[-1])[:300]}")
            playing = Counter((e["tiers"].get("active"), e["tiers"].get("delay_ms")) for e in tele if e.get("tiers"))
            print(f"  playing tier @ reported delay_ms: {dict(playing)}")
        acq = [(e.get("t"), e["acquisition"]) for e in tele if e.get("acquisition")]
        if acq:
            isolated = next((t for t, a in acq if a.get("state") == "locked"), None)
            print(f"  acquisition (last): {acq[-1][1].get('state')} '{acq[-1][1].get('detail')}' · "
                  f"first isolated at t={isolated if isolated is None else round(isolated, 1)}")
        stage_ms = Counter()
        for e in tele:
            for k, v in (e.get("stages") or {}).items():
                if isinstance(v, (int, float)):
                    stage_ms[k] += v
        if stage_ms:
            print("  per-stage ms (mean): " + ", ".join(f"{k} {v/len(tele):.1f}" for k, v in stage_ms.most_common()))
        quiet = [e["quiet"] for e in tele if isinstance(e.get("quiet"), dict)]
        if quiet:  # T-149: Quiet's caption feed (focus = the isolated target, paused = captions held)
            print(f"  quiet: ticks {len(quiet)} · feed {dict(Counter(q.get('feed') for q in quiet))} · "
                  f"reasons {dict(Counter(q.get('reason') for q in quiet).most_common(3))}")
        tgt = [e.get("target") for e in tele if e.get("target")]
        faces = [len(e.get("faces") or []) for e in tele]
        if faces:
            print(f"  faces per tick: {dict(Counter(faces))}")
        if tgt:
            locked = sum(1 for t in tgt if t.get("locked"))
            conf = np.array([t.get("confidence", 0) for t in tgt], dtype=float)
            ids = Counter(t.get("track_id") for t in tgt)
            print(f"  target: ticks with target {len(tgt)}/{len(tele)} · locked {locked} · conf p50 {pct(conf,50):.2f} · track ids {dict(ids)}")
        else:
            print(f"  target: never chosen ({len(tele)} ticks)")
        speak = [f.get("p_speaking") for e in tele for f in (e.get("faces") or []) if f.get("p_speaking") is not None]
        if speak:
            s = np.array(speak, dtype=float)
            print(f"  p_speaking: p50 {pct(s,50):.2f} p90 {pct(s,90):.2f}  (>0.5 in {np.mean(s>0.5)*100:.0f}% of face reports)")
        voice = [e.get("voice") for e in tele if e.get("voice")]
        if voice:
            print(f"  voice: {Counter(json.dumps(v, sort_keys=True) for v in voice).most_common(3)}")
        print("\n  timeline (every ~2 s):")
        last = -9.0
        prev_sig, prev_lat, skipped = None, None, 0
        for e in tele:
            if e["t"] - last < 2:
                continue
            last = e["t"]
            lat = e.get("latency_ms") or {}
            row = {"t": round(e["t"], 1), "mode": e.get("effective_mode") or e.get("mode"), "tier": e.get("tier"),
                   "lat": lat.get("total") if isinstance(lat, dict) else lat, "srv": lat.get("server") if isinstance(lat, dict) else None}
            if e.get("transition"):
                row["transition"] = e["transition"]
            tg = e.get("target") or {}
            row["target"] = f"{tg.get('track_id')}/{'L' if tg.get('locked') else 'u'}/{tg.get('confidence',0):.2f}" if tg else "-"
            row["faces"] = len(e.get("faces") or [])
            if isinstance(e.get("remote"), dict):  # T-156: the GPU tier timeline
                row["gpu"] = f"{e['remote'].get('source')}/{e['remote'].get('remote_tier')}"
            sig = json.dumps({k: v for k, v in row.items() if k not in ("t", "lat", "srv")})
            if sig == prev_sig and abs((row["lat"] or 0) - (prev_lat or 0)) < 5:
                skipped += 1
                continue
            if skipped:
                print(f"    ... {skipped} unchanged ticks")
                skipped = 0
            prev_sig, prev_lat = sig, row["lat"]
            print("   ", json.dumps(row))
        if skipped:
            print(f"    ... {skipped} unchanged ticks")
    else:
        print("  none")

    print(f"\n## captions ({len(caps)})")
    for c in caps[:40]:
        print(f"  {c}")

    print(f"\n## errors/degradations ({len(errs)})")
    for e in Counter(json.dumps(e, sort_keys=True) for e in errs).most_common(10):
        print(f"  x{e[1]} {e[0][:200]}")

    print("\n## audio")
    x, sr = wav(d / "in.wav")
    y, _ = wav(d / "out.wav")
    if x is None or not len(x):
        print("  no input audio")
        return
    def stats(name, a):
        rms = float(np.sqrt(np.mean(a**2))) if len(a) else 0
        clip = float(np.mean(np.abs(a) > 0.99)) if len(a) else 0
        frame = sr // 50
        n = len(a) // frame
        f = a[: n * frame].reshape(n, frame)
        frms = np.sqrt(np.mean(f**2, axis=1))
        print(f"  {name}: {len(a)/sr:6.2f} s  rms {20*np.log10(rms+1e-9):6.1f} dBFS  peak {20*np.log10(np.abs(a).max()+1e-9):6.1f} dBFS  clipped {clip*100:.2f}%  exact-zero 20ms frames {level_stats(a, sr)['zero_pct']:.1f}%  below -80 dBFS (not zero) {level_stats(a, sr)['below_80_pct']:.1f}%")
        return frms
    fx = stats("in ", x)
    segs = hello_segments(rows, sr)
    if len(segs) > 1:
        print("  input by client (hello) segment:")
        for i, (client, a, b) in enumerate(segs):
            print(format_level(f"  #{i} {client}", level_stats(x[a:b], sr)))
    if y is not None and len(y):
        fy = stats("out", y)
        n = min(len(fx), len(fy))
        if n:
            active = fx[:n] >= pct(fx[:n], 30)
            red = 20 * np.log10((fy[:n][active] + 1e-6) / (fx[:n][active] + 1e-6))
            print(f"  level change on active frames: median {np.median(red):+.1f} dB  (p10 {pct(red,10):+.1f}, p90 {pct(red,90):+.1f})")
            drop = (fy[:n] < 1e-4) & (fx[:n] > 1e-3)
            if drop.any():
                runs = np.diff(np.flatnonzero(np.diff(np.r_[0, drop.astype(int), 0])))[::2]
                print(f"  !! output dropouts: {int(drop.sum())} frames where input had signal but output was silent; longest run {runs.max()*20} ms")
        print(f"  length delta out-in: {(len(y)-len(x))/sr*1000:+.0f} ms")
    else:
        print("  !! no output audio recorded")


if __name__ == "__main__":
    main(sys.argv)
