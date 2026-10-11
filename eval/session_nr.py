"""No-reference score of a recorded session of a real-room clip (T-090).

There are no clean stems for a phone recording, so the clip's manifest (``eval.real_clip``) stands
in for them: its ASD speaking timeline says when only the target talks and when only the other
person talks. Against a session recorded by the tap (``HEARING_SESSION_LOG``: ``in.wav``,
``out.wav``, ``events.jsonl``) this measures:

1. DNSMOS P.835 SIG / BAK / OVRL of the output vs the input;
2. other-talker leak: output energy / input energy during other-only segments (dB, lower = better);
3. target preserved: the same ratio during target-only segments (dB, 0 = untouched);
4. dropouts: 20 ms frames inside target-only speech where the input had signal and the output was
   digitally silent (count, total and longest run);
5. time to Isolated: first telemetry with ``acquisition.state == "locked"`` after audio started;
plus the tier timeline (every change of the playing tier).

The output is shifted by its measured lag behind the input (envelope cross-correlation, bounded
to :data:`MAX_LAG_S`) before any segment is compared, and the input is mapped onto clip time
modulo the clip length, so looped replays score every pass.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

FRAME_S = 0.02
MAX_LAG_S = 3.0  # Focus HQ plays ~1-2 s late; anything later is not this output
SILENT = 1e-4  # output frame RMS below this = digital silence (session_report's threshold)
SIGNAL = 1e-3  # input frame RMS above this = there was something to play


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return x, sr


def frame_rms(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * FRAME_S)
    n = len(x) // hop
    return np.sqrt(np.mean(np.square(x[: n * hop].reshape(n, hop), dtype=np.float64), axis=1))


def estimate_lag(x: np.ndarray, y: np.ndarray, sr: int, max_lag_s: float = MAX_LAG_S) -> int:
    """Samples ``y`` lags ``x`` by, from 20 ms log-envelope cross-correlation (0 when either is silent)."""
    ex, ey = frame_rms(x, sr), frame_rms(y, sr)
    n = min(len(ex), len(ey))
    if n < 10 or ex[:n].max() < SIGNAL or ey[:n].max() < SIGNAL:
        return 0
    ex, ey = np.log(ex[:n] + 1e-4), np.log(ey[:n] + 1e-4)
    ex, ey = ex - ex.mean(), ey - ey.mean()
    best, lag = -np.inf, 0
    for k in range(min(int(max_lag_s / FRAME_S), n - 10) + 1):
        c = float(np.dot(ex[: n - k], ey[k:]) / (n - k))
        if c > best:
            best, lag = c, k
    return int(lag * FRAME_S * sr)


def masks(manifest: dict, n_frames: int, offset_s: float = 0.0) -> dict[str, np.ndarray]:
    """Per 20 ms session frame: ``target_only`` / ``other_only`` / ``both`` from the clip timeline."""
    dur = float(manifest["duration_s"])
    t = (np.arange(n_frames) * FRAME_S + FRAME_S / 2 - offset_s) % dur
    valid = np.arange(n_frames) * FRAME_S >= offset_s
    tid = manifest["target"]["track_id"]

    def active(segs):
        a = np.zeros(n_frames, dtype=bool)
        for s, e in segs:
            a |= (t >= s) & (t < e)
        return a & valid

    target = active(next(f["speaking"] for f in manifest["faces"] if f["track_id"] == tid))
    other = np.zeros(n_frames, dtype=bool)
    for f in manifest["faces"]:
        if f["track_id"] != tid:
            other |= active(f["speaking"])
    return {"target_only": target & ~other, "other_only": other & ~target, "both": target & other}


def energy_ratio_db(y_rms: np.ndarray, x_rms: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    ex = float(np.sum(x_rms[mask] ** 2))
    ey = float(np.sum(y_rms[mask] ** 2))
    return round(10 * np.log10((ey + 1e-12) / (ex + 1e-12)), 2)


def runs(mask: np.ndarray) -> np.ndarray:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return edges[1::2] - edges[::2]


def telemetry(rows: list[dict]) -> dict:
    """Time to Isolated / locked-visual and the tier timeline, on the clock of the first audio frame."""
    start = next((r["t"] for r in rows if r["event"].get("type") == "binary" and r["event"].get("kind") == 1
                  and r["dir"] == "in"), None)
    tele = [(r["t"], r["event"]) for r in rows if r["dir"] == "out" and r["event"].get("type") == "telemetry"]
    if start is None:
        start = tele[0][0] if tele else 0.0
    first = {}
    tiers, last = [], None
    for t, e in tele:
        state = (e.get("acquisition") or {}).get("state")
        if state and state not in first:
            first[state] = round(t - start, 2)
        tier = (e.get("tiers") or {}).get("active") or e.get("tier")
        if tier != last:
            tiers.append([round(t - start, 2), tier])
            last = tier
    return {"time_to_isolated_s": first.get("locked"), "time_to_locked_visual_s": first.get("locked-visual"),
            "acquisition_first_s": first, "tier_timeline": tiers, "telemetry_ticks": len(tele)}


def score(session: Path, manifest: dict, dnsmos: str = "auto", offset_s: float = 0.0) -> dict:
    from eval.dnsmos import get_dnsmos

    session = Path(session)
    x, sr = read_wav(session / "in.wav")
    y, _ = read_wav(session / "out.wav") if (session / "out.wav").exists() else (np.zeros(0, np.float32), sr)
    lag = estimate_lag(x, y, sr)
    y_al = y[lag:]
    n = min(len(x), len(y_al))
    x_rms, y_rms = frame_rms(x[:n], sr), frame_rms(y_al[:n], sr)
    m = masks(manifest, len(x_rms), offset_s)
    drop = m["target_only"] & (y_rms < SILENT) & (x_rms > SIGNAL)
    drop_runs = runs(drop)
    scorer = get_dnsmos(dnsmos)
    dn_in, dn_out = scorer.score(x[:n], sr), scorer.score(y_al[:n], sr) if n else None
    rows = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines() if line.strip()] \
        if (session / "events.jsonl").exists() else []
    return {
        "session": session.name, "clip": manifest.get("name"), "target_track": manifest["target"]["track_id"],
        "scored_s": round(n / sr, 2), "output_lag_ms": round(1000 * lag / sr),
        "segments_s": {k: round(float(v.sum()) * FRAME_S, 2) for k, v in m.items()},
        "dnsmos": {"scorer": scorer.name,
                   "in": {k: round(getattr(dn_in, k), 2) for k in ("sig", "bak", "ovrl")},
                   "out": {k: round(getattr(dn_out, k), 2) for k in ("sig", "bak", "ovrl")} if dn_out else None},
        "other_leak_db": energy_ratio_db(y_rms, x_rms, m["other_only"]),
        "target_preserved_db": energy_ratio_db(y_rms, x_rms, m["target_only"]),
        "dropouts": {"frames": int(drop.sum()), "total_ms": int(drop.sum() * FRAME_S * 1000),
                     "longest_ms": int(drop_runs.max() * FRAME_S * 1000) if drop_runs.size else 0},
        **telemetry(rows),
    }


def format_report(r: dict) -> str:
    d = r["dnsmos"]
    out = d["out"] or {"sig": float("nan"), "bak": float("nan"), "ovrl": float("nan")}

    def fmt(v, unit="", sign="+"):
        return "n/a" if v is None else f"{v:{sign}.1f}{unit}"

    lines = [
        f"# no-reference score: session {r['session']} · clip {r['clip']} · target track {r['target_track']}",
        f"  scored {r['scored_s']} s · output lag {r['output_lag_ms']} ms · segments (s) {json.dumps(r['segments_s'])}",
        (f"1. DNSMOS ({d['scorer']})  SIG {d['in']['sig']:.2f} -> {out['sig']:.2f} · "
         f"BAK {d['in']['bak']:.2f} -> {out['bak']:.2f} · OVRL {d['in']['ovrl']:.2f} -> {out['ovrl']:.2f}"),
        f"2. other-talker leak      {fmt(r['other_leak_db'], ' dB')}  (out/in energy, other-only segments; lower is better)",
        f"3. target preserved       {fmt(r['target_preserved_db'], ' dB')}  (out/in energy, target-only segments; 0 = untouched)",
        (f"4. dropouts in target     {r['dropouts']['frames']} frames · {r['dropouts']['total_ms']} ms · "
         f"longest {r['dropouts']['longest_ms']} ms"),
        f"5. time to Isolated       {fmt(r['time_to_isolated_s'], ' s', '')}  (locked-visual at {fmt(r['time_to_locked_visual_s'], ' s', '')})",
        "   tier timeline: " + (" -> ".join(f"{t}s {tier}" for t, tier in r["tier_timeline"]) or "no telemetry"),
    ]
    return "\n".join(lines)
