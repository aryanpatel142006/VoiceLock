"""Enhanced Replay research (T-195): does a longer TSE context improve the same raw take?

    swarm-lock --exclusive -- .venv/bin/python -m eval.enhanced_replay            # all takes, one run
    .venv/bin/python -m eval.enhanced_replay --takes t141:demo-team-easy --contexts 1,full

Every take is re-rendered **offline** with the enrolled/profile candidate (``tse:enrolled``, the
saved ``aryan`` profile as target and ``joseph`` as the known negative, standard isolation) through
:class:`hearing.tse.SlidingWindowExtractor` at window = context (1 / 5 / 15 s, hop = window / 2) and
one window over the whole take (``full``), each with and without the existing post-filter (the
``dfn`` stage of ``enrolled_dfn``'s ``tse>dfn>mixer``; mixer at s = 1 is the identity). Takes:

* ``t141:<scene>``: T-141 team scenes (``$HEARING_FIXTURES_DIR/t141/aryan/<scene>``) with close-talk
  stems, so leak dB, SI-SDRi and true dropouts are measured against references. No live recording
  exists for them: **live** is the live window (balanced preset, 1.0 s / 0.5 s), the ``1`` row.
* ``session:<stamp>``: today's judge sessions (``results/sessions/<stamp>/in.wav`` = raw take,
  ``out.wav`` = what was heard live, delay-aligned here). No stems: leak dB / SI-SDRi are n/a and
  speaker routing is the ECAPA proxy only. A ``--excerpt-s`` crop where the live ladder played the
  enrolled / focus_hq tier the most keeps the render bounded; ``full`` means the whole excerpt.

Scores per render: ``leak_db`` (least-squares stem gains: other x rms(other) over target x
rms(target), in the overlap section), ``si_sdri`` (vs the clean target stem), DNSMOS P.835
(SIG/BAK/OVRL), ``dropouts`` (0.25 s windows where the target is active - stem within 20 dB of its
speech level, or for sessions the input's ECAPA proxy says the target - but the output is more than
20 dB below its own active level, in runs), ``wrong_pct`` (ECAPA proxy: % of voiced 1.5 s output
windows closer to the other profile than to the target's) and wall-clock render seconds. Each
render writes ``manifest.json`` (input path + sha256, config, context, device, render s); every take
exports level-matched (-26 dBFS active RMS) ``raw`` / ``LIVE`` / ``REPLAY_ENHANCED_<ctx>`` wavs for
headphone A/B. Everything lands in ``<main checkout>/results/enhanced_<stamp>/`` (gitignored). These
files are a research REPLAY, not live output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
CONTEXTS = ("1", "5", "15", "full")
LIVE_WINDOW = (1.0, 0.5)  # latency_preset balanced (hearing/latency_preset.py): the shipped Mac window
WIN_S = 0.25
ACTIVE_DB = -20.0
DROP_DB = -20.0
PROXY_WIN_S, PROXY_HOP_S = 1.5, 0.75
PROXY_VOICED_DB = -30.0  # a proxy window is voiced when within 30 dB of the loudest window
MATCH_DBFS = -26.0
DEFAULT_TAKES = ("t141:demo-team-easy", "t141:demo-team-hard", "t141:demo-team-same",
                 "session:20261009T191504Z-2", "session:20261009T203531Z-2", "session:20261009T211905Z-8")


def main_checkout() -> Path:
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return x.mean(axis=1)


def frame_rms(x: np.ndarray, win_s: float = WIN_S) -> np.ndarray:
    hop = int(SR * win_s)
    n = len(x) // hop
    return np.sqrt(np.mean(np.square(x[: n * hop].reshape(n, hop), dtype=np.float64), axis=1))


def db(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-12)))


def active_level(x: np.ndarray) -> float:
    """RMS over 0.25 s windows within 20 dB of the loudest decile (the speech level)."""
    r = frame_rms(x)
    if not r.size or r.max() <= 1e-9:
        return 0.0
    ref = np.percentile(r, 90)
    act = r[r >= ref * 10 ** (ACTIVE_DB / 20)]
    return float(np.sqrt(np.mean(act**2)))


def level_match(x: np.ndarray, dbfs: float = MATCH_DBFS) -> np.ndarray:
    lvl = active_level(x)
    if lvl <= 1e-9:
        return x
    y = x * (10 ** (dbfs / 20) / lvl)
    peak = float(np.abs(y).max())
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def stem_gains(y: np.ndarray, stems: list[np.ndarray]) -> np.ndarray:
    """Least-squares gains of ``y ~ sum g_i stems_i``."""
    a = np.stack(stems, axis=1).astype(np.float64)
    g, *_ = np.linalg.lstsq(a, y.astype(np.float64), rcond=None)
    return g


def leak_db(y: np.ndarray, target: np.ndarray, other: np.ndarray, noise: np.ndarray | None = None) -> float:
    """Other talker's level in ``y`` relative to the target's (dB; lower is better)."""
    stems = [target, other] + ([noise] if noise is not None else [])
    g = stem_gains(y, stems)
    rt = np.sqrt(np.mean(target.astype(np.float64) ** 2))
    ro = np.sqrt(np.mean(other.astype(np.float64) ** 2))
    return db(abs(g[1]) * ro) - db(abs(g[0]) * rt)


def runs(mask: np.ndarray) -> int:
    m = np.asarray(mask, bool).astype(np.int8)
    return int(np.sum(np.diff(np.concatenate([[0], m])) == 1))


def dropouts(y: np.ndarray, active: np.ndarray) -> tuple[int, float]:
    """(runs, seconds) of target-active 0.25 s windows where ``y`` is > 20 dB below its active level."""
    r = frame_rms(y)
    n = min(len(r), len(active))
    r, active = r[:n], np.asarray(active[:n], bool)
    lvl = active_level(y)
    if lvl <= 1e-9:
        return runs(active), float(active.sum() * WIN_S)
    drop = active & (r < lvl * 10 ** (DROP_DB / 20))
    return runs(drop), float(drop.sum() * WIN_S)


def stem_active(stem: np.ndarray) -> np.ndarray:
    r = frame_rms(stem)
    return r >= active_level(stem) * 10 ** (ACTIVE_DB / 20)


def proxy_windows(x: np.ndarray, embed, t_emb: np.ndarray, o_emb: np.ndarray) -> list[dict]:
    """ECAPA cos to target / other per voiced 1.5 s window (hop 0.75 s)."""
    w, h = int(PROXY_WIN_S * SR), int(PROXY_HOP_S * SR)
    starts = range(0, max(len(x) - w, 0) + 1, h)
    rms = [float(np.sqrt(np.mean(x[s:s + w].astype(np.float64) ** 2))) for s in starts]
    top = max(rms) if rms else 0.0
    out = []
    for s, r in zip(starts, rms, strict=True):
        if top <= 1e-9 or r < top * 10 ** (PROXY_VOICED_DB / 20):
            continue
        e = embed(x[s:s + w])
        out.append({"t": s / SR, "cos_t": float(t_emb @ e), "cos_o": float(o_emb @ e)})
    return out


def wrong_pct(windows: list[dict]) -> float | None:
    if not windows:
        return None
    return 100.0 * sum(w["cos_o"] > w["cos_t"] for w in windows) / len(windows)


def proxy_active(windows: list[dict], n: int) -> np.ndarray:
    """0.25 s target-active mask from the input's proxy windows (target closer than the other)."""
    m = np.zeros(n, bool)
    per = round(PROXY_HOP_S / WIN_S)
    for w in windows:
        if w["cos_t"] > w["cos_o"]:
            i = round(w["t"] / WIN_S) + per // 2
            m[i:i + per] = True
    return m


class WithOther:
    """``EnrolledCandidate`` with a fixed negative enrollment, as the tse stage passes it."""

    def __init__(self, cand, other: np.ndarray):
        self.cand, self.other = cand, other
        self.name, self.licence, self.needs = cand.name, cand.licence, frozenset({"enroll"})

    @property
    def device(self):
        return self.cand.device

    def load(self):
        self.cand.load()

    def extract(self, mix, sr, *, enroll=None, **_):
        return self.cand.extract(mix, sr, enroll=enroll, other=self.other)


def session_excerpt(session: Path, excerpt_s: float) -> tuple[float, float]:
    """Start/end (s) of the ``excerpt_s`` crop where the live ladder played enrolled / focus_hq most."""
    t0, ticks = None, []
    with open(session / "events.jsonl") as f:
        for line in f:
            r = json.loads(line)
            e = r["event"]
            if r["dir"] == "in" and e.get("type") == "binary" and t0 is None:
                t0 = r["t"]
            if r["dir"] == "out" and e.get("type") == "telemetry" and t0 is not None:
                ticks.append((r["t"] - t0, e.get("tier") in ("enrolled", "focus_hq")))
    dur = sf.info(str(session / "in.wav")).duration
    if dur <= excerpt_s:
        return 0.0, dur
    best, best_start = -1, 0.0
    for start in np.arange(0.0, dur - excerpt_s, 1.0):
        n = sum(ok for t, ok in ticks if start <= t < start + excerpt_s)
        if n > best:
            best, best_start = n, float(start)
    return best_start, best_start + excerpt_s


def load_take(spec: str, root: Path, fixtures: Path, excerpt_s: float) -> dict:
    kind, name = spec.split(":", 1)
    if kind == "t141":
        d = fixtures / "t141" / "aryan" / name
        man = json.loads((d / "manifest.json").read_text())
        st = {k: load(d / v) for k, v in man["stems"].items()}
        ov = next((s for s in man.get("sections", []) if s["name"] == "overlap"), None)
        return {"id": spec, "kind": kind, "path": d / man["stems"]["mix"], "mix": st["mix"],
                "live": None, "stems": st, "overlap": (ov["start_s"], ov["end_s"]) if ov else None,
                "crop_s": [0.0, len(st["mix"]) / SR]}
    if kind == "session":
        d = root / "results" / "sessions" / name
        raw, live = load(d / "in.wav"), load(d / "out.wav")
        from eval.session_nr import estimate_lag

        lag = estimate_lag(raw, live, SR)
        live = np.concatenate([live[lag:], np.zeros(lag, np.float32)])
        a, b = session_excerpt(d, excerpt_s)
        i, j = int(a * SR), int(b * SR)
        return {"id": spec, "kind": kind, "path": d / "in.wav", "mix": raw[i:j], "live": live[i:j],
                "live_lag_ms": 1000.0 * lag / SR, "stems": None, "overlap": None, "crop_s": [a, b]}
    raise ValueError(f"unknown take {spec!r} (t141:<scene> or session:<stamp>)")


def render(cand_w, mix: np.ndarray, enroll: np.ndarray, ctx: str) -> tuple[np.ndarray, float, tuple]:
    from hearing.tse.windowed import SlidingWindowExtractor

    if ctx == "full":
        window, hop = len(mix) / SR, len(mix) / SR
    elif ctx == "1":
        window, hop = LIVE_WINDOW
    else:
        window = float(ctx)
        hop = window / 2
    ext = SlidingWindowExtractor(cand_w, window_s=window, hop_s=hop)
    cand_w.cand.reset()
    t = time.perf_counter()
    y = ext.extract(mix, SR, enroll=enroll)
    return y, time.perf_counter() - t, (window, hop)


def postfilter(y: np.ndarray) -> tuple[np.ndarray, float]:
    from eval.bench import build, run_stream

    t = time.perf_counter()
    out = run_stream(build("dfn"), y, SR).output
    return np.asarray(out[: len(y)], np.float32), time.perf_counter() - t


def score(y, take, scorer, embed, t_emb, o_emb, active) -> dict:
    row: dict = {}
    st = take["stems"]
    if st is not None:
        sl = slice(None)
        if take["overlap"]:
            sl = slice(int(take["overlap"][0] * SR), int(take["overlap"][1] * SR))
        row["leak_db"] = leak_db(y[sl], st["target"][sl], st["other"][sl], st["noise"][sl])
        from eval.metrics import si_sdri

        row["si_sdri"] = float(si_sdri(y, take["mix"], st["target"]))
    d = scorer.score(y, SR)
    row.update(dnsmos_sig=d.sig, dnsmos_bak=d.bak, dnsmos_ovrl=d.ovrl)
    row["dropouts"], row["dropout_s"] = dropouts(y, active)
    row["wrong_pct"] = wrong_pct(proxy_windows(y, embed, t_emb, o_emb))
    return row


def run(args) -> dict:
    from eval.dnsmos import get_dnsmos
    from hearing.tse.enroll import profile_audio_dir
    from hearing.tse.enrolled import EnrolledCandidate, _unit_embedding

    root = main_checkout()
    fixtures = Path(os.environ.get("HEARING_FIXTURES_DIR", root / "eval" / "fixtures" / "generated"))
    pdir = Path(args.profiles or fixtures / "profiles")
    enroll = load(min(profile_audio_dir(pdir / args.target).glob("*.wav")))
    other = load(min(profile_audio_dir(pdir / args.other).glob("*.wav")))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out or root / "results" / f"enhanced_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)
    cand = EnrolledCandidate()
    cand.load()
    cw = WithOther(cand, other)

    def embed(x):
        return _unit_embedding(cand.embedder, x)

    t_emb, o_emb = embed(enroll), embed(other)
    scorer = get_dnsmos(args.dnsmos)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    load_avg = os.getloadavg()
    result = {"run": out_root.name, "stamp": stamp, "git": git, "host": platform.node(),
              "device": str(cand.device), "load_avg": load_avg, "dnsmos": getattr(scorer, "name", "?"),
              "target_profile": str(pdir / args.target), "other_profile": str(pdir / args.other),
              "contexts": args.contexts, "excerpt_s": args.excerpt_s, "takes": []}
    for spec in args.takes:
        take = load_take(spec, root, fixtures, args.excerpt_s)
        tdir = out_root / spec.replace(":", "_")
        tdir.mkdir(exist_ok=True)
        if take["stems"] is not None:
            active = stem_active(take["stems"]["target"])
        else:
            active = proxy_active(proxy_windows(take["mix"], embed, t_emb, o_emb), len(frame_rms(take["mix"])))
        info = {"id": spec, "input": str(take["path"]), "sha256": sha256(take["path"]), "crop_s": take["crop_s"],
                "duration_s": len(take["mix"]) / SR, "target_active_s": float(active.sum() * WIN_S),
                "rows": []}
        sf.write(tdir / "raw.wav", level_match(take["mix"]), SR, subtype="PCM_16")
        base = score(take["mix"], take, scorer, embed, t_emb, o_emb, active)
        info["rows"].append({"render": "raw", **base})
        if take["live"] is not None:
            info["live_lag_ms"] = take["live_lag_ms"]
            row = score(take["live"], take, scorer, embed, t_emb, o_emb, active)
            info["rows"].append({"render": "live (heard)", **row})
            sf.write(tdir / "A_LIVE.wav", level_match(take["live"]), SR, subtype="PCM_16")
        for ctx in args.contexts:
            y, secs, (window, hop) = render(cw, take["mix"], enroll, ctx)
            for post in (False, True):
                yy, psecs = postfilter(y) if post else (y, 0.0)
                label = f"{ctx}{'+dfn' if post else ''}"
                row = score(yy, take, scorer, embed, t_emb, o_emb, active)
                row.update(render=label, context=ctx, window_s=window, hop_s=hop, postfilter="dfn" if post else None,
                           render_s=secs + psecs, rtf=(secs + psecs) / (len(take["mix"]) / SR))
                info["rows"].append(row)
                name = f"B_REPLAY_ENHANCED_ctx{label.replace('+', '_')}.wav"
                if take["live"] is None and ctx == "1" and not post:
                    name = "A_LIVE_window1s.wav"  # T-141 has no recording: live = the live window
                sf.write(tdir / name, level_match(yy), SR, subtype="PCM_16")
                manifest = {"label": "REPLAY / ENHANCED (offline research render; not live output)",
                            "take": spec, "input": str(take["path"]), "input_sha256": info["sha256"],
                            "crop_s": take["crop_s"], "context": ctx, "window_s": window, "hop_s": hop,
                            "config": {"candidate": "enrolled", "kwargs": "defaults (standard isolation)",
                                       "target_profile": args.target, "other_profile": args.other,
                                       "postfilter": "dfn" if post else None},
                            "device": str(cand.device), "render_s": secs + psecs, "host": platform.node(),
                            "git": git, "wav": name, "level_match_dbfs": MATCH_DBFS}
                (tdir / f"manifest_ctx{label.replace('+', '_')}.json").write_text(json.dumps(manifest, indent=2))
                print(spec, label, json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                              for k, v in row.items()}), flush=True)
        result["takes"].append(info)
        (out_root / "results.json").write_text(json.dumps(result, indent=2))
    result["load_avg_end"] = os.getloadavg()
    (out_root / "results.json").write_text(json.dumps(result, indent=2))
    (out_root / "README.txt").write_text(
        "REPLAY / ENHANCED research renders (T-195). A_* = live (or the live 1 s window for T-141 scenes), "
        "B_REPLAY_ENHANCED_* = offline re-render of the SAME raw take; all level-matched to "
        f"{MATCH_DBFS} dBFS active RMS. Not live output; nothing here goes on stage without rehearsal.\n")
    return result


def _f(v, fmt="{:.2f}"):
    return "n/a" if v is None else fmt.format(v)


def format_table(result: dict) -> str:
    """Markdown rows: take x render, the BENCHMARKS.md table."""
    lines = [("| take | render | context s | leak dB | SI-SDRi dB | DNSMOS SIG / BAK / OVRL | dropouts (runs / s) "
              "| wrong-voice % (ECAPA proxy) | render s | run |"),
             "|---|---|---|---|---|---|---|---|---|---|"]
    for take in result["takes"]:
        for r in take["rows"]:
            ctx = "-" if "window_s" not in r else f"{r['window_s']:g} / hop {r['hop_s']:g}"
            lines.append(
                f"| `{take['id']}` | {r['render']} | {ctx} | {_f(r.get('leak_db'), '{:.1f}')} "
                f"| {_f(r.get('si_sdri'))} | {r['dnsmos_sig']:.2f} / {r['dnsmos_bak']:.2f} / {r['dnsmos_ovrl']:.2f} "
                f"| {r['dropouts']} / {r['dropout_s']:.2f} | {_f(r.get('wrong_pct'), '{:.0f}')} "
                f"| {_f(r.get('render_s'), '{:.1f}')} | `{result['run']}` |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", default=",".join(DEFAULT_TAKES))
    ap.add_argument("--contexts", default=",".join(CONTEXTS))
    ap.add_argument("--excerpt-s", type=float, default=45.0)
    ap.add_argument("--target", default="aryan")
    ap.add_argument("--other", default="joseph")
    ap.add_argument("--profiles", default=None, help="profiles root (default $HEARING_FIXTURES_DIR/profiles)")
    ap.add_argument("--dnsmos", default="auto", choices=["auto", "dnsmos", "mock"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", type=Path, default=None, help="print the markdown table of a results.json and exit")
    args = ap.parse_args(argv)
    if args.table:
        print(format_table(json.loads(args.table.read_text())))
        return 0
    args.takes = [t for t in args.takes.split(",") if t]
    args.contexts = [c for c in args.contexts.split(",") if c]
    bad = [c for c in args.contexts if c not in CONTEXTS]
    if bad:
        ap.error(f"unknown contexts {bad}; pick from {CONTEXTS}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
