#!/usr/bin/env python3
"""Score a recorded session against the replayed clip's clean stems (what the judge hears).

    python scripts/session_score.py [results/sessions/<dir> ...]   # default: newest
    python scripts/session_score.py <dir> --json out.json --dnsmos mock

A session recorded with ``HEARING_SESSION_LOG`` while ``hearing.replay`` fed the demo clip
(``eval/fixtures/demo``) holds ``in.wav`` (the clip, looped) and ``out.wav`` (what the app
played). The clip is a looped GRID mixture whose stems (target, other talker, noise) are on
disk, so the output can be scored like a benchmark:

1. **input -> stems**: each loop of the source mixture is found in ``in.wav`` by
   cross-correlation (the AAC loop in the MP4 drifts by up to ~50 ms per loop, so one global
   offset is wrong); the stems are placed at the same positions with the fitted gain.
2. **output -> input**: piecewise delay. Each run of output samples played at one delay ``d``
   (one tier) is mapped back onto input ``[start - d, end - d)``; input a switch skipped is
   left unscored, input a switch replayed is scored once (the later tier). The prior per telemetry tick is the delay the
   engine puts into the sample stream (``latency_ms.algorithmic + resample``, or the
   playing tier's ``tiers.delay_ms``; ``latency_ms.total`` also counts server compute and
   capture buffering, which are not in the recorded stream, and is reported next to it).
   Every run of constant prior is refined by cross-correlation within +-150 ms; a weak
   peak keeps the prior. Each run is also verified against an open search (0-4 s): when that
   finds a clearly stronger peak > 100 ms away from the reported delay, the run is aligned on the
   measured lag and the score says so (``alignment``; T-108: the remote path once reported the
   local fallback's delay). Prior and measured delays are both reported.
3. **metrics** on the aligned output: target SI-SDR / SI-SDRi vs the input; other-talker
   leak (energy of the other stem's least-squares projection in the output over the same in
   the input, dB, per 200 ms frame) and seconds where it dominates the output (louder than the
   target's part, >= 10 % of the output, ducked < 10 dB); DNSMOS P.835 (not for a silent output);
   true dropouts (target active in its clean stem but the output silent) apart from
   intended ducking (target silent, other talking, output reduced); time to first isolated
   (acquisition ``locked``); tier and target switch counts.

The one-line **demo score** = SI-SDRi, leak dB, true-dropout s, DNSMOS OVRL, delay ms.
Read-only: it never touches the recording.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000
FRAME = 320  # 20 ms frames for VAD / dropouts
LS_FRAME = 3200  # 200 ms frames for the leak projection
SEARCH_MS = 150  # delay refinement window around the telemetry prior
MIN_CORR = 0.2  # normalised xcorr peak below this keeps the prior
VERIFY_MAX_MS = 4000  # open lag search per run (Focus HQ AV over the remote: 3.2 s)
DISAGREE_MS = 100  # reported vs measured beyond this: align on the measured lag, and say so
VERIFY_GAIN = 0.1  # ... only when the open search's peak is this much stronger
MIN_FIT_DB = 10.0  # a loop whose source fit is worse than this is not the clip
ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.popen(f"git -C '{ROOT}' rev-parse --path-format=absolute --git-common-dir 2>/dev/null")
            .read().strip() or ROOT / ".git").parent


# --- I/O ------------------------------------------------------------------------------------


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from hearing.audio.io import resample

        x = resample(x, sr, SR)
    return np.asarray(x, dtype=np.float32)


@dataclass
class Stems:
    mix: np.ndarray
    target: np.ndarray
    other: np.ndarray
    noise: np.ndarray
    fixture: str = ""
    target_face: dict = field(default_factory=dict)


def demo_dir() -> Path:
    for d in (ROOT / "eval/fixtures/demo", MAIN / "eval/fixtures/demo"):
        if (d / "manifest.json").exists() or (d / "demo-mf-easy/manifest.json").exists():
            return d
    return ROOT / "eval/fixtures/demo"


def main_fallback(path: Path) -> Path:
    """``path`` if it exists, else the same path in the main checkout (gitignored demo clips and stems
    live only there; a worktree has none unless swarm links them), else ``path`` unchanged."""
    path = Path(path)
    if path.exists():
        return path
    try:
        alt = MAIN / path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return path
    return alt if alt.exists() else path


def session_dirs(paths: list[Path]) -> list[Path]:
    """Session dirs from the command line: a dir holding ``events.jsonl`` is a session; any other dir
    (a ``regress-<stamp>/`` folder) stands for its subfolders that hold one (not ``server.log`` etc.)."""
    out = []
    for p in paths:
        if p.is_dir() and not (p / "events.jsonl").exists():
            out += sorted(q for q in p.iterdir() if q.is_dir() and (q / "events.jsonl").exists())
        else:
            out.append(p)
    return out


def fixtures_dir() -> Path:
    env = os.environ.get("HEARING_FIXTURES_DIR")
    return Path(env) if env else MAIN / "eval/fixtures/generated"


def resolve_stems(manifest: Path | None = None) -> Stems:
    """The demo manifest's ``stems`` (written by ``make_fixtures --demo``) or, for manifests
    written before that, ``source_fixture`` under ``$HEARING_FIXTURES_DIR/<set>``."""
    if manifest is None:
        d = demo_dir()
        manifest = d / "demo-mf-easy/manifest.json" if (d / "demo-mf-easy/manifest.json").exists() else d / "manifest.json"
    manifest = main_fallback(Path(manifest))
    if not manifest.exists():
        raise FileNotFoundError(f"no demo manifest at {manifest} (nor in {MAIN}); run eval.make_fixtures --demo "
                                "or pass --clip/--manifest")
    m = json.loads(manifest.read_text())
    src = m.get("source") or {}
    stems = m.get("stems") or {}
    paths = {k: main_fallback(Path(manifest).parent / v) for k, v in stems.items()}
    if not all(p.exists() for p in paths.values()) or not paths:
        fx = fixtures_dir() / src.get("set", "av_full") / m["source_fixture"]
        files = src.get("paths") or {}
        paths = {"mix": fx / files.get("mix", "mix.wav"), "target": fx / files.get("target", "target.wav"),
                 "other": fx / files.get("interferer", "interferer.wav"), "noise": fx / files.get("noise", "noise.wav")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"clip stems missing: {missing} (looked in this checkout and {MAIN}; run "
                                f"eval.make_fixtures --av --full, then --demo, or pass --clip/--manifest)")
    face = m.get("target") or next((f for f in src.get("faces", []) if f.get("role") == "target"), {})
    return Stems(**{k: read_wav(paths[k]) for k in ("mix", "target", "other", "noise")},
                 fixture=m.get("source_fixture", ""), target_face=face)


# --- alignment ------------------------------------------------------------------------------


def _xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c[k] = sum_n a[n + k] * b[n] for k in [0, len(a) - len(b)] (FFT)."""
    from scipy.signal import fftconvolve

    return fftconvolve(a, b[::-1], mode="valid")


def place_stems(x: np.ndarray, stems: Stems, search: int = SR // 2) -> tuple[dict, list[dict]]:
    """Find every loop of the source mixture in ``x``; return stems laid out on x's timeline
    (zero where no loop fits) and one fit record per loop."""
    period = len(stems.mix)
    out = {k: np.zeros(len(x), np.float32) for k in ("mix", "target", "other", "noise")}
    covered = np.zeros(len(x), bool)
    loops, expect = [], 0
    while expect < len(x) - period // 4:
        lo, hi = max(0, expect - search), expect + period + search
        seg = np.pad(x[lo:hi], (0, max(0, hi - max(lo, len(x)))))  # a loop cut by the clip end still fits
        pos = lo + int(np.argmax(_xcorr(seg, stems.mix)))
        n = min(period, len(x) - pos)
        ref = stems.mix[:n]
        y = x[pos:pos + n]
        g = float(np.dot(y, ref) / (np.dot(ref, ref) + 1e-12))
        fit = 10 * np.log10(np.sum(y**2) / (np.sum((y - g * ref) ** 2) + 1e-12) + 1e-12)
        loops.append({"start_s": round(pos / SR, 4), "gain": round(g, 4), "fit_db": round(float(fit), 1)})
        if fit >= MIN_FIT_DB and n > 0:
            for k, a in out.items():
                a[pos:pos + n] = g * getattr(stems, k)[:n]
            covered[pos:pos + n] = True
        expect = pos + period
    out["covered"] = covered
    return out, loops


def _ticks(rows: list[dict]) -> list[dict]:
    return [dict(r["event"], t=r["t"]) for r in rows
            if r.get("dir") == "out" and r["event"].get("type") == "telemetry"]


def prior_delay_ms(tick: dict) -> float:
    """Delay the engine puts into the sample stream at this tick."""
    tiers = tick.get("tiers") or {}
    if isinstance(tiers.get("delay_ms"), (int, float)):
        return float(tiers["delay_ms"])
    lat = tick.get("latency_ms") or {}
    if isinstance(lat, dict):
        return float(lat.get("algorithmic") or 0) + float(lat.get("resample") or 0)
    return 0.0


def delay_segments(ticks: list[dict], n: int, split_tier: bool = False) -> list[dict]:
    """Runs of constant prior delay (within 5 ms) in input samples [start, end); ``split_tier`` also cuts at tier changes."""
    segs: list[dict] = []
    for tk in ticks:
        d = round(prior_delay_ms(tk), 1)
        total = (tk.get("latency_ms") or {}).get("total") if isinstance(tk.get("latency_ms"), dict) else None
        at = min(n, max(0, int(tk["t"] * SR)))
        tier = (tk.get("tiers") or {}).get("active") or tk.get("tier")
        if segs and abs(segs[-1]["prior_ms"] - d) <= 5 and not (split_tier and segs[-1]["tier"] != tier):
            segs[-1]["totals"].append(total)
            continue
        if segs:
            segs[-1]["end"] = at
        segs.append({"start": at if segs else 0, "end": n, "prior_ms": d, "totals": [total],
                     "tier": tier})
    if not segs:
        segs = [{"start": 0, "end": n, "prior_ms": 0, "totals": [], "tier": None}]
    return [s for s in segs if s["end"] > s["start"]]


def refine(x: np.ndarray, y: np.ndarray, start: int, end: int, prior: int,
           search: int = SEARCH_MS * SR // 1000) -> tuple[int, float]:
    """Delay d (samples) maximising corr(x[n], y[n + d]) over n in [start, end), |d - prior| <= search."""
    lo_d = max(0, prior - search)
    hi_d = prior + search
    end = min(end, len(y) - hi_d)
    seg = x[start:end]
    win = y[start + lo_d:end + hi_d]
    if len(seg) < SR // 4 or len(win) < len(seg) or np.sum(seg**2) < 1e-6:
        return prior, 0.0
    c = _xcorr(win, seg)
    k = int(np.argmax(np.abs(c)))
    e = np.sqrt(np.sum(seg**2) * np.sum(win[k:k + len(seg)] ** 2)) + 1e-12
    corr = float(abs(c[k]) / e)
    return (lo_d + k, corr) if corr >= MIN_CORR else (prior, corr)


def align_output(x: np.ndarray, y: np.ndarray, ticks: list[dict], *, only_tier: str | None = None,
                 ride: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """y on x's timeline: y_al[n] = y[n + d(n)]; ``valid`` marks samples with output.

    ``only_tier`` aligns every run of that tier on its own and leaves the rest invalid. ``ride`` =
    ``(y2, out)``: ``out`` (zeros, len(x)) receives y2 shifted by y's delays (out_pre_agc.wav, T-179)."""
    y_al = np.zeros(len(x), np.float32)
    valid = np.zeros(len(x), bool)
    report = []
    for s in delay_segments(ticks, len(y), split_tier=only_tier is not None):
        # A segment is a run of OUTPUT samples [start, end) played by one tier at delay d, i.e.
        # input [start - d, end - d) (T-095: cutting the input at the output times scored the
        # last d of every 1.5 s enrolled run against the next tier's audio).
        if only_tier is not None and s["tier"] != only_tier:
            continue
        prior = round(s["prior_ms"] * SR / 1000)
        i0, i1 = max(0, s["start"] - prior), max(0, s["end"] - prior)
        d, corr = refine(x, y, i0, i1, prior)
        half = VERIFY_MAX_MS * SR // 2000
        d_open, c_open = refine(x, y, i0, i1, half, search=half)
        note = None
        if (c_open >= MIN_CORR and c_open >= corr + VERIFY_GAIN
                and abs(d_open - prior) > DISAGREE_MS * SR // 1000):
            note = (f"reported {s['prior_ms']:.0f} ms disagrees with measured {1000 * d_open / SR:.0f} ms "
                    f"(corr {c_open:.2f} vs {corr:.2f}): aligned on the measured lag")
            d, corr = d_open, c_open
        a, b = max(0, s["start"] - d), min(s["end"] - d, len(x), len(y) - d)
        if b > a:
            y_al[a:b] = y[a + d:b + d]
            valid[a:b] = True
            if ride is not None:
                ride[1][a:b] = ride[0][a + d:b + d]
        totals = [t for t in s["totals"] if isinstance(t, (int, float))]
        report.append({"start_s": round(s["start"] / SR, 2), "end_s": round(s["end"] / SR, 2), "tier": s["tier"],
                       "prior_ms": s["prior_ms"], "measured_ms": round(1000 * d / SR, 1), "corr": round(corr, 3),
                       "reported_total_ms": round(float(np.median(totals)), 1) if totals else None,
                       **({"note": note} if note else {})})
    return y_al, valid, report


# --- metrics --------------------------------------------------------------------------------


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR in dB; NaN for a silent estimate (text mode plays nothing, not "0 dB")."""
    if np.mean(np.square(est, dtype=np.float64)) < 1e-10:
        return float("nan")
    est = est.astype(np.float64) - est.mean()
    ref = ref.astype(np.float64) - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t = a * ref
    return float(10 * np.log10((np.sum(t**2) + 1e-12) / (np.sum((est - t) ** 2) + 1e-12)))


def _frames(a: np.ndarray, n: int) -> np.ndarray:
    m = len(a) // n
    return a[: m * n].reshape(m, n)


def _rms(a: np.ndarray, n: int = FRAME) -> np.ndarray:
    return np.sqrt(np.mean(_frames(a, n).astype(np.float64) ** 2, axis=1))


def vad(clean: np.ndarray, floor_db: float = -35.0, n: int = FRAME) -> np.ndarray:
    """Energy VAD on a clean stem: frame RMS within ``floor_db`` of the stem's 95th percentile."""
    r = _rms(clean, n)
    if not len(r) or r.max() <= 1e-6:
        return np.zeros(len(r), bool)
    return r > max(1e-4, np.percentile(r, 95) * 10 ** (floor_db / 20))


def onset_clipping(t_act: np.ndarray, silent: np.ndarray, gap: int = 10, cut: int = 3) -> tuple[int, int]:
    """(onsets, clipped): target onsets after >= ``gap`` inactive 20 ms frames, and those whose
    first ``cut`` frames (60 ms, i.e. cut by > 50 ms) of output were all silent."""
    onsets = clipped = 0
    quiet = gap
    for i, a in enumerate(t_act):
        if a and quiet >= gap:
            onsets += 1
            w = silent[i : i + cut]
            clipped += int(len(w) == cut and bool(w.all()))
        quiet = 0 if a else quiet + 1
    return onsets, clipped


def _runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only runs of True at least ``min_len`` long."""
    out = np.zeros_like(mask)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if b - a >= min_len:
            out[a:b] = True
    return out


def projection(y: np.ndarray, refs: list[np.ndarray], n: int = LS_FRAME) -> np.ndarray:
    """Per-frame least-squares gains of y on refs: (frames, len(refs))."""
    yf = _frames(y, n).astype(np.float64)
    rf = np.stack([_frames(r, n).astype(np.float64) for r in refs], axis=2)
    g = np.zeros((len(yf), len(refs)))
    for i in range(len(yf)):
        a = rf[i]
        lam = 1e-6 * (np.trace(a.T @ a) + 1e-9)
        g[i] = np.linalg.solve(a.T @ a + lam * np.eye(len(refs)), a.T @ yf[i])
    return g


@dataclass
class Score:
    session: str
    mode: str
    seconds: float
    si_sdr_db: float
    si_sdr_in_db: float
    si_sdri_db: float
    leak_db: float
    other_dominant_s: float
    true_dropout_s: float
    ducking_s: float
    target_active_s: float
    dnsmos_ovrl: float | None
    dnsmos_sig: float | None
    dnsmos_bak: float | None
    dnsmos_scorer: str
    delay_ms: float
    reported_total_ms: float | None
    first_isolated_s: float | None
    tier_switches: int
    target_switches: int
    captions: int
    input_fit_db: float
    fixture: str = ""
    target_agreement: float | None = None
    onsets: int = 0  # target speech onsets (>= 200 ms of target silence before)
    onset_clips: int = 0  # ... whose first 60 ms (> 50 ms) of output were silent (T-096)
    delays: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    alignment: str = "reported"  # "measured: ..." when a run's reported delay was wrong (T-108)
    # T-135 (Q-350): separation SI-SDRi on the output *before* the judge profile's output AGC
    # (out_pre_agc.wav); every other number (leak, other-dominant, DNSMOS, loudness) is on out.wav.
    si_sdri_pre_agc_db: float | None = None
    tier_switches_after_isolated: int | None = None  # ladder changes from first_isolated_s on (T-130 bar: 0)
    # T-179: out.wav minus out_pre_agc.wav (energy over the 20 ms frames where the target is silent): what
    # the output AGC did to the room in pauses (bar <= +3 dB); speech_level_db: out.wav over target-active frames
    pause_lift_db: float | None = None
    speech_level_db: float | None = None
    # T-179 (Q-485): out_pre_agc.wav scored on the very same delay-aligned crop as out.wav (out.wav's delays,
    # out.wav's kept frames), so SI-SDRi before/after the AGC differ only by the AGC; speech_level_pre_agc_db:
    # out_pre_agc.wav over the same target-active frames; speech_level_p90_db: 90th percentile of out.wav's
    # target-active 20 ms frame levels (the loud frames the AGC's 50 ms-attack envelope levels to -20 dBFS)
    si_sdri_pre_agc_same_crop_db: float | None = None
    speech_level_pre_agc_db: float | None = None
    speech_level_p90_db: float | None = None

    def line(self) -> str:
        f = (lambda v, fmt: "-" if v is None or not np.isfinite(v) else format(v, fmt))
        pre = ("" if self.si_sdri_pre_agc_db is None
               else f" (separation SI-SDRi, before output leveling {self.si_sdri_pre_agc_db:+.1f} dB)")
        lift = "" if self.pause_lift_db is None else f" | pause lift {self.pause_lift_db:+.1f} dB"
        return (f"demo score [{self.mode}] SI-SDRi {f(self.si_sdri_db, '+.1f')} dB{pre}{lift} | leak {f(self.leak_db, '+.1f')} dB | "
                f"true dropouts {self.true_dropout_s:.2f} s | onset clips {self.onset_clips}/{self.onsets} | DNSMOS OVRL {f(self.dnsmos_ovrl, '.2f')}"
                f"{'' if self.dnsmos_scorer in ('dnsmos', 'none') else ' (' + self.dnsmos_scorer + ')'} | delay {self.delay_ms:.0f} ms"
                f" | target agreement {f(self.target_agreement, '.0%')}"
                f"{' WRONG TARGET' if wrong_target(self.target_agreement) else ''}"
                f"{'' if self.alignment == 'reported' else ' | ALIGNMENT ' + self.alignment}")

    def as_dict(self) -> dict:
        return {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in asdict(self).items()}


WRONG_TARGET_BELOW = 0.8  # target agreement under this: the engine isolated someone else


def wrong_target(agreement: float | None) -> bool:
    return agreement is not None and agreement < WRONG_TARGET_BELOW


def target_agreement(ticks: list[dict], target_face: dict) -> float | None:
    """Share of ticks with a target track whose box centre is on the manifest target's side of the
    frame (track ids differ per server, sides do not). None without a manifest box or any targeted tick."""
    box = target_face.get("box")
    if not box:
        return None
    left = box[0] + box[2] / 2 < 0.5
    hits = []
    for tk in ticks:
        tid = (tk.get("target") or {}).get("track_id")
        face = next((f for f in tk.get("faces") or [] if f.get("track_id") == tid), None) if tid is not None else None
        if face is None or not face.get("box"):
            continue
        fx, _, fw, _ = face["box"]
        hits.append((fx + fw / 2 < 0.5) == left)
    return round(float(np.mean(hits)), 3) if hits else None


def score_arrays(x: np.ndarray, y: np.ndarray, stems: Stems, rows: list[dict] | None = None, *,
                 session: str = "", dnsmos: str = "auto", skip_s: float = 0.0, enrolled_phase: bool = False,
                 y_pre: np.ndarray | None = None) -> Score:
    rows = rows or []
    ticks = _ticks(rows)
    placed, loops = place_stems(x, stems)
    # enrolled_phase: only the ticks where tier == enrolled, aligned on their own; the input is
    # cropped with the very same mask (xs/ys below share ``idx``) so SI-SDR in/out compare
    pre_al = None if y_pre is None else np.zeros(len(x), np.float32)
    y_al, valid, delays = align_output(x, y, ticks, only_tier="enrolled" if enrolled_phase else None,
                                       ride=None if y_pre is None else (y_pre, pre_al))
    use = placed["covered"] & valid
    use[: int(skip_s * SR)] = False
    n = len(x) // LS_FRAME * LS_FRAME
    use[n:] = False
    # whole 200 ms frames only, so every per-frame metric sees the same samples
    fm = _frames(use[:n], LS_FRAME).all(axis=1)
    keep = np.repeat(fm, LS_FRAME)
    idx = np.flatnonzero(keep)
    t, o, nz = (placed[k][:n] for k in ("target", "other", "noise"))
    xs, ys = x[:n], y_al[:n]
    def sel(a: np.ndarray) -> np.ndarray:
        return a[idx]

    if len(idx) < SR // 2:
        raise ValueError(f"{session}: < 0.5 s of output aligned to the clip (input fit {loops})")
    sdr_out, sdr_in = si_sdr(sel(ys), sel(t)), si_sdr(sel(xs), sel(t))

    gy = projection(ys, [t, o, nz])[fm]
    gx = projection(xs, [t, o, nz])[fm]
    eo = np.sum(_frames(o, LS_FRAME)[fm].astype(np.float64) ** 2, axis=1)
    et = np.sum(_frames(t, LS_FRAME)[fm].astype(np.float64) ** 2, axis=1)
    leak = float(10 * np.log10((np.sum(gy[:, 1] ** 2 * eo) + 1e-12) / (np.sum(gx[:, 1] ** 2 * eo) + 1e-12)))
    ey = np.sum(_frames(ys, LS_FRAME)[fm].astype(np.float64) ** 2, axis=1)
    # the other talker dominates a frame when its part of the output beats the target's, makes up
    # >= 10 % of the output and is ducked by < 10 dB from the input (an audible competing voice)
    eo_y, eo_x = gy[:, 1] ** 2 * eo, gx[:, 1] ** 2 * eo
    dom = (eo_y > gy[:, 0] ** 2 * et) & (eo_y > 0.1 * ey) & (eo_y > 0.1 * eo_x) & (ey > LS_FRAME * 1e-7)
    other_dom_s = float(dom.sum() * LS_FRAME / SR)

    # dropouts on 20 ms frames inside the kept region
    k20 = _frames(keep, FRAME).all(axis=1)
    t_act, o_act = vad(t)[k20], vad(o)[k20]
    ry, rx, rt = _rms(ys)[k20], _rms(xs)[k20], _rms(t)[k20]
    silent = ry < np.maximum(1e-4, rt * 10 ** (-25 / 20))
    true_drop = _runs(t_act & silent, 3)
    ducked = (~t_act) & o_act & (ry < rx * 10 ** (-6 / 20))
    onsets, onset_clips = onset_clipping(t_act, silent)
    ey20 = _rms(ys)[k20] ** 2
    speech_level = float(10 * np.log10(np.mean(ey20[t_act]) + 1e-12)) if t_act.any() else None
    ep20 = None if pre_al is None else _rms(pre_al[:n])[k20] ** 2
    lift = None if ep20 is None else pause_lift(ey20, ep20, ~t_act)
    speech_p90 = float(np.percentile(10 * np.log10(ey20[t_act] + 1e-12), 90)) if t_act.any() else None
    speech_pre = (float(10 * np.log10(np.mean(ep20[t_act]) + 1e-12))
                  if ep20 is not None and t_act.any() else None)
    sdri_pre_same = None if pre_al is None else si_sdr(pre_al[:n][idx], sel(t)) - sdr_in

    score = None
    if dnsmos != "none" and np.sqrt(np.mean(np.square(sel(ys), dtype=np.float64))) > 1e-4:  # not for silence
        from eval.dnsmos import get_dnsmos

        scorer = get_dnsmos(dnsmos)
        clip = sel(ys)[: 60 * SR]
        s = scorer.score(clip, SR)
        score = (s.ovrl, s.sig, s.bak, s.scorer)
    seg_w = np.array([d["end_s"] - d["start_s"] for d in delays])
    meas = np.array([d["measured_ms"] for d in delays])
    totals = [d["reported_total_ms"] for d in delays if d["reported_total_ms"] is not None]
    delay = float(meas[np.argmax(seg_w)]) if len(meas) else 0.0
    if len(meas):  # duration-weighted median
        order = np.argsort(meas)
        cw = np.cumsum(seg_w[order])
        pick = order[np.searchsorted(cw, cw[-1] / 2)]
        delay = float(meas[pick])
        totals = [delays[pick]["reported_total_ms"]] if delays[pick]["reported_total_ms"] is not None else totals

    ctrl = [r["event"] for r in rows if r.get("dir") == "in" and r["event"].get("type") == "control"
            and r["event"].get("mode")]
    mode = ctrl[-1]["mode"] if ctrl else "?"
    first = first_isolated(ticks)
    tiers = [(tk.get("tiers") or {}).get("active") or tk.get("tier") for tk in ticks]
    tiers = [v for v in tiers if v]
    tiers_after = [v for v in ((tk.get("tiers") or {}).get("active") or tk.get("tier") for tk in ticks
                               if first is not None and tk["t"] >= first) if v]
    tracks = [(tk.get("target") or {}).get("track_id") for tk in ticks]
    tracks = [v for v in tracks if v is not None]
    def switches(v: list) -> int:
        return sum(a != b for a, b in itertools.pairwise(v))

    caps = sum(1 for r in rows if r.get("dir") == "out" and r["event"].get("type") == "caption")
    return Score(
        session=session, mode=mode, seconds=round(len(idx) / SR, 2),
        si_sdr_db=round(sdr_out, 2), si_sdr_in_db=round(sdr_in, 2), si_sdri_db=round(sdr_out - sdr_in, 2),
        leak_db=round(leak, 2), other_dominant_s=round(other_dom_s, 2),
        true_dropout_s=round(float(true_drop.sum() * FRAME / SR), 2), ducking_s=round(float(ducked.sum() * FRAME / SR), 2),
        target_active_s=round(float(t_act.sum() * FRAME / SR), 2),
        dnsmos_ovrl=None if score is None else round(score[0], 2), dnsmos_sig=None if score is None else round(score[1], 2),
        dnsmos_bak=None if score is None else round(score[2], 2), dnsmos_scorer="none" if score is None else score[3],
        delay_ms=round(delay, 1), reported_total_ms=round(float(np.median(totals)), 1) if totals else None,
        first_isolated_s=None if first is None else round(first, 2),
        tier_switches=switches(tiers), target_switches=switches(tracks), captions=caps,
        input_fit_db=round(float(np.median([lp["fit_db"] for lp in loops])), 1) if loops else float("nan"),
        fixture=stems.fixture, target_agreement=target_agreement(ticks, stems.target_face),
        onsets=onsets, onset_clips=onset_clips, delays=delays, loops=loops, alignment=_alignment(delays),
        tier_switches_after_isolated=None if first is None else switches(tiers_after),
        pause_lift_db=None if lift is None else round(lift, 2),
        speech_level_db=None if speech_level is None else round(speech_level, 2),
        si_sdri_pre_agc_same_crop_db=None if sdri_pre_same is None else round(float(sdri_pre_same), 2),
        speech_level_pre_agc_db=None if speech_pre is None else round(speech_pre, 2),
        speech_level_p90_db=None if speech_p90 is None else round(speech_p90, 2))


def pause_lift(e_out: np.ndarray, e_pre: np.ndarray, target_silent: np.ndarray) -> float | None:
    """dB of output energy over pre-AGC energy on the target-silent frames (T-179); None without any
    (or when both are digital silence there)."""
    a, b = float(np.sum(e_out[target_silent])), float(np.sum(e_pre[target_silent]))
    if not target_silent.any() or b < 1e-10:
        return None
    return 10 * np.log10((a + 1e-12) / b)


def _alignment(delays: list[dict]) -> str:
    """``reported`` when every run played at its reported delay, else which runs did not."""
    bad = [f"{d['start_s']:.1f}-{d['end_s']:.1f} s {d['tier']}: {d['note']}" for d in delays if d.get("note")]
    return "reported" if not bad else "measured: " + "; ".join(bad)


def first_isolated(ticks: list[dict]) -> float | None:
    """Time of the first tick that isolates someone: the camera lock, or (camera-free modes such as
    Name Call have no acquisition) ``name_call.state == 'isolated'``, else the first enrolled-tier tick."""
    for tk in ticks:
        if (tk.get("acquisition") or {}).get("state") == "locked":
            return tk["t"]
    for tk in ticks:
        if (tk.get("name_call") or {}).get("state") == "isolated":
            return tk["t"]
    return next((tk["t"] for tk in ticks
                 if ((tk.get("tiers") or {}).get("active") or tk.get("tier")) == "enrolled"), None)


def score_session(d: Path, stems: Stems | None = None, *, dnsmos: str = "auto", skip_s: float = 0.0,
                  enrolled_phase: bool = False, pre_agc: bool = False) -> Score:
    """``pre_agc`` (Q-471): every metric on ``out_pre_agc.wav`` (the output before leveling) instead of ``out.wav``."""
    rows = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines() if line.strip()]
    x, y = read_wav(d / "in.wav"), read_wav(d / ("out_pre_agc.wav" if pre_agc else "out.wav"))
    stems = stems or resolve_stems()
    pre = d / "out_pre_agc.wav"
    y_pre = read_wav(pre) if pre.exists() and not pre_agc else None
    if y_pre is not None and len(y_pre) != len(y):  # a resampled session: not sample-synchronous with out.wav
        y_pre = None
    s = score_arrays(x, y, stems, rows, session=d.name, dnsmos=dnsmos, skip_s=skip_s, enrolled_phase=enrolled_phase,
                     y_pre=y_pre)
    if not pre_agc:
        s.si_sdri_pre_agc_db = pre_agc_sdri(d, x, len(y), stems, rows, skip_s=skip_s, enrolled_phase=enrolled_phase)
    return s


def pre_agc_sdri(d: Path, x: np.ndarray, n_out: int, stems: Stems, rows: list[dict], *, skip_s: float = 0.0,
                 enrolled_phase: bool = False) -> float | None:
    """SI-SDRi of ``out_pre_agc.wav`` (the judge profile's output before its AGC, T-135), scored like
    ``out.wav``; None without the file or when it does not cover the same samples (a resampled session)."""
    path = d / "out_pre_agc.wav"
    if not path.exists():
        return None
    y = read_wav(path)
    if abs(len(y) - n_out) > SR // 100:
        return None
    return score_arrays(x, y, stems, rows, session=d.name, dnsmos="none", skip_s=skip_s,
                        enrolled_phase=enrolled_phase).si_sdri_db


HEADER = ("| session | mode | s | SI-SDR in | SI-SDR out | SI-SDRi | leak dB | other-dominant s | true dropouts s "
          "| ducking s | DNSMOS OVRL | delay ms (reported total) | first isolated s | tier sw | target sw | target agreement |")


def table(scores: list[Score]) -> str:
    f = (lambda v, fmt="": "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt))
    lines = [HEADER, "|" + "---|" * HEADER.count(" | ") + "---|"]
    for s in scores:
        mos = f(s.dnsmos_ovrl, ".2f") + ("" if s.dnsmos_scorer in ("dnsmos", "none") else f" ({s.dnsmos_scorer})")
        lines.append(f"| {s.session} | {s.mode} | {s.seconds:.1f} | {f(s.si_sdr_in_db, '+.2f')} | {f(s.si_sdr_db, '+.2f')} | "
                     f"**{f(s.si_sdri_db, '+.2f')}** | **{s.leak_db:+.1f}** | {s.other_dominant_s:.1f} | **{s.true_dropout_s:.2f}** | "
                     f"{s.ducking_s:.2f} | **{mos}** | **{s.delay_ms:.0f}** ({f(s.reported_total_ms, '.0f')}) | "
                     f"{f(s.first_isolated_s, '.1f')} | {s.tier_switches} | {s.target_switches} | "
                     f"{f(s.target_agreement, '.0%')}{' **WRONG TARGET**' if wrong_target(s.target_agreement) else ''} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path, help="session dirs (default: newest under results/sessions)")
    ap.add_argument("--manifest", type=Path, default=None, help="demo manifest (default eval/fixtures/demo/demo-mf-easy/manifest.json, else the legacy loop's)")
    ap.add_argument("--dnsmos", default="auto", choices=["auto", "dnsmos", "mock", "none"])
    ap.add_argument("--skip", type=float, default=0.0, help="ignore the first N seconds of input")
    ap.add_argument("--enrolled-phase", action="store_true", help="score only the ticks where tier == enrolled (own alignment)")
    ap.add_argument("--json", type=Path, default=None, help="write all scores as JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-segment delays and loop fits")
    args = ap.parse_args(argv)
    sessions = session_dirs(args.sessions)
    if args.sessions and not sessions:
        print(f"no session (a dir with events.jsonl) in {[str(p) for p in args.sessions]}", file=sys.stderr)
        return 2
    if not sessions:
        root = next((r for r in (ROOT / "results/sessions", MAIN / "results/sessions") if r.exists()), None)
        if root is None:
            print("no results/sessions; pass a session dir", file=sys.stderr)
            return 2
        sessions = [max((p for p in root.glob("*/") if (p / "events.jsonl").exists()), key=lambda p: p.stat().st_mtime)]
    try:
        stems = resolve_stems(main_fallback(args.manifest) if args.manifest else None)
    except FileNotFoundError as e:
        print(f"!! {e}", file=sys.stderr)
        return 2
    scores, failed = [], 0
    for d in sessions:
        try:
            s = score_session(d, stems, dnsmos=args.dnsmos, skip_s=args.skip, enrolled_phase=args.enrolled_phase)
        except (ValueError, FileNotFoundError) as e:
            print(f"!! {d.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        scores.append(s)
        print(f"{d.name}: {s.line()}")
        if args.verbose:
            for seg in s.delays:
                print(f"    delay {seg}")
            print(f"    input loops: {s.loops}")
    if scores:
        print()
        print(table(scores))
    if args.json:
        args.json.write_text(json.dumps([s.as_dict() for s in scores], indent=2) + "\n")
    return 1 if failed and not scores else 0


if __name__ == "__main__":
    sys.exit(main())
