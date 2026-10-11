"""Offline oracle diagnostic for optional two-speaker isolation (T-200).

Why does "isolate one of two voices" not reliably work? Four mechanisms can each produce a bad
output, and the live counters cannot tell them apart. This tool replays labelled two-stem
mixtures through the same window loop the live wrapper uses (window ``--window-s``, hop
``--hop-s``, triangular overlap-add), keeps BOTH separator streams, and classifies every
target-active window by the first mechanism that explains it:

1. ``deadline``    window compute > hop: the live wrapper would play the raw mix (raw fallback).
2. ``separation``  even the ORACLE stream (picked with the clean target, evaluation only) is
                   below ``--sep-min-db`` SI-SDR: the separator lost the target before selection.
3. ``selector``    an oracle stream is good but the deployable choice (ECAPA/LDA) took the other
                   one (or silenced the window): ``actual < oracle - --drop-db``.
4. ``mix_back``    the chosen stream is good but the final mixer/AGC output is ``--drop-db`` worse
                   (residual leaked back, gain pumping).
5. ``ok``

Oracle rows use the clean target and are NOT deployable; rows are labelled so in the table and
manifest. Target-absent windows get a leak metric (energy of the output relative to the mix),
never SI-SDR against zero. Fake (dry-run) results are labelled ``fake: true`` and are not
measured separation.

    python -m eval.isolation_diagnostic --dry-run
    python -m eval.isolation_diagnostic --scene <dir with manifest.json + stems/> \\
        --profiles <dir>/<name>/profile-*.wav --target-name aryan --other-name joseph
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval import metrics

SR = 16000
ACTIVE_RMS = 0.003  # a stem window below this RMS (about -50 dBFS) counts as silent
CATEGORIES = ("deadline", "separation", "selector", "mix_back", "ok")
DEFAULT_RATIOS = (0.0, -5.0, -10.0, -15.0)
SAME_VOICE = 0.7  # hearing.tse.enrolled.SAME_VOICE: negative profile closer than this to target is dropped


def _rms(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-9)))


def scale_target(target, other, ratio_db: float):
    """Scale ``target`` to ``ratio_db`` dB relative to ``other`` (whole-clip RMS). 0 = equal levels."""
    t, o = np.asarray(target, np.float32), np.asarray(other, np.float32)
    gain = _rms(o) * 10 ** (ratio_db / 20.0) / max(_rms(t), 1e-9)
    return (t * gain).astype(np.float32), o


# --- window runners -------------------------------------------------------------------------


@dataclass
class WindowOut:
    streams: np.ndarray  # (2, n): both separator outputs, retained
    choice: int | None  # None = selector returned silence (gated)
    margin: float | None = None
    floor: float | None = None
    compute_ms: float = 0.0
    extra: dict = field(default_factory=dict)  # held / paused / scoring / similarities ...


class EnrolledRunner:
    """Real runner: the live ``EnrolledCandidate`` with a recording separator so both streams are kept."""

    def __init__(self, enroll, other, separator=None, embedder=None, **candidate_kw):
        from hearing.tse.blind import BlindSeparator
        from hearing.tse.enrolled import EnrolledCandidate

        class _Recorder:
            def __init__(self, inner):
                self.inner, self.last = inner, None
                self.device = getattr(inner, "device", "")

            def load(self):
                self.inner.load()
                self.device = self.inner.device

            def separate(self, mix, sr=SR):
                self.last = self.inner.separate(mix, sr)
                return self.last

        self.rec = _Recorder(separator if separator is not None else BlindSeparator())
        self.cand = EnrolledCandidate(separator=self.rec, embedder=embedder, **candidate_kw)
        self.enroll, self.other = enroll, other
        self.device = self.rec.device

    def run(self, mix, start):
        t0 = time.perf_counter()
        self.cand.extract(mix, SR, enroll=self.enroll, other=self.other)
        ms = (time.perf_counter() - t0) * 1000
        c = self.cand
        return WindowOut(
            np.asarray(self.rec.last), None if c.gated else int(c.selected_stream), c.margin_score, c.floor, ms,
            {"held": c.held, "paused": c.paused, "vetoed": c.vetoed, "scoring": c.scoring,
             "similarities": c.similarities, "confidence": c.confidence},
        )


def _voiced(f0, n, seed, rate):
    """Harmonic stand-in for a voice: F0 plus 8 harmonics, syllable-like envelope at ``rate`` Hz."""
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    x = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 9))
    env = np.clip(np.sin(2 * np.pi * rate * t) + 0.3, 0, None)
    return (0.15 * x * env / 3).astype(np.float32)


def fake_stems(seconds=8.0):
    """Two constructed 'speakers' with different pitch and syllable rate. Not speech, not a scene
    generator: just enough structure for the classifier tests and the dry run."""
    n = int(seconds * SR)
    return _voiced(120, n, 1, 1.7), _voiced(190, n, 2, 1.1)


class FakeRunner:
    """Dry-run runner with a constructed defect; reads the stems it was built from, never a model."""

    CASES = ("ok", "separator_missing_target", "wrong_selector", "mix_back_leak", "missed_deadline")
    fake = True

    def __init__(self, case, target, other, hop_s):
        if case not in self.CASES:
            raise ValueError(f"unknown fake case {case!r}")
        self.case, self.target, self.other, self.hop_s = case, target, other, hop_s
        self.device = "fake"
        self.mixer_s = 0.2 if case == "mix_back_leak" else 1.0

    def run(self, mix, start):
        n = len(mix)
        t, o = self.target[start:start + n], self.other[start:start + n]
        rng = np.random.default_rng(start)
        if self.case == "separator_missing_target":  # both streams are the other voice (+ hiss)
            streams = np.stack([o, 0.5 * o + 0.01 * rng.standard_normal(n).astype(np.float32)])
        else:
            streams = np.stack([t, o])
        choice = 0
        if self.case == "wrong_selector":
            choice = 1
        ms = self.hop_s * 1500 if self.case == "missed_deadline" else 120.0
        return WindowOut(streams.astype(np.float32), choice, 0.4, None, ms, {"fake": True})


def mix_final(est, mix, s=1.0, agc=False):
    """Diagnostic proxy of the live mixer/AGC: ``est + (1-s)*(mix-est)`` (``hearing.fusion.mixer``'s
    documented target + (1-s)*residual), optionally AGC'd back to the mix's RMS within +-6 dB."""
    out = est + (1.0 - s) * (mix - est)
    if agc and _rms(out) > 1e-9:
        out = out * float(np.clip(_rms(mix) / _rms(out), 0.5, 2.0))
    return out.astype(np.float32)


# --- replay ---------------------------------------------------------------------------------


@dataclass
class Replay:
    windows: list[dict]
    estimate: np.ndarray  # overlap-added selected estimate (raw mix where a window missed its deadline)
    final: np.ndarray  # mixer/AGC output
    oracle_est: np.ndarray  # overlap-added oracle-picked estimate
    window_s: float
    hop_s: float


def _slice_ola(windows_out, n, win):
    out = np.zeros(n, np.float64)
    wsum = np.zeros(n, np.float64)
    w = np.bartlett(win + 2)[1:-1] + 1e-3
    for start, seg in windows_out:
        out[start:start + len(seg)] += seg * w[: len(seg)]
        wsum[start:start + len(seg)] += w[: len(seg)]
    return (out / np.maximum(wsum, 1e-9)).astype(np.float32)


def replay(runner, mix, target, other, *, window_s=2.0, hop_s=1.0, sep_min_db=4.0, drop_db=3.0,
           mixer_s=None, agc=False):
    """Run the live window loop over ``mix`` and classify each window. ``target``/``other`` are the
    clean stems (evaluation only; the runner only sees ``mix``)."""
    win, hop = int(window_s * SR), int(hop_s * SR)
    n = len(mix)
    mixer_s = getattr(runner, "mixer_s", 1.0) if mixer_s is None else mixer_s
    starts = list(range(0, max(n - win, 0) + 1, hop))
    recs, est_parts, ora_parts = [], [], []
    for i, st in enumerate(starts):
        m, t, o = mix[st:st + win], target[st:st + win], other[st:st + win]
        out = runner.run(m, st)
        sis = [metrics.si_sdr(s, t) for s in out.streams]
        oracle = int(np.argmax(sis))
        missed = out.compute_ms > hop_s * 1000.0
        silenced = out.choice is None
        est = m if (missed or silenced) else out.streams[out.choice]
        # a silenced window plays only the attenuated residual; the raw mix stands in for the estimate
        # only on deadline; here it is zeros so leak/absent metrics see what the selector really emitted
        if silenced and not missed:
            est = np.zeros_like(m)
        est_parts.append((st, np.asarray(est, np.float32)))
        ora_parts.append((st, out.streams[oracle]))
        recs.append({
            "i": i, "start_s": st / SR, "target_active": _rms(t) > ACTIVE_RMS, "other_active": _rms(o) > ACTIVE_RMS,
            "oracle_choice": oracle, "oracle_sisdr": sis[oracle], "stream_sisdr": sis,
            "choice": out.choice, "actual_sisdr": None if silenced else sis[out.choice],
            "margin": out.margin, "floor": out.floor, "compute_ms": out.compute_ms,
            "deadline_missed": missed, "raw_fallback": missed, "skipped": silenced, "extra": out.extra,
        })
    if not starts:
        raise ValueError("clip shorter than one window")
    estimate = _slice_ola(est_parts, n, win)
    oracle_est = _slice_ola(ora_parts, n, win)
    final = mix_final(estimate, mix, mixer_s, agc)
    for r, st in zip(recs, starts, strict=True):
        t = target[st:st + win]
        r["final_sisdr"] = metrics.si_sdr(final[st:st + win], t) if r["target_active"] else None
        r["category"] = _classify(r, sep_min_db, drop_db) if r["target_active"] else None
        if not r["target_active"]:  # target absent: energy leak only, no SI-SDR against zero
            r["leak_db"] = _db(_rms(final[st:st + win])) - _db(max(_rms(mix[st:st + win]), 1e-9))
    return Replay(recs, estimate, final, oracle_est, window_s, hop_s)


def _classify(r, sep_min_db, drop_db):
    if r["deadline_missed"]:
        return "deadline"
    if r["oracle_sisdr"] < sep_min_db:
        return "separation"
    if r["actual_sisdr"] is None or r["actual_sisdr"] < r["oracle_sisdr"] - drop_db:
        return "selector"
    if r["final_sisdr"] < r["actual_sisdr"] - drop_db:
        return "mix_back"
    return "ok"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(rep: Replay, mix, target, other, split="all"):
    """Rows keyed by estimator; each carries its denominator (windows) and the sample rate."""
    recs = rep.windows
    if split == "heldout":
        recs = recs[len(recs) // 2:]
    act = [r for r in recs if r["target_active"]]
    solo = [r for r in act if not r["other_active"]]
    both = [r for r in act if r["other_active"]]
    absent = [r for r in recs if not r["target_active"]]
    cats = {c: sum(1 for r in act if r["category"] == c) for c in CATEGORIES}
    rows = [
        {"estimator": "oracle_pick", "oracle": True, "deployable": False, "n_windows": len(act),
         "sisdr_mean_db": _mean(r["oracle_sisdr"] for r in act)},
        {"estimator": "actual_choice", "oracle": False, "deployable": True, "n_windows": len(act),
         "sisdr_mean_db": _mean(r["actual_sisdr"] for r in act),
         "n_silenced": sum(r["skipped"] for r in act)},
        {"estimator": "final_mixer_agc", "oracle": False, "deployable": True, "n_windows": len(act),
         "sisdr_mean_db": _mean(r["final_sisdr"] for r in act)},
        {"estimator": "target_only_windows(final)", "oracle": False, "deployable": True, "n_windows": len(solo),
         "sisdr_mean_db": _mean(r["final_sisdr"] for r in solo)},
        {"estimator": "overlap_windows(final)", "oracle": False, "deployable": True, "n_windows": len(both),
         "sisdr_mean_db": _mean(r["final_sisdr"] for r in both)},
        {"estimator": "target_absent(final leak vs mix)", "oracle": False, "deployable": True,
         "n_windows": len(absent), "leak_mean_db": _mean(r.get("leak_db") for r in absent), "sisdr_mean_db": None},
    ]
    for row in rows:
        row["sample_rate_hz"] = SR
    n_oracle_pick = sum(1 for r in act if r["choice"] == r["oracle_choice"])
    sel_ok = (n_oracle_pick / len(act)) if act else None
    c = {k: v for k, v in cats.items() if k != "ok"}
    dominant = max(c, key=c.get) if any(c.values()) else "none"
    return {
        "rows": rows, "categories": cats, "dominant_failure": dominant,
        "selector_agrees_with_oracle": sel_ok,
        "n_deadline_missed": sum(r["deadline_missed"] for r in recs),
        "n_raw_fallback": sum(r["raw_fallback"] for r in recs),
        "n_skipped": sum(r["skipped"] for r in recs),
        "compute_ms_mean": _mean(r["compute_ms"] for r in recs),
        "compute_ms_max": max((r["compute_ms"] for r in recs), default=None),
        "window_s": rep.window_s, "hop_s": rep.hop_s, "split": split, "n_windows_total": len(recs),
        "ceiling_gap_db": (rows[0]["sisdr_mean_db"] - rows[1]["sisdr_mean_db"])
        if rows[0]["sisdr_mean_db"] is not None and rows[1]["sisdr_mean_db"] is not None else None,
    }


# --- embedding probes (why embeddings have not reliably worked) -------------------------------


def _unit(v):
    v = np.asarray(v, np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def embedding_probes(embedder, prof_t, prof_o, stem_t, stem_o, crops_s=(0.5, 1.0, 2.0, 4.0),
                     contam=(0.0, 0.1, 0.3, 0.5)):
    """Cosine margins ``cos(target profile, x) - cos(other profile, x)`` for clean target-stem crops.
    Positive = the embedder would pick correctly. Reports crop-duration, contamination (the other voice
    mixed into the crop at the given energy share), the profile-vs-scene-mic mismatch (profile vs the
    clean stem it should match) and the negative-profile gate (cos(target, other) > 0.7 drops it)."""
    et, eo = _unit(embedder.embed(prof_t)), _unit(embedder.embed(prof_o))

    def margin(x):
        e = _unit(embedder.embed(np.ascontiguousarray(x, np.float32)))
        return float(et @ e - eo @ e)

    active = np.flatnonzero(np.abs(stem_t) > ACTIVE_RMS)
    centre = int(active[len(active) // 2]) if active.size else len(stem_t) // 2
    out = {"crop_margin": {}, "contamination_margin": {}}
    for d in crops_s:
        half = int(d * SR / 2)
        a = max(centre - half, 0)
        out["crop_margin"][f"{d}s"] = margin(stem_t[a:a + 2 * half])
    seg = stem_t[max(centre - SR, 0):centre + SR]
    oth = stem_o[max(centre - SR, 0):centre + SR]
    oth = oth[: len(seg)]
    seg = seg[: len(oth)]
    for share in contam:
        k = np.sqrt(share / max(1 - share, 1e-9)) * _rms(seg) / max(_rms(oth), 1e-9)
        out["contamination_margin"][f"{int(share * 100)}%"] = margin(seg + k * oth)
    out["profile_vs_scene_target_cos"] = float(et @ _unit(embedder.embed(stem_t[: 4 * SR])))
    out["profile_vs_scene_other_cos"] = float(et @ _unit(embedder.embed(stem_o[: 4 * SR])))
    cos_to = float(et @ eo)
    out["profile_target_vs_other_cos"] = cos_to
    out["negative_profile_dropped_at_0.7"] = cos_to > SAME_VOICE
    return out


class FakeEmbedder:
    """Deterministic band-energy embedding for the dry run (distinguishes the constructed pitches)."""

    def embed(self, audio):
        x = np.asarray(audio, np.float64)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=2048)) ** 2
        return np.log1p(np.add.reduceat(spec, np.linspace(0, 1024, 33, dtype=int)[:-1]))


# --- table / manifest -------------------------------------------------------------------------


def _fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(results):
    lines = ["| condition | estimator | kind | n windows | SR | SI-SDR dB | leak vs mix dB |", "|---|---|---|---|---|---|---|"]
    for res in results:
        for row in res["summary"]["rows"]:
            kind = "ORACLE (not deployable)" if row["oracle"] else "deployable"
            lines.append(f"| {res['condition']} | {row['estimator']} | {kind} | {row['n_windows']} | {row['sample_rate_hz']} "
                         f"| {_fmt(row['sisdr_mean_db'])} | {_fmt(row.get('leak_mean_db'))} |")
    lines += ["", "| condition | dominant failure | " + " | ".join(CATEGORIES) + " | oracle==choice | missed/raw/skipped | compute ms mean/max | window/hop s |",
              "|---|---|" + "---|" * (len(CATEGORIES) + 4)]
    for res in results:
        s = res["summary"]
        cats = " | ".join(str(s["categories"][c]) for c in CATEGORIES)
        lines.append(f"| {res['condition']} | {s['dominant_failure']} | {cats} | {_fmt(s['selector_agrees_with_oracle'], 2)} "
                     f"| {s['n_deadline_missed']}/{s['n_raw_fallback']}/{s['n_skipped']} "
                     f"| {_fmt(s['compute_ms_mean'])}/{_fmt(s['compute_ms_max'])} | {s['window_s']}/{s['hop_s']} |")
    return "\n".join(lines)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _read(path):
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from hearing.audio.io import resample

        x = resample(x, sr, SR)
    return np.ascontiguousarray(x, np.float32)


def run_conditions(make_runner, target, other, ratios, roles, args, noise=None):
    results = []
    for role in roles:
        t0, o0 = (target, other) if role == "A" else (other, target)
        for ratio in ratios:
            t, o = scale_target(t0, o0, ratio)
            n = min(len(t), len(o))
            t, o = t[:n], o[:n]
            mix = (t + o + (noise[:n] if noise is not None else 0)).astype(np.float32)
            runner = make_runner(role, t, o)
            rep = replay(runner, mix, t, o, window_s=args.window_s, hop_s=args.hop_s,
                         sep_min_db=args.sep_min_db, drop_db=args.drop_db, agc=args.agc)
            results.append({"condition": f"role{role} target{ratio:+.0f}dB", "role": role, "ratio_db": ratio,
                            "summary": summarize(rep, mix, t, o, args.split)})
    return results


def dry_run(args):
    t, o = fake_stems(args.fake_seconds)
    out = []
    for case in FakeRunner.CASES:
        res = run_conditions(lambda role, tt, oo, case=case: FakeRunner(case, tt, oo, args.hop_s), t, o,
                             [0.0], ["A"], args)
        res[0]["condition"] = f"FAKE:{case}"
        out.append(res[0])
    probes = embedding_probes(FakeEmbedder(), t[:SR * 3], o[:SR * 3], t, o)
    return {"mode": "dry_run", "fake": True, "label": "FAKE models and constructed stems: classifier check, NOT measured separation",
            "results": out, "probes": probes, "inputs": {}, "device": "fake"}


def real_run(args):
    scene = Path(args.scene)
    man = json.loads((scene / "manifest.json").read_text())
    stems = {k: scene / v for k, v in man["stems"].items()}
    tdir, odir = Path(args.profiles) / args.target_name, Path(args.profiles) / args.other_name
    prof = {k: sorted(d.glob("profile-*.wav")) for k, d in (("target", tdir), ("other", odir))}
    missing = [str(p) for p in [*stems.values()] if not p.exists()] + [f"profile wavs in {d}" for d, k in ((tdir, "target"), (odir, "other")) if not prof[k]]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))
    t, o = _read(stems["target"]), _read(stems["other"])
    noise = _read(stems["noise"]) if args.noise and "noise" in stems else None
    pt, po = _read(prof["target"][0]), _read(prof["other"][0])
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enrolled import EnrolledCandidate

    sep = BlindSeparator()
    probe_cand = EnrolledCandidate(separator=sep)
    probe_cand.load()
    probes = embedding_probes(probe_cand.embedder, pt, po, t, o)

    def make_runner(role, tt, oo):
        a, b = (pt, po) if role == "A" else (po, pt)
        return EnrolledRunner(a, b, separator=sep, embedder=probe_cand.embedder)

    roles = ["A", "B"] if args.roles == "both" else [args.roles]
    results = run_conditions(make_runner, t, o, args.ratios, roles, args, noise)
    inputs = {"scene": str(scene), "scene_manifest_sha": _sha(scene / "manifest.json"),
              **{f"stem_{k}": _sha(p) for k, p in stems.items()},
              "profile_target": _sha(prof["target"][0]), "profile_other": _sha(prof["other"][0])}
    return {"mode": "real", "fake": False, "results": results, "probes": probes, "inputs": inputs,
            "device": str(getattr(sep, "device", ""))}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true", help="fake models + constructed stems (classifier check)")
    p.add_argument("--scene", help="dir with manifest.json and stems/{mix,target,other,noise}.wav")
    p.add_argument("--profiles", help="dir with <name>/profile-*.wav")
    p.add_argument("--target-name", default="aryan")
    p.add_argument("--other-name", default="joseph")
    p.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS), help="quiet-target dB vs the other voice")
    p.add_argument("--roles", choices=("A", "B", "both"), default="both", help="A: scene target is the target; B: swapped")
    p.add_argument("--split", choices=("all", "heldout"), default="heldout", help="heldout = second half of the windows")
    p.add_argument("--noise", action="store_true", help="add the scene's noise stem")
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--hop-s", type=float, default=1.0)
    p.add_argument("--sep-min-db", type=float, default=4.0)
    p.add_argument("--drop-db", type=float, default=3.0)
    p.add_argument("--agc", action="store_true")
    p.add_argument("--fake-seconds", type=float, default=8.0)
    p.add_argument("--out", help="manifest path (default $HEARING_RESULTS_DIR/isolation_diagnostic/<stamp>.json)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    if not args.dry_run and not (args.scene and args.profiles):
        print("BLOCKED: real run needs --scene and --profiles (or use --dry-run)", file=sys.stderr)
        return 2
    try:
        data = dry_run(args) if args.dry_run else real_run(args)
    except FileNotFoundError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    data.update({"stamp": datetime.now(UTC).isoformat(timespec="seconds"), "host": platform.node(),
                 "config": {k: v for k, v in vars(args).items()}})
    print(("** " + data["label"] + " **\n") if data["fake"] else "", end="")
    print(table(data["results"]))
    print("\nprobes:", json.dumps(data["probes"], indent=1))
    out = Path(args.out) if args.out else Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / "isolation_diagnostic" / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + (".dryrun" if args.dry_run else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, default=float))
    print("manifest:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
