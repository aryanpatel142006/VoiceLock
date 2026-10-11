"""Where do the enrolled tier's wrong-speaker windows come from? (T-323)

    .venv/bin/python -m eval.wrong_speaker_audit --profiles DIR --scene TARGET:OTHER:SCENE_DIR [--scene ...]

Replays :class:`hearing.tse.enrolled.EnrolledCandidate` offline (real weights, the saved profiles, closed-set
LDA, default ``margin`` -> hold threshold) over a T-141 ``demo_scene`` (``stems/{mix,target,other,noise}.wav``)
in 1.0 s windows every 0.5 s, each window contributing its last hop as the live wrapper streams it. Every window
logs the pick (``scores``, ``margin_score``, the hold threshold, ``held``, ``vetoed``, ``gated``), the stream
the previous choice continues (``follow``: what a sticky pick would play) and the **oracle** stream (the one
correlating more with the clean target stem than with the other's; evaluation only, not deployable).

The output is what the live ``tse>mixer`` plays at focus_hq's slider 100: the picked stream plus the residual
(window mix minus stream) at the confidence floor (``hearing.fusion.mixer.confidence_floor_db``: 6 dB down at
confidence 0, 40 dB at 1). A window is **wrong** when one of its output's 200 ms frames is other-dominant by the
``session_score`` rule (the other stem's least-squares part beats the target's, is >= 10 % of the output and is
ducked < 10 dB). Each wrong window gets one cause:

* ``mix_back``: the pick was the oracle stream and the stream alone is not other-dominant; the residual played
  back at a low-confidence floor carries the other voice.
* ``separator``: the pick was the oracle stream; the stream itself carries the other voice (T-317 idea C).
* ``high_margin``: a wrong pick with ``margin_score >= hold``: the hold rule never saw it (T-317 idea A).
* ``low_margin_held``: a wrong pick with ``margin_score < hold`` where the hold followed the previous stream.
* ``low_margin``: a wrong pick with ``margin_score < hold``, the hold fired but agreed with the scores (or
  found no previous stream).
* ``vetoed``: the F0 cue overturned the scores.

``sticky_fixes`` counts wrong picks whose ``follow`` stream was the oracle (keeping the previous stream would
have been right). Results: ``$HEARING_RESULTS_DIR/wrong_speaker_audit_<stamp>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SR = 16000
WIN_S, HOP_S = 1.0, 0.5
LS_FRAME = 3200  # 200 ms, as scripts/session_score.py
FRAME = 320  # 20 ms frames for true dropouts, as scripts/session_score.py
CAUSES = ("mix_back", "separator", "high_margin", "low_margin_held", "low_margin", "vetoed")


def _frames(a: np.ndarray, n: int = LS_FRAME) -> np.ndarray:
    m = len(a) // n
    return np.asarray(a[: m * n], np.float64).reshape(m, n)


def _ls_gains(z: np.ndarray, t: np.ndarray, o: np.ndarray, nz: np.ndarray) -> np.ndarray:
    """Per 200 ms frame: least-squares gains of the (target, other, noise) stems in ``z``."""
    zf, rf = _frames(z), np.stack([_frames(r) for r in (t, o, nz)], axis=2)
    g = np.zeros((len(zf), 3))
    for i in range(len(zf)):
        a = rf[i]
        lam = 1e-6 * (np.trace(a.T @ a) + 1e-9)
        g[i] = np.linalg.solve(a.T @ a + lam * np.eye(3), a.T @ zf[i])
    return g


def target_atten(y: np.ndarray, x: np.ndarray, t: np.ndarray, o: np.ndarray, nz: np.ndarray) -> dict:
    """T-326 check: target-active 200 ms frames (stem RMS > 0.1 x its speech level) where the output's target
    part is >= 10 dB (6 dB) under the input's; the projection is onto the target stem."""
    from eval.namecall_eval import _speech_level

    gy, gx = _ls_gains(y, t, o, nz), _ls_gains(x, t, o, nz)
    act = np.sqrt(np.mean(_frames(t) ** 2, axis=1)) > 0.1 * _speech_level(t)
    db = 20.0 * np.log10(np.maximum(np.abs(gy[:, 0]), 1e-6) / np.maximum(np.abs(gx[:, 0]), 1e-6))
    sec = LS_FRAME / SR
    return {"target_active_s": round(float(act.sum() * sec), 2),
            "target_down_10dB_s": round(float(np.sum(act & (db <= -10)) * sec), 2),
            "target_down_6dB_s": round(float(np.sum(act & (db <= -6)) * sec), 2),
            "median_target_gain_db": round(float(np.median(db[act])), 2) if act.any() else None}


def other_dominant(y: np.ndarray, x: np.ndarray, t: np.ndarray, o: np.ndarray, nz: np.ndarray) -> np.ndarray:
    """Per 200 ms frame: the other talker dominates the output ``y`` (``session_score.score_arrays`` rule)."""
    gy, gx = _ls_gains(y, t, o, nz), _ls_gains(x, t, o, nz)
    eo, et, ey = (np.sum(_frames(s) ** 2, axis=1) for s in (o, t, y))
    eo_y, eo_x = gy[:, 1] ** 2 * eo, gx[:, 1] ** 2 * eo
    return (eo_y > gy[:, 0] ** 2 * et) & (eo_y > 0.1 * ey) & (eo_y > 0.1 * eo_x) & (ey > LS_FRAME * 1e-7)


def true_dropout_s(y: np.ndarray, t: np.ndarray) -> float:
    """Seconds the target talks (energy VAD on its stem) while the output is silent (< 25 dB under the
    stem), in runs of >= 3 frames of 20 ms (``session_score`` rule)."""
    r = np.sqrt(np.mean(_frames(t, FRAME) ** 2, axis=1))
    ry = np.sqrt(np.mean(_frames(y, FRAME) ** 2, axis=1))
    act = r > max(1e-4, np.percentile(r, 95) * 10 ** (-35 / 20)) if r.max() > 1e-6 else np.zeros(len(r), bool)
    mask = act & (ry < np.maximum(1e-4, r * 10 ** (-25 / 20)))
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return float(sum(b - a for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= 3) * FRAME / SR)


def window_of_frames(n_frames: int, win: int, hop: int) -> np.ndarray:
    """Index of the window whose streamed hop covers each 200 ms frame's centre (window 0 also owns [0, hop))."""
    centre = np.arange(n_frames) * LS_FRAME + LS_FRAME // 2
    return np.maximum((centre - (win - hop)) // hop, 0)


def cause(row: dict) -> str:
    if row["vetoed"]:
        return "vetoed"
    if row["selected"] == row["oracle"]:
        return "separator" if row["dom_stream_s"] > 0 else "mix_back"
    if row["margin"] is None or row["margin"] >= row["hold"]:
        return "high_margin"
    return "low_margin_held" if row["held"] else "low_margin"


def summarize(rows: list[dict]) -> dict:
    """Wrong windows by cause, with the margin split the T-323 question asks for."""
    wrong = [r for r in rows if r["wrong"]]
    picks = [r for r in wrong if r["cause"] not in ("separator", "mix_back")]
    counts = {c: sum(r["cause"] == c for r in wrong) for c in CAUSES}
    above = sum(r["margin"] is None or r["margin"] >= r["hold"] for r in wrong)
    return {"windows": len(rows), "wrong": len(wrong), "wrong_s": round(sum(r["dom_s"] for r in wrong), 2),
            "causes": counts,
            "above_hold": above, "below_hold": len(wrong) - above,
            "held_fired": sum(bool(r["held"]) for r in wrong),
            "wrong_picks": len(picks),
            "sticky_fixes": sum(r["follow"] is not None and r["follow"] == r["oracle"] for r in picks),
            "pick_errors_all": sum(r["selected"] != r["oracle"] and r["target_active"] for r in rows)}


class SeparatorCache:
    """Memoise a separator by window content so the frozen and adaptive arms (T-380) separate each window once."""

    def __init__(self, inner):
        self.inner, self.memo = inner, {}

    @property
    def device(self):
        return self.inner.device

    def load(self):
        self.inner.load()

    def separate(self, mix, sr=SR):
        key = hash(np.asarray(mix).tobytes())
        if key not in self.memo:
            self.memo[key] = self.inner.separate(mix, sr)
        return self.memo[key]


class _Capture:
    """Separator wrapper keeping the last window's streams (the oracle needs both)."""

    def __init__(self, inner):
        self.inner, self.last = inner, None

    @property
    def device(self):
        return self.inner.device

    def load(self):
        self.inner.load()

    def separate(self, mix, sr=SR):
        self.last = self.inner.separate(mix, sr)
        return self.last


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(np.dot(a, b)) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


ARMS = ("unset", "gated20", "gated20+overlap")  # T-337: the residual floor of each arm


def _arm_floor_db(arm: str, confidence: float, overlap: bool) -> float:
    from hearing.fusion.mixer import UNSURE_DB, confidence_floor_db

    if confidence == 0.0 and arm != "unset" and (arm == "gated20" or overlap):
        return 20.0
    return UNSURE_DB if confidence == 0.0 else confidence_floor_db(confidence)


def leak_db(y: np.ndarray, x: np.ndarray, t: np.ndarray, o: np.ndarray, nz: np.ndarray) -> float:
    """The other talker's energy in ``y`` vs in the input ``x`` (dB, 200 ms least-squares projections, whole clip)."""
    gy, gx = _ls_gains(y, t, o, nz), _ls_gains(x, t, o, nz)
    eo = np.sum(_frames(o) ** 2, axis=1)
    return round(float(10.0 * np.log10((np.sum(gy[:, 1] ** 2 * eo) + 1e-12) / (np.sum(gx[:, 1] ** 2 * eo) + 1e-12))), 2)


def run(scene: Path, enroll: np.ndarray, other: np.ndarray, wav_out: Path | None = None, adaptive: bool = False,
        separator=None) -> dict:
    from eval.contrast_sweep import load_scene
    from eval.namecall_eval import _rms, _speech_level
    from hearing.audio import io as aio
    from hearing.fusion.mixer import confidence_floor_db
    from hearing.fusion.stream_vad import StreamVad
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enrolled import EnrolledCandidate

    sc = load_scene(scene)
    mix, t, o = sc["mix"], sc["A"], sc["B"]
    nz = aio.load(scene / "stems" / "noise.wav", SR)[0][: len(mix)]
    nz = np.pad(nz, (0, len(mix) - len(nz)))
    sep = _Capture(separator if separator is not None else BlindSeparator())
    if adaptive:  # T-380: the target class learns from confident solo windows (knob off by default)
        from eval.adaptive_target import AdaptiveTargetCandidate

        cand = AdaptiveTargetCandidate(separator=sep)
    else:
        cand = EnrolledCandidate(separator=sep)
    cand.prepare(enroll, other, SR)
    thr_t = 0.1 * _speech_level(t)
    win, hop = round(WIN_S * SR), round(HOP_S * SR)
    out, played = np.zeros_like(mix), np.zeros_like(mix)
    arms = {a: np.zeros_like(mix) for a in ARMS}
    vad = StreamVad()
    rows = []
    for k, i in enumerate(range(0, len(mix) - win + 1, hop)):
        sl = slice(i, i + win)
        follow = None
        if cand._prev is not None:  # the stream a sticky pick would keep (the hold's own lookup)
            follow = cand._follow(mix[sl], sep.inner.separate(mix[sl], SR))
        y = cand.extract(mix[sl], SR, enroll=enroll, other=other)
        g = 10.0 ** (-confidence_floor_db(cand.confidence) / 20.0)
        z = y + g * (mix[sl] - y)
        act = vad.window(mix[sl], sep.last)
        learned = cand.learn(sep.last[cand.selected_stream], act) if adaptive else False
        if k == 0:
            out[:hop], played[:hop] = y[:hop], z[:hop]
        out[i + hop : i + win], played[i + hop : i + win] = y[hop:], z[hop:]
        for a, arr in arms.items():
            za = y + 10.0 ** (-_arm_floor_db(a, float(cand.confidence), act["overlap"]) / 20.0) * (mix[sl] - y)
            if k == 0:
                arr[:hop] = za[:hop]
            arr[i + hop : i + win] = za[hop:]
        oracle = int(np.argmax([_corr(s, t[sl]) - _corr(s, o[sl]) for s in sep.last]))
        scale = 4.0 if cand.scoring == "lda" else 1.0
        rows.append({"k": k, "t": round(i / SR, 2), "scores": [round(float(s), 3) for s in cand.scores],
                     "margin": None if cand.margin_score is None else round(float(cand.margin_score), 3),
                     "hold": round(cand.margin * scale, 3), "held": bool(cand.held), "vetoed": bool(cand.vetoed),
                     "gated": bool(cand.gated), "soft": bool(getattr(cand, "soft_gated", False)), "confidence": round(float(cand.confidence), 3),
                     "selected": int(cand.selected_stream),
                     "raw": int(np.argmax(cand.scores)), "follow": follow, "oracle": oracle,
                     "target_active": bool(_rms(t[sl]) > thr_t), "scoring": cand.scoring,
                     "overlap": act["overlap"], "both_s": act["both_s"], "any_s": act["any_s"],
                     "learned": learned,
                     "other_active": bool(_rms(o[sl]) > 0.1 * _speech_level(o))})
    dom, dom_stream = other_dominant(played, mix, t, o, nz), other_dominant(out, mix, t, o, nz)
    owner = window_of_frames(len(dom), win, hop)
    for r in rows:
        n = int(np.sum(dom & (owner == r["k"])))
        r["dom_s"] = round(n * LS_FRAME / SR, 2)
        r["dom_stream_s"] = round(int(np.sum(dom_stream & (owner == r["k"]))) * LS_FRAME / SR, 2)
        r["wrong"] = n > 0
        r["cause"] = cause(r) if n else None
    arm_stats = {}
    for a, arr in arms.items():
        d = other_dominant(arr, mix, t, o, nz)
        arm_stats[a] = {"other_dominant_s": round(float(d.sum() * LS_FRAME / SR), 2),
                        "true_dropout_s": round(true_dropout_s(arr, t), 2), **target_atten(arr, mix, t, o, nz)}
    if wav_out is not None:  # T-370: the played output (tse + room at the window's floor) for listening
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        aio.save(wav_out, played.astype(np.float32), SR)
    played_stats = {"leak_db": leak_db(played, mix, t, o, nz), **target_atten(played, mix, t, o, nz),
                    "soft_windows": sum(r["soft"] for r in rows)}
    return {"scene": str(scene), "adaptive": adaptive, "learned_windows": sum(r["learned"] for r in rows),
            "gated_windows": sum(r["gated"] for r in rows), "target_muted_s": target_atten(played, mix, t, o, nz),
            "wrong_speaker_pct": round(100.0 * float(dom.sum() * LS_FRAME / SR) / (len(dom) * LS_FRAME / SR), 2),
            "arms": arm_stats, "played": played_stats, "other_dominant_s": round(float(dom.sum() * LS_FRAME / SR), 2),
            "other_dominant_stream_s": round(float(dom_stream.sum() * LS_FRAME / SR), 2),
            "true_dropout_s": round(true_dropout_s(played, t), 2),
            "seconds": round(len(dom) * LS_FRAME / SR, 2), "summary": summarize(rows), "rows": rows}


def compare_line(frozen: dict, adaptive: dict) -> str:
    """One A/B line (T-380): wrong-speaker %, gated windows, target-muted seconds (>= 10 dB down), learned windows."""
    f, a = frozen, adaptive
    return (f"wrong {f['wrong_speaker_pct']:.2f}% -> {a['wrong_speaker_pct']:.2f}% "
            f"({a['wrong_speaker_pct'] - f['wrong_speaker_pct']:+.2f} pt)  gated {f['gated_windows']} -> {a['gated_windows']}  "
            f"target-muted {f['target_muted_s']['target_down_10dB_s']:.2f} -> {a['target_muted_s']['target_down_10dB_s']:.2f} s  "
            f"learned {a['learned_windows']}")


def kill_rule(frozen: dict, adaptive: dict, max_rise_pt: float = 1.0) -> bool:
    """T-380 kill rule: True when the adaptive arm raised wrong-speaker time by more than ``max_rise_pt`` points."""
    return adaptive["wrong_speaker_pct"] - frozen["wrong_speaker_pct"] > max_rise_pt


def fixed_gate_arm(scene: Path, enroll: np.ndarray, other: np.ndarray, separator) -> dict:
    """T-402: the same scene with ``HEARING_FIXED_GATE=1`` (the fixed-slot gate), environment restored after."""
    from hearing.tse.fixed_gate import FIXED_GATE_ENV

    old = os.environ.get(FIXED_GATE_ENV)
    os.environ[FIXED_GATE_ENV] = "1"
    try:
        return run(scene, enroll, other, separator=separator)
    finally:
        if old is None:
            os.environ.pop(FIXED_GATE_ENV, None)
        else:
            os.environ[FIXED_GATE_ENV] = old


def session_atten(session: Path, clip: Path, skip_s: float = 3.9) -> dict:
    """T-326's target-attenuation check on a live ``demo_regress`` session: the pre-AGC output
    (``out_pre_agc.wav``) aligned like ``session_score`` and projected onto the placed target stem, after the
    cold start (``skip_s``)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("session_score", Path(__file__).parents[1] / "scripts" / "session_score.py")
    ss = sys.modules.setdefault("session_score", importlib.util.module_from_spec(spec))
    spec.loader.exec_module(ss)
    x, y, y_pre = (ss.read_wav(session / f) for f in ("in.wav", "out.wav", "out_pre_agc.wav"))
    rows = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines() if line.strip()]
    stems = ss.resolve_stems(clip / "manifest.json")
    placed, _ = ss.place_stems(x, stems)
    pre_al = np.zeros(len(x), np.float32)
    _, valid, _ = ss.align_output(x, y, ss._ticks(rows), ride=(y_pre, pre_al))
    use = placed["covered"] & valid
    use[: int(skip_s * SR)] = False
    return target_atten(pre_al[use], x[use], *(placed[k][use] for k in ("target", "other", "noise")))


def main(argv: list[str] | None = None) -> int:
    from eval.contrast_sweep import load_profile
    from hearing import device

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", type=Path, help="dir with <target>/ and <other>/ profile wavs")
    ap.add_argument("--scene", action="append", default=[], help="TARGET:OTHER:SCENE_DIR (repeatable)")
    ap.add_argument("--session", action="append", default=[],
                    help="LABEL:SESSION_DIR:CLIP_DIR: only the target-attenuation check on a live session (T-337)")
    ap.add_argument("--adaptive-target", action="store_true",
                    help="T-380 A/B: per scene also run the adaptive-target arm (default off) and print it beside the frozen profile")
    ap.add_argument("--fixed-gate", action="store_true",
                    help="T-402 A/B: per scene also run with HEARING_FIXED_GATE=1 (the fixed-slot gate) beside the run as set")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wav-dir", type=Path, default=None,
                    help="T-370: write each scene's played output to <dir>/<target>-<scene>.wav")
    args = ap.parse_args(argv)
    if args.session:
        res = []
        for spec in args.session:
            label, sess, clip = spec.split(":", 2)
            res.append({"row": label, "session": sess, **session_atten(Path(sess), Path(clip))})
            print(json.dumps(res[-1]), flush=True)
        if args.out:
            args.out.write_text(json.dumps(res, indent=1))
        return 0
    results = []
    for spec in args.scene:
        target, other, path = spec.split(":", 2)
        wav = None if args.wav_dir is None else args.wav_dir / f"{target}-{Path(path).name}.wav"
        enroll, neg = load_profile(args.profiles, target), load_profile(args.profiles, other)
        cache = SeparatorCache(__import__("hearing.tse.blind", fromlist=["BlindSeparator"]).BlindSeparator())
        res = run(Path(path), enroll, neg, wav, separator=cache)
        res["target"], res["other"] = target, other
        if args.adaptive_target:
            res["adaptive_arm"] = ad = run(Path(path), enroll, neg, adaptive=True, separator=cache)
            print(f"  A/B {Path(path).name} {target}:{other}  " + compare_line(res, ad), flush=True)
        if args.fixed_gate:
            res["fixed_gate_arm"] = fg = fixed_gate_arm(Path(path), enroll, neg, cache)
            print(f"  fixed-gate A/B {Path(path).name} {target}:{other}  " + compare_line(res, fg).rsplit("  learned", 1)[0],
                  flush=True)
        s = res["summary"]
        print(f"{Path(path).name:16s} {target}:{other}  other-dominant {res['other_dominant_s']:.1f} s (stream alone "
              f"{res['other_dominant_stream_s']:.1f}) / "
              f"{res['seconds']:.1f} s  true dropouts {res['true_dropout_s']:.2f} s  wrong windows {s['wrong']}  causes {s['causes']}  above/below hold "
              f"{s['above_hold']}/{s['below_hold']}  held {s['held_fired']}  sticky would fix "
              f"{s['sticky_fixes']}/{s['wrong_picks']}", flush=True)
        g = [r for r in res["rows"] if r["gated"]]
        both = [r for r in g if r["target_active"] and r["other_active"]]
        solo = [r for r in g if r["target_active"] != r["other_active"]]
        print(f"  gated {len(g)}: oracle overlap {len(both)} (VAD overlap {sum(r['overlap'] for r in both)}), oracle solo "
              f"{len(solo)} (VAD overlap {sum(r['overlap'] for r in solo)}), gated wrong {sum(r['wrong'] for r in g)} "
              f"(VAD overlap {sum(r['wrong'] and r['overlap'] for r in g)})", flush=True)
        print(f"  played           {json.dumps(res['played'])}", flush=True)
        for a, st in res["arms"].items():
            print(f"  {a:16s} {json.dumps(st)}", flush=True)
        results.append(res)
    total = summarize([r for res in results for r in res["rows"]])
    print("total", json.dumps(total))
    out = args.out or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / (
        f"wrong_speaker_audit_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": device.pick(), "sticky": os.environ.get("HEARING_PICK_STICKY"),
                               "total": total, "scenes": results}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
