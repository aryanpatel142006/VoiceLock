"""Time-to-isolated (T-092): replay :class:`hearing.tse.enroll.AutoEnroller` gate settings over cached vision.

No TSE runs here: only the enrollment gate, so a config takes well under a second per scene. For each
association scene (``eval/scenes/association``, the vision states ``eval.association_eval
--vision-cache`` wrote) and for the demo clip (``eval/fixtures/demo/grid-two-face.mp4`` replayed
through the session's ``VisionEngine`` + ``TargetSelector``, looped), it reports

* ``ready_s`` time-to-isolated: clip time when the first enrollment of the target is complete (the
  Enrolled tier becomes eligible; the live pill adds one window + hop of extraction on top),
* ``purity`` share of the kept 10 ms chunks in which the enrolled person's clean stem is the louder one
  (scenes: as ``association_eval``; demo: the share where the manifest's target stem is louder; 0.5
  is chance),
* the duty cycle of the gate: per 10 ms chunk while a target is set and the clip is not ready, why it
  was rejected (``AutoEnroller.reason``: ``no_target`` / ``unvoiced`` / ``target_quiet`` /
  ``other_talking``) or ``kept``.

    .venv/bin/python -m eval.enroll_eval --vision-cache /tmp/assoc-vc --configs t061,final
    .venv/bin/python -m eval.enroll_eval --demo --configs t061,final     # + the demo clip (vision ~1 min)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from eval import association_eval as ae
from hearing.types import StageContext, TargetState

# T-061's gate (the defaults before T-092) and the ones tried here; ``final`` = the current defaults.
CONFIGS: dict[str, dict] = {
    "t061": {"seconds": 2.0, "target_p": 0.7, "other_p": 0.7, "relax_target_p": None},
    "other08": {"target_p": 0.7, "other_p": 0.8, "relax_target_p": None},
    "other08_hold": {"target_p": 0.7, "other_p": 0.8, "other_hold_s": 0.3, "relax_target_p": None},
    "relax": {"target_p": 0.7, "other_p": 0.7, "relax_target_p": 0.6, "relax_other_p": 0.3},
    "short15": {"target_p": 0.7, "other_p": 0.7, "relax_target_p": None, "seconds": 1.5},
    "final": {},  # the current defaults (T-092)
}
REASONS = ("kept", "unvoiced", "target_quiet", "other_talking", "no_target")


def replay(audio: np.ndarray, states: list, kw: dict) -> dict:
    """Feed ``audio`` in 10 ms chunks with the per-frame vision ``states`` (lagging one frame, as live)."""
    from hearing.tse.enroll import AutoEnroller

    enr = AutoEnroller(sr=ae.SR, **kw)
    ctx = StageContext.default()
    reasons: Counter = Counter()
    first_target = ready = None
    spans: list = []
    tid = None
    for i in range(min(len(states), len(audio) // ae.SPF)):
        st = states[i - 1] if i > 0 else ae.FrameState([], None, {})
        ctx.tracks = st.tracks
        ctx.target = TargetState(st.target_id, 1.0 if st.target_id is not None else 0.0, st.target_id is not None, 0.0)
        for c in range(ae.SPF // ae.HOP):
            pos = i * ae.SPF + c * ae.HOP
            before = enr.ready
            enr.update(audio[pos : pos + ae.HOP], ctx)
            if st.target_id is not None and first_target is None:
                first_target = pos / ae.SR
            if ready is None:
                reasons[enr.reason] += 1
            if enr.ready and not before and ready is None:
                ready = (pos + ae.HOP) / ae.SR
                spans, tid = list(enr.spans), enr.track_id
    total = sum(v for k, v in reasons.items() if k != "no_target") or 1
    return {"ready_s": ready, "first_target_s": first_target, "spans": spans, "track_id": tid,
            "duty": {k: round(reasons[k] / total, 3) for k in REASONS if k != "no_target"},
            "no_target_s": reasons["no_target"] / 100}


def _purity_scene(r, states, spans, tid) -> float | None:
    sides: dict[int, str] = {}
    for st in states:
        sides.update(st.sides)
    who = {r.side_of("A"): 0, r.side_of("B"): 1}.get(sides.get(tid, ""))
    if who is None or not spans:
        return None
    stems = (r.stems["A"], r.stems["B"])
    pure = sum(ae._energy(stems[who][p : p + n]) > ae._energy(stems[1 - who][p : p + n]) for p, n in spans)
    return pure / len(spans)


class _Unpickler(pickle.Unpickler):
    """``association_eval`` run as ``__main__`` pickles its ``FrameState`` under that module name."""

    def find_class(self, module, name):
        if module == "__main__":
            module = "eval.association_eval"
        return super().find_class(module, name)


def scene_inputs(cache: Path, pair: str = "mf", snr: float = 5.0, noise: str = "babble", seed: int = 0):
    scenes = ae.load_scenes()
    talkers = {s: ae.load_talker(s) for s in ae.PAIRS[pair]}
    for si, scene in enumerate(scenes):
        for variant in ("ab", "ba"):
            r = ae.render(scene, variant, talkers, snr, noise, seed + si, pair)
            spec = [dataclasses.asdict(scene), variant, snr, noise, seed + si, "light_asd", "mediapipe"]
            if pair != "mf":
                spec.append(ae.PAIRS[pair])
            key = hashlib.sha1(json.dumps(spec).encode()).hexdigest()[:16]
            tag = "" if pair == "mf" else f"{pair}_"
            path = cache / f"{tag}{scene.name}_{variant}_{key}.pkl"
            if not path.exists():
                states, _ = ae.run_vision(r)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps((states, {})))
            else:
                with path.open("rb") as f:
                    states, _ = _Unpickler(f).load()
            yield r, states


def demo_states(clip_path: Path, loops: int, cache: Path | None) -> tuple[np.ndarray, list, dict]:
    """Audio, per-frame states and stems of the demo clip looped ``loops`` times, through the session's
    VisionEngine (faces + Light-ASD + TargetSelector, unthreaded: every frame is applied)."""
    from eval.fixtures import video
    from hearing.server.vision import VisionEngine

    frames, _ = video.read_video(clip_path, fps=ae.FPS)
    audio = video.read_audio(clip_path)
    n = min(len(frames), len(audio) // ae.SPF)
    frames, audio = frames[:n], audio[: n * ae.SPF]
    stems = {}
    man = clip_path.with_name("manifest.json")
    meta = json.loads(man.read_text()) if man.exists() else {}
    files = {}
    if "stems" in meta:  # v2 demo scenes (T-097, eval/fixtures/demo/demo-*/): stems next to the clip
        files = {"target": man.parent / meta["stems"]["target"], "interferer": man.parent / meta["stems"]["other"]}
    elif meta:
        d = ae.grid_root().parents[1] / "av_full" / str(meta.get("source_fixture"))
        files = {k: d / f"{k}.wav" for k in ("target", "interferer")}
    if files and all(f.exists() for f in files.values()):
        import soundfile as sf

        for k, f in files.items():
            x, _ = sf.read(f, dtype="float32")
            stems[k] = np.tile(x, -(-len(audio) // len(x)))[: len(audio)]
    audio_l = np.tile(audio, loops)
    stems = {k: np.tile(v, loops) for k, v in stems.items()}
    key = hashlib.sha1(f"{clip_path.resolve()}:{clip_path.stat().st_mtime}:{loops}".encode()).hexdigest()[:12]
    path = cache / f"demo_{key}.pkl" if cache else None
    if path is not None and path.exists():
        return audio_l, pickle.loads(path.read_bytes()), stems
    from eval.association_eval import _jpeg

    engine = VisionEngine("mediapipe", "light_asd", threaded=False, lips=False)
    ctx = StageContext.default()
    states = []
    try:
        for i in range(n * loops):
            ts = (i + 1) / ae.FPS - 0.005
            for c in range(ae.SPF // ae.HOP):
                pos = i * ae.SPF + c * ae.HOP
                engine.push_audio(audio_l[pos : pos + ae.HOP], pos / ae.SR)
            engine.submit(i, ts, _jpeg(frames[i % n]))
            engine.poll(ctx)
            states.append(ae.FrameState([replace(t) for t in ctx.tracks], ctx.target.track_id, {}))
    finally:
        engine.close()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(states))
    return audio_l, states, stems


def _purity_demo(stems: dict, spans: list) -> float | None:
    if not spans or set(stems) != {"target", "interferer"}:
        return None
    t = sum(ae._energy(stems["target"][p : p + n]) > ae._energy(stems["interferer"][p : p + n]) for p, n in spans)
    return t / len(spans)  # share where the manifest's target stem is the louder (was max(t, n - t): flattering)


def solo_before(manifest: Path, t: float | None) -> float | None:
    """Seconds of the target speaking alone (manifest ``faces[].speaking``, v2 demo scenes) before clip time
    ``t``: the "Isolated <= 8 s of the target talking alone" criterion. ``None`` without that manifest."""
    if t is None or not manifest.exists():
        return None
    faces = json.loads(manifest.read_text()).get("faces") or []
    tgt = [f["speaking"] for f in faces if f.get("role") == "target" and "speaking" in f]
    oth = [iv for f in faces if f.get("role") != "target" for iv in f.get("speaking", [])]
    if not tgt:
        return None
    grid = np.arange(0.0, t, 0.01) + 0.005
    on = np.zeros(len(grid), bool)
    for a, b in tgt[0]:
        on |= (grid >= a) & (grid < b)
    for a, b in oth:
        on &= ~((grid >= a) & (grid < b))
    return float(on.sum() * 0.01)


# --- T-145: the built-in MacBook mic, the teammates' real takes at laptop-mic levels -----------------
TEAM_LEVELS = (-20.0, -35.0, -45.0, -50.0)  # active-speech RMS, dBFS; -20 is the reference run
TEAM_FLOOR_DB = -58.0  # room floor (pink noise RMS), the T-119 live session's mic
TEAM_LEAD_S = 1.5  # room alone before the target starts talking (the gate learns its floor)
TEAM_RUN_S = 12.0  # speech offered per run
TEAM_STARTS_S = (0.0, 15.0, 30.0)  # one run per start offset in the take (seed = run index)
METER_FRAME_MS = 20  # judge.js FRAME_MS: the meter is fed one peak per sent frame
METER_RELEASE_DB_PER_S = 24.0  # devices.js RELEASE_DB_PER_S
METER_PAINT_MS = 50  # devices.js paints every 50 ms
METER_FLOOR_DB = -60.0
ASD_HANG_S = 0.3  # ASD keeps "speaking" this long over a pause between words
PICK_WIN_S, PICK_HOP_S = 1.5, 0.75  # closed-set pick windows


def _chunk_db(x: np.ndarray, hop: int = 160) -> np.ndarray:
    n = len(x) // hop
    c = np.asarray(x[: n * hop], np.float64).reshape(n, hop)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(c**2, axis=1)), 1e-6))


def active_chunks(x: np.ndarray, hop: int = 160) -> np.ndarray:
    """10 ms chunks of a clean take that are speech (``eval.demo_team``'s VAD rule: within 30 dB of the
    take's loud chunks, 8 dB above its quiet ones, never under -50 dBFS at the take's own level)."""
    from eval.demo_team import VAD_ABOVE_NOISE_DB, VAD_RANGE_DB

    db = _chunk_db(x, hop)
    if not len(db):
        return np.zeros(0, bool)
    loud, quiet = np.percentile(db, 95), np.percentile(db, 20)
    return db >= max(loud - VAD_RANGE_DB, quiet + VAD_ABOVE_NOISE_DB)


def scale_to_active(x: np.ndarray, level_db: float, active: np.ndarray, hop: int = 160) -> np.ndarray:
    """``x`` scaled so its active-speech RMS (the chunks in ``active``) is ``level_db`` dBFS."""
    n = len(active) * hop
    c = np.asarray(x[:n], np.float64).reshape(-1, hop)[active]
    rms = float(np.sqrt(np.mean(c**2))) if len(c) else 0.0
    return (np.asarray(x, np.float32) * np.float32(10 ** (level_db / 20) / max(rms, 1e-9))).astype(np.float32)


def room_noise(n: int, floor_db: float, seed: int) -> np.ndarray:
    from eval.fixtures.sources import pink_noise

    return (pink_noise(n, np.random.default_rng(seed)) * np.float32(10 ** (floor_db / 20))).astype(np.float32)


def meter_levels(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """What the /judge meter shows (devices.js): per 20 ms frame the peak (dBFS, floor -90) lifts the level,
    which falls 24 dB/s; read every 50 ms. One value per paint."""
    f = round(sr * METER_FRAME_MS / 1000)
    level, out, t_paint = METER_FLOOR_DB, [], 0.0
    dt = METER_FRAME_MS / 1000
    for i in range(len(x) // f):
        level = max(METER_FLOOR_DB, level - METER_RELEASE_DB_PER_S * dt)
        peak = float(np.max(np.abs(x[i * f : (i + 1) * f])))
        level = max(level, max(-90.0, 20 * np.log10(peak + 1e-9)))
        t = (i + 1) * dt
        while t_paint + METER_PAINT_MS / 1000 <= t + 1e-9:
            t_paint += METER_PAINT_MS / 1000
            out.append(level)
    return np.asarray(out)


def asd_track(active: np.ndarray, hang_s: float = ASD_HANG_S) -> np.ndarray:
    """ASD of the lone target from the clean take's activity: speaking during words and ``hang_s`` after."""
    hang = round(hang_s * 100)
    on = active.copy()
    last = -10**9
    for i, a in enumerate(active):
        if a:
            last = i
        on[i] = i - last <= hang
    return on


def team_scene(take: np.ndarray, level_db: float, floor_db: float, start_s: float, seconds: float, seed: int,
               sr: int = 16000) -> dict:
    """``seconds`` of ``take`` from ``start_s`` at ``level_db`` (active RMS of the whole take) after
    ``TEAM_LEAD_S`` of room, over pink room noise at ``floor_db``; per 10 ms chunk the ASD verdict."""
    hop = sr // 100
    act_all = active_chunks(take, hop)
    scaled = scale_to_active(take, level_db, act_all, hop)
    a, b = round(start_s * sr) // hop * hop, round((start_s + seconds) * sr) // hop * hop
    seg = scaled[a:b]
    lead = round(TEAM_LEAD_S * sr) // hop * hop
    x = np.concatenate([np.zeros(lead, np.float32), seg])
    x = x + room_noise(len(x), floor_db, seed)
    asd = np.concatenate([np.zeros(lead // hop, bool), asd_track(act_all[a // hop : b // hop])])
    return {"audio": x, "asd": asd[: len(x) // hop], "active": np.concatenate(
        [np.zeros(lead // hop, bool), act_all[a // hop : b // hop]])[: len(x) // hop], "lead": lead}


def team_replay(scene: dict, sr: int = 16000) -> dict:
    """The session enrollment path (``AutoEnroller``, its ``VoicedGate``) over a lone-target scene, plus a
    2 s calibration window (``AutoCalibrator``'s) opened at the first word, and the meter during speech."""
    from hearing.tse.enroll import AutoEnroller
    from hearing.types import FaceTrack

    hop = sr // 100
    x, asd = scene["audio"], scene["asd"]
    ctx = StageContext.default()
    ctx.target = TargetState(1, 1.0, False, 0.0)
    track = FaceTrack(1, (0.4, 0.4, 0.2, 0.3), None, 0.05, 0.0)
    ctx.tracks = [track]
    en, cal = AutoEnroller(sr=sr, other_s=0.0), AutoEnroller(sr=sr, other_s=0.0)
    first = int(np.argmax(asd)) if asd.any() else None
    ready = ready_clip = cal_s = None
    talking = kept = 0
    for i in range(len(x) // hop):
        track.p_speaking = 0.9 if asd[i] else 0.05
        chunk = x[i * hop : (i + 1) * hop]
        if first is not None and i == first:
            cal.calibrate(2.0)
        if i >= scene["lead"] // hop and not cal.calibrating and cal_s is None and first is not None and i > first:
            cal_s = cal.collected_s if cal.calibrated else 0.0
        cal.update(chunk, ctx)
        was = en.ready
        en.update(chunk, ctx)
        if not was:
            if asd[i]:
                talking += 1
                kept += en.reason == "kept"
            if en.ready and ready is None:
                ready, ready_clip = talking / 100, (i + 1 - first) / 100
    m = meter_levels(x, sr)
    step = METER_PAINT_MS // 10  # paint k reads the level after chunk (k + 1) * step - 1
    talk = asd[step - 1 :: step][: len(m)]
    m_talk = m[: len(talk)][talk] if talk.any() else m
    f = round(sr * METER_FRAME_MS / 1000)
    act = scene["active"]
    peaks = [20 * np.log10(float(np.max(np.abs(x[j * f : (j + 1) * f]))) + 1e-9)
             for j in range(len(x) // f) if act[j * f // hop : (j + 1) * f // hop].mean() >= 0.5]
    return {"ready_s": ready, "ready_clip_s": ready_clip, "kept_share": kept / talking if talking else None,
            "floor_db": en.gate.floor_db, "threshold_db": en.gate.threshold_db,
            "calibrated_s": cal_s, "clip": en.clip(),
            "meter_p10_db": float(np.percentile(m_talk, 10)), "meter_p50_db": float(np.percentile(m_talk, 50)),
            "frame_peak_p10_db": float(np.percentile(peaks, 10)) if peaks else None}


def _pick_windows(take: np.ndarray, level_db: float, floor_db: float, seed: int, sr: int = 16000):
    hop = sr // 100
    act = active_chunks(take, hop)
    x = scale_to_active(take, level_db, act, hop) + room_noise(len(take), floor_db, seed)
    w, h = round(PICK_WIN_S * sr), round(PICK_HOP_S * sr)
    return [x[a : a + w] for a in range(0, len(x) - w + 1, h) if act[a // hop : (a + w) // hop].mean() >= 0.5]


def closed_set_pick(takes: dict, profiles: dict, levels, floor_db: float, enrolled: dict | None = None) -> dict:
    """Closed-set pick over the two teammates (T-134): per window of each take at each level, the LDA fitted
    on both saved profiles' piece embeddings says whose voice it is; with ``enrolled`` ({level: {name:
    clip}}) also the cosine to each voice's session-enrolled clip at that level. Accuracy per level."""
    from hearing import models
    from hearing.tse import enrolled as tse
    from hearing.tse.enroll import split_pieces

    backend = models.load(tse.MODEL)
    names = sorted(takes)
    emb = {n: [tse._unit_embedding(backend, p) for p in split_pieces(profiles[n])] for n in names}
    lda = tse.fit_lda(emb[names[0]], emb[names[1]])
    out = {}
    for lv in levels:
        right = right_cos = n_win = 0
        ref = None
        if enrolled and enrolled.get(lv) and all(enrolled[lv].get(n) is not None for n in names):
            ref = {n: tse._unit_embedding(backend, enrolled[lv][n]) for n in names}
        for k, n in enumerate(names):
            for win in _pick_windows(takes[n], lv, floor_db, seed=100 + k):
                e = tse._unit_embedding(backend, win)
                n_win += 1
                if lda is not None:
                    right += (float(lda[0] @ e - lda[1]) > 0) == (k == 0)
                if ref is not None:
                    right_cos += (float(ref[names[0]] @ e) > float(ref[names[1]] @ e)) == (k == 0)
        out[lv] = {"windows": n_win, "lda_acc": right / n_win if n_win and lda is not None else None,
                   "enrolled_acc": right_cos / n_win if n_win and ref is not None else None}
    return out


def team_main(team: Path, levels, floor_db: float, runs: int, pick: bool, out_path: Path | None) -> int:
    """``--team DIR --level-db ...``: ready time, kept share, calibration and meter per voice per level."""
    import os

    from hearing.audio.io import load

    dirs = sorted(d for d in team.iterdir() if d.is_dir() and (d / "fixture-01.wav").exists())
    takes = {d.name: load(d / "fixture-01.wav", ae.SR)[0] for d in dirs}
    profiles = {d.name: load(d / "profile-01.wav", ae.SR)[0] for d in dirs if (d / "profile-01.wav").exists()}
    rows, enrolled = [], {}
    print(f"{'voice':<8}{'level':>6}{'run':>4}{'ready s':>9}{'kept':>6}{'floor':>7}{'thr':>6}{'cal s':>7}"
          f"{'meter p10':>10}{'p50':>6}{'peak p10':>9}")
    for lv in levels:
        for name, take in takes.items():
            for r in range(runs):
                start = TEAM_STARTS_S[r % len(TEAM_STARTS_S)]
                res = team_replay(team_scene(take, lv, floor_db, start, TEAM_RUN_S, seed=r))
                clip = res.pop("clip")
                if r == 0:
                    enrolled.setdefault(lv, {})[name] = clip
                res.update(voice=name, level_db=lv, run=r, start_s=start)
                rows.append(res)
                print(f"{name:<8}{lv:>6.0f}{r:>4}{_fmt(res['ready_s']):>9}{_fmt(res['kept_share']):>6}"
                      f"{_fmt(res['floor_db'], '{:.1f}'):>7}{res['threshold_db']:>6.1f}{_fmt(res['calibrated_s']):>7}"
                      f"{res['meter_p10_db']:>10.1f}{res['meter_p50_db']:>6.1f}{_fmt(res['frame_peak_p10_db'], '{:.1f}'):>9}")
    picks = None
    if pick and len(takes) == 2 and len(profiles) == 2:
        picks = closed_set_pick(takes, profiles, levels, floor_db, enrolled)
        for lv, p in picks.items():
            print(f"closed-set pick at {lv:.0f} dBFS: LDA over profiles {_fmt(p['lda_acc'], '{:.3f}')}, "
                  f"cosine to session-enrolled clips {_fmt(p['enrolled_acc'], '{:.3f}')} ({p['windows']} windows)")
    m45 = [r["meter_p10_db"] for r in rows if r["level_db"] == -45.0]
    if m45:
        print(f"meter p10 while talking at -45 dBFS: {min(m45):.1f} dBFS (worst run) -> TOO_QUIET_DB {min(m45) - 3:.1f}")
    out_path = out_path or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / (
        f"enroll_level_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"team": str(team), "floor_db": floor_db, "levels": list(levels), "rows": rows,
                                    "picks": {str(k): v for k, v in (picks or {}).items()}}, indent=1))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def _fmt(v, f="{:.2f}"):
    return "never" if v is None else f.format(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="t061,final")
    ap.add_argument("--define", action="append", default=[], metavar="NAME=JSON", help="extra AutoEnroller kwargs")
    ap.add_argument("--vision-cache", type=Path, default=Path("/tmp/assoc-vc"))
    ap.add_argument("--pairs", default="mf")
    ap.add_argument("--demo", action="store_true", help="also the demo clip replay")
    ap.add_argument("--no-scenes", action="store_true")
    ap.add_argument("--clip", type=Path, default=None)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--team", type=Path, default=None, help="T-145: <dir>/<name>/fixture-01.wav (+ profile-01.wav)")
    ap.add_argument("--level-db", default=",".join(f"{v:g}" for v in TEAM_LEVELS),
                    help="active-speech RMS levels for --team, dBFS")
    ap.add_argument("--floor-db", type=float, default=TEAM_FLOOR_DB)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-pick", action="store_true", help="--team without the ECAPA closed-set pick")
    args = ap.parse_args(argv)
    if args.team is not None:
        levels = [float(v) for v in args.level_db.split(",") if v]
        return team_main(args.team, levels, args.floor_db, args.runs, not args.no_pick, args.out)
    for d in args.define:
        name, _, spec = d.partition("=")
        CONFIGS[name] = json.loads(spec)
    configs = [c for c in args.configs.split(",") if c]
    out: dict = {c: {"scenes": [], "demo": None} for c in configs}
    t0 = time.perf_counter()
    if not args.no_scenes:
        for pair in [p for p in args.pairs.split(",") if p]:
            for r, states in scene_inputs(args.vision_cache, pair):
                for c in configs:
                    res = replay(r.mix, states, CONFIGS[c])
                    res["purity"] = _purity_scene(r, states, res.pop("spans"), res["track_id"])
                    res.update(scene=r.scene.name, variant=r.variant, pair=pair)
                    out[c]["scenes"].append(res)
    if args.demo:
        clip = args.clip or ae.grid_root().parents[1].parent / "demo" / "grid-two-face.mp4"
        audio, states, stems = demo_states(clip, args.loops, args.vision_cache)
        for c in configs:
            res = replay(audio, states, CONFIGS[c])
            res["purity"] = _purity_demo(stems, res.pop("spans"))
            res["solo_s"] = solo_before(clip.with_name("manifest.json"), res["ready_s"]) if args.loops == 1 else None
            out[c]["demo"] = res
    print(f"{'config':<14}{'n':>4}{'ready mean':>11}{'worst':>7}{'never':>6}{'purity':>8}{'worst':>7}"
          f"   duty (kept/unvoiced/target_quiet/other_talking)   demo ready / purity / duty")
    for c in configs:
        sc = out[c]["scenes"]
        ready = [s["ready_s"] for s in sc if s["ready_s"] is not None]
        pur = [s["purity"] for s in sc if s["purity"] is not None]
        duty = {k: float(np.mean([s["duty"][k] for s in sc])) if sc else 0.0 for k in REASONS[:-1]}
        d = out[c]["demo"]
        summ = {"ready_mean": float(np.mean(ready)) if ready else None, "ready_worst": max(ready, default=None),
                "never": len(sc) - len(ready), "purity_mean": float(np.mean(pur)) if pur else None,
                "purity_worst": min(pur, default=None), "duty": duty}
        out[c]["summary"] = summ
        demo = "" if d is None else (f"{_fmt(d['ready_s'])} s / {_fmt(d['purity'])} / "
                                     + "/".join(f"{d['duty'][k]:.2f}" for k in REASONS[:-1])
                                     + ("" if d.get("solo_s") is None else f" / solo {d['solo_s']:.2f} s"))
        print(f"{c:<14}{len(sc):>4}{_fmt(summ['ready_mean']):>11}{_fmt(summ['ready_worst']):>7}{summ['never']:>6}"
              f"{_fmt(summ['purity_mean'], '{:.3f}'):>8}{_fmt(summ['purity_worst']):>7}   "
              + "/".join(f"{duty[k]:.2f}" for k in REASONS[:-1]) + f"   {demo}")
    print(f"wall {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"configs": {c: CONFIGS[c] for c in configs}, "results": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
