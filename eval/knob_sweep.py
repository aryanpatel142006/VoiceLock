"""Broad offline sweep of the existing enrolled-tier knobs on a tuning set disjoint from the test scenes (T-340).

    .venv/bin/python -m eval.knob_sweep build [--out DIR]                # tuning scenes from the enrollment cuts only
    .venv/bin/python -m eval.knob_sweep sweep --scene T:O:DIR [...]      # every knob setting -> JSON + Pareto table
    .venv/bin/python -m eval.knob_sweep sweep --tuning DIR               # every scene under DIR (manifest roles)

**Tuning set.** Two-talker aryan + joseph scenes in the ``demo_scene`` layout (``stems/{mix,target,other,noise}.wav``
+ ``manifest.json``) built ONLY from ``team/<name>/profile-01.wav`` and the T-306 augmentations whose manifest
``source_file`` is ``profile-01.wav`` and whose ``source_sha256`` equals that file's hash. Anything derived from a
``fixture-*`` cut is refused, because the T-141 test scenes (``t141/<target>/demo-team-*``) come from those cuts
(:func:`provenance`). Each scene is 24 s: target alone 0-4 s, alternating turns 4-12 s, overlap 12-20 s with the
other talker at the scene's SIR (0 / -5 / -10 dB: the other is that much LOUDER), target alone 20-24 s; with no noise
or DEMAND PCAFETER cafeteria noise at SNR 10 dB under the target's speech. Both target roles.

**Sweep.** The pick depends on ``HEARING_ENROLLED_CONTRAST`` and ``HEARING_GATED_PLAY``; the residual floor on
``HEARING_GATED_FLOOR_DB`` / ``HEARING_UNSURE_DB`` only through :func:`hearing.fusion.mixer.confidence_floor_db`.
So each scene is separated once per window (cached), the extractor replays once per (contrast, play) and the
played output (``stream + floor * (window mix - stream)``, as ``eval.wrong_speaker_audit``) is recomposed for every
(floor, unsure). Per setting and scene: ``wrong_s`` (``wrong_speaker_audit.other_dominant``: the session_score
rule), ``leak_db`` (the other stem's least-squares energy in the output over the input, session_score's formula),
``atten_s`` (T-326's check: 200 ms frames with clear target energy whose target projection is >= 10 dB below its
input level) and ``dropout_s`` (session_score's true dropouts). Results: ``$HEARING_RESULTS_DIR/knob_sweep_<stamp>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

SR = 16000
WIN_S, HOP_S = 1.0, 0.5
LS_FRAME = 3200
SPEECH_LEVEL_DB = -18.5
PEAK = 0.95
SIRS = (0.0, -5.0, -10.0)
NOISES = ("none", "cafe")
NAMES = ("aryan", "joseph")
CAFE_SNR_DB = 10.0
CAFE_ZIP, CAFE_MEMBER = "public_voice/PCAFETER_16k.zip", "PCAFETER/ch01.wav"
AUGS = ("clean", "reverb_02", "reverb_05", "opus16", "opus32")  # pitch/combo shift the voice itself: not used
# (section, start s, end s, who): the demo_scene shape, shortened to 24 s
SECTIONS = (("target_alone", 0.0, 4.0, "target"), ("alternate", 4.0, 12.0, "alternate"),
            ("overlap", 12.0, 20.0, "both"), ("target_alone_2", 20.0, 24.0, "target"))
DUR_S = 24.0
TURN_S = 2.0

CONTRASTS = (0.0, -0.15, -0.3, -0.45)
FLOORS = (None, 12.0, 20.0)
UNSURES = (None, 10.0, 14.0)
PLAYS = (None, "stream")
ENVS = ("HEARING_ENROLLED_CONTRAST", "HEARING_GATED_FLOOR_DB", "HEARING_UNSURE_DB", "HEARING_GATED_PLAY")


def fixtures_dir() -> Path:
    return Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated"))


def augmented_dir() -> Path:
    main = Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")).resolve().parent
    return Path(os.environ.get("HEARING_AUGMENTED_DIR", main / "augmented" / "T-306"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(manifest: dict, profile_sha: dict[str, str]) -> tuple[list[dict], dict]:
    """T-306 items allowed in the tuning set: ``source_file == profile-01.wav`` with the profile's own hash, in
    :data:`AUGS`. Returns ``(items, check)``; ``check`` counts what was kept and refused (fixture-derived etc.)."""
    keep, refused = [], {}
    for f in manifest["files"]:
        ok = (f["source_file"] == "profile-01.wav" and profile_sha.get(f["speaker"]) == f["source_sha256"]
              and f["augmentation"] in AUGS)
        if ok:
            keep.append(f)
        else:
            key = f"{f['source_file']}:{f['augmentation']}"
            refused[key] = refused.get(key, 0) + 1
    if any(not f["source_file"].startswith("profile-") for f in keep):
        raise AssertionError("a fixture-derived item reached the tuning set")
    kept = {}
    for f in keep:
        kept[f"{f['speaker']}:{f['augmentation']}"] = kept.get(f"{f['speaker']}:{f['augmentation']}", 0) + 1
    return keep, {"kept": kept, "refused": refused, "profile_sha256": profile_sha}


def speech_chunks(x: np.ndarray, min_s: float = 0.8, max_s: float = TURN_S) -> list[np.ndarray]:
    """Cut ``x`` into speech pieces of ``min_s``..``max_s`` (20 ms energy VAD, -35 dB under the 95th percentile)."""
    f = 320
    m = len(x) // f
    r = np.sqrt(np.mean(np.asarray(x[: m * f], np.float64).reshape(m, f) ** 2, axis=1))
    act = r > max(1e-4, np.percentile(r, 95) * 10 ** (-35 / 20))
    out, i = [], 0
    while i < m:
        if not act[i]:
            i += 1
            continue
        j = i
        while j < m and act[j] and (j - i) * f < max_s * SR:
            j += 1
        if (j - i) * f >= min_s * SR:
            out.append(np.asarray(x[i * f: j * f], np.float32))
        i = j
    return out


def lay(pool: list[np.ndarray], start: int, end: int, cursor: int, track: np.ndarray, gap: int = SR // 4) -> int:
    """Place pool pieces (cyclic from ``cursor``) into ``track[start:end]``; returns the new cursor."""
    c = start
    while c < end and pool:
        p = pool[cursor % len(pool)]
        n = min(len(p), end - c)
        if n < SR // 2:
            break
        track[c: c + n] += p[:n]
        c += n + gap
        cursor += 1
    return cursor


def compose(target_pool, other_pool, sir_db: float, noise: np.ndarray | None):
    """Stems of one 24 s scene in the demo shape; levels as ``demo_scene.compose`` plus the SIR on the other."""
    n = round(DUR_S * SR)
    t, o = np.zeros(n, np.float32), np.zeros(n, np.float32)
    ct = co = 0
    for _, a, b, who in SECTIONS:
        s, e = round(a * SR), round(b * SR)
        if who == "target":
            ct = lay(target_pool, s, e, ct, t)
        elif who == "both":
            ct = lay(target_pool, s, e, ct, t)
            co = lay(other_pool, s, e, co, o)
        else:
            for k, ts in enumerate(range(s, e, round(TURN_S * SR))):
                te = min(e, ts + round(TURN_S * SR))
                if k % 2 == 0:
                    ct = lay(target_pool, ts, te, ct, t)
                else:
                    co = lay(other_pool, ts, te, co, o)
    ref = 10 ** (SPEECH_LEVEL_DB / 20)
    t *= ref / _active_rms(t)
    o *= ref * 10 ** (-sir_db / 20) / _active_rms(o)
    nz = np.zeros(n, np.float32) if noise is None else np.resize(noise, n).astype(np.float32)
    if noise is not None:
        nz *= ref / (float(np.sqrt(np.mean(nz.astype(np.float64) ** 2))) + 1e-12) / 10 ** (CAFE_SNR_DB / 20)
    mix = t + o + nz
    g = min(1.0, PEAK / float(np.max(np.abs(mix))))
    return {"mix": mix * g, "target": t * g, "other": o * g, "noise": nz * g}


def _active_rms(x: np.ndarray) -> float:
    a = x[np.abs(x) > 1e-6]
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12) if len(a) else 1e-12


def cafe_noise(root: Path) -> np.ndarray:
    import soundfile as sf

    with zipfile.ZipFile(root / CAFE_ZIP) as z:
        x, sr = sf.read(io.BytesIO(z.read(CAFE_MEMBER)), dtype="float32")
    if sr != SR:
        from hearing.audio import io as aio

        x = aio.resample(x, sr, SR)
    return x[60 * SR: 60 * SR + round(DUR_S * SR)]  # 24 s from minute 1 (past any start-up)


def build(out_root: Path) -> dict:
    """Write the 12 tuning scenes (target x SIR x noise) and ``tuning_manifest.json``; returns that manifest."""
    from hearing.audio import io as aio

    fx = fixtures_dir()
    aug = augmented_dir()
    man = json.loads((aug / "manifest.json").read_text())
    profile_sha = {n: sha256(fx / "team" / n / "profile-01.wav") for n in NAMES}
    items, check = provenance(man, profile_sha)
    pools = {}
    for n in NAMES:
        pieces = []
        for f in sorted((f for f in items if f["speaker"] == n), key=lambda f: (f["augmentation"], f["path"])):
            pieces += [(f["path"], p) for p in speech_chunks(aio.load(aug / f["path"], SR)[0])]
        rng = np.random.default_rng(340 + len(n))
        order = rng.permutation(len(pieces))
        pools[n] = [pieces[i] for i in order]
    cafe = cafe_noise(fx)
    scenes = []
    for k, (target, sir, noise) in enumerate(itertools.product(NAMES, SIRS, NOISES)):
        other = next(x for x in NAMES if x != target)
        shift = 7 * k  # each scene starts the pools at a different piece
        tp = [p for _, p in pools[target][shift:] + pools[target][:shift]]
        op = [p for _, p in pools[other][shift:] + pools[other][:shift]]
        stems = compose(tp, op, sir, cafe if noise == "cafe" else None)
        name = f"tune-{target}-sir{int(sir):+d}-{noise}".replace("+0", "0")
        d = out_root / name
        (d / "stems").mkdir(parents=True, exist_ok=True)
        for key, x in stems.items():
            aio.save(d / "stems" / f"{key}.wav", x.astype(np.float32), SR)
        meta = {"version": 2, "name": name, "kind": "tuning_scene", "task": "T-340", "target_name": target,
                "other_name": other, "sir_db_overlap": sir, "noise": noise,
                "snr_db": CAFE_SNR_DB if noise == "cafe" else None, "duration_s": DUR_S, "sample_rate": SR,
                "speech_level_dbfs": SPEECH_LEVEL_DB, "stems": {key: f"stems/{key}.wav" for key in stems},
                "sections": [{"name": s, "start_s": a, "end_s": b} for s, a, b, _ in SECTIONS],
                "source": {"speech": "team/<name>/profile-01.wav + T-306 augmentations of it", "augmentations": AUGS,
                           "noise": f"DEMAND {CAFE_MEMBER}" if noise == "cafe" else None}}
        (d / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        scenes.append(name)
    out = {"task": "T-340", "scenes": scenes, "provenance": check,
           "pool_pieces": {n: len(pools[n]) for n in NAMES},
           "pool_sources": {n: sorted({s for s, _ in pools[n]}) for n in NAMES}}
    (out_root / "tuning_manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


# --- sweep ---------------------------------------------------------------------------------------------------


class CachedSeparator:
    """Separator that remembers each window's streams by key: the separation does not depend on any knob."""

    def __init__(self, inner):
        self.inner, self.cache, self.key, self.last = inner, {}, None, None

    @property
    def device(self):
        return self.inner.device

    def load(self):
        self.inner.load()

    def separate(self, mix, sr=SR):
        if self.key is None or self.key not in self.cache:
            out = self.inner.separate(mix, sr)
            if self.key is not None:
                self.cache[self.key] = out
        else:
            out = self.cache[self.key]
        self.last = out
        return [np.array(s, copy=True) for s in out]


def set_env(contrast=None, floor=None, unsure=None, play=None) -> None:
    for k, v in zip(ENVS, (contrast, floor, unsure, play), strict=True):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def setting_name(contrast, floor, unsure, play) -> str:
    f = lambda v: "unset" if v is None else f"{v:g}"
    return f"c{contrast:g}_f{f(floor)}_u{f(unsure)}_{'stream' if play else 'zero'}"


def replay(scene: dict, sep: CachedSeparator, enroll, other, contrast: float, play) -> list[dict]:
    """One pass of the extractor over the scene's windows; per window the streamed hop's ``y``, mix and confidence."""
    from hearing.tse.enrolled import EnrolledCandidate

    set_env(contrast=contrast, play=play)
    cand = EnrolledCandidate(separator=sep, contrast=contrast)
    cand.prepare(enroll, other, SR)
    mix = scene["mix"]
    win, hop = round(WIN_S * SR), round(HOP_S * SR)
    out = []
    for k, i in enumerate(range(0, len(mix) - win + 1, hop)):
        sep.key = (scene["id"], i)
        y = cand.extract(mix[i: i + win], SR, enroll=enroll, other=other)
        out.append({"k": k, "i": i, "y": y, "conf": float(cand.confidence), "gated": bool(cand.gated)})
    sep.key = None
    return out


def compose_output(scene: dict, wins: list[dict]) -> np.ndarray:
    """``stream + g * (window mix - stream)`` per window with g from the CURRENT env's ``confidence_floor_db``."""
    from hearing.fusion.mixer import confidence_floor_db

    mix = scene["mix"]
    win, hop = round(WIN_S * SR), round(HOP_S * SR)
    played = np.zeros_like(mix)
    for w in wins:
        i, y = w["i"], w["y"]
        g = 10.0 ** (-confidence_floor_db(w["conf"]) / 20.0)
        z = y + g * (mix[i: i + win] - y)
        if w["k"] == 0:
            played[:hop] = z[:hop]
        played[i + hop: i + win] = z[hop:]
    return played


def lstsq_gains(z: np.ndarray, refs: list[np.ndarray]) -> np.ndarray:
    from eval.wrong_speaker_audit import _frames

    zf, rf = _frames(z), np.stack([_frames(r) for r in refs], axis=2)
    g = np.zeros((len(zf), len(refs)))
    for i in range(len(zf)):
        a = rf[i]
        lam = 1e-6 * (np.trace(a.T @ a) + 1e-9)
        g[i] = np.linalg.solve(a.T @ a + lam * np.eye(len(refs)), a.T @ zf[i])
    return g


def score(played: np.ndarray, scene: dict, atten_db: float = 10.0) -> dict:
    """wrong-speaker s, leak dB, target-active s pushed down >= ``atten_db`` (T-326), true dropouts s."""
    from eval.namecall_eval import _speech_level
    from eval.wrong_speaker_audit import _frames, other_dominant, true_dropout_s

    mix, t, o, nz = scene["mix"], scene["A"], scene["B"], scene["noise"]
    n = len(_frames(mix))
    dom = other_dominant(played, mix, t, o, nz)
    gy, gx = lstsq_gains(played, [t, o, nz]), lstsq_gains(mix, [t, o, nz])
    eo = np.sum(_frames(o) ** 2, axis=1)
    leak = float(10 * np.log10((np.sum(gy[:, 1] ** 2 * eo) + 1e-12) / (np.sum(gx[:, 1] ** 2 * eo) + 1e-12)))
    et = np.sqrt(np.mean(_frames(t) ** 2, axis=1))
    active = et > 0.1 * _speech_level(t)  # clear target energy (the audit's target_active rule per 200 ms frame)
    rel = np.abs(gy[:, 0]) / np.maximum(np.abs(gx[:, 0]), 1e-9)
    down = active & (rel <= 10 ** (-atten_db / 20))
    return {"wrong_s": round(float(dom.sum() * LS_FRAME / SR), 2), "leak_db": round(leak, 2),
            "atten_s": round(float(down.sum() * LS_FRAME / SR), 2), "dropout_s": round(true_dropout_s(played, t), 2),
            "target_active_s": round(float(active.sum() * LS_FRAME / SR), 2), "seconds": round(n * LS_FRAME / SR, 2)}


def load_scene(path: Path) -> dict:
    from eval.contrast_sweep import load_scene as ls
    from hearing.audio import io as aio

    sc = ls(path)
    nz = aio.load(path / "stems" / "noise.wav", SR)[0][: len(sc["mix"])]
    sc["noise"] = np.pad(nz, (0, len(sc["mix"]) - len(nz)))
    sc["id"] = str(path)
    return sc


def settings(contrasts=CONTRASTS, floors=FLOORS, unsures=UNSURES, plays=PLAYS):
    return list(itertools.product(contrasts, floors, unsures, plays))


def sweep(specs: list[tuple[str, str, Path]], profiles: Path, grid=None, keep_audio: dict | None = None) -> dict:
    """``{setting: {scene: metrics}}`` over ``grid`` (default: every combination)."""
    from eval.contrast_sweep import load_profile
    from hearing.tse.blind import BlindSeparator

    grid = grid or settings()
    sep = CachedSeparator(BlindSeparator())
    res: dict[str, dict] = {}
    saved = {k: os.environ.get(k) for k in ENVS}
    try:
        for target, other, path in specs:
            sc = load_scene(path)
            enroll, oth = load_profile(profiles, target), load_profile(profiles, other)
            for contrast, play in sorted({(c, p) for c, _, _, p in grid}, key=str):
                t0 = time.monotonic()
                wins = replay(sc, sep, enroll, oth, contrast, play)
                for c, f, u, p in grid:
                    if (c, p) != (contrast, play):
                        continue
                    set_env(c, f, u, p)
                    played = compose_output(sc, wins)
                    m = score(played, sc)
                    m["gated_windows"] = sum(w["gated"] for w in wins)
                    name = setting_name(c, f, u, p)
                    res.setdefault(name, {})[path.name if target in path.name else f"{target}/{path.name}"] = m
                    if keep_audio is not None:
                        keep_audio[(name, str(path))] = played
                print(f"{path.name:24s} c={contrast:+.2f} play={play or 'zero':6s} {time.monotonic() - t0:5.1f} s",
                      flush=True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return res


def totals(res: dict) -> dict:
    out = {}
    for name, per in res.items():
        v = list(per.values())
        out[name] = {"wrong_s": round(sum(m["wrong_s"] for m in v), 2),
                     "leak_db": round(float(np.mean([m["leak_db"] for m in v])), 2),
                     "atten_s": round(sum(m["atten_s"] for m in v), 2),
                     "dropout_s": round(sum(m["dropout_s"] for m in v), 2)}
    return out


def pareto(tot: dict, keys=("wrong_s", "leak_db", "atten_s")) -> list[str]:
    """Settings no other setting beats on every key (lower is better) and strictly on one."""
    names = list(tot)
    front = []
    for a in names:
        dominated = any(all(tot[b][k] <= tot[a][k] for k in keys) and any(tot[b][k] < tot[a][k] for k in keys)
                        for b in names if b != a)
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda n: (tot[n]["wrong_s"], tot[n]["atten_s"], tot[n]["leak_db"]))


BASELINE = setting_name(0.0, None, None, None)


def pick_best(res: dict, n: int = 2, atten_slack_s: float = 1.0, baseline: str = BASELINE) -> list[str]:
    """The ``n`` settings with the least wrong-speaker time whose target attenuation stays within ``atten_slack_s``
    of the baseline on EVERY scene (the T-340 test bar, applied on the tuning set); ties by leak."""
    tot = totals(res)
    base = res[baseline]
    ok = [s for s in res if s != baseline and all(res[s][k]["atten_s"] <= base[k]["atten_s"] + atten_slack_s
                                                   for k in base)]
    ok.sort(key=lambda s: (tot[s]["wrong_s"], tot[s]["leak_db"], tot[s]["atten_s"]))
    return ok[:n]


def table(res: dict, rows: list[str]) -> str:
    tot = totals(res)
    lines = ["| setting | wrong-speaker s | mean leak dB | target >= 10 dB down s | true dropouts s |",
             "|---|---|---|---|---|"]
    lines += [f"| `{s}` | {tot[s]['wrong_s']:.1f} | {tot[s]['leak_db']:+.2f} | {tot[s]['atten_s']:.1f} | "
              f"{tot[s]['dropout_s']:.2f} |" for s in rows]
    return "\n".join(lines)


def _session_score():
    import importlib.util
    import sys

    if "_t340_session_score" in sys.modules:
        return sys.modules["_t340_session_score"]
    path = Path(__file__).resolve().parents[1] / "scripts" / "session_score.py"
    spec = importlib.util.spec_from_file_location("_t340_session_score", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def live_atten(d: Path, manifest: Path, skip_s: float = 3.9, atten_db: float = 10.0) -> dict:
    """T-326's live check on one ``demo_regress`` session: target-active 200 ms frames (after the cold start)
    whose pre-AGC target projection is >= ``atten_db`` below the input's; the output aligned as session_score."""
    from eval.namecall_eval import _speech_level

    ss = _session_score()
    rows = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines() if line.strip()]
    x, y, y_pre = (ss.read_wav(d / f) for f in ("in.wav", "out.wav", "out_pre_agc.wav"))
    placed, _ = ss.place_stems(x, ss.resolve_stems(manifest))
    pre_al = np.zeros(len(x), np.float32)
    _, valid, _ = ss.align_output(x, y, ss._ticks(rows), ride=(y_pre, pre_al))
    n = len(x) // LS_FRAME * LS_FRAME
    use = placed["covered"] & valid
    use[: int(skip_s * SR)] = False
    fm = use[:n].reshape(-1, LS_FRAME).all(axis=1)
    t, o, nz = (placed[k][:n] for k in ("target", "other", "noise"))
    gy, gx = ss.projection(pre_al[:n], [t, o, nz]), ss.projection(x[:n], [t, o, nz])
    et = np.sqrt(np.mean(t.astype(np.float64).reshape(-1, LS_FRAME) ** 2, axis=1))
    active = fm & (et > 0.1 * _speech_level(t))
    rel = np.abs(gy[:, 0]) / np.maximum(np.abs(gx[:, 0]), 1e-9)
    gain_db = 20 * np.log10(np.maximum(rel[active], 1e-9))
    return {"target_active_s": round(float(active.sum() * LS_FRAME / SR), 2),
            "target_down_10dB_s": round(float((active & (rel <= 10 ** (-atten_db / 20))).sum() * LS_FRAME / SR), 2),
            "median_target_gain_db": round(float(np.median(gain_db)), 2) if len(gain_db) else None}


def _specs(args) -> list[tuple[str, str, Path]]:
    specs = []
    for spec in args.scene or []:
        target, other, path = spec.split(":", 2)
        specs.append((target, other, Path(path)))
    if args.tuning:
        for m in sorted(args.tuning.glob("*/manifest.json")):
            meta = json.loads(m.read_text())
            specs.append((meta["target_name"], meta["other_name"], m.parent))
    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=fixtures_dir() / "tuning_t340")
    s = sub.add_parser("sweep")
    s.add_argument("--profiles", type=Path, default=fixtures_dir() / "profiles")
    s.add_argument("--scene", action="append", help="TARGET:OTHER:SCENE_DIR (repeatable)")
    s.add_argument("--tuning", type=Path, default=None, help="every tuning scene under this dir")
    s.add_argument("--only", action="append", default=None, help="setting name(s) to run (default: the full grid)")
    s.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(build(args.out), indent=1))
        return 0
    grid = settings()
    if args.only:
        grid = [g for g in grid if setting_name(*g) in set(args.only)]
    res = sweep(_specs(args), args.profiles, grid)
    tot = totals(res)
    front = pareto(tot)
    best = pick_best(res) if BASELINE in res else []
    print(table(res, sorted(tot, key=lambda n: (tot[n]["wrong_s"], tot[n]["atten_s"]))))
    print("pareto", front)
    print("best", best)
    out = args.out or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / (
        f"knob_sweep_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"totals": tot, "pareto": front, "best": best, "per_scene": res}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
