"""Standard augmented team-audio dataset for robustness experiments (T-306).

Takes the team recordings (``$HEARING_FIXTURES_DIR/team/{aryan,joseph}/*.wav``, 16 kHz mono),
cuts them into ~10 s segments and writes nine groups::

    <out>/{clean,pitch_pm2,pitch_pm4,reverb_02,reverb_05,reverb_08,opus16,opus32,combo}/<speaker>/<file>.wav
    <out>/manifest.json   per file: speaker, augmentation, params, source_file, sha256 (+ peak/duration)

Everything is deterministic (fixed seeds, librosa pitch shift, synthetic exponential-decay RIRs,
ffmpeg libopus round trip).  Every output is peak-capped at -3 dBFS so no file reaches -1 dBFS.
Audio stays out of git; the default output is the main checkout's ``eval/augmented/T-306``.
Reproduce: ``.venv/bin/python -m eval.augment_team`` (see docs/research/AUGMENTED_DATASET.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
SEGMENT_S = 10.0
MIN_TAIL_S = 4.0
PEAK_CAP_DBFS = -3.0
CLIP_LIMIT_DBFS = -1.0
COMBO_FRACTION = 0.2
SPEAKERS = ("aryan", "joseph")
GROUPS = (
    "clean",
    "pitch_pm2",
    "pitch_pm4",
    "reverb_02",
    "reverb_05",
    "reverb_08",
    "opus16",
    "opus32",
    "combo",
)
REVERB = {"reverb_02": 0.2, "reverb_05": 0.5, "reverb_08": 0.8}
OPUS = {"opus16": 16, "opus32": 32}
PITCH = {"pitch_pm2": 2, "pitch_pm4": 4}
COMBO_PITCH = 4
COMBO_RT60 = 0.8


def _main_checkout() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()
    return Path(out).parent


def default_source_dir() -> Path:
    env = os.environ.get("HEARING_FIXTURES_DIR")
    base = Path(env) if env else _main_checkout() / "eval" / "fixtures" / "generated"
    return base / "team"


def default_out_dir() -> Path:
    return _main_checkout() / "eval" / "augmented" / "T-306"


def peak_dbfs(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


def cap_peak(x: np.ndarray, cap_dbfs: float = PEAK_CAP_DBFS) -> tuple[np.ndarray, float]:
    """Scale down (never up) so the peak is at most ``cap_dbfs``; returns (audio, gain_db)."""
    p = peak_dbfs(x)
    if p <= cap_dbfs:
        return np.asarray(x, dtype=np.float32), 0.0
    gain_db = cap_dbfs - p
    return np.asarray(x * 10.0 ** (gain_db / 20.0), dtype=np.float32), gain_db


def segment(x: np.ndarray, sr: int = SR, seg_s: float = SEGMENT_S, min_tail_s: float = MIN_TAIL_S):
    """Cut into ``seg_s`` pieces; a tail shorter than ``min_tail_s`` joins the previous piece."""
    n = round(seg_s * sr)
    bounds = list(range(0, len(x), n))
    pieces = [[b, min(b + n, len(x))] for b in bounds]
    if len(pieces) > 1 and pieces[-1][1] - pieces[-1][0] < round(min_tail_s * sr):
        tail = pieces.pop()
        pieces[-1][1] = tail[1]
    return [(a, b) for a, b in pieces]


def difficulty_db(x: np.ndarray, sr: int = SR) -> float:
    """Speech-to-floor proxy: p95 minus p10 of 20 ms frame RMS (dB). Lower = harder."""
    hop = sr // 50
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    rms = 10.0 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return float(np.percentile(rms, 95) - np.percentile(rms, 10))


def synthetic_rir(rt60: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Direct path + exponentially decaying Gaussian tail with the given RT60 (-60 dB at rt60)."""
    rng = np.random.default_rng(seed)
    n = round(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)  # ln(1000) = 60 dB amplitude
    tail[: round(0.005 * sr)] = 0.0  # 5 ms pre-delay before the first reflections
    h = 0.5 * tail / np.sqrt(np.sum(tail**2))
    h[0] = 1.0
    return h.astype(np.float64)


def reverb(x: np.ndarray, rt60: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    from scipy.signal import fftconvolve

    y = fftconvolve(x.astype(np.float64), synthetic_rir(rt60, sr, seed))[: len(x)]
    rms_in = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    rms_out = np.sqrt(np.mean(y**2))
    if rms_out > 0:
        y *= rms_in / rms_out  # keep the dry loudness
    return y.astype(np.float32)


def pitch_shift(x: np.ndarray, semitones: float, sr: int = SR) -> np.ndarray:
    import librosa

    return np.asarray(librosa.effects.pitch_shift(x.astype(np.float32), sr=sr, n_steps=semitones), dtype=np.float32)


def opus_roundtrip(x: np.ndarray, kbps: int, sr: int = SR, ffmpeg: str | None = None) -> np.ndarray:
    """Encode with ffmpeg libopus at ``kbps`` (VBR off, voip) and decode back to ``sr`` mono."""
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; install it (brew install ffmpeg) for the opus groups")
    with tempfile.TemporaryDirectory(prefix="T-306-opus-") as tmp:
        src, enc, dec = Path(tmp, "in.wav"), Path(tmp, "enc.ogg"), Path(tmp, "out.wav")
        sf.write(src, x, sr, subtype="FLOAT")
        quiet = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        subprocess.run(
            [*quiet, "-i", str(src), "-c:a", "libopus", "-b:a", f"{kbps}k", "-vbr", "off",
             "-application", "voip", str(enc)],
            check=True,
        )
        subprocess.run([*quiet, "-i", str(enc), "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le", str(dec)], check=True)
        y, _ = sf.read(dec, dtype="float32")
    y = y[: len(x)]
    return np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source_dir: Path, out_dir: Path, *, speakers=SPEAKERS, combo_fraction: float = COMBO_FRACTION) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    for speaker in speakers:
        for wav in sorted((source_dir / speaker).glob("*.wav")):
            x, sr = sf.read(wav, dtype="float32", always_2d=True)
            if sr != SR:
                raise ValueError(f"{wav}: expected {SR} Hz, got {sr}")
            x = x.mean(axis=1)
            for i, (a, b) in enumerate(segment(x)):
                seg = x[a:b]
                segments.append({
                    "speaker": speaker,
                    "source": wav,
                    "stem": f"{wav.stem}_s{i:02d}",
                    "start_s": a / SR,
                    "end_s": b / SR,
                    "audio": seg,
                    "difficulty_db": difficulty_db(seg),
                })
    # hardest 20%: lowest speech-to-floor range; ties broken by name for determinism
    ranked = sorted(segments, key=lambda s: (s["difficulty_db"], s["speaker"], s["stem"]))
    n_combo = max(1, math.ceil(combo_fraction * len(segments))) if segments else 0
    combo = {(s["speaker"], s["stem"]): rank for rank, s in enumerate(ranked[:n_combo])}

    files = []

    def emit(group: str, seg: dict, suffix: str, y: np.ndarray, params: dict) -> None:
        y, gain_db = cap_peak(y)
        rel = Path(group) / seg["speaker"] / f"{seg['stem']}{suffix}.wav"
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, y, SR, subtype="PCM_16")
        back, _ = sf.read(path, dtype="float32")
        files.append({
            "path": rel.as_posix(),
            "speaker": seg["speaker"],
            "augmentation": group,
            "params": {**params, "peak_cap_gain_db": round(gain_db, 3)},
            "source_file": seg["source"].name,
            "source_sha256": seg["source_sha256"],
            "source_start_s": round(seg["start_s"], 3),
            "source_end_s": round(seg["end_s"], 3),
            "difficulty_db": round(seg["difficulty_db"], 2),
            "sample_rate": SR,
            "duration_s": round(len(back) / SR, 3),
            "peak_dbfs": round(peak_dbfs(back), 2),
            "sha256": _sha256(path),
        })

    src_sha = {}
    for seg in segments:
        src = seg["source"]
        if src not in src_sha:
            src_sha[src] = _sha256(src)
        seg["source_sha256"] = src_sha[src]
        x, key = seg["audio"], (seg["speaker"], seg["stem"])
        seed = int(hashlib.sha256(f"{seg['speaker']}/{seg['stem']}".encode()).hexdigest()[:8], 16)
        emit("clean", seg, "", x, {})
        for group, st in PITCH.items():
            for sign in (+1, -1):
                emit(group, seg, f"_{'p' if sign > 0 else 'm'}{st}", pitch_shift(x, sign * st), {"semitones": sign * st})
        for group, rt60 in REVERB.items():
            emit(group, seg, "", reverb(x, rt60, seed=seed), {"rt60_s": rt60, "rir": "synthetic_exp", "rir_seed": seed})
        clean_capped, _ = cap_peak(x)
        for group, kbps in OPUS.items():
            emit(group, seg, "", opus_roundtrip(clean_capped, kbps),
                 {"codec": "libopus", "kbps": kbps, "vbr": "off", "application": "voip"})
        if key in combo:
            st = COMBO_PITCH if combo[key] % 2 == 0 else -COMBO_PITCH
            y = reverb(pitch_shift(x, st), COMBO_RT60, seed=seed)
            emit("combo", seg, f"_{'p' if st > 0 else 'm'}{abs(st)}_rt08", y,
                 {"semitones": st, "rt60_s": COMBO_RT60, "rir": "synthetic_exp", "rir_seed": seed,
                  "difficulty_rank": combo[key]})

    manifest = {
        "task": "T-306",
        "sample_rate": SR,
        "segment_s": SEGMENT_S,
        "peak_cap_dbfs": PEAK_CAP_DBFS,
        "groups": list(GROUPS),
        "combo_fraction": combo_fraction,
        "n_segments": len(segments),
        "n_files": len(files),
        "counts": {g: sum(f["augmentation"] == g for f in files) for g in GROUPS},
        "max_peak_dbfs": max((f["peak_dbfs"] for f in files), default=None),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def spot_check(manifest: dict, out_dir: Path, per_group: int = 5) -> list[dict]:
    """Re-read up to ``per_group`` files per group (evenly spread) and check sr, duration, peak, hash."""
    rows = []
    for group in manifest["groups"]:
        entries = [f for f in manifest["files"] if f["augmentation"] == group]
        idx = np.unique(np.linspace(0, len(entries) - 1, min(per_group, len(entries))).round().astype(int))
        for i in idx if entries else []:
            f = entries[i]
            path = out_dir / f["path"]
            info = sf.info(path)
            x, _ = sf.read(path, dtype="float32")
            p = peak_dbfs(x)
            rows.append({
                "group": group,
                "path": f["path"],
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_s": round(info.duration, 3),
                "peak_dbfs": round(p, 2),
                "ok": info.samplerate == SR and info.channels == 1 and p < CLIP_LIMIT_DBFS
                and _sha256(path) == f["sha256"],
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=None, help="dir with <speaker>/*.wav (default $HEARING_FIXTURES_DIR/team)")
    ap.add_argument("--out", type=Path, default=None, help="output root (default <main checkout>/eval/augmented/T-306)")
    ap.add_argument("--manifest-copy", type=Path, default=None, help="also copy manifest.json here (e.g. the repo)")
    ap.add_argument("--check-only", action="store_true", help="skip rendering; spot-check an existing dataset")
    args = ap.parse_args(argv)
    out = args.out or default_out_dir()
    if args.check_only:
        manifest = json.loads((out / "manifest.json").read_text())
    else:
        manifest = render(args.source or default_source_dir(), out)
    rows = spot_check(manifest, out)
    for r in rows:
        print(f"{r['group']:10s} {r['path']:60s} {r['sample_rate']} Hz ch={r['channels']} "
              f"{r['duration_s']:6.2f} s peak {r['peak_dbfs']:6.2f} dBFS {'ok' if r['ok'] else 'FAIL'}")
    print(f"{manifest['n_files']} files, {manifest['n_segments']} segments, counts {manifest['counts']}, "
          f"max peak {manifest['max_peak_dbfs']} dBFS -> {out}")
    (out / "spot_check.json").write_text(json.dumps(rows, indent=1) + "\n")
    if args.manifest_copy and args.manifest_copy.resolve() != out.resolve():
        args.manifest_copy.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "manifest.json", args.manifest_copy / "manifest.json")
        shutil.copy2(out / "spot_check.json", args.manifest_copy / "spot_check.json")
    return 0 if rows and all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
