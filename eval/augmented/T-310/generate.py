#!/usr/bin/env python3
"""T-310: generate the standard augmented team-audio dataset.

Sources: eval/fixtures/tiny/speech_a.wav (LibriSpeech spk 1580, F) and
speech_b.wav (spk 61, M) as STAND-INS. Real team recordings (Aryan/Joseph)
live on laptop-b (docs/TEAM_RECORDINGS.md) and are NOT in this checkout.
Manifests use neutral speaker_a/speaker_b labels -- never real names on
stand-in data.

Groups (9): clean, pitch_pm2, pitch_pm4, reverb_02, reverb_05, reverb_08,
opus16, opus32, combo (= pitch_pm4 + reverb_08).

Usage: python3 eval/augmented/T-310/generate.py [--out eval/augmented/T-310]
Requires: numpy, scipy, ffmpeg (with rubberband + libopus).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[3]
SRC = {
    # neutral labels; true LibriSpeech speaker ids recorded in manifest for provenance
    "speaker_a": ("eval/fixtures/tiny/speech_a.wav", "1580", "F"),
    "speaker_b": ("eval/fixtures/tiny/speech_b.wav", "61", "M"),
}
SR = 16000


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}... failed:\n{r.stderr[-800:]}")
    return r


def read_wav(path):
    sr, x = wavfile.read(str(path))
    assert sr == SR, f"{path}: sr {sr} != {SR}"
    return x.astype(np.float32) / 32768.0


def write_wav(path, x, peak_limit_db=-1.0):
    peak = np.max(np.abs(x))
    limit = 10 ** (peak_limit_db / 20)
    if peak > limit:
        x = x * (limit / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-12))


def has_rubberband():
    r = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, check=False)
    return "rubberband" in r.stdout


def pitch_shift_ffmpeg(src, dst, semitones):
    # rubberband pitch shift, duration-preserving
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-af", f"rubberband=pitch={2 ** (semitones / 12):.6f}",
         "-ar", str(SR), "-ac", "1", str(dst)])


def pitch_shift_librosa(src, dst, semitones):
    import librosa
    sr, x = wavfile.read(str(src))
    x = x.astype(np.float32) / 32768.0
    y = librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones)
    write_wav(dst, y)


def pitch_shift(src, dst, semitones):
    if has_rubberband():
        pitch_shift_ffmpeg(src, dst, semitones)
        return "rubberband"
    pitch_shift_librosa(src, dst, semitones)
    return "librosa"


def synth_rir(sr, rt60, seed=0, drr_db=9.0):
    """Exponentially-decaying noise RIR with a unit direct-path impulse.

    Tail energy is scaled for the target direct-to-reverberant ratio (DRR).
    rir[0] is exactly 1.0; the tail decays -60 dB at t = rt60.
    """
    rng = np.random.default_rng(seed)
    n = int(rt60 * 1.5 * sr)
    t = np.arange(n) / sr
    # amplitude envelope 10^(-3t/rt60) -> ENERGY decays 10^(-6t/rt60),
    # i.e. -60 dB energy at t = rt60
    tail = rng.standard_normal(n) * 10 ** (-3 * t / rt60)
    tail[0] = 0.0
    e_tail = np.sum(tail ** 2) + 1e-12
    tail = tail * np.sqrt(10 ** (-drr_db / 10) / e_tail)
    rir = tail
    rir[0] = 1.0
    return rir


def apply_reverb(x, rt60):
    rir = synth_rir(SR, rt60)
    y = np.convolve(x, rir)[: len(x)]
    return y


def opus_roundtrip(src, dst, bitrate_k):
    # opus cannot live in a WAV container: encode to .opus, then decode back
    enc = dst.with_suffix(".opus")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-c:a", "libopus", "-b:a", f"{bitrate_k}k", "-ar", "48000", "-ac", "1",
         str(enc)])
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(enc),
         "-ar", str(SR), "-ac", "1", str(dst)])
    enc.unlink()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/augmented/T-310")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out

    groups = ["clean", "pitch_pm2", "pitch_pm4", "reverb_02", "reverb_05",
              "reverb_08", "opus16", "opus32", "combo"]
    manifest = {"task": "T-310", "sample_rate": SR,
                "sources_are_standins": True,
                "note": "speech_a/b are LibriSpeech stand-ins (spk 1580/F, 61/M). "
                        "Real team recordings live on laptop-b; rerun with --src to swap.",
                "groups": {}}
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    peak_report = {}

    for label, (rel, spk_id, sex) in SRC.items():
        src = REPO / rel
        x = read_wav(src)
        for g in groups:
            dst = out / g / label / (Path(rel).stem + ".wav")
            params = {}
            if g == "clean":
                peak_db = write_wav(dst, x)
            elif g in ("pitch_pm2", "pitch_pm4"):
                st = 2 if g == "pitch_pm2" else 4
                tmpw = tmp / f"{label}_{g}.wav"
                method = pitch_shift(src, tmpw, st)
                params = {"semitones": st, "method": method}
                peak_db = write_wav(dst, read_wav(tmpw))
            elif g.startswith("reverb_"):
                rt60 = float(g.split("_")[1]) / 100
                params = {"rt60_s": rt60, "method": "synthetic_exp_decay_rir"}
                peak_db = write_wav(dst, apply_reverb(x, rt60))
            elif g in ("opus16", "opus32"):
                kb = int(g[4:])
                params = {"bitrate_kbps": kb, "codec": "libopus"}
                tmpw = tmp / f"{label}_{g}.wav"
                opus_roundtrip(src, tmpw, kb)
                peak_db = write_wav(dst, read_wav(tmpw))
            elif g == "combo":
                tmpw = tmp / f"{label}_combo.wav"
                method = pitch_shift(src, tmpw, 4)
                params = {"semitones": 4, "rt60_s": 0.8,
                          "pitch_method": method,
                          "note": "hardest-20pct rule applies with more sources; "
                                  "applied to all sources here (n=2)"}
                peak_db = write_wav(dst, apply_reverb(read_wav(tmpw), 0.8))

            entry = {"file": str(dst.relative_to(out)), "speaker_label": label,
                     "librispeech_speaker_id": spk_id, "sex": sex,
                     "standin": True, "params": params,
                     "sha256": sha256(dst), "peak_dbfs": round(peak_db, 2)}
            manifest["groups"].setdefault(g, []).append(entry)
            peak_report.setdefault(g, []).append(peak_db)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("groups:")
    for g in groups:
        peaks = peak_report[g]
        print(f"  {g:10s} n={len(peaks)} peak_dbfs=[{min(peaks):.1f}, {max(peaks):.1f}]")
    # cleanup tmp
    for f in tmp.glob("*.wav"):
        f.unlink()
    print("wrote", out / "manifest.json")


if __name__ == "__main__":
    sys.exit(main())
