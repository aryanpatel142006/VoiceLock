"""DNSMOS: Microsoft's no-reference MOS predictor (P.835 SIG/BAK/OVRL + P.808) for real clips.

    .venv/bin/python -m eval.dnsmos --download                       # once: the two ONNX models (network)
    .venv/bin/python -m eval.dnsmos                                  # every WAV under eval/fixtures/real/
    .venv/bin/python -m eval.dnsmos clip.wav other/ --stages passthrough --stages gtcrn

Real phone clips have no clean reference, so SI-SDR/PESQ/STOI cannot score them; DNSMOS can.
It is the reference ``dnsmos_local.py`` of the DNS Challenge (commit ``591184a9``) re-implemented
on our I/O: 16 kHz, clips shorter than 9.01 s are tiled, one 9.01 s segment per second of audio,
SIG/BAK/OVRL from ``sig_bak_ovr.onnx`` through the published non-personalised polynomials, P.808
from ``model_v8.onnx`` on a 120-band log-mel. Scores are means over segments (1 = bad, 5 = excellent).

``get_dnsmos("auto")`` returns the ONNX scorer when the models, onnxruntime and librosa are present,
else the deterministic :class:`MockDnsmos` (flagged ``name == "mock"``, never a quality claim).
Models live in ``$HEARING_MODELS_DIR/dnsmos/``; scoring never downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from hearing import models
from hearing.audio import io as aio

SR = 16000
INPUT_LENGTH_S = 9.01
COMMIT = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/{commit}/DNSMOS/DNSMOS/{name}"
PRIMARY = "sig_bak_ovr.onnx"
P808 = "model_v8.onnx"
SHA256 = {
    PRIMARY: "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    P808: "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
}
# Non-personalised P.835 calibration polynomials from dnsmos_local.py (highest power first).
POLY_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
POLY_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
POLY_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
REAL_DIR = Path(__file__).resolve().parent / "fixtures" / "real"
AUDIO_EXT = (".wav", ".flac", ".ogg")


@dataclass
class DnsmosScore:
    sig: float  # speech signal quality
    bak: float  # background noise intrusiveness (higher = less noise)
    ovrl: float  # overall
    p808: float  # P.808 overall MOS
    segments: int
    duration_s: float
    scorer: str  # "dnsmos" or "mock"

    def as_dict(self) -> dict:
        return asdict(self)


# --- models ---------------------------------------------------------------------------------


def model_dir() -> Path:
    return models.cache_dir() / "dnsmos"


def _verify(name: str, data: bytes) -> None:
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise ValueError(f"DNSMOS {name} checksum mismatch; re-provision the pinned model")


def download_models() -> Path:
    """Explicit network adapter: fetch both ONNX files at the pinned commit, verify, install atomically."""
    import httpx

    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name in SHA256:
        path = root / name
        if path.exists():
            _verify(name, path.read_bytes())
            continue
        response = httpx.get(URL.format(commit=COMMIT, name=name), timeout=60, follow_redirects=True)
        response.raise_for_status()
        _verify(name, response.content)
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as f:
            tmp = Path(f.name)
            f.write(response.content)
        try:
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return root


def available() -> bool:
    """Both models on disk plus onnxruntime and librosa importable."""
    root = model_dir()
    files = all((root / name).is_file() for name in SHA256)
    return files and find_spec("onnxruntime") is not None and find_spec("librosa") is not None


# --- scoring --------------------------------------------------------------------------------


def segments(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """``N x 144160`` float32 windows exactly as dnsmos_local.py cuts them (tile short clips, 1 s hop)."""
    audio = np.asarray(x, dtype=np.float32).ravel()
    if audio.size == 0:
        raise ValueError("DNSMOS needs a non-empty clip")
    n = int(INPUT_LENGTH_S * sr)
    while len(audio) < n:
        audio = np.append(audio, audio)
    hops = int(np.floor(len(audio) / sr) - INPUT_LENGTH_S) + 1
    segs = [audio[int(i * sr) : int((i + INPUT_LENGTH_S) * sr)] for i in range(hops)]
    return np.stack([s for s in segs if len(s) >= n]).astype(np.float32)


def _prepare(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:  # (n, channels) from soundfile
        x = x.mean(axis=1)
    return aio.resample(x, sr, SR) if sr != SR else x


class OnnxDnsmos:
    """The real scorer (CPU onnxruntime; the models are small CNNs, ~10 ms per 9 s segment)."""

    name = "dnsmos"

    def __init__(self):
        if not available():
            raise RuntimeError(
                "DNSMOS models absent (or onnxruntime/librosa missing); run python -m eval.dnsmos --download"
            )
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        root, providers = model_dir(), ["CPUExecutionProvider"]
        self.primary = ort.InferenceSession(str(root / PRIMARY), opts, providers=providers)
        self.p808 = ort.InferenceSession(str(root / P808), opts, providers=providers)

    @staticmethod
    def melspec(seg: np.ndarray) -> np.ndarray:
        """120-band log-mel as dnsmos_local.audio_melspec: n_fft 321, hop 160, (dB(ref=max) + 40) / 40."""
        import librosa

        mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    def score(self, x: np.ndarray, sr: int = SR) -> DnsmosScore:
        audio = _prepare(x, sr)
        segs = segments(audio)
        raw = self.primary.run(None, {"input_1": segs})[0]  # N x 3: sig, bak, ovr
        mels = np.stack([self.melspec(s[:-160]) for s in segs]).astype(np.float32)
        p808 = self.p808.run(None, {"input_1": mels})[0][:, 0]
        return DnsmosScore(
            sig=float(np.mean(POLY_SIG(raw[:, 0]))),
            bak=float(np.mean(POLY_BAK(raw[:, 1]))),
            ovrl=float(np.mean(POLY_OVR(raw[:, 2]))),
            p808=float(np.mean(p808)),
            segments=len(segs),
            duration_s=len(audio) / SR,
            scorer=self.name,
        )


class MockDnsmos:
    """Deterministic offline stand-in: maps clip loudness onto 1-5 so plumbing tests are stable.

    Never a quality claim; reports carry ``scorer: "mock"``.
    """

    name = "mock"

    def score(self, x: np.ndarray, sr: int = SR) -> DnsmosScore:
        audio = _prepare(x, sr)
        segs = segments(audio)
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
        level = float(np.clip((20 * math.log10(rms + 1e-10) + 60) / 15, 1.0, 5.0))  # -45 dBFS -> 1, 0 dBFS -> 5
        return DnsmosScore(level, level, level, level, len(segs), len(audio) / SR, self.name)


def get_dnsmos(name: str = "auto"):
    """``dnsmos`` (raises if unavailable), ``mock``, or ``auto`` (real when available, else mock)."""
    if name == "mock":
        return MockDnsmos()
    if name == "dnsmos" or (name == "auto" and available()):
        return OnnxDnsmos()
    if name != "auto":
        raise ValueError(f"unknown DNSMOS scorer {name!r}; use auto, dnsmos or mock")
    return MockDnsmos()


# --- real clips -----------------------------------------------------------------------------


def find_clips(paths: list[Path]) -> list[Path]:
    """Audio files named directly or found (recursively, sorted) under the given directories."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT))
        elif p.is_file():
            out.append(p)
    return out


def score_clips(clips: list[Path], specs: list[str], scorer) -> list[dict]:
    """One row per clip x pipeline: the clip streamed through the pipeline (eval.bench), then DNSMOS."""
    from eval import bench

    rows = []
    for clip in clips:
        x, sr = aio.load(clip, sr=SR)
        for spec in specs:
            row = {"clip": str(clip), "stages": spec}
            try:
                y = bench.run_stream(bench.build(spec), x, sr).output if spec != "input" else x
                row.update(scorer.score(y, sr).as_dict())
            except Exception as e:  # noqa: BLE001 - one bad clip/pipeline is a row, not a crash
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    def cell(r: dict, k: str) -> str:
        v = r.get(k)
        return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.2f}"

    lines = ["| clip | stages | SIG | BAK | OVRL | P808 | s | scorer |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dur = f"{r['duration_s']:.1f}" if "duration_s" in r else "-"
        lines.append(
            f"| {Path(r['clip']).name} | {r['stages']} | {cell(r, 'sig')} | {cell(r, 'bak')} | "
            f"{cell(r, 'ovrl')} | {cell(r, 'p808')} | {dur} | {r.get('scorer', r.get('error', '-'))} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="clips or directories (default eval/fixtures/real/)")
    ap.add_argument("--stages", action="append", help='pipeline spec to score after (repeatable); "input" = raw clip')
    ap.add_argument("--scorer", choices=("auto", "dnsmos", "mock"), default="auto")
    ap.add_argument("--download", action="store_true", help="fetch the pinned ONNX models (network) and exit")
    ap.add_argument("--json", type=Path, default=None, help="also write the rows to this JSON file")
    args = ap.parse_args(argv)
    if args.download:
        print(f"DNSMOS models in {download_models()}")
        return 0
    clips = find_clips(args.paths or [REAL_DIR])
    if not clips:
        print(f"dnsmos: no audio under {', '.join(map(str, args.paths or [REAL_DIR]))}", file=sys.stderr)
        return 2
    scorer = get_dnsmos(args.scorer)
    rows = score_clips(clips, ["input", *(args.stages or [])], scorer)
    print(f"scorer={scorer.name} clips={len(clips)}")
    print(format_table(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
