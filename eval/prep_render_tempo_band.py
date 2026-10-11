"""Render a bounded, provenance-preserving tempo/bandwidth stress shard.

This is preparation only: tempo uses resampling, so every non-1.0 tempo render
is explicitly marked as a pitch-changing stress case and must not be treated as
an identity-preserving training example.

Example:
    .venv/bin/python -m eval.prep_render_tempo_band \
      --input "$HEARING_FIXTURES_DIR/team" --out "$HEARING_FIXTURES_DIR/tempo-band"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

NATIVE_SR = 16000
MAX_VARIANTS = 600


@dataclass(frozen=True)
class Variant:
    tempo: float = 1.0
    sample_rate: int = NATIVE_SR
    bandlimit_hz: int | None = None
    clip: float | None = None
    gain_db: float = 0.0
    channels: str = "mono"

    @classmethod
    def baseline(cls) -> Variant:
        return cls()

    @property
    def baseline_flag(self) -> bool:
        return self == type(self).baseline()

    @property
    def tempo_pitch_change(self) -> bool:
        return not np.isclose(self.tempo, 1.0)

    def slug(self) -> str:
        tempo = "baseline" if self.baseline_flag else f"tempo{self.tempo:g}".replace(".", "")
        parts = [tempo]
        if self.sample_rate != NATIVE_SR:
            parts.append(f"sr{self.sample_rate}")
        if self.bandlimit_hz is not None:
            parts.append(f"band{self.bandlimit_hz}")
        if self.clip is not None:
            parts.append(f"clip{self.clip:g}".replace(".", ""))
        if self.gain_db:
            parts.append(f"gain{'p' if self.gain_db > 0 else 'm'}{abs(self.gain_db):g}".replace(".", ""))
        if self.channels != "mono":
            parts.append(self.channels)
        return "__".join(parts)


def enumerate_variants(
    origins: list[str] | tuple[str, ...],
    *,
    seed: int = 0,
    max_variants: int = MAX_VARIANTS,
) -> list[Variant]:
    """Return a deterministic bounded plan shared by all origin recordings."""
    del origins  # The plan is shared so every origin group sees the same transforms.
    if not 1 <= max_variants <= MAX_VARIANTS:
        raise ValueError(f"max_variants must be between 1 and {MAX_VARIANTS}")
    values = []
    for tempo in (0.9, 1.0, 1.1):
        for sample_rate in (NATIVE_SR, 8000, 24000, 48000):
            for bandlimit in (None, 4000, 8000):
                for clip in (None, 0.8):
                    for gain_db in (0.0, -3.0, 3.0):
                        for channels in ("mono", "stereo"):
                            values.append(Variant(tempo, sample_rate, bandlimit, clip, gain_db, channels))
    baseline = Variant.baseline()
    values = [v for v in values if v != baseline]
    np.random.default_rng(seed).shuffle(values)
    return [baseline, *values[: max_variants - 1]]


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    ratio = Fraction(target_sr, source_sr).limit_denominator()
    return signal.resample_poly(audio, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def apply_variant(audio: np.ndarray, variant: Variant, source_sr: int = NATIVE_SR) -> np.ndarray:
    """Apply transforms in a fixed order, returning native-rate float32 audio."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1, dtype=np.float32)
    x = _resample(x, source_sr, NATIVE_SR)
    if variant.tempo != 1.0:
        ratio = Fraction(1.0 / variant.tempo).limit_denominator(100)
        x = signal.resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)
    if variant.sample_rate != NATIVE_SR:
        x = _resample(_resample(x, NATIVE_SR, variant.sample_rate), variant.sample_rate, NATIVE_SR)
    if variant.bandlimit_hz is not None:
        cutoff = min(float(variant.bandlimit_hz), NATIVE_SR / 2 - 1)
        sos = signal.butter(6, cutoff, btype="lowpass", fs=NATIVE_SR, output="sos")
        x = signal.sosfiltfilt(sos, x).astype(np.float32) if len(x) > 32 else x
    if variant.gain_db:
        x = x * np.float32(10 ** (variant.gain_db / 20))
    if variant.clip is not None:
        x = np.clip(x, -variant.clip, variant.clip)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    if variant.channels == "stereo":
        x = np.column_stack((x, x))
    return x


def _transcript(source: Path) -> str | None:
    for candidate in (source.parent / "manifest.json", source.with_suffix(".json")):
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        value = data.get("transcript")
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            target = value.get("target")
            if isinstance(target, str):
                return target
    return None


def _source_hash(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def render_source(source: Path, out: Path, variants: list[Variant], *, seed: int = 0) -> dict:
    """Render one source, skipping valid completed sidecars for checkpoint/resume."""
    source = Path(source)
    out.mkdir(parents=True, exist_ok=True)
    audio, source_sr = sf.read(source, dtype="float32", always_2d=False)
    source_digest = _source_hash(source)
    origin_group = source.parent.name
    transcript = _transcript(source)
    written = skipped = 0
    checkpoint = out / "checkpoint.jsonl"
    with checkpoint.open("a", encoding="utf-8") as journal:
        for variant in variants:
            stem = f"{source.stem}__{variant.slug()}"
            wav_path, meta_path = out / f"{stem}.wav", out / f"{stem}.json"
            if wav_path.exists() and meta_path.exists():
                try:
                    old = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError):
                    old = {}
                if old.get("source_sha256") == source_digest and old.get("variant") == _variant_dict(variant):
                    skipped += 1
                    continue
            rendered = apply_variant(audio, variant, source_sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=out, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                sf.write(temporary, rendered, NATIVE_SR, subtype="PCM_16")
                temporary.replace(wav_path)
            finally:
                temporary.unlink(missing_ok=True)
            metadata = {
                "source": str(source),
                "source_sha256": source_digest,
                "origin_group": origin_group,
                "transcript": transcript,
                "sample_rate": NATIVE_SR,
                "seed": seed,
                "baseline": variant.baseline_flag,
                "tempo_pitch_change": variant.tempo_pitch_change,
                "identity_preserving": not variant.tempo_pitch_change,
                "variant": _variant_dict(variant),
                "wav": wav_path.name,
            }
            _atomic_json(meta_path, metadata)
            journal.write(json.dumps({"wav": wav_path.name, "source_sha256": source_digest}) + "\n")
            journal.flush()
            written += 1
    return {"source": str(source), "written": written, "skipped": skipped}


def _variant_dict(variant: Variant) -> dict:
    return {field.name: getattr(variant, field.name) for field in fields(variant)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, action="append", required=True, help="recording or directory (repeatable)")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("HEARING_FIXTURES_DIR", ".")) / "tempo-band")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-variants", type=int, default=MAX_VARIANTS)
    args = parser.parse_args(argv)
    sources = sorted(
        path for root in args.input for path in ([root] if root.is_file() else root.rglob("*.wav"))
    )
    if not sources:
        parser.error("no .wav recordings found under --input")
    variants = enumerate_variants([str(path) for path in sources], seed=args.seed, max_variants=args.max_variants)
    summaries = [render_source(source, args.out / source.stem, variants, seed=args.seed) for source in sources]
    print(json.dumps({"sources": len(sources), "variants_per_source": len(variants), "results": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
