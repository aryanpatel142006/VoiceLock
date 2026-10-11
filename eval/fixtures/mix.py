"""Mixing math and manifest schema for the fixture bank.

Definitions (all over the whole clip, after fitting every stem to the target length):

* ``snr_db = 20 log10(rms(target) / rms(noise))``
* ``sir_db = 20 log10(rms(target) / rms(interferer))``; noise-only fixtures have no
  interferer: an all-zero interferer stem and ``sir_db`` ``None`` (``null`` in JSON)
* ``mix = target + interferer + noise`` and the mix is peak-normalised to ``PEAK`` with the
  *same* gain applied to every stem, so the saved stems still sum exactly to the mix.

Stems are float32 mono at 16 kHz; the maths runs in float64.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from hearing.audio import io as aio

PEAK = 0.9
STEM_NAMES = ("mix", "target", "interferer", "noise")
_EPS = 1e-12


def rms(x: np.ndarray) -> float:
    """Root-mean-square of a 1-D signal (0.0 for an empty one)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def db(ratio: float) -> float:
    return float(20.0 * np.log10(max(ratio, _EPS)))


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim ``x`` to ``n`` samples or loop it (tiling, no crossfade) until it is ``n`` long."""
    x = np.asarray(x)
    if n <= 0:
        return x[:0]
    if x.size == 0:
        raise ValueError("cannot fit an empty signal to a non-zero length")
    if x.size >= n:
        return x[:n]
    reps = -(-n // x.size)
    return np.tile(x, reps)[:n]


def gain_for_level(reference: np.ndarray, signal: np.ndarray, level_db: float) -> float:
    """Gain so that ``rms(reference) / rms(gain * signal)`` equals ``level_db``."""
    r_sig = rms(signal)
    if r_sig <= 0.0:
        raise ValueError("signal is silent; cannot set a level relative to it")
    return float(rms(reference) / r_sig / 10.0 ** (level_db / 20.0))


@dataclass
class Stems:
    """One fixture's audio: ``mix == target + interferer + noise`` (float32, 16 kHz)."""

    mix: np.ndarray
    target: np.ndarray
    interferer: np.ndarray
    noise: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def measured_snr_db(self) -> float:
        return db(rms(self.target) / rms(self.noise))

    def measured_sir_db(self) -> float:
        """``inf`` for a noise-only fixture (all-zero interferer)."""
        r_int = rms(self.interferer)
        return db(rms(self.target) / r_int) if r_int > 0.0 else math.inf


def make_mixture(
    target: np.ndarray,
    interferer: np.ndarray | None,
    noise: np.ndarray,
    snr_db: float,
    sir_db: float | None,
    peak: float = PEAK,
) -> Stems:
    """Scale ``interferer`` and ``noise`` against ``target``, sum, and peak-normalise.

    ``interferer=None`` (``sir_db`` ignored) makes a noise-only mix with an all-zero
    interferer stem. The interferer and noise are trimmed or looped to the target length first. The
    normalising gain is applied to all three stems, so ``stems.mix`` equals their sum to
    float precision.
    """
    t = np.asarray(target, dtype=np.float64)
    if t.size == 0 or rms(t) <= 0.0:
        raise ValueError("target must be non-empty and non-silent")
    n = t.size
    if interferer is None:
        i = np.zeros(n)
    else:
        if sir_db is None:
            raise ValueError("sir_db is required when an interferer is given")
        i = fit_length(np.asarray(interferer, dtype=np.float64), n)
        i = i * gain_for_level(t, i, sir_db)
    z = fit_length(np.asarray(noise, dtype=np.float64), n)
    z = z * gain_for_level(t, z, snr_db)
    mix = t + i + z
    g = peak / max(float(np.max(np.abs(mix))), _EPS)
    t, i, z = t * g, i * g, z * g
    mix = t + i + z
    return Stems(
        mix=mix.astype(np.float32),
        target=t.astype(np.float32),
        interferer=i.astype(np.float32),
        noise=z.astype(np.float32),
    )


@dataclass
class Manifest:
    """Per-fixture metadata (``manifest.json``); ``index.json`` holds a list of these.

    ``paths`` are relative to the directory holding ``manifest.json``; in ``index.json`` they
    are prefixed with the fixture id so they resolve from the set root.
    """

    id: str
    set: str
    sample_rate: int
    duration_s: float
    snr_db: float
    sir_db: float | None  # None: noise-only fixture (no interferer)
    target_id: str
    interferer_id: str  # "" for a noise-only fixture
    noise_type: str
    transcript: str
    paths: dict[str, str]
    seed: int
    source: dict = field(default_factory=dict)
    interferer_transcript: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        return cls(**d)


def write_fixture(set_root: str | os.PathLike, manifest: Manifest, stems: Stems) -> Path:
    """Write the four stems and ``manifest.json`` under ``<set_root>/<id>/``."""
    out = Path(set_root) / manifest.id
    out.mkdir(parents=True, exist_ok=True)
    for name in STEM_NAMES:
        aio.save(out / manifest.paths[name], stems[name], manifest.sample_rate)
    (out / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return out


def write_index(set_root: str | os.PathLike, set_name: str, seed: int, manifests: list[Manifest]) -> Path:
    """Write ``index.json`` listing every fixture, paths relative to the set root."""
    root = Path(set_root)
    entries = []
    for m in manifests:
        d = m.to_dict()
        d["paths"] = {k: f"{m.id}/{v}" for k, v in m.paths.items()}
        entries.append(d)
    index = {
        "set": set_name,
        "seed": seed,
        "sample_rate": manifests[0].sample_rate if manifests else aio.PIPELINE_SR,
        "count": len(entries),
        "fixtures": entries,
    }
    path = root / "index.json"
    path.write_text(json.dumps(index, indent=2) + "\n")
    return path


def load_fixture(fixture_dir: str | os.PathLike) -> tuple[Manifest, Stems]:
    """Read a fixture directory back as ``(Manifest, Stems)`` at 16 kHz float32."""
    d = Path(fixture_dir)
    manifest = Manifest.from_dict(json.loads((d / "manifest.json").read_text()))
    arrays = {}
    for name in STEM_NAMES:
        x, _ = aio.load(d / manifest.paths[name], manifest.sample_rate)
        arrays[name] = x
    return manifest, Stems(**arrays)
