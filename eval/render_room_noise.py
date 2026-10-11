"""Render deterministic room-response and additive-noise evaluation shards.

The renderer is deliberately offline: room impulse responses are analytic presets and the
committed tiny speech/noise clips are the only source material. Audio is generated into the
requested output directory while JSON manifests are kept as the durable index.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from eval.fixtures.mix import rms
from hearing.audio import io as aio

SR = aio.PIPELINE_SR
DEFAULT_OUT = Path(__file__).resolve().parent / "shards" / "room_noise"
ROOMS = {
    "small_room": (0.012, ((0.003, 0.45), (0.007, 0.25), (0.011, 0.14))),
    "medium_room": (0.018, ((0.004, 0.38), (0.010, 0.22), (0.016, 0.12))),
    "large_room": (0.028, ((0.006, 0.34), (0.015, 0.18), (0.025, 0.10))),
}
SNRS = (-5.0, 0.0, 5.0, 10.0)


def room_impulse_response(name: str, sample_rate: int = SR) -> np.ndarray:
    """Return a deterministic, unit-direct-path synthetic room response."""
    try:
        tail_s, reflections = ROOMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown room {name!r}; choose from {sorted(ROOMS)}") from exc
    n = max(1, round(tail_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    ir = np.exp(-t / (tail_s / 3.0)).astype(np.float64)
    ir[0] = 1.0
    for delay_s, gain in reflections:
        delay = round(delay_s * sample_rate)
        if delay < n:
            ir[delay] += gain
    ir /= np.max(np.abs(ir))
    return ir.astype(np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        raise ValueError("audio source cannot be empty")
    return np.resize(np.asarray(x, dtype=np.float32), n).astype(np.float32)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def render_example(
    speech: np.ndarray,
    noise: np.ndarray,
    *,
    snr_db: float,
    rir_name: str,
    seed: int,
    example_id: str,
) -> dict:
    """Render one reverberated speech + additive-noise example."""
    del seed  # retained in the contract so recipes can be extended without changing manifests
    speech = np.asarray(speech, dtype=np.float32)
    noise = _fit(np.asarray(noise, dtype=np.float32), len(speech))
    ir = room_impulse_response(rir_name)
    target = np.convolve(speech, ir, mode="full")[: len(speech)].astype(np.float32)
    target_rms, noise_rms = rms(target), rms(noise)
    if target_rms <= 0 or noise_rms <= 0:
        raise ValueError("speech and noise must both be non-silent")
    noise = noise * (target_rms / noise_rms / 10 ** (snr_db / 20))
    mix = target + noise
    peak = max(float(np.max(np.abs(mix))), 1e-12)
    if peak > 0.9:
        target = target * (0.9 / peak)
        noise = noise * (0.9 / peak)
        mix = target + noise
    return {
        "id": example_id,
        "sample_rate": SR,
        "duration_s": round(len(speech) / SR, 4),
        "snr_db": float(snr_db),
        "measured_snr_db": float(20 * np.log10(rms(target) / rms(noise))),
        "room_ir": {"name": rir_name, "sample_rate": SR, "length": len(ir)},
        "target": target.astype(np.float32),
        "noise": noise.astype(np.float32),
        "mix": mix.astype(np.float32),
    }


def _write_audio(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".wav", dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        aio.save(temporary, x, SR)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def render_shards(
    out: str | os.PathLike = DEFAULT_OUT,
    *,
    count: int = 12,
    shards: int = 3,
    seed: int = 0,
    resume: bool = True,
) -> dict:
    """Render ``count`` examples split over ``shards`` atomically checkpointed directories."""
    if count < 1 or shards < 1:
        raise ValueError("count and shards must be positive")
    out = Path(out)
    from eval.fixtures.sources import TinySource

    source = TinySource()
    speech = source.speech()
    _, base_noise, _ = source.noise()
    checkpoint_path = out / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text())
        if resume and checkpoint_path.exists()
        else {"schema_version": 1, "seed": seed, "count": count, "shards": shards, "completed_shards": []}
    )
    if (checkpoint["seed"], checkpoint["count"], checkpoint["shards"]) != (seed, count, shards):
        raise ValueError("existing checkpoint does not match requested recipe")
    completed = set(checkpoint["completed_shards"])
    for shard_index in range(shards):
        if shard_index in completed:
            continue
        indices = range(shard_index * count // shards, (shard_index + 1) * count // shards)
        entries = []
        shard_root = out / f"shard-{shard_index:04d}"
        for index in indices:
            target = speech[index % len(speech)].audio
            noise = base_noise if index % 2 == 0 else np.random.default_rng(seed + index).standard_normal(len(target)).astype(np.float32)
            room = tuple(ROOMS)[index % len(ROOMS)]
            snr = SNRS[index % len(SNRS)]
            example_id = f"{index:05d}_{room}_snr{snr:g}"
            rendered = render_example(target, noise, snr_db=snr, rir_name=room, seed=seed + index, example_id=example_id)
            example_root = shard_root / example_id
            _write_audio(example_root / "mix.wav", rendered["mix"])
            _write_audio(example_root / "target.wav", rendered["target"])
            _write_audio(example_root / "noise.wav", rendered["noise"])
            entries.append({k: v for k, v in rendered.items() if k not in {"target", "noise", "mix"}} | {
                "paths": {"mix": f"{example_id}/mix.wav", "target": f"{example_id}/target.wav", "noise": f"{example_id}/noise.wav"}
            })
        _atomic_json(shard_root / "manifest.json", {"schema_version": 1, "shard": shard_index, "examples": entries})
        completed.add(shard_index)
        checkpoint["completed_shards"] = sorted(completed)
        _atomic_json(checkpoint_path, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = render_shards(args.out, count=args.count, shards=args.shards, seed=args.seed, resume=not args.no_resume)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
