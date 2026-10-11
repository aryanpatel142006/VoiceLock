"""Prepare bounded, offline acoustic augmentation recipes.

This module inventories only local files.  It never downloads audio or renders
mixtures; consumers (T211) can use the emitted recipes to do that later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PIPELINE_RATE = 16_000
MAX_RECIPES = 24
MAX_INPUTS = 500
SNR_CHOICES = (-5.0, 0.0, 5.0, 10.0)


@dataclass(frozen=True)
class RirInspection:
    path: str
    valid: bool
    reason: str
    sample_rate: int | None = None
    duration_s: float | None = None
    sha256: str | None = None
    peak_sample: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_rir(path: Path, required_rate: int = PIPELINE_RATE) -> RirInspection:
    """Validate a local mono impulse response without changing its samples."""
    try:
        info = sf.info(path)
        if info.samplerate != required_rate:
            return RirInspection(str(path), False, f"rate {info.samplerate} != {required_rate}")
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError) as exc:
        return RirInspection(str(path), False, f"decode failed: {exc}")
    mono = np.asarray(audio).mean(axis=1)
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        return RirInspection(str(path), False, "empty or non-finite")
    peak = int(np.argmax(np.abs(mono)))
    if float(np.max(np.abs(mono))) <= 0:
        return RirInspection(str(path), False, "zero impulse")
    return RirInspection(
        str(path),
        True,
        "ok",
        required_rate,
        round(mono.size / required_rate, 6),
        _sha256(path),
        peak,
    )


def _metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        value = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _kind(path: Path, metadata: dict[str, Any]) -> str | None:
    value = str(metadata.get("kind", "")).lower()
    if value in {"noise", "rir"}:
        return value
    name = path.name.lower()
    if any(token in name for token in ("rir", "impulse", "room")):
        return "rir"
    if any(token in name for token in ("noise", "babble", "pink")):
        return "noise"
    return None


def _discover(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rirs: list[dict[str, Any]] = []
    noises: list[dict[str, Any]] = []
    if not root.exists():
        return rirs, noises
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".wav", ".flac"}:
            continue
        metadata = _metadata(path)
        kind = _kind(path, metadata)
        if kind is None:
            continue
        if not metadata.get("licence"):
            continue
        if kind == "rir":
            checked = inspect_rir(path)
            if checked.valid:
                rirs.append({"inspection": asdict(checked), "metadata": metadata})
        else:
            try:
                info = sf.info(path)
            except (OSError, RuntimeError, ValueError):
                continue
            if info.samplerate == PIPELINE_RATE and info.frames > 0:
                noises.append(
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                        "sample_rate": info.samplerate,
                        "duration_s": round(info.frames / info.samplerate, 6),
                        "metadata": metadata,
                    }
                )
    return rirs[:MAX_INPUTS], noises[:MAX_INPUTS]


def _synthetic_rir(variant: int) -> dict[str, Any]:
    tails = (0.0, 0.08, 0.18)
    return {
        "synthetic": True,
        "generator": "unit impulse with declared exponential tail",
        "variant": variant,
        "sample_rate": PIPELINE_RATE,
        "alignment": {"peak_sample": 0, "tail_s": tails[variant], "peak_scale": 1.0},
        "licence": "project-generated",
    }


def recipe_hash(recipe: dict[str, Any]) -> str:
    body = {key: value for key, value in recipe.items() if key != "config_hash"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_plan(root: Path, *, seed: int = 239, max_recipes: int = MAX_RECIPES) -> dict[str, Any]:
    """Return a deterministic recipe manifest; no audio output is produced."""
    if not 1 <= max_recipes <= MAX_RECIPES:
        raise ValueError(f"max_recipes must be between 1 and {MAX_RECIPES}")
    rirs, noises = _discover(root)
    planned = max_recipes
    recipes: list[dict[str, Any]] = []
    for index in range(planned):
        noise = noises[index % len(noises)] if noises else None
        rir = rirs[index % len(rirs)] if rirs else {"inspection": None, "metadata": {}, "synthetic": True}
        rir_spec = (
            {
                "synthetic": False,
                "path": rir["inspection"]["path"],
                "sha256": rir["inspection"]["sha256"],
                "sample_rate": PIPELINE_RATE,
                "alignment": {
                    "peak_sample": rir["inspection"]["peak_sample"],
                    "tail_s": rir["inspection"]["duration_s"],
                    "peak_scale": 1.0,
                },
                "licence": rir["metadata"]["licence"],
            }
            if not rir.get("synthetic")
            else _synthetic_rir(index % 3)
        )
        noise_spec = (
            {
                "path": noise["path"],
                "sha256": noise["sha256"],
                "sample_rate": PIPELINE_RATE,
                "licence": noise["metadata"]["licence"],
                "source": noise["metadata"].get("source", noise["path"]),
                "speaker": noise["metadata"].get("speaker", ""),
                "utterance": noise["metadata"].get("utterance", ""),
                "interval": noise["metadata"].get("interval", ""),
            }
            if noise
            else {"synthetic": True, "generator": "none (consumer must supply noise)", "licence": "not-applicable"}
        )
        recipe = {
            "id": f"acoustic_{index:02d}",
            "seed": seed + index,
            "snr_db": SNR_CHOICES[index % len(SNR_CHOICES)],
            "fixed_original_group": f"original_{index % 4:02d}",
            "rir": rir_spec,
            "noise": {"snr_db": SNR_CHOICES[index % len(SNR_CHOICES)], "source": noise_spec},
            "provenance": {
                "rir": rir_spec,
                "noise": noise_spec,
                "source_root": str(root),
            },
            "caps": {"max_duration_s": 20 * 60, "max_bytes": 1_000_000_000},
        }
        recipe["config_hash"] = recipe_hash(recipe)
        recipes.append(recipe)
    return {
        "schema": "overnight-acoustic-plan.v1",
        "seed": seed,
        "pipeline_sample_rate": PIPELINE_RATE,
        "available": {"rir": len(rirs), "noise": len(noises), "cache_root": str(root)},
        "planned": {"recipes": planned, "actual_cached_pairings": min(len(rirs), len(noises))},
        "rejected": {"unknown_licence": "excluded during discovery", "invalid_rir": "excluded during discovery"},
        "recipes": recipes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated")))
    parser.add_argument("--output", type=Path, default=Path("eval/results/overnight_prep_acoustic.json"))
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--max-recipes", type=int, default=MAX_RECIPES)
    parser.add_argument("--resume", action="store_true", help="reuse output when every recipe hash verifies")
    args = parser.parse_args(argv)
    if args.resume and args.output.is_file():
        try:
            prior = json.loads(args.output.read_text())
            recipes = prior["recipes"]
            if recipes and all(recipe_hash(recipe) == recipe.get("config_hash") for recipe in recipes):
                print(json.dumps({"output": str(args.output), "resumed": True, "recipes": len(recipes)}))
                return 0
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    plan = build_plan(args.cache_root, seed=args.seed, max_recipes=args.max_recipes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "available": plan["available"], "recipes": len(plan["recipes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
