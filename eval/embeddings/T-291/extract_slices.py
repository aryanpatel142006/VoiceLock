"""Extract and package ECAPA embeddings and projection head outputs for team audio (T-291).

Extracts 192-d ECAPA embeddings and 32-d TeamProjectionHead outputs for 0.5s, 1.0s, and 2.0s
windows over Aryan and Joseph recordings.
Computes per-window metadata (SNR, stock cosine margin, projected head margin).
Saves to eval/embeddings/T-291/team_slices.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.projection_head_pilot import TeamProjectionHead
from hearing.tse.enrolled import _load_encoder, _unit_embedding

SR = 16000
WINDOWS = {
    "0.5s": 0.5,
    "1.0s": 1.0,
    "2.0s": 2.0,
}
SPEAKERS = ("aryan", "joseph")


def resolve_team_fixtures_dir() -> Path:
    """Find the directory containing team recordings."""
    env = os.environ.get("HEARING_FIXTURES_DIR")
    candidates = []
    if env:
        candidates.append(Path(env) / "team")
        candidates.append(Path(env))
    candidates.extend([
        ROOT / "eval/fixtures/generated/team",
    ])
    for cand in candidates:
        if cand.is_dir() and (cand / "aryan").is_dir() and (cand / "joseph").is_dir():
            return cand
    raise FileNotFoundError(
        f"Team audio directory not found in {candidates}. "
        "Set HEARING_FIXTURES_DIR to point to generated fixtures."
    )


def resolve_head_ckpt() -> Path:
    """Find models/team_projection_head.pt."""
    env = os.environ.get("HEARING_MODELS_DIR")
    candidates = []
    if env:
        candidates.append(Path(env) / "team_projection_head.pt")
    candidates.extend([
        ROOT / "models/team_projection_head.pt",
    ])
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"team_projection_head.pt not found in {candidates}. "
        "Set HEARING_MODELS_DIR or check models/."
    )


def estimate_snr_db(wav: np.ndarray, frame_ms: float = 20.0) -> float:
    """Estimate SNR in dB comparing overall speech RMS against 10th-percentile frame RMS."""
    f = int(frame_ms * 1e-3 * SR)
    if len(wav) < f:
        return 0.0
    n_frames = len(wav) // f
    frames = wav[: n_frames * f].reshape(-1, f)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    noise_floor = float(np.percentile(frame_rms, 10))
    speech_rms = float(np.sqrt(np.mean(wav**2) + 1e-12))
    return float(20.0 * np.log10(speech_rms / max(noise_floor, 1e-12)))


def extract_slices_for_window(
    win_s: float,
    audio_by_speaker: dict[str, list[tuple[str, np.ndarray]]],
    encoder,
    head: TeamProjectionHead,
) -> dict[str, np.ndarray]:
    """Slice audio for a given window length and compute embeddings, projections, and margins."""
    win_samples = int(win_s * SR)
    raw_slices = []

    for spk_idx, spk_name in enumerate(SPEAKERS):
        recordings = audio_by_speaker[spk_name]
        for sess_idx, (rel_name, samples) in enumerate(recordings):
            # Non-overlapping windows for independent statistical evaluation
            for start in range(0, len(samples) - win_samples + 1, win_samples):
                chunk = samples[start : start + win_samples]
                peak = float(np.max(np.abs(chunk)))
                rms = float(np.sqrt(np.mean(chunk**2)))
                # Standard voice activity gate (peak >= 0.02, RMS >= 0.005)
                if peak < 0.02 or rms < 0.005:
                    continue

                snr = estimate_snr_db(chunk)
                e = _unit_embedding(encoder, chunk).astype(np.float32)

                raw_slices.append({
                    "speaker": spk_idx,
                    "speaker_name": spk_name,
                    "session": sess_idx,
                    "recording": rel_name,
                    "start_s": float(start / SR),
                    "window_length": float(win_s),
                    "snr_db": snr,
                    "embedding": e,
                })

    n = len(raw_slices)
    if n == 0:
        raise ValueError(f"No voiced slices found for window {win_s}s")

    embeddings = np.array([s["embedding"] for s in raw_slices], dtype=np.float32)
    with torch.no_grad():
        head_out = head(torch.from_numpy(embeddings)).numpy().astype(np.float32)

    speakers = np.array([s["speaker"] for s in raw_slices], dtype=np.int64)
    speaker_names = np.array([s["speaker_name"] for s in raw_slices], dtype=object)
    sessions = np.array([s["session"] for s in raw_slices], dtype=np.int64)
    recordings = np.array([s["recording"] for s in raw_slices], dtype=object)
    starts = np.array([s["start_s"] for s in raw_slices], dtype=np.float32)
    win_lens = np.array([s["window_length"] for s in raw_slices], dtype=np.float32)
    snrs = np.array([s["snr_db"] for s in raw_slices], dtype=np.float32)

    # Compute per-window stock cosine margin and projected head margin
    stock_margins = np.zeros(n, dtype=np.float32)
    head_margins = np.zeros(n, dtype=np.float32)

    for i in range(n):
        same_mask = (speakers == speakers[i])
        same_mask[i] = False
        cross_mask = (speakers != speakers[i])

        stock_same = float(np.mean(embeddings[i] @ embeddings[same_mask].T))
        stock_cross = float(np.mean(embeddings[i] @ embeddings[cross_mask].T))
        stock_margins[i] = stock_same - stock_cross

        head_same = float(np.mean(head_out[i] @ head_out[same_mask].T))
        head_cross = float(np.mean(head_out[i] @ head_out[cross_mask].T))
        head_margins[i] = head_same - head_cross

    return {
        "embeddings": embeddings,
        "head_out": head_out,
        "speaker": speakers,
        "speaker_id": speaker_names,
        "speaker_name": speaker_names,
        "session": sessions,
        "recording": recordings,
        "start_s": starts,
        "window_length": win_lens,
        "snr_db": snrs,
        "cosine_margin_stock": stock_margins,
        "projected_margin_head": head_margins,
    }


def compute_summary_stats(margins: np.ndarray) -> dict[str, float]:
    """Compute standard statistical summaries of a margin distribution."""
    return {
        "mean": float(np.mean(margins)),
        "std": float(np.std(margins)),
        "min": float(np.min(margins)),
        "p05": float(np.percentile(margins, 5)),
        "p25": float(np.percentile(margins, 25)),
        "p50": float(np.percentile(margins, 50)),
        "p75": float(np.percentile(margins, 75)),
        "p95": float(np.percentile(margins, 95)),
        "max": float(np.max(margins)),
    }


def build_team_slices_dataset(
    fixtures_dir: Path | None = None,
    head_ckpt: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Extract and build the team_slices.npz dataset."""
    fixtures_dir = fixtures_dir or resolve_team_fixtures_dir()
    head_ckpt = head_ckpt or resolve_head_ckpt()
    out_path = out_path or (ROOT / "eval/embeddings/T-291/team_slices.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading team audio from {fixtures_dir}...")
    audio_by_speaker: dict[str, list[tuple[str, np.ndarray]]] = {}
    for spk in SPEAKERS:
        spk_dir = fixtures_dir / spk
        spk_files = sorted(spk_dir.glob("*.wav"))
        audio_by_speaker[spk] = []
        for f in spk_files:
            data, _sr = sf.read(str(f), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            audio_by_speaker[spk].append((f.name, mono))
            print(f"  {spk}: {f.name} ({len(mono)/SR:.2f}s)")

    print("Loading ECAPA encoder...")
    encoder = _load_encoder()

    print(f"Loading projection head from {head_ckpt}...")
    head = TeamProjectionHead(in_dim=192, hidden_dim=64, out_dim=32)
    head.load_state_dict(torch.load(head_ckpt, map_location="cpu"))
    head.eval()

    store: dict[str, np.ndarray | str] = {
        "windows": np.array(list(WINDOWS.keys())),
        "speakers": np.array(list(SPEAKERS)),
    }

    summary: dict[str, dict] = {}

    for w_name, win_s in WINDOWS.items():
        print(f"Extracting slices for window {w_name} ({win_s}s)...")
        w_data = extract_slices_for_window(win_s, audio_by_speaker, encoder, head)
        n_slices = len(w_data["embeddings"])
        print(f"  {w_name}: {n_slices} voiced slices extracted")

        store[f"embeddings/{w_name}"] = w_data["embeddings"]
        store[f"head_out/{w_name}"] = w_data["head_out"]
        store[f"meta/{w_name}/speaker"] = w_data["speaker"]
        store[f"meta/{w_name}/speaker_id"] = w_data["speaker_id"]
        store[f"meta/{w_name}/speaker_name"] = w_data["speaker_name"]
        store[f"meta/{w_name}/session"] = w_data["session"]
        store[f"meta/{w_name}/recording"] = w_data["recording"]
        store[f"meta/{w_name}/start_s"] = w_data["start_s"]
        store[f"meta/{w_name}/window_length"] = w_data["window_length"]
        store[f"meta/{w_name}/snr_db"] = w_data["snr_db"]
        store[f"meta/{w_name}/cosine_margin_stock"] = w_data["cosine_margin_stock"]
        store[f"meta/{w_name}/projected_margin_head"] = w_data["projected_margin_head"]

        stock_stats = compute_summary_stats(w_data["cosine_margin_stock"])
        head_stats = compute_summary_stats(w_data["projected_margin_head"])

        summary[w_name] = {
            "n_slices": n_slices,
            "window_length_s": win_s,
            "stock_cosine_margin": stock_stats,
            "projected_head_margin": head_stats,
            "ratio_head_over_stock_mean": head_stats["mean"] / stock_stats["mean"],
        }
        print(
            f"  Stock margin mean={stock_stats['mean']:.3f} (p50={stock_stats['p50']:.3f}), "
            f"Head margin mean={head_stats['mean']:.3f} (p50={head_stats['p50']:.3f})"
        )

    provenance = {
        "dataset_name": "T-291 team_slices",
        "description": "ECAPA-TDNN (192-d) embeddings and TeamProjectionHead (32-d) outputs on real team audio",
        "source_recordings": {
            spk: [name for name, _ in audio_by_speaker[spk]] for spk in SPEAKERS
        },
        "models": {
            "ecapa": "speechbrain/spkrec-ecapa-voxceleb (192-d)",
            "head": "models/team_projection_head.pt (Linear 192->64 -> LayerNorm -> GELU -> Linear 64->32 -> L2)",
        },
        "windows": list(WINDOWS.keys()),
        "slice_stride": "non-overlapping (hop = window_length)",
        "vad_gate": "peak >= 0.02 and rms >= 0.005",
        "device": str(getattr(encoder, "device", "mps")),
    }

    store["provenance_json"] = np.array(json.dumps(provenance))
    store["margins_json"] = np.array(json.dumps(summary))

    np.savez_compressed(out_path, **store)
    file_size_kb = out_path.stat().st_size / 1024
    print(f"Saved dataset to {out_path} ({file_size_kb:.1f} KB)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=None, help="Path to team fixtures")
    parser.add_argument("--head-ckpt", type=Path, default=None, help="Path to team_projection_head.pt")
    parser.add_argument("--out", type=Path, default=None, help="Output .npz path")
    args = parser.parse_args()

    build_team_slices_dataset(
        fixtures_dir=args.fixtures_dir,
        head_ckpt=args.head_ckpt,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
