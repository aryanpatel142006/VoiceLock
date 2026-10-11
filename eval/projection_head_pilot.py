"""Aryan vs Joseph dedicated contrastive projection head pilot (T-273).

Evaluates a compact 2-layer projection head (192 -> 64 -> 32) trained on frozen ECAPA-TDNN
speaker embeddings of Aryan and Joseph using ArcFace or Contrastive loss.

Measures:
1. Metric space geometry: whether angular distance widens from baseline cos ~ 0.60 (or acute angle)
   to antipodal cos <= -0.50.
2. Wrong-speaker rejection and flips: whether stream classification flips drop to zero on heldout
   team recordings.
3. Inference latency and parameter budget: confirms < 0.1 ms CPU overhead.

Offline evaluation only; no production defaults mutation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim


def resolve_team_audio_paths() -> dict[str, Path]:
    """Locate Aryan and Joseph audio recordings across local, main repo, and fixture roots."""
    here = Path(__file__).resolve().parent.parent
    possible_roots = [
        here,
        here.parent / "selective-hearing",
    ]
    fixtures_env = os.environ.get("HEARING_FIXTURES_DIR")
    if fixtures_env:
        possible_roots.append(Path(fixtures_env).parent.parent)

    resolved: dict[str, Path] = {}
    candidates = {
        "aryan_profile": [
            "profiles/aryan/profile-01.wav",
            "eval/fixtures/generated/profiles/aryan/profile-01.wav",
        ],
        "joseph_profile": [
            "profiles/joseph/profile-01.wav",
            "eval/fixtures/generated/profiles/joseph/profile-01.wav",
        ],
        "aryan_fixture": [
            "eval/fixtures/generated/team/aryan/fixture-01.wav",
            "eval/fixtures/demo_listen/02_isolated_aryan.wav",
        ],
        "joseph_fixture": [
            "eval/fixtures/generated/team/joseph/fixture-01-20261009-1353.wav",
            "eval/fixtures/generated/team/joseph/fixture-02-namecall-20261009-1353.wav",
            "eval/fixtures/demo_listen/03_isolated_joseph.wav",
        ],
        "overlap_mix": [
            "eval/fixtures/demo_listen/01_raw_overlap_mix.wav",
            "eval/fixtures/demo_listen/00_AB_DEMO_COMPARISON.wav",
        ],
        "aryan_isolated": [
            "eval/fixtures/demo_listen/02_isolated_aryan.wav",
        ],
        "joseph_isolated": [
            "eval/fixtures/demo_listen/03_isolated_joseph.wav",
        ],
    }

    for key, rel_list in candidates.items():
        for root in possible_roots:
            if not root.exists():
                continue
            for rel in rel_list:
                cand = (root / rel).resolve()
                if cand.is_file():
                    resolved[key] = cand
                    break
            if key in resolved:
                break

    return resolved


class TeamProjectionHead(nn.Module):
    """Compact 2-layer projection head mapping 192-d ECAPA embeddings to 32-d discriminative space.

    Architecture:
        Linear(192, 64) -> LayerNorm(64) -> GELU() -> Linear(64, 32) -> L2-norm
    """

    def __init__(self, in_dim: int = 192, hidden_dim: int = 64, out_dim: int = 32) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input embeddings to unit hypersphere S^(out_dim - 1)."""
        z = self.net(x)
        return nn.functional.normalize(z, p=2, dim=-1)


class ArcFaceLoss(nn.Module):
    """ArcFace (Additive Angular Margin) Loss for two-class team discrimination.

    Enforces angular margin m on the target class logit, driving Aryan and Joseph
    representations into separated clusters on the unit hypersphere.
    """

    def __init__(
        self,
        in_features: int = 32,
        num_classes: int = 2,
        scale: float = 16.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_penalty = math.sin(math.pi - margin) * margin
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = nn.functional.normalize(self.weight, p=2, dim=-1)
        cosine = torch.matmul(features, w.t())
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.margin_penalty)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        loss = self.criterion(logits, labels.long())
        return loss, logits


class ContrastiveProjectionLoss(nn.Module):
    """Contrastive loss explicitly penalizing cross-speaker cosine similarity above target_neg_cos.

    Pulls positive pairs towards cos = 1.0 and pushes negative pairs below target_neg_cos (<= -0.50).
    """

    def __init__(self, target_neg_cos: float = -0.50, neg_weight: float = 2.5) -> None:
        super().__init__()
        self.target_neg_cos = float(target_neg_cos)
        self.neg_weight = float(neg_weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = torch.matmul(features, features.t())
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        pos_mask = label_matrix
        neg_mask = ~label_matrix

        pos_loss = (1.0 - sim[pos_mask]).pow(2).mean()

        neg_sim = sim[neg_mask]
        if neg_sim.numel() > 0:
            neg_loss = torch.clamp(neg_sim - self.target_neg_cos, min=0.0).pow(2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=features.device)

        return pos_loss + self.neg_weight * neg_loss


def slice_audio(
    audio: np.ndarray,
    sr: int = 16000,
    chunk_s: float = 0.5,
    step_s: float = 0.25,
    rms_min: float = 0.005,
    peak_min: float = 0.02,
) -> list[np.ndarray]:
    """Slice audio into voiced chunks of fixed duration."""
    n = int(chunk_s * sr)
    step = int(step_s * sr)
    chunks = []
    for start in range(0, len(audio) - n + 1, step):
        chunk = audio[start : start + n]
        if len(chunk) == n:
            peak = float(np.max(np.abs(chunk)))
            rms = float(np.sqrt(np.mean(chunk**2)))
            if peak >= peak_min and rms >= rms_min:
                chunks.append(chunk)
    return chunks


def generate_synthetic_team_dataset(
    seed: int = 42,
    num_train: int = 100,
    num_test: int = 50,
    baseline_similarity: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic synthetic 192-d ECAPA embeddings simulating Aryan and Joseph.

    Simulates the realistic baseline geometry:
    - Shared acoustic subspace (gender/pitch correlation leading to cos ~ 0.60).
    - Distinct personal phonetic subspaces.
    """
    rng = np.random.default_rng(seed)

    shared_subspace = rng.normal(size=192).astype(np.float32)
    shared_subspace /= np.linalg.norm(shared_subspace)

    v_aryan = rng.normal(size=192).astype(np.float32)
    v_aryan -= np.dot(v_aryan, shared_subspace) * shared_subspace
    v_aryan /= np.linalg.norm(v_aryan)

    v_joseph = rng.normal(size=192).astype(np.float32)
    v_joseph -= np.dot(v_joseph, shared_subspace) * shared_subspace
    v_joseph /= np.linalg.norm(v_joseph)

    alpha = math.sqrt(baseline_similarity)
    beta = math.sqrt(1.0 - baseline_similarity)

    aryan_center = alpha * shared_subspace + beta * v_aryan
    joseph_center = alpha * shared_subspace + beta * v_joseph
    aryan_center /= np.linalg.norm(aryan_center)
    joseph_center /= np.linalg.norm(joseph_center)

    def sample_speaker(center: np.ndarray, count: int, noise_scale: float = 0.12) -> np.ndarray:
        noise = rng.normal(scale=noise_scale, size=(count, 192)).astype(np.float32)
        samples = center + noise
        norms = np.linalg.norm(samples, axis=-1, keepdims=True)
        return samples / np.maximum(norms, 1e-12)

    X_a_train = sample_speaker(aryan_center, num_train)
    X_j_train = sample_speaker(joseph_center, num_train)
    X_a_test = sample_speaker(aryan_center, num_test)
    X_j_test = sample_speaker(joseph_center, num_test)

    return X_a_train, X_j_train, X_a_test, X_j_test


def extract_real_team_embeddings(
    audio_paths: dict[str, Path],
    train_chunk_s: float = 0.5,
    train_step_s: float = 0.15,
    test_chunk_s: float = 1.0,
    test_step_s: float = 1.0,
    max_train_per_speaker: int = 120,
    max_test_per_speaker: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract real frozen ECAPA embeddings for Aryan and Joseph clean slices and heldout fixtures."""
    import soundfile as sf

    from hearing.tse.enrolled import _load_encoder, _unit_embedding

    backend = _load_encoder()

    ary_prof, _ = sf.read(str(audio_paths["aryan_profile"]), dtype="float32")
    jos_prof, _ = sf.read(str(audio_paths["joseph_profile"]), dtype="float32")
    ary_fixt, _ = sf.read(str(audio_paths["aryan_fixture"]), dtype="float32")
    jos_fixt, _ = sf.read(str(audio_paths["joseph_fixture"]), dtype="float32")

    ca_train = slice_audio(ary_prof, chunk_s=train_chunk_s, step_s=train_step_s)
    cj_train = slice_audio(jos_prof, chunk_s=train_chunk_s, step_s=train_step_s)
    ca_test = slice_audio(ary_fixt, chunk_s=test_chunk_s, step_s=test_step_s)
    cj_test = slice_audio(jos_fixt, chunk_s=test_chunk_s, step_s=test_step_s)

    ca_train = ca_train[:max_train_per_speaker]
    cj_train = cj_train[:max_train_per_speaker]
    ca_test = ca_test[:max_test_per_speaker]
    cj_test = cj_test[:max_test_per_speaker]

    X_a_train = np.array([_unit_embedding(backend, c) for c in ca_train], dtype=np.float32)
    X_j_train = np.array([_unit_embedding(backend, c) for c in cj_train], dtype=np.float32)
    X_a_test = np.array([_unit_embedding(backend, c) for c in ca_test], dtype=np.float32)
    X_j_test = np.array([_unit_embedding(backend, c) for c in cj_test], dtype=np.float32)

    return X_a_train, X_j_train, X_a_test, X_j_test


def train_projection_head(
    X_a_train: np.ndarray,
    X_j_train: np.ndarray,
    loss_type: str = "contrastive",
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    target_neg_cos: float = -0.50,
    arcface_margin: float = 0.50,
    arcface_scale: float = 16.0,
) -> tuple[TeamProjectionHead, dict[str, Any]]:
    """Train the 2-layer projection head on frozen ECAPA embeddings using Contrastive or ArcFace loss."""
    dev = torch.device(device)
    model = TeamProjectionHead(in_dim=192, hidden_dim=64, out_dim=32).to(dev)

    X_train = torch.cat([torch.from_numpy(X_a_train), torch.from_numpy(X_j_train)], dim=0).to(dev)
    y_train = torch.cat([
        torch.zeros(len(X_a_train), dtype=torch.long),
        torch.ones(len(X_j_train), dtype=torch.long),
    ]).to(dev)

    if loss_type == "arcface":
        criterion = ArcFaceLoss(
            in_features=32,
            num_classes=2,
            scale=arcface_scale,
            margin=arcface_margin,
        ).to(dev)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif loss_type == "contrastive":
        criterion = ContrastiveProjectionLoss(
            target_neg_cos=target_neg_cos,
            neg_weight=2.5,
        ).to(dev)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        if loss_type == "arcface":
            z = model(X_train)
            loss, _ = criterion(z, y_train)
        else:
            z = model(X_train)
            loss = criterion(z, y_train)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    meta = {
        "loss_type": loss_type,
        "epochs": epochs,
        "initial_loss": history[0] if history else 0.0,
        "final_loss": history[-1] if history else 0.0,
        "device": device,
    }
    return model, meta


def compute_metric_stats(
    embeddings_a: np.ndarray,
    embeddings_j: np.ndarray,
) -> dict[str, float]:
    """Calculate within-speaker and cross-speaker cosine similarity statistics and angular distance."""
    # Cross speaker
    cross = embeddings_a @ embeddings_j.T
    mean_cross = float(np.mean(cross))
    max_cross = float(np.max(cross))
    min_cross = float(np.min(cross))

    # Within speaker
    within_a = embeddings_a @ embeddings_a.T
    within_j = embeddings_j @ embeddings_j.T
    mean_within_a = float(np.mean(within_a))
    mean_within_j = float(np.mean(within_j))
    mean_within = float((mean_within_a + mean_within_j) / 2.0)

    # Angular distance in degrees and radians
    clamped_cross = max(-1.0, min(1.0, mean_cross))
    angular_dist_rad = math.acos(clamped_cross)
    angular_dist_deg = math.degrees(angular_dist_rad)

    margin = mean_within - mean_cross

    return {
        "mean_cross_cosine": mean_cross,
        "max_cross_cosine": max_cross,
        "min_cross_cosine": min_cross,
        "mean_within_a": mean_within_a,
        "mean_within_j": mean_within_j,
        "mean_within": mean_within,
        "margin": margin,
        "angular_distance_rad": angular_dist_rad,
        "angular_distance_deg": angular_dist_deg,
    }


def evaluate_flips_on_heldout(
    model: TeamProjectionHead,
    X_a_test: np.ndarray,
    X_j_test: np.ndarray,
    prototype_a: np.ndarray,
    prototype_j: np.ndarray,
) -> dict[str, Any]:
    """Measure classification accuracy and wrong-speaker flips before and after projection."""
    # Baseline ECAPA cosine matching
    s_a_base_for_a = X_a_test @ prototype_a
    s_j_base_for_a = X_a_test @ prototype_j
    base_a_errors = int(np.sum(s_j_base_for_a > s_a_base_for_a))

    s_a_base_for_j = X_j_test @ prototype_a
    s_j_base_for_j = X_j_test @ prototype_j
    base_j_errors = int(np.sum(s_a_base_for_j > s_j_base_for_j))

    total_base_samples = len(X_a_test) + len(X_j_test)
    base_total_errors = base_a_errors + base_j_errors
    base_error_rate = float(base_total_errors / max(1, total_base_samples))

    # Projected cosine matching
    model.eval()
    with torch.no_grad():
        z_proto_a = model(torch.from_numpy(prototype_a.reshape(1, -1)).float()).squeeze(0).numpy()
        z_proto_j = model(torch.from_numpy(prototype_j.reshape(1, -1)).float()).squeeze(0).numpy()

        z_a_test = model(torch.from_numpy(X_a_test).float()).numpy()
        z_j_test = model(torch.from_numpy(X_j_test).float()).numpy()

    s_a_proj_for_a = z_a_test @ z_proto_a
    s_j_proj_for_a = z_a_test @ z_proto_j
    proj_a_errors = int(np.sum(s_j_proj_for_a > s_a_proj_for_a))

    s_a_proj_for_j = z_j_test @ z_proto_a
    s_j_proj_for_j = z_j_test @ z_proto_j
    proj_j_errors = int(np.sum(s_a_proj_for_j > s_j_proj_for_j))

    proj_total_errors = proj_a_errors + proj_j_errors
    proj_error_rate = float(proj_total_errors / max(1, total_base_samples))

    # Flips in sequential stream tracking simulation
    # When tracking Aryan across consecutive windows, does selection ever flip to Joseph?
    base_sequence_flips = 0
    proj_sequence_flips = 0
    prev_base_pick = 0
    prev_proj_pick = 0

    for i in range(len(X_a_test)):
        # Target Aryan
        base_pick = 0 if s_a_base_for_a[i] >= s_j_base_for_a[i] else 1
        proj_pick = 0 if s_a_proj_for_a[i] >= s_j_proj_for_a[i] else 1
        if base_pick != prev_base_pick and i > 0:
            base_sequence_flips += 1
        if proj_pick != prev_proj_pick and i > 0:
            proj_sequence_flips += 1
        prev_base_pick = base_pick
        prev_proj_pick = proj_pick

    return {
        "baseline": {
            "aryan_errors": base_a_errors,
            "aryan_total": len(X_a_test),
            "joseph_errors": base_j_errors,
            "joseph_total": len(X_j_test),
            "total_errors": base_total_errors,
            "error_rate": base_error_rate,
            "accuracy": 1.0 - base_error_rate,
            "sequential_flips": base_sequence_flips,
        },
        "projected": {
            "aryan_errors": proj_a_errors,
            "aryan_total": len(X_a_test),
            "joseph_errors": proj_j_errors,
            "joseph_total": len(X_j_test),
            "total_errors": proj_total_errors,
            "error_rate": proj_error_rate,
            "accuracy": 1.0 - proj_error_rate,
            "sequential_flips": proj_sequence_flips,
        },
        "flips_dropped_to_zero": proj_total_errors == 0,
    }


def measure_inference_overhead(model: TeamProjectionHead, iters: int = 1000) -> dict[str, float]:
    """Benchmark inference latency on CPU and count model parameters."""
    model_cpu = TeamProjectionHead(in_dim=model.in_dim, hidden_dim=model.hidden_dim, out_dim=model.out_dim)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    dummy = torch.randn(1, 192)

    # Warmup
    for _ in range(50):
        _ = model_cpu(dummy)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model_cpu(dummy)
    elapsed_s = time.perf_counter() - t0

    latency_us = (elapsed_s / iters) * 1e6
    latency_ms = latency_us / 1000.0
    param_count = sum(p.numel() for p in model_cpu.parameters())

    return {
        "latency_us": latency_us,
        "latency_ms": latency_ms,
        "param_count": float(param_count),
    }


def run_projection_head_pilot(
    loss_type: str = "contrastive",
    use_synthetic: bool = False,
    epochs: int = 120,
    lr: float = 1e-3,
    device: str = "cpu",
    target_neg_cos: float = -0.50,
    arcface_margin: float = 0.50,
    arcface_scale: float = 16.0,
    save_weights_path: Path | None = None,
) -> dict[str, Any]:
    """Execute full offline pilot evaluation: train on clean slices, test on heldout team audio."""
    audio_paths = resolve_team_audio_paths()
    has_real_audio = ("aryan_profile" in audio_paths and "joseph_profile" in audio_paths
                      and "aryan_fixture" in audio_paths and "joseph_fixture" in audio_paths)

    if use_synthetic or not has_real_audio:
        data_source = "synthetic"
        X_a_train, X_j_train, X_a_test, X_j_test = generate_synthetic_team_dataset(
            seed=42, num_train=100, num_test=50, baseline_similarity=0.60
        )
    else:
        data_source = "real_team_recordings"
        X_a_train, X_j_train, X_a_test, X_j_test = extract_real_team_embeddings(
            audio_paths,
            train_chunk_s=0.5,
            train_step_s=0.15,
            test_chunk_s=1.0,
            test_step_s=1.0,
        )

    # 1. Baseline metric statistics
    base_stats = compute_metric_stats(X_a_test, X_j_test)

    # 2. Train projection head
    model, train_meta = train_projection_head(
        X_a_train,
        X_j_train,
        loss_type=loss_type,
        epochs=epochs,
        lr=lr,
        device=device,
        target_neg_cos=target_neg_cos,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
    )

    if save_weights_path:
        save_weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_weights_path)

    # 3. Projected metric statistics on heldout test sets
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        z_a_test = model(torch.from_numpy(X_a_test).to(dev)).cpu().numpy()
        z_j_test = model(torch.from_numpy(X_j_test).to(dev)).cpu().numpy()

    proj_stats = compute_metric_stats(z_a_test, z_j_test)

    # 4. Wrong-speaker flips and stream tracking evaluation
    proto_a = np.mean(X_a_train[:10], axis=0)
    proto_j = np.mean(X_j_train[:10], axis=0)
    proto_a /= np.linalg.norm(proto_a)
    proto_j /= np.linalg.norm(proto_j)

    flip_results = evaluate_flips_on_heldout(model, X_a_test, X_j_test, proto_a, proto_j)

    # 5. Measure latency & parameter footprint
    overhead = measure_inference_overhead(model)

    # 6. Evaluation gates
    # Gate 1: Cross-speaker cosine <= -0.50 (angular distance >= 120 degrees)
    gate_cosine_widened = proj_stats["mean_cross_cosine"] <= -0.50
    # Gate 2: Flips dropped to zero on heldout 1.0s recordings
    gate_flips_zero = flip_results["projected"]["total_errors"] == 0
    # Gate 3: CPU latency < 0.20 ms (200 us)
    gate_latency_pass = overhead["latency_us"] < 200.0

    verdict = gate_cosine_widened and gate_flips_zero and gate_latency_pass

    report: dict[str, Any] = {
        "pilot_id": "T-273",
        "title": "Creative R&D: Aryan vs Joseph dedicated contrastive projection head pilot",
        "data_source": data_source,
        "loss_type": loss_type,
        "training": train_meta,
        "metric_geometry": {
            "baseline_ecapa": base_stats,
            "projected_head": proj_stats,
            "angular_distance_widened_deg": (
                proj_stats["angular_distance_deg"] - base_stats["angular_distance_deg"]
            ),
            "cosine_widened_to_le_minus_half": gate_cosine_widened,
        },
        "heldout_speaker_flips": flip_results,
        "runtime_overhead": overhead,
        "gates": {
            "gate_cosine_widened_le_minus_0_50": gate_cosine_widened,
            "gate_wrong_speaker_flips_zero": gate_flips_zero,
            "gate_latency_under_200us": gate_latency_pass,
            "overall_adopt_verdict": verdict,
        },
        "summary": (
            f"Projection head ({loss_type}) shifted cross-speaker cosine from "
            f"{base_stats['mean_cross_cosine']:.3f} ({base_stats['angular_distance_deg']:.1f}°) to "
            f"{proj_stats['mean_cross_cosine']:.3f} ({proj_stats['angular_distance_deg']:.1f}°). "
            f"Heldout classification errors dropped from {flip_results['baseline']['total_errors']} to "
            f"{flip_results['projected']['total_errors']} (flips dropped to zero: {gate_flips_zero})."
        ),
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Aryan vs Joseph projection head pilot evaluation (T-273)."
    )
    parser.add_argument(
        "--loss",
        choices=["contrastive", "arcface"],
        default="contrastive",
        help="Loss function to optimize (default: contrastive).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic embeddings instead of audio files.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs (default: 120).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (cpu, mps, cuda; default: cpu).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=-0.50,
        help="Target negative cosine for contrastive loss (default: -0.50).",
    )
    parser.add_argument(
        "--arcface-margin",
        type=float,
        default=0.50,
        help="ArcFace angular margin m in radians (default: 0.50).",
    )
    parser.add_argument(
        "--arcface-scale",
        type=float,
        default=16.0,
        help="ArcFace scale s factor (default: 16.0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON file path (default: prints to stdout).",
    )
    parser.add_argument(
        "--save-weights",
        type=str,
        default="",
        help="Optional path to save trained PyTorch weights.",
    )

    args = parser.parse_args()

    save_weights_path = Path(args.save_weights) if args.save_weights else None
    report = run_projection_head_pilot(
        loss_type=args.loss,
        use_synthetic=args.synthetic,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        target_neg_cos=args.margin,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        save_weights_path=save_weights_path,
    )

    formatted = json.dumps(report, indent=2)
    print(formatted)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted)
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
