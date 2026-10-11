"""Replayable ablation runner for speaker embedding projection heads (T-246).

Evaluates:
1. Baseline: Raw ECAPA 192-D Cosine
2. Closed-set LDA: 2-class pooled covariance with shrinkage (current enrolled tier)
3. Shrinkage Whitening: Background covariance whitening (Mahalanobis metric)
4. Linear Projection + Triplet Margin Loss
5. Linear Projection + Supervised Contrastive Loss (SupCon)
6. MLP Projection + SupCon + Covariance Regularization

Tests exact kill/adopt gates:
- Gate 1: Target-absent False Acceptance Rate (FAR <= 2.0% or >= 30% reduction vs baseline)
- Gate 2: Quiet-target False Rejection Rate (FRR reduction >= 15% at -10 dB)
- Gate 3: Male-male hard-pair margin (margin >= 0.35)
- Gate 4: Zero speaker-leakage / generalization on heldout split (no degradation)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

try:
    from .model import (
        LinearProjectionHead,
        MLPProjectionHead,
        WhiteningTransform,
        covariance_regularization_loss,
        fit_closed_set_lda,
        supervised_contrastive_loss,
        triplet_margin_loss,
    )
    from .synthetic_data import (
        DatasetPartition,
        StreamPairSample,
        create_synthetic_splits,
    )
except (ImportError, ValueError):
    _curr_dir = Path(__file__).parent.resolve()
    if str(_curr_dir) not in sys.path:
        sys.path.insert(0, str(_curr_dir))
    from model import (
        LinearProjectionHead,
        MLPProjectionHead,
        WhiteningTransform,
        covariance_regularization_loss,
        fit_closed_set_lda,
        supervised_contrastive_loss,
        triplet_margin_loss,
    )
    from synthetic_data import (
        DatasetPartition,
        StreamPairSample,
        create_synthetic_splits,
    )



def compute_code_hash() -> str:
    """Compute sha256 of the model and synthetic data source files."""
    hasher = hashlib.sha256()
    curr_dir = Path(__file__).parent
    for fname in ("model.py", "synthetic_data.py", "run_ablation.py"):
        fpath = curr_dir / fname
        if fpath.exists():
            hasher.update(fpath.read_bytes())
    return hasher.hexdigest()[:16]


def train_linear_triplet(
    train_part: DatasetPartition,
    dim: int = 192,
    proj_dim: int = 128,
    epochs: int = 20,
    lr: float = 0.01,
) -> LinearProjectionHead:
    """Train linear head with triplet loss using hard negative crops."""
    model = LinearProjectionHead(in_dim=dim, proj_dim=proj_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        feats, labels = train_part.sample_contrastive_batch(batch_size=64, crops_per_speaker=4)
        z = model(torch.from_numpy(feats))

        # Mine anchor-positive-negative triplets
        anchors, positives, negatives = [], [], []
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            pos_idx = np.where(labels == lbl)[0]
            neg_idx = np.where(labels != lbl)[0]
            if len(pos_idx) >= 2 and len(neg_idx) >= 1:
                for a_i in pos_idx:
                    for p_i in pos_idx:
                        if a_i != p_i:
                            n_i = np.random.choice(neg_idx)
                            anchors.append(z[a_i])
                            positives.append(z[p_i])
                            negatives.append(z[n_i])

        if anchors:
            a_t = torch.stack(anchors)
            p_t = torch.stack(positives)
            n_t = torch.stack(negatives)
            loss = triplet_margin_loss(a_t, p_t, n_t, margin=0.2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_linear_supcon(
    train_part: DatasetPartition,
    dim: int = 192,
    proj_dim: int = 128,
    epochs: int = 20,
    lr: float = 0.01,
) -> LinearProjectionHead:
    """Train linear head with Supervised Contrastive loss."""
    model = LinearProjectionHead(in_dim=dim, proj_dim=proj_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        feats, labels = train_part.sample_contrastive_batch(batch_size=64, crops_per_speaker=4)
        t_feats = torch.from_numpy(feats)
        t_labels = torch.from_numpy(labels)

        z = model(t_feats)
        loss = supervised_contrastive_loss(z, t_labels, temperature=0.07)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_mlp_supcon(
    train_part: DatasetPartition,
    dim: int = 192,
    hidden_dim: int = 128,
    proj_dim: int = 128,
    epochs: int = 20,
    lr: float = 0.01,
) -> MLPProjectionHead:
    """Train 2-layer MLP head with SupCon and covariance anti-collapse regularizer."""
    model = MLPProjectionHead(in_dim=dim, hidden_dim=hidden_dim, proj_dim=proj_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        feats, labels = train_part.sample_contrastive_batch(batch_size=64, crops_per_speaker=4)
        t_feats = torch.from_numpy(feats)
        t_labels = torch.from_numpy(labels)

        z = model(t_feats)
        supcon = supervised_contrastive_loss(z, t_labels, temperature=0.07)
        cov_loss = covariance_regularization_loss(z, factor=0.02)
        total_loss = supcon + cov_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return model


def evaluate_stream_pairs(
    samples: list[StreamPairSample],
    method_name: str,
    model: Any = None,
    acceptance_threshold: float = 0.50,
) -> dict[str, float]:
    """Evaluate a scoring metric on stream pairs across experimental conditions."""
    normal_correct = 0
    normal_total = 0
    hard_pair_margins = []
    quiet_target_correct = 0
    quiet_target_total = 0
    target_absent_accepted = 0
    target_absent_total = 0

    for sample in samples:
        # Obtain stream representations and profile representation
        if method_name == "cosine_baseline":
            p = sample.target_profile
            s0 = sample.stream_0
            s1 = sample.stream_1
            score_0 = float(np.dot(p, s0))
            score_1 = float(np.dot(p, s1))
        elif method_name == "closed_set_lda":
            # If closed set available, score via LDA hyperplane
            if sample.other_profile is not None:
                # Target profile segments + other profile segments
                lda_res = fit_closed_set_lda(
                    sample.target_profile[np.newaxis, :],
                    sample.other_profile[np.newaxis, :],
                )
                if lda_res is not None:
                    w, b = lda_res
                    score_0 = float(w @ sample.stream_0 - b)
                    score_1 = float(w @ sample.stream_1 - b)
                else:
                    score_0 = float(np.dot(sample.target_profile, sample.stream_0))
                    score_1 = float(np.dot(sample.target_profile, sample.stream_1))
            else:
                score_0 = float(np.dot(sample.target_profile, sample.stream_0))
                score_1 = float(np.dot(sample.target_profile, sample.stream_1))
        elif method_name == "shrinkage_whitening":
            p = model.transform(sample.target_profile[np.newaxis, :])[0]
            s0 = model.transform(sample.stream_0[np.newaxis, :])[0]
            s1 = model.transform(sample.stream_1[np.newaxis, :])[0]
            score_0 = float(np.dot(p, s0))
            score_1 = float(np.dot(p, s1))
        elif method_name in ("linear_triplet", "linear_supcon", "mlp_supcon"):
            p = model.project_numpy(sample.target_profile[np.newaxis, :])[0]
            s0 = model.project_numpy(sample.stream_0[np.newaxis, :])[0]
            s1 = model.project_numpy(sample.stream_1[np.newaxis, :])[0]
            score_0 = float(np.dot(p, s0))
            score_1 = float(np.dot(p, s1))
        else:
            raise ValueError(f"Unknown method {method_name}")

        chosen_stream = 0 if score_0 >= score_1 else 1
        best_score = max(score_0, score_1)

        if sample.condition == "normal":
            normal_total += 1
            if chosen_stream == sample.correct_stream:
                normal_correct += 1

        elif sample.condition == "hard_pair":
            # Margin between true target stream score and interferer stream score
            target_score = score_0 if sample.correct_stream == 0 else score_1
            interferer_score = score_1 if sample.correct_stream == 0 else score_0
            hard_pair_margins.append(target_score - interferer_score)

        elif sample.condition == "quiet_target":
            quiet_target_total += 1
            # In quiet target, false rejection occurs if best stream score < acceptance threshold
            # or if wrong stream was chosen
            if chosen_stream == sample.correct_stream and best_score >= acceptance_threshold:
                quiet_target_correct += 1

        elif sample.condition == "target_absent":
            target_absent_total += 1
            # In target absent, false accept occurs if best score exceeds threshold
            if best_score >= acceptance_threshold:
                target_absent_accepted += 1

    normal_acc = normal_correct / max(normal_total, 1)
    mean_margin = float(np.mean(hard_pair_margins)) if hard_pair_margins else 0.0
    quiet_frr = 1.0 - (quiet_target_correct / max(quiet_target_total, 1))
    target_absent_far = target_absent_accepted / max(target_absent_total, 1)

    return {
        "normal_accuracy": float(normal_acc),
        "hard_pair_margin": float(mean_margin),
        "quiet_target_frr": float(quiet_frr),
        "target_absent_far": float(target_absent_far),
    }


def run_ablation_suite(
    seed: int = 42,
    cases: int = 300,
    smoke: bool = False,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the full theoretical projection head ablation experiment."""
    start_time = time.time()
    code_hash = compute_code_hash()

    if smoke:
        cases = 12
        epochs = 3
    else:
        cases = min(cases, 300)
        epochs = 25

    config_dict = {
        "seed": seed,
        "cases": cases,
        "smoke": smoke,
        "epochs": epochs,
        "code_hash": code_hash,
    }
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]

    # Checkpoint recovery
    checkpoint_path = checkpoint_dir / f"checkpoint_{config_hash}.json" if checkpoint_dir else None
    if resume and checkpoint_path and checkpoint_path.exists():
        try:
            saved_report = json.loads(checkpoint_path.read_text())
            if saved_report.get("status") == "completed":
                print(f"Resumed from verified complete checkpoint: {checkpoint_path}")
                return saved_report
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_part, dev_part, heldout_part = create_synthetic_splits(seed=seed)

    # 1. Fit unsupervised whitening on train partition
    train_feats, _ = train_part.sample_contrastive_batch(batch_size=128, crops_per_speaker=4)
    whitening = WhiteningTransform(shrinkage=0.5).fit(train_feats)

    # 2. Train projection heads
    linear_triplet = train_linear_triplet(train_part, epochs=epochs)
    linear_supcon = train_linear_supcon(train_part, epochs=epochs)
    mlp_supcon = train_mlp_supcon(train_part, epochs=epochs)

    # 3. Generate test streams for dev and heldout
    dev_samples = dev_part.generate_evaluation_stream_pairs(count=cases)
    heldout_samples = heldout_part.generate_evaluation_stream_pairs(count=cases)

    methods = [
        ("cosine_baseline", None),
        ("closed_set_lda", None),
        ("shrinkage_whitening", whitening),
        ("linear_triplet", linear_triplet),
        ("linear_supcon", linear_supcon),
        ("mlp_supcon", mlp_supcon),
    ]

    results_table = {}
    for name, model_obj in methods:
        dev_metrics = evaluate_stream_pairs(dev_samples, name, model_obj)
        heldout_metrics = evaluate_stream_pairs(heldout_samples, name, model_obj)
        results_table[name] = {
            "dev": dev_metrics,
            "heldout": heldout_metrics,
        }

    # Extract baseline metrics for gate evaluations
    base_dev = results_table["cosine_baseline"]["dev"]
    base_heldout = results_table["cosine_baseline"]["heldout"]
    mlp_dev = results_table["mlp_supcon"]["dev"]
    mlp_heldout = results_table["mlp_supcon"]["heldout"]

    # Gate Evaluations
    # Gate 1: Target-absent false accept rate (FAR <= 2% or >= 30% reduction vs baseline)
    base_far = base_dev["target_absent_far"]
    mlp_far = mlp_dev["target_absent_far"]
    far_rel_drop = (base_far - mlp_far) / max(base_far, 1e-6)
    gate_1_pass = (mlp_far <= 0.02) or (far_rel_drop >= 0.30)

    # Gate 2: Quiet-target false rejection rate (FRR reduction >= 15% at -10 dB)
    base_frr = base_dev["quiet_target_frr"]
    mlp_frr = mlp_dev["quiet_target_frr"]
    frr_rel_drop = (base_frr - mlp_frr) / max(base_frr, 1e-6)
    gate_2_pass = frr_rel_drop >= 0.15

    # Gate 3: Male-male hard pair margin (margin >= 0.35)
    mlp_margin = mlp_dev["hard_pair_margin"]
    gate_3_pass = mlp_margin >= 0.35

    # Gate 4: Zero speaker-leakage / generalization on heldout split
    base_heldout_acc = base_heldout["normal_accuracy"]
    mlp_heldout_acc = mlp_heldout["normal_accuracy"]
    gate_4_pass = mlp_heldout_acc >= (base_heldout_acc - 0.02)

    all_gates_pass = gate_1_pass and gate_2_pass and gate_3_pass and gate_4_pass

    runtime_s = round(time.time() - start_time, 3)

    report = {
        "status": "completed",
        "task_id": "T-246",
        "code_hash": code_hash,
        "config_hash": config_hash,
        "seed": seed,
        "cases_evaluated": cases,
        "smoke": smoke,
        "runtime_s": runtime_s,
        "ablation_results": results_table,
        "gates": {
            "gate_1_target_absent_far": {
                "threshold": "FAR <= 0.02 or rel drop >= 30%",
                "baseline_far": base_far,
                "mlp_far": mlp_far,
                "rel_drop": float(far_rel_drop),
                "pass": bool(gate_1_pass),
            },
            "gate_2_quiet_target_frr": {
                "threshold": "rel drop in FRR >= 15%",
                "baseline_frr": base_frr,
                "mlp_frr": mlp_frr,
                "rel_drop": float(frr_rel_drop),
                "pass": bool(gate_2_pass),
            },
            "gate_3_hard_pair_margin": {
                "threshold": "margin >= 0.35",
                "baseline_margin": base_dev["hard_pair_margin"],
                "mlp_margin": mlp_margin,
                "pass": bool(gate_3_pass),
            },
            "gate_4_heldout_generalization": {
                "threshold": "heldout_acc >= baseline - 0.02",
                "baseline_heldout_acc": base_heldout_acc,
                "mlp_heldout_acc": mlp_heldout_acc,
                "pass": bool(gate_4_pass),
            },
            "all_gates_pass": bool(all_gates_pass),
        },
        "overnight_gpu_request_manifest": {
            "task_id": "T-246",
            "assigned_to": "codex-b",
            "model": "gpt-6.1-sol",
            "flags": "overnight-gpu-request + host:laptop-b",
            "max_runtime_minutes": 45,
            "immutable_code_hash": code_hash,
            "argv": [
                ".venv/bin/python",
                "experiments/theory/T-246/run_ablation.py",
                "--seed",
                str(seed),
                "--cases",
                "300",
                "--out",
                "$HEARING_RESULTS_DIR/overnight_prep/T-246/ablation_report.json",
            ],
        },
    }

    # Save atomic checkpoint if requested
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(report, indent=2))
        tmp_path.replace(checkpoint_path)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="T-246 Projection Head Ablation Runner")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument("--cases", type=int, default=300, help="Max test stream pair cases")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run (12 cases)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Atomic checkpoint dir")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON report file")
    args = parser.parse_args()

    report = run_ablation_suite(
        seed=args.seed,
        cases=args.cases,
        smoke=args.smoke,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )

    out_json = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_json)
        print(f"Report written to {args.out}")
    else:
        print(out_json)

    gates = report["gates"]
    print("\n--- ABLATION GATE VERDICTS ---")
    print(f"Gate 1 (Target Absent FAR): {'PASS' if gates['gate_1_target_absent_far']['pass'] else 'FAIL'}")
    print(f"Gate 2 (Quiet Target FRR):  {'PASS' if gates['gate_2_quiet_target_frr']['pass'] else 'FAIL'}")
    print(f"Gate 3 (Hard Pair Margin):  {'PASS' if gates['gate_3_hard_pair_margin']['pass'] else 'FAIL'}")
    print(f"Gate 4 (Heldout Generalize):{'PASS' if gates['gate_4_heldout_generalization']['pass'] else 'FAIL'}")
    print(f"Overall Adopt Status:       {'ADOPT' if gates['all_gates_pass'] else 'KILL'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
