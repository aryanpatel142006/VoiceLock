"""T-247 separator fine-tune planning prototype (stdlib only, no training, no weights, no GPU).

Subcommands (all offline, deterministic):
  estimate  theoretical memory table from checkpoint bytes (arithmetic, not a measurement)
  split     source-group-disjoint dev/heldout split of a group list (hash-based, seeded)
  gate      evaluate success/rollback gates on a metrics JSON (frozen baseline vs candidate)
  manifest  emit pilot manifest with config hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

# Checkpoint size is documented in docs/MODELS.md (639 MB fp32 weights) -> params ~ bytes / 4.
CKPT_BYTES = 639e6
PARAMS = CKPT_BYTES / 4

ARMS = {
    "selector_only": {
        "trainable_frac": 0.002,
        "note": "frozen separator, tiny selector head",
    },
    "adapter": {
        "trainable_frac": 0.03,
        "note": "LoRA/bottleneck adapters on the top 6 blocks",
    },
    "top_layers": {
        "trainable_frac": 0.25,
        "note": "unfreeze top 6 of 24 MossFormer blocks + decoder",
    },
    "full": {"trainable_frac": 1.0, "note": "full fine-tune"},
}

CONFIG = {
    "seed": 247,
    "arm": "adapter",
    "lr": 1e-5,
    "optimizer": "AdamW",
    "max_minutes": 45,
    "max_steps": 3000,
    "batch": 8,
    "clip_s": 4.0,
    "loss": {
        "si_sdr_pit": 1.0,
        "target_pick_ce": 0.3,
        "absent_energy": 0.2,
        "kd_to_frozen": 0.5,
    },
    "target_absent_frac": 0.25,
    "deploy": False,
}

GATES = {
    # candidate must beat frozen baseline on dev AND hold on frozen heldout
    "min_si_sdri_gain_db": 0.5,
    "max_public_regress_db": 0.3,  # catastrophic-forgetting guard on public speakers
    "max_weak_target_erasure_db": 1.0,  # drop in quietest-target-tercile SI-SDRi
    "max_absent_leak_db": 3.0,  # allowed rise in output energy when target absent (dB)
    "max_wrong_speaker_pct": 25.0,  # T-141 observed range 6-25%
}


def chash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def estimate(arm: str, batch: int = 8, act_gb_per_item: float = 1.5) -> dict:
    """Theoretical VRAM (GB): fp32 weights + grads/Adam for trainable + activations. Not measured."""
    t = PARAMS * ARMS[arm]["trainable_frac"]
    weights = PARAMS * 4
    opt = t * (4 + 8)  # grad + Adam m,v
    act = batch * act_gb_per_item * 1e9 * (0.4 if arm != "full" else 1.0)
    total = weights + opt + act
    return {
        "arm": arm,
        "trainable_params_m": round(t / 1e6, 1),
        "weights_gb": round(weights / 1e9, 2),
        "optimizer_gb": round(opt / 1e9, 2),
        "activations_gb_assumed": round(act / 1e9, 1),
        "total_gb": round(total / 1e9, 1),
        "fits_a100_40gb": total < 40e9,
        "theoretical": True,
    }


def split(groups: list[str], seed: int, heldout_pct: int = 20) -> dict:
    """Whole source groups go to one side; assignment depends only on (seed, group id)."""
    out: dict[str, list[str]] = {"dev": [], "heldout": []}
    for g in sorted(set(groups)):
        h = int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), 16) % 100
        out["heldout" if h < heldout_pct else "dev"].append(g)
    return out


def gate(m: dict) -> dict:
    """m: {baseline:{dev,heldout_team,heldout_public,weak_tercile,absent_db,wrong_pct}, candidate:{...}}"""
    b, c = m["baseline"], m["candidate"]
    checks = {
        "dev_gain": c["dev_si_sdri"] - b["dev_si_sdri"] >= GATES["min_si_sdri_gain_db"],
        "heldout_team_gain": c["heldout_team_si_sdri"] - b["heldout_team_si_sdri"]
        >= GATES["min_si_sdri_gain_db"],
        "public_no_regress": b["heldout_public_si_sdri"] - c["heldout_public_si_sdri"]
        <= GATES["max_public_regress_db"],
        "weak_target_kept": b["weak_tercile_si_sdri"] - c["weak_tercile_si_sdri"]
        <= GATES["max_weak_target_erasure_db"],
        "absent_leak": c["absent_out_db"] - b["absent_out_db"]
        <= GATES["max_absent_leak_db"],
        "wrong_speaker": c["wrong_speaker_pct"] <= GATES["max_wrong_speaker_pct"],
    }
    hard_fail = not (
        checks["public_no_regress"]
        and checks["weak_target_kept"]
        and checks["wrong_speaker"]
    )
    verdict = (
        "success"
        if all(checks.values())
        else ("rollback" if hard_fail else "inconclusive")
    )
    return {"checks": checks, "verdict": verdict}


def manifest(cfg: dict) -> dict:
    return {
        "task": "T-247",
        "config": cfg,
        "config_hash": chash(cfg),
        "gates": GATES,
        "gates_hash": chash(GATES),
        "estimate": estimate(cfg["arm"], cfg["batch"]),
        "argv_hint": ["<pilot trainer: not implemented here>", "--config", "<path>"],
        "launched": False,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    sp = sub.add_parser("split")
    sp.add_argument("groups_file", help="text file, one source-group id per line")
    sp.add_argument("--seed", type=int, default=CONFIG["seed"])
    g = sub.add_parser("gate")
    g.add_argument("metrics_json")
    sub.add_parser("manifest")
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        res = [estimate(k) for k in ARMS]
    elif a.cmd == "split":
        with open(a.groups_file) as f:
            res = split([x.strip() for x in f if x.strip()], a.seed)
    elif a.cmd == "gate":
        with open(a.metrics_json) as f:
            res = gate(json.load(f))
    else:
        res = manifest(CONFIG)
    json.dump(res, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
