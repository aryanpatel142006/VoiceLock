"""T-307 GPU run: AP-loss vs CE head trained on many LibriSpeech speakers, scored on unseen speakers.

The laptop run (``eval/heads/T-307/ap_head.py real``) trains on only 30 test-clean speakers, and both
heads then lose to raw ECAPA on held-out speakers. This script removes that bottleneck: ECAPA features
for ``--train-splits`` (default train-clean-100, 251 speakers; add train-clean-360 for 921), heads
trained for ``--steps`` on the GPU, scored on all 40 test-clean speakers (never trained on), their
male/male hardest pairs, and the T-306 team set if ``--team`` exists. ~30 min on one A100
(feature extraction dominates; features are cached in ``--out`` so a re-run only trains).

    python experiments/gpu/T-307/train_ap_head.py --librispeech /freespace/.../LibriSpeech --out runs/T-307
    python experiments/gpu/T-307/train_ap_head.py --dry-run   # tiny local smoke test, no GPU needed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("ap_head_t307", ROOT / "eval" / "heads" / "T-307" / "ap_head.py")
ap = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ap
_spec.loader.exec_module(ap)


def _team_audio_present(root: Path) -> bool:
    """The manifest is in git, the wavs are not: rsync <main>/eval/augmented/T-306/ to the GPU box to score them."""
    if not (root / "manifest.json").exists():
        return False
    files = json.loads((root / "manifest.json").read_text())["files"]
    return all((root / f["path"]).exists() for f in files)


def evaluate(train: dict, test: dict, team: dict | None, cfg, seeds: int) -> dict:
    sex_of = dict(zip(test["y"].tolist(), test["sex"].tolist(), strict=True))
    enroll = np.zeros(len(test["y"]), bool)
    for s in np.unique(test["y"]):
        enroll[np.flatnonzero(test["y"] == s)[:10]] = True
    hx, hy = test["x"], test["y"]
    partner = ap.hardest_partners(hx[enroll], hy[enroll], sex_of)
    sets = {
        "librispeech_test_clean": (hx[enroll], hy[enroll], hx[~enroll], hy[~enroll], partner),
        "librispeech_male_male": (hx[enroll], hy[enroll], hx[~enroll], hy[~enroll], {s: p for s, p in partner.items() if sex_of[s] == "M"}),
    }
    if team is not None:
        ids = {"aryan": -1, "joseph": -2}
        ty = np.array([ids[s] for s in team["speaker"]])
        prof = np.char.startswith(team["source"].astype(str), "profile")
        clean = team["aug"] == "clean"
        sets["team_all_aug"] = (team["x"][prof & clean], ty[prof & clean], team["x"][~prof], ty[~prof], {-1: -2, -2: -1})
        sets["team_clean"] = (team["x"][prof & clean], ty[prof & clean], team["x"][~prof & clean], ty[~prof & clean], {-1: -2, -2: -1})
    out = {}
    for seed in range(seeds):
        out[f"seed{seed}"] = ap.compare(train["x"], train["y"], sets, ap.TrainConfig(**{**cfg.__dict__, "seed": seed}))
    out["summary"] = ap.summarize(out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--librispeech", type=Path, help="LibriSpeech root holding SPEAKERS.TXT and the split dirs")
    p.add_argument("--train-splits", default="train-clean-100")
    p.add_argument("--team", type=Path, default=ROOT / "eval" / "augmented" / "T-306")
    p.add_argument("--out", type=Path, default=ap._results_dir() / "T-307")
    p.add_argument("--utts", type=int, default=60, help="utterances per training speaker")
    p.add_argument("--crop-s", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--episode-speakers", type=int, default=128)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="synthetic features, 200 steps, CPU")
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        x, y, _ = ap.make_synthetic_speakers(60, 10, seed=0)
        tx, ty, _ = ap.make_synthetic_speakers(10, 20, seed=1, label_offset=1000)
        train, test, team = {"x": x, "y": y}, {"x": tx, "y": ty, "sex": np.array(["M"] * len(ty))}, None
        cfg = ap.TrainConfig(steps=200, speakers_per_episode=32)
        device = "cpu"
    else:
        from hearing import device as hdev

        device = hdev.pick()
        if a.librispeech is None:
            p.error("--librispeech is required (or --dry-run)")
        if "test-clean" in a.train_splits.split(","):
            print("WARNING: test-clean is also the evaluation split; numbers are not open-set", flush=True)
        parts = [
            ap.librispeech_features(a.librispeech, a.crop_s, a.utts, a.out / f"feat_{s}_c{a.crop_s}.npz", split=s)
            for s in a.train_splits.split(",")
        ]
        train = {k: np.concatenate([q[k] for q in parts]) for k in ("x", "y", "sex")}
        test = ap.librispeech_features(a.librispeech, a.crop_s, 40, a.out / f"feat_test-clean_c{a.crop_s}.npz")
        team = ap.team_features(a.team, a.crop_s, a.out / f"team_c{a.crop_s}.npz") if _team_audio_present(a.team) else None
        cfg = ap.TrainConfig(steps=a.steps, speakers_per_episode=a.episode_speakers, batch_size=512, device=device)
    print(f"features ready in {time.perf_counter() - t0:.0f} s: train {len(train['x'])} crops / {len(np.unique(train['y']))} speakers", flush=True)
    res = evaluate(train, test, team, cfg, a.seeds)
    res["meta"] = {
        "host": platform.node(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_splits": a.train_splits,
        "steps": cfg.steps,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    path = a.out / f"ap_head_gpu_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=1, default=str))
    json.dump(res["summary"], sys.stdout, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
