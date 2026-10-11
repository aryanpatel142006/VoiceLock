"""T-305: reproduce the projection-head margin claims on the real Aryan/Joseph recordings.

Claim under test (docs/research/theory/projection_head_theory.md, T-294): "0.5 s-window speaker margin
8x (0.15 -> 1.19) with 97.5% held-out accuracy". This script measures, on real team audio only:

1. stock ECAPA vs the T-273 projection head (contrastive, 192->64->32), trained with the pilot recipe
   on the profile cuts, evaluated on the disjoint fixture cuts, at 0.5 / 1.0 / 2.0 s windows;
   margin = mean intra-speaker cosine - mean cross-speaker cosine, with 95% bootstrap CIs over
   windows, plus nearest-prototype accuracy with Wilson CIs;
2. the same per fixture take (time distance from the training cut within the recording);
3. the session check: every labelled clip of a speaker comes from ONE voice memo (session == speaker),
   so the cross-session margin cannot be measured; instead it measures how far the recording itself
   (room/mic/codec, no speech) identifies the "speaker": non-speech windows are scored by the
   speech-trained prototypes and by a noise-only nearest-centroid classifier.

Run: .venv/bin/python eval/evidence/T-305/reproduce_margins.py [--seeds 5] [--out results.json]
Writes the trained seed-0 head to $HEARING_MODELS_DIR/team_projection_head.pt (gitignored).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from eval.projection_head_pilot import slice_audio, train_projection_head

SR = 16000
WINDOWS = (0.5, 1.0, 2.0)
N_BOOT = 2000
# Source offsets inside each speaker's single voice memo (team/*/meta.json, joseph processing-report.json).
RECORDINGS = {
    "aryan": {
        "session": "aryan-voicememo-2026-10-09 (Mac Voice Memos)",
        "profile": ("aryan/profile-01.wav", 0.0, 50.0),
        "fixtures": [("aryan/fixture-01.wav", 50.0, 108.7)],
    },
    "joseph": {
        "session": "joseph-voicememo-2026-10-09 (Q-372 pack, joseph-original.m4a)",
        "profile": ("joseph/profile-01.wav", 8.3, 24.0),
        "fixtures": [
            ("joseph/fixture-01.wav", 29.85, 78.0),
            ("joseph/fixture-02-namecall.wav", 85.9, 113.7),
        ],
    },
}


def team_root() -> Path:
    env = os.environ.get("HEARING_FIXTURES_DIR")
    cands = [Path(env) / "team"] if env else []
    cands.append(ROOT / "eval/fixtures/generated/team")
    for c in cands:
        if (c / "aryan/profile-01.wav").is_file():
            return c
    raise SystemExit(f"team audio not found in {cands}; set HEARING_FIXTURES_DIR")


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path}: expected {SR} Hz, got {sr}")
    return x.mean(axis=1)


def voiced_windows(x: np.ndarray, win_s: float, hop_s: float) -> tuple[list[np.ndarray], list[float]]:
    """Pilot gate (peak >= 0.02, RMS >= 0.005); returns windows and their start times (s)."""
    n, hop = int(win_s * SR), int(hop_s * SR)
    out, starts = [], []
    for s in range(0, len(x) - n + 1, hop):
        c = x[s : s + n]
        if float(np.max(np.abs(c))) >= 0.02 and float(np.sqrt(np.mean(c**2))) >= 0.005:
            out.append(c)
            starts.append(s / SR)
    return out, starts


def silent_windows(x: np.ndarray, win_s: float) -> list[np.ndarray]:
    """Non-overlapping windows with no 10 ms frame above 2x the recording's 10th-percentile frame RMS."""
    f = 160
    frames = x[: len(x) // f * f].reshape(-1, f)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    thr = 2.0 * max(float(np.percentile(rms, 10)), 1e-6)
    per = int(win_s * SR) // f
    out = []
    for i in range(0, len(rms) - per + 1, per):
        if float(rms[i : i + per].max()) <= thr:
            out.append(x[i * f : (i + per) * f])
    return out


class Embedder:
    def __init__(self) -> None:
        from hearing.tse.enrolled import _load_encoder, _unit_embedding

        self._backend = _load_encoder()
        self._unit = _unit_embedding
        self.device = str(getattr(self._backend, "device", "?"))

    def __call__(self, chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, 192), dtype=np.float32)
        return np.array([self._unit(self._backend, c) for c in chunks], dtype=np.float32)


def project(head, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0, 32), dtype=np.float32)
    with torch.no_grad():
        return head(torch.from_numpy(x)).numpy()


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def margins(a: np.ndarray, j: np.ndarray, ia=None, ij=None) -> dict[str, float]:
    """Intra/cross cosines. `margin` excludes self-pairs; `margin_pilot` keeps them (T-273 convention).

    With bootstrap indices `ia`/`ij`, pairs of the same source window are dropped from the intra mean
    (a resampled duplicate would add a cosine of exactly 1 and bias the CI upward).
    """
    def intra_mean(x: np.ndarray, idx) -> float:
        idx = np.arange(len(x)) if idx is None else idx
        m = x[idx] @ x[idx].T
        keep = idx[:, None] != idx[None, :]
        return float(m[keep].mean()) if keep.any() else float("nan")

    xa = a if ia is None else a[ia]
    xj = j if ij is None else j[ij]
    intra = (intra_mean(a, ia) + intra_mean(j, ij)) / 2
    intra_pilot = (float((xa @ xa.T).mean()) + float((xj @ xj.T).mean())) / 2
    cross = float((xa @ xj.T).mean())
    return {"intra": intra, "cross": cross, "margin": intra - cross, "margin_pilot": intra_pilot - cross}


def boot_ci(a: np.ndarray, j: np.ndarray, rng: np.random.Generator, key: str = "margin") -> list[float]:
    vals = []
    for _ in range(N_BOOT):
        ia = rng.integers(0, len(a), len(a))
        ij = rng.integers(0, len(j), len(j))
        vals.append(margins(a, j, ia, ij)[key])
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def accuracy(a: np.ndarray, j: np.ndarray, pa: np.ndarray, pj: np.ndarray) -> dict:
    """Nearest-prototype accuracy; a rows should go to pa, j rows to pj."""
    ok_a = int(np.sum(a @ pa > a @ pj)) if len(a) else 0
    ok_j = int(np.sum(j @ pj > j @ pa)) if len(j) else 0
    n = len(a) + len(j)
    k = ok_a + ok_j
    return {"correct": k, "n": n, "acc": k / n if n else float("nan"), "ci95": wilson(k, n),
            "aryan_correct": ok_a, "aryan_n": len(a), "joseph_correct": ok_j, "joseph_n": len(j)}


def summarise(a, j, pa, pj, rng) -> dict:
    m = margins(a, j)
    m["margin_ci95"] = boot_ci(a, j, rng)
    m["margin_pilot_ci95"] = boot_ci(a, j, rng, "margin_pilot")
    m["n_aryan"], m["n_joseph"] = len(a), len(j)
    m["accuracy"] = accuracy(a, j, pa, pj)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5, help="head training seeds (default 5)")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("margins_raw.json"))
    ap.add_argument("--save-head", type=Path, default=None,
                    help="default: $HEARING_MODELS_DIR/team_projection_head.pt")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    root = team_root()
    emb = Embedder()
    rng = np.random.default_rng(305)

    audio = {}
    for spk, rec in RECORDINGS.items():
        for rel, _, _ in [rec["profile"], *rec["fixtures"]]:
            audio[rel] = load(root / rel)

    # Training data: the T-273 pilot recipe (0.5 s voiced crops, 0.15 s step, first 120 per speaker).
    train = {spk: emb(slice_audio(audio[rec["profile"][0]], chunk_s=0.5, step_s=0.15)[:120])
             for spk, rec in RECORDINGS.items()}
    heads = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        head, meta = train_projection_head(train["aryan"], train["joseph"], loss_type="contrastive",
                                           epochs=120, lr=1e-3, device="cpu")
        head.eval()
        heads.append((head, meta))
    save = args.save_head or Path(os.environ.get("HEARING_MODELS_DIR", ROOT / "models")) / "team_projection_head.pt"
    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads[0][0].state_dict(), save)

    proto = {"stock": {s: unit(train[s].mean(0)) for s in train}}
    for i, (h, _) in enumerate(heads):
        proto[f"head{i}"] = {s: unit(project(h, train[s]).mean(0)) for s in train}

    def spaces(x: np.ndarray) -> dict[str, np.ndarray]:
        d = {"stock": x}
        for i, (h, _) in enumerate(heads):
            d[f"head{i}"] = project(h, x)
        return d

    results: dict = {
        "task": "T-305",
        "host": platform.node(), "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(), "torch": torch.__version__, "ecapa_device": emb.device,
        "team_root": str(root), "head_saved_to": str(save), "seeds": args.seeds, "n_boot": N_BOOT,
        "recordings": RECORDINGS,
        "train": {s: {"n_windows": len(train[s]), "window_s": 0.5, "step_s": 0.15} for s in train},
        "head_training": [m for _, m in heads],
        "windows": {}, "per_take": {}, "session_probe": {},
    }

    for w in WINDOWS:
        test = {}
        starts = {}
        for spk, rec in RECORDINGS.items():
            chunks, st = [], []
            for rel, off, _ in rec["fixtures"]:
                c, s = voiced_windows(audio[rel], w, w)  # non-overlapping windows: independent for the CI
                chunks += c
                st += [(rel, off + t) for t in s]
            test[spk] = emb(chunks)
            starts[spk] = st
        sp_a, sp_j = spaces(test["aryan"]), spaces(test["joseph"])
        row = {}
        for name in sp_a:
            row[name] = summarise(sp_a[name], sp_j[name], proto[name]["aryan"], proto[name]["joseph"], rng)
        head_m = [row[f"head{i}"]["margin"] for i in range(args.seeds)]
        head_acc = [row[f"head{i}"]["accuracy"]["acc"] for i in range(args.seeds)]
        row["head_seed_spread"] = {"margin_min": min(head_m), "margin_max": max(head_m),
                                   "margin_mean": float(np.mean(head_m)),
                                   "acc_min": min(head_acc), "acc_max": max(head_acc)}
        row["ratio_head0_over_stock"] = row["head0"]["margin"] / row["stock"]["margin"]
        # Bootstrap the ratio with paired resampling.
        ratios = []
        for _ in range(N_BOOT):
            ia = rng.integers(0, len(test["aryan"]), len(test["aryan"]))
            ij = rng.integers(0, len(test["joseph"]), len(test["joseph"]))
            ms = margins(sp_a["stock"], sp_j["stock"], ia, ij)["margin"]
            mh = margins(sp_a["head0"], sp_j["head0"], ia, ij)["margin"]
            ratios.append(mh / ms if ms > 0 else float("nan"))
        row["ratio_ci95"] = [float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))]
        results["windows"][f"{w}s"] = row

        # Per fixture take vs the other speaker's full test set.
        per = {}
        for spk, rec in RECORDINGS.items():
            other = "joseph" if spk == "aryan" else "aryan"
            for rel, off, end in rec["fixtures"]:
                idx = [i for i, (r, _) in enumerate(starts[spk]) if r == rel]
                gap = off - rec["profile"][2]
                entry = {"source_span_s": [off, end], "gap_after_profile_s": gap, "n": len(idx)}
                for name in ("stock", "head0"):
                    mine = (sp_a if spk == "aryan" else sp_j)[name][idx]
                    oth = (sp_j if spk == "aryan" else sp_a)[name]
                    a_, j_ = (mine, oth) if spk == "aryan" else (oth, mine)
                    mm = margins(a_, j_)
                    pm, po = proto[name][spk], proto[name][other]
                    k = int(np.sum(mine @ pm > mine @ po))
                    entry[name] = {"margin": mm["margin"], "acc": k / len(idx) if idx else float("nan"),
                                   "correct": k, "acc_ci95": wilson(k, len(idx))}
                per[rel] = entry
        results["per_take"][f"{w}s"] = per

    # Session probe: non-speech windows from every clip (room / mic / codec only).
    for w in (0.5, 1.0):
        noise = {spk: {} for spk in RECORDINGS}
        for spk, rec in RECORDINGS.items():
            for rel, _, _ in [rec["profile"], *rec["fixtures"]]:
                noise[spk][rel] = emb(silent_windows(audio[rel], w))
        na = np.concatenate(list(noise["aryan"].values()))
        nj = np.concatenate(list(noise["joseph"].values()))
        probe = {"n_aryan": len(na), "n_joseph": len(nj),
                 "n_per_clip": {rel: len(v) for spk in noise for rel, v in noise[spk].items()}}
        if len(na) and len(nj):
            sa, sj = spaces(na), spaces(nj)
            probe["speech_prototypes"] = {
                name: accuracy(sa[name], sj[name], proto[name]["aryan"], proto[name]["joseph"])
                for name in ("stock", "head0")
            }
            # Noise-only classifier: centroids from the profile clips' silences, tested on the fixture clips'.
            tr_a = noise["aryan"][RECORDINGS["aryan"]["profile"][0]]
            tr_j = noise["joseph"][RECORDINGS["joseph"]["profile"][0]]
            te_a = np.concatenate([noise["aryan"][r] for r, _, _ in RECORDINGS["aryan"]["fixtures"]])
            te_j = np.concatenate([noise["joseph"][r] for r, _, _ in RECORDINGS["joseph"]["fixtures"]])
            if len(tr_a) and len(tr_j) and (len(te_a) or len(te_j)):
                probe["noise_only_centroid"] = {
                    "n_train": [len(tr_a), len(tr_j)],
                    "stock": accuracy(te_a, te_j, unit(tr_a.mean(0)), unit(tr_j.mean(0))),
                }
            probe["noise_margin"] = {name: margins(sa[name], sj[name]) for name in ("stock", "head0")}
        results["session_probe"][f"{w}s"] = probe

    results["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=float) + "\n")
    for w, row in results["windows"].items():
        s, h = row["stock"], row["head0"]
        print(f"{w}: N={s['n_aryan']}+{s['n_joseph']} stock margin {s['margin']:.3f} {s['margin_ci95']} "
              f"acc {s['accuracy']['acc']:.3f} | head margin {h['margin']:.3f} {h['margin_ci95']} "
              f"acc {h['accuracy']['acc']:.3f} | ratio {row['ratio_head0_over_stock']:.2f} {row['ratio_ci95']}")
    print(f"wrote {args.out} in {results['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
