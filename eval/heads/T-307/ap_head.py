"""Angular prototypical (AP) loss head vs a cross-entropy head on frozen ECAPA features (T-307).

A 192 -> 64 -> 32 MLP maps frozen ECAPA embeddings into a unit-norm 32-d space. Two heads with the
same architecture are trained on the same features:

* ``ap``: angular prototypical loss (Chung et al., Interspeech 2020, "In defence of metric learning
  for speaker recognition"). Each episode samples N speakers x M crops; crop 0 of each speaker is the
  query, the mean of crops 1..M-1 (in the 32-d space) is its prototype, the logits are
  ``w * cos(query, prototype) + b`` (learnable ``w > 0``, ``b``) and the loss is the cross entropy of
  the true prototype, i.e. ``-log softmax`` over the cosine to every prototype in the episode.
* ``ce``: the same MLP plus a linear softmax classifier over the training speakers; the classifier is
  discarded and the normalised 32-d output is the embedding.

Evaluation is open-set like the live enrolled tier: prototypes come from enrollment crops of speakers
the head never trained on (or, for the team set, from the clean profile recordings) and each query
crop is scored by cosine against every prototype.

* ``margin``: cos(query, own prototype) - max cos(query, other prototype), averaged over queries.
* ``flip_rate``: fraction of queries whose margin is < 0, i.e. the enrolled stage, choosing the
  stream most similar to the target profile, would pick the other speaker for that crop.
* ``hard_*``: the same two numbers restricted to each speaker's hardest partner (the other speaker
  of the same sex whose raw-ECAPA prototype is closest; for the team set the other team member).

Run: ``python eval/heads/T-307/ap_head.py synthetic`` (no weights) or ``... real`` (ECAPA weights,
LibriSpeech test-clean and the T-306 augmented team set). Results JSON goes to ``$HEARING_RESULTS_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

INPUT_DIM = 192
HIDDEN_DIM = 64
OUTPUT_DIM = 32


class MLPHead(nn.Module):
    """192 -> 64 -> 32 MLP; ``forward`` returns unit-norm 32-d embeddings, ``raw`` the pre-norm ones."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, output_dim: int = OUTPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def raw(self, x: Tensor) -> Tensor:
        return self.net(x)

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)


class AngularPrototypicalLoss(nn.Module):
    """AP loss on an episode shaped (N speakers, M >= 2 crops, D); crop 0 is the query."""

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))

    def logits(self, episode: Tensor) -> Tensor:
        if episode.ndim != 3 or episode.shape[0] < 2 or episode.shape[1] < 2:
            raise ValueError("episode must be (N >= 2 speakers, M >= 2 crops, D)")
        query = F.normalize(episode[:, 0], dim=-1)
        prototypes = F.normalize(episode[:, 1:].mean(dim=1), dim=-1)
        cosine = query @ prototypes.T
        return self.w.clamp_min(1e-6) * cosine + self.b

    def forward(self, episode: Tensor) -> Tensor:
        logits = self.logits(episode)
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


@dataclass
class TrainConfig:
    steps: int = 1500
    speakers_per_episode: int = 32
    crops_per_speaker: int = 4
    batch_size: int = 128  # CE head: crops per step
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"  # training device; the returned head is always on the CPU


def _episode_indices(by_speaker: list[np.ndarray], n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    eligible = [i for i, idx in enumerate(by_speaker) if len(idx) >= m]
    if len(eligible) < 2:
        raise ValueError(f"need >= 2 speakers with >= {m} crops each")
    chosen = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    return np.stack([rng.choice(by_speaker[s], size=m, replace=False) for s in chosen])


def _group(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == s) for s in np.unique(labels)]


def train_ap_head(features: Tensor, labels: Tensor, cfg: TrainConfig | None = None) -> MLPHead:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    head, criterion = MLPHead(features.shape[1]).to(cfg.device), AngularPrototypicalLoss().to(cfg.device)
    features = features.to(cfg.device)
    opt = torch.optim.Adam([*head.parameters(), *criterion.parameters()], lr=cfg.lr, weight_decay=cfg.weight_decay)
    by_speaker = _group(labels.numpy())
    head.train()
    for _ in range(cfg.steps):
        idx = torch.from_numpy(_episode_indices(by_speaker, cfg.speakers_per_episode, cfg.crops_per_speaker, rng)).to(cfg.device)
        n, m = idx.shape
        emb = head(features[idx.reshape(-1)]).reshape(n, m, -1)
        loss = criterion(emb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.cpu().eval()


def train_ce_head(features: Tensor, labels: Tensor, cfg: TrainConfig | None = None) -> MLPHead:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    classes, y = torch.unique(labels, return_inverse=True)
    head = MLPHead(features.shape[1]).to(cfg.device)
    classifier = nn.Linear(head.net[-1].out_features, len(classes)).to(cfg.device)
    features, y = features.to(cfg.device), y.to(cfg.device)
    opt = torch.optim.Adam([*head.parameters(), *classifier.parameters()], lr=cfg.lr, weight_decay=cfg.weight_decay)
    head.train()
    for _ in range(cfg.steps):
        idx = torch.from_numpy(rng.choice(len(features), size=min(cfg.batch_size, len(features)), replace=False)).to(cfg.device)
        loss = F.cross_entropy(classifier(head.raw(features[idx])), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.cpu().eval()


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def evaluate_prototypes(
    enroll: np.ndarray,
    enroll_labels: np.ndarray,
    queries: np.ndarray,
    query_labels: np.ndarray,
    hard_partner: dict | None = None,
) -> dict:
    """Margin and flip rate of queries against per-speaker enrollment prototypes (all in one space)."""
    speakers = np.unique(enroll_labels)
    if len(speakers) < 2:
        raise ValueError("need prototypes for >= 2 speakers")
    protos = _unit(np.stack([_unit(enroll[enroll_labels == s]).mean(axis=0) for s in speakers]))
    col = {s: i for i, s in enumerate(speakers)}
    keep = np.isin(query_labels, speakers)
    q, ql = _unit(queries[keep]), query_labels[keep]
    cos = q @ protos.T
    own_idx = np.array([col[s] for s in ql])
    own = cos[np.arange(len(q)), own_idx]
    others = cos.copy()
    others[np.arange(len(q)), own_idx] = -np.inf
    margin = own - others.max(axis=1)
    out = {
        "n_queries": len(q),
        "n_speakers": len(speakers),
        "margin": float(margin.mean()),
        "flip_rate": float((margin < 0).mean()),
        "own_cos": float(own.mean()),
    }
    if hard_partner:
        partner = np.array([col.get(hard_partner.get(s), -1) for s in ql])
        ok = partner >= 0
        hard = own[ok] - cos[np.flatnonzero(ok), partner[ok]]
        out |= {"hard_n": int(ok.sum()), "hard_margin": float(hard.mean()), "hard_flip_rate": float((hard < 0).mean())}
    return out


def hardest_partners(features: np.ndarray, labels: np.ndarray, groups: dict | None = None) -> dict:
    """Each speaker's closest other speaker by raw prototype cosine (within the same group if given)."""
    speakers = np.unique(labels)
    protos = _unit(np.stack([_unit(features[labels == s]).mean(axis=0) for s in speakers]))
    sim = protos @ protos.T
    np.fill_diagonal(sim, -np.inf)
    if groups:
        g = np.array([groups.get(s) for s in speakers])
        sim[g[:, None] != g[None, :]] = -np.inf
    return {s: speakers[int(np.argmax(sim[i]))] for i, s in enumerate(speakers) if np.isfinite(sim[i]).any()}


def embed(head: MLPHead | None, x: np.ndarray) -> np.ndarray:
    if head is None:
        return _unit(x)
    with torch.no_grad():
        return head(torch.from_numpy(np.asarray(x, np.float32))).numpy()


# ------------------------------------------------------------------ synthetic speakers


def make_synthetic_speakers(
    n_pairs: int,
    crops: int,
    *,
    dim: int = INPUT_DIM,
    pair_cos: float = 0.8,
    noise: float = 0.05,
    nuisance_rank: int = 8,
    nuisance_scale: float = 0.35,
    seed: int = 0,
    label_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Speakers in confusable pairs (centre cosine ``pair_cos``, like two adult male voices) plus a
    shared low-rank nuisance (channel/reverb/codec) subspace and isotropic within-speaker noise.

    The nuisance basis is drawn from a fixed seed so train and test speakers share it, as recording
    conditions are shared across real speakers. Returns (features, labels, partner map).
    """
    basis = np.linalg.qr(np.random.default_rng(12345).standard_normal((dim, nuisance_rank)))[0].T
    rng = np.random.default_rng(seed)
    feats, labels, partner = [], [], {}
    for p in range(n_pairs):
        a = _unit(rng.standard_normal(dim))
        r = rng.standard_normal(dim)
        r = _unit(r - (r @ a) * a)
        b = pair_cos * a + np.sqrt(1 - pair_cos**2) * r
        sa, sb = label_offset + 2 * p, label_offset + 2 * p + 1
        partner[sa], partner[sb] = sb, sa
        for s, c in ((sa, a), (sb, b)):
            x = c + noise * rng.standard_normal((crops, dim)) + nuisance_scale * rng.standard_normal((crops, nuisance_rank)) @ basis
            feats.append(x)
            labels += [s] * crops
    return np.concatenate(feats).astype(np.float32), np.array(labels), partner


def compare(
    train_x: np.ndarray, train_y: np.ndarray, sets: dict[str, tuple], cfg: TrainConfig
) -> dict:
    """Train the AP and CE heads on (train_x, train_y); score raw/AP/CE on each eval set.

    ``sets[name] = (enroll_x, enroll_y, query_x, query_y, hard_partner)``.
    """
    tx, ty = torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.int64))
    t0 = time.perf_counter()
    heads = {"raw": None, "ap": train_ap_head(tx, ty, cfg)}
    t_ap = time.perf_counter() - t0
    heads["ce"] = train_ce_head(tx, ty, cfg)
    t_ce = time.perf_counter() - t0 - t_ap
    out = {"train": {"crops": len(train_x), "speakers": len(np.unique(train_y)), "ap_s": round(t_ap, 2), "ce_s": round(t_ce, 2)}}
    for name, (ex, ey, qx, qy, hp) in sets.items():
        out[name] = {h: evaluate_prototypes(embed(m, ex), ey, embed(m, qx), qy, hp) for h, m in heads.items()}
    return out


def run_synthetic(seeds: tuple[int, ...] = (0, 1, 2), cfg: TrainConfig | None = None, pair_cos: float = 0.8) -> dict:
    cfg = cfg or TrainConfig()
    results = {}
    for seed in seeds:
        x, y, _ = make_synthetic_speakers(100, 20, pair_cos=pair_cos, seed=seed)
        tx, ty, partner = make_synthetic_speakers(20, 20, pair_cos=pair_cos, seed=1000 + seed, label_offset=10_000)
        enroll = np.zeros(len(ty), bool)
        for s in np.unique(ty):
            enroll[np.flatnonzero(ty == s)[:5]] = True
        sets = {"held_out": (tx[enroll], ty[enroll], tx[~enroll], ty[~enroll], partner)}
        results[f"seed{seed}"] = compare(x, y, sets, TrainConfig(**{**cfg.__dict__, "seed": seed}))
    return results


# ------------------------------------------------------------------ real ECAPA features


def _results_dir() -> Path:
    main = Path(__file__).resolve().parents[3]
    return Path(os.environ.get("HEARING_RESULTS_DIR", main / "eval" / "results"))


def ecapa_crops(wavs: list[Path], crop_s: float, hop_s: float, max_crops: int | None = None, batch: int = 64) -> tuple[np.ndarray, list[int]]:
    """ECAPA embeddings of fixed-length crops of each file; returns (embeddings, file index per crop)."""
    import soundfile as sf

    from hearing import models
    from hearing.tse import enrolled

    if not enrolled.available():
        raise RuntimeError("ECAPA weights or speechbrain missing (set HEARING_MODELS_DIR)")
    backend = models.load(enrolled.MODEL)
    n, h = int(crop_s * 16000), int(hop_s * 16000)
    crops, owner = [], []
    for i, path in enumerate(wavs):
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != 16000:
            raise ValueError(f"{path}: {sr} Hz, expected 16000")
        x = x.mean(axis=1)
        starts = list(range(0, len(x) - n + 1, h))[:max_crops]
        for s in starts:
            c = x[s : s + n]
            if np.sqrt(np.mean(c**2)) > 10 ** (-50 / 20):  # skip near-silent crops
                crops.append(c)
                owner.append(i)
    out = []
    with torch.inference_mode():
        for k in range(0, len(crops), batch):
            xb = torch.from_numpy(np.stack(crops[k : k + batch])).to(backend.device)
            out.append(backend.encoder.encode_batch(xb).detach().cpu().numpy().reshape(len(xb), -1))
    return np.concatenate(out).astype(np.float32), owner


def librispeech_features(
    root: Path, crop_s: float, utts_per_speaker: int, cache: Path, split: str = "test-clean", crops_per_utt: int = 2
) -> dict:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    sex = {}
    for line in (root / "SPEAKERS.TXT").read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0].isdigit():
            sex[int(parts[0])] = parts[1]
    wavs, spk = [], []
    for d in sorted(p for p in (root / split).iterdir() if p.is_dir()):
        files = sorted(d.rglob("*.flac"))[:utts_per_speaker]
        wavs += files
        spk += [int(d.name)] * len(files)
    x, owner = ecapa_crops(wavs, crop_s, crop_s, max_crops=crops_per_utt)
    labels = np.array([spk[i] for i in owner])
    data = {"x": x, "y": labels, "sex": np.array([sex[s] for s in labels])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def team_features(root: Path, crop_s: float, cache: Path) -> dict:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    manifest = json.loads((root / "manifest.json").read_text())
    files = manifest["files"]
    x, owner = ecapa_crops([root / f["path"] for f in files], crop_s, crop_s / 2)
    data = {
        "x": x,
        "speaker": np.array([files[i]["speaker"] for i in owner]),
        "aug": np.array([files[i]["augmentation"] for i in owner]),
        "source": np.array([re.sub(r"_s\d+.*$", "", Path(files[i]["path"]).stem) for i in owner]),
    }
    np.savez(cache, **data)
    return data


def run_real(librispeech: Path, team: Path, cfg: TrainConfig, crop_s: float = 2.0, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    cache = _results_dir() / "T-307"
    ls = librispeech_features(librispeech, crop_s, 40, cache / f"librispeech_c{crop_s}.npz")
    tm = team_features(team, crop_s, cache / f"team_c{crop_s}.npz")
    speakers = np.unique(ls["y"])
    sex_of = dict(zip(ls["y"].tolist(), ls["sex"].tolist(), strict=True))
    team_ids = {"aryan": -1, "joseph": -2}
    ty = np.array([team_ids[s] for s in tm["speaker"]])
    profile = np.char.startswith(tm["source"].astype(str), "profile")
    clean = tm["aug"] == "clean"
    team_partner = {-1: -2, -2: -1}
    team_set = (tm["x"][profile & clean], ty[profile & clean], tm["x"][~profile], ty[~profile], team_partner)
    team_clean_q = (tm["x"][profile & clean], ty[profile & clean], tm["x"][~profile & clean], ty[~profile & clean], team_partner)
    results = {"crop_s": crop_s, "librispeech_crops": len(ls["x"]), "team_crops": len(tm["x"])}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        held = set()
        for s in ("M", "F"):  # 5 male + 5 female held out per seed
            pool = [k for k in speakers if sex_of[k] == s]
            held |= set(rng.choice(pool, size=5, replace=False).tolist())
        is_test = np.isin(ls["y"], list(held))
        hx, hy = ls["x"][is_test], ls["y"][is_test]
        enroll = np.zeros(len(hy), bool)
        for s in np.unique(hy):
            enroll[np.flatnonzero(hy == s)[:10]] = True
        partner = hardest_partners(hx[enroll], hy[enroll], sex_of)
        male_partner = {s: p for s, p in partner.items() if sex_of[s] == "M"}
        sets = {
            "librispeech_held_out": (hx[enroll], hy[enroll], hx[~enroll], hy[~enroll], partner),
            "librispeech_male_male": (hx[enroll], hy[enroll], hx[~enroll], hy[~enroll], male_partner),
            "team_all_aug": team_set,
            "team_clean": team_clean_q,
        }
        results[f"seed{seed}"] = compare(ls["x"][~is_test], ls["y"][~is_test], sets, TrainConfig(**{**cfg.__dict__, "seed": seed}))
    return results


def summarize(results: dict) -> dict:
    """Mean over seeds of every (set, head, metric)."""
    runs = [v for k, v in results.items() if k.startswith("seed")]
    out: dict = {}
    for name in runs[0]:
        if name == "train":
            continue
        for head in runs[0][name]:
            for metric, val in runs[0][name][head].items():
                if isinstance(val, float):
                    vals = [r[name][head][metric] for r in runs]
                    out.setdefault(name, {}).setdefault(head, {})[metric] = round(float(np.mean(vals)), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    main_checkout = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("which", choices=["synthetic", "real"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--crop-s", type=float, default=2.0)
    p.add_argument("--librispeech", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "T-307-ls" / "LibriSpeech")
    p.add_argument("--team", type=Path, default=main_checkout / "eval" / "augmented" / "T-306")
    a = p.parse_args(argv)
    cfg, seeds = TrainConfig(steps=a.steps), tuple(range(a.seeds))
    t0 = time.perf_counter()
    if a.which == "synthetic":
        res = run_synthetic(seeds, cfg)
    else:
        res = run_real(a.librispeech, a.team, cfg, a.crop_s, seeds)
    res["summary"] = summarize(res)
    res["meta"] = {"host": platform.node(), "torch": torch.__version__, "steps": a.steps, "wall_s": round(time.perf_counter() - t0, 1)}
    out = _results_dir() / "T-307" / f"ap_head_{a.which}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    json.dump(res["summary"], sys.stdout, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
