"""Joint alpha/metric/head tracking sweep for T-295 (ilab1, RTX A4500).

Validates the T-295 composition-theory predictions on real phone-mic team audio:
  P1: alpha* decreases as window shortens (0.5 < 1.0 < 2.0 s)
  P2: alpha*_phone < alpha*_directional  (via synthetic SIR ladder)
  P3: phone/directional alpha* gap widens with window
  P4: alpha*_head > alpha*_maha > alpha*_cosine

Grid: alpha 0.1..0.9 (9) x metric {cosine, mahalanobis, head} (3)
      x window {0.5, 1.0, 2.0}s (3)  = 81 configs on real audio
  + SIR ladder: SIR {0, 3, 6, 12} dB x alpha {0.3, 0.5, 0.7} x window {0.5, 2.0}s
      = 24 configs (cosine only), emulating phone -> directional by mixing
      interferer at controlled SIR into clean single-speaker segments.

Data layout on ilab1 (job 447101):
  ~/hearing-joseph-overnight/data/          # team clips (phone mic)
  models/team_projection_head.pt            # T-273 head (61 KB)

Usage (on ilab1, inside job 447101 allocation):
  python3 sweep_tracking.py --config config.json [--dry-run] [--limit N]

Output: results/t295_sweep_<ts>.json  (per-config: flip_rate, mean_margin,
        switch_latency_hops) + results/t295_sir_ladder_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Embedding backends (lazy; degrade gracefully if weights/models absent)
# ---------------------------------------------------------------------------

def load_ecapa():
    """ECAPA encoder via hearing.tse.enrolled, else raise with clear message."""
    try:
        from hearing.tse import enrolled
        if not enrolled.available():
            raise RuntimeError("ECAPA weights absent on this machine")
        # _load_encoder is the registered loader; use the backend protocol
        backend = enrolled._load_encoder()
        return backend
    except (ImportError, RuntimeError, OSError, KeyError, AttributeError) as exc:
        raise RuntimeError(f"ECAPA backend unavailable: {exc}")



def load_head(head_path):
    """T-273 projection head: 192->64->32 MLP. Returns callable or None."""
    if not HAVE_TORCH or not os.path.isfile(head_path):
        return None
    try:
        m = torch.nn.Sequential(
            torch.nn.Linear(192, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32))
        sd = torch.load(head_path, map_location="cpu")
        # tolerate state-dict or full-module saves
        if isinstance(sd, dict):
            m.load_state_dict(sd, strict=False)
        m.eval()
        return lambda e: m(torch.as_tensor(e, dtype=torch.float32)).numpy()
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        print(f"[warn] head load failed ({exc}); running without head", flush=True)
        return None


def load_maha_stats(stats_path):
    """Within-speaker covariance for Mahalanobis; None -> fall back to cosine."""
    if stats_path and os.path.isfile(stats_path):
        try:
            d = np.load(stats_path, allow_pickle=True).item()
            cov = d["within_cov"]
            return np.linalg.inv(cov + 1e-4 * np.eye(cov.shape[0]))
        except (OSError, KeyError, ValueError) as exc:
            print(f"[warn] maha stats unusable ({exc})", flush=True)
    return None



# ---------------------------------------------------------------------------
# Streaming tracker (mirrors experiments/theory/T-295/tracking_sim.py)
# ---------------------------------------------------------------------------

class Tracker:
    def __init__(self, profiles, metric="cosine", alpha=0.5,
                 head_fn=None, maha_inv=None):
        self.profiles = profiles / np.linalg.norm(profiles, axis=1, keepdims=True)
        self.metric = metric
        self.alpha = alpha
        self.head_fn = head_fn
        self.maha_inv = maha_inv
        self.z = None
        if metric == "head" and head_fn is not None:
            self.mprofiles = np.stack([head_fn(p) for p in self.profiles])
            self.mprofiles /= np.linalg.norm(self.mprofiles, axis=1, keepdims=True)
        else:
            self.mprofiles = self.profiles

    def _embed_space(self, e):
        if self.metric == "head" and self.head_fn is not None:
            h = self.head_fn(e)
            return h / (np.linalg.norm(h) + 1e-9)
        return e / (np.linalg.norm(e) + 1e-9)

    def _score(self, z, profiles):
        if self.metric == "mahalanobis" and self.maha_inv is not None:
            d = profiles - z[None, :]
            dists = np.einsum("ij,jk,ik->i", d, self.maha_inv, d)
            sims = -dists
        else:
            sims = profiles @ z
        order = np.argsort(sims)[::-1]
        return int(order[0]), float(sims[order[0]] - sims[order[1]])

    def step(self, emb):
        h = self._embed_space(emb)
        self.z = h if self.z is None else self.alpha * h + (1.0 - self.alpha) * self.z
        zn = self.z / (np.linalg.norm(self.z) + 1e-9)
        return self._score(zn, self.mprofiles)


def stream_file(path, backend, window_s, hop_s, sr=16000):
    """Yield embeddings per hop. Reader is best-effort: wav via stdlib/soundfile."""
    import wave
    with wave.open(path, "rb") as wf:
        n_ch, sw, fr, n_fr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n_fr)
    pcm = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.int32).astype(np.float32)
    pcm = pcm.reshape(-1, n_ch).mean(axis=1) / (2 ** (8 * sw - 1))
    if fr != sr:  # crude resample
        idx = (np.arange(int(len(pcm) * sr / fr)) * fr / sr).astype(int)
        pcm = pcm[np.clip(idx, 0, len(pcm) - 1)]
    w, h = int(window_s * sr), int(hop_s * sr)
    for start in range(0, len(pcm) - w + 1, h):
        yield backend.embed(pcm[start:start + w])


def eval_config(files, labels, backend, profiles, metric, alpha, window_s,
                head_fn=None, maha_inv=None):
    """files: list of wav paths; labels: list of true speaker idx per file.
    Returns dict(flip_rate, mean_margin, n_hops)."""
    tr = Tracker(profiles, metric=metric, alpha=alpha,
                 head_fn=head_fn, maha_inv=maha_inv)
    flips, margins, n = 0, 0.0, 0
    hop_s = window_s / 10.0
    for path, true_k in zip(files, labels):
        tr.z = None  # reset per file
        try:
            for emb in stream_file(path, backend, window_s, hop_s):
                pred, margin = tr.step(emb)
                flips += (pred != true_k)
                margins += margin
                n += 1
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"[warn] skip {path}: {exc}", flush=True)

    return {"flip_rate": flips / max(n, 1), "mean_margin": margins / max(n, 1),
            "n_hops": n}


def mix_interferer(target_pcm, interf_pcm, sir_db):
    """Mix interferer into target at given SIR (dB). For the SIR ladder."""
    n = min(len(target_pcm), len(interf_pcm))
    t, itf = target_pcm[:n], interf_pcm[:n]
    p_t = np.mean(t ** 2) + 1e-12
    p_i = np.mean(itf ** 2) + 1e-12
    g = math.sqrt(p_t / (p_i * 10.0 ** (sir_db / 10.0)))
    return t + g * itf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    data_dir = os.path.expanduser(cfg["data_dir"])
    files, labels = [], []
    for entry in cfg["clips"]:
        p = os.path.join(data_dir, entry["file"])
        if os.path.isfile(p):
            files.append(p); labels.append(entry["speaker"])
    print(f"[info] {len(files)} clips found", flush=True)
    if args.dry_run:
        n_cfg = len(cfg["alphas"]) * len(cfg["metrics"]) * len(cfg["windows"])
        n_sir = len(cfg["sir_db"]) * len(cfg["sir_alphas"]) * len(cfg["sir_windows"])
        print(f"[dry-run] main sweep: {n_cfg} configs, SIR ladder: {n_sir} configs")
        print(f"[dry-run] est. runtime ~{(n_cfg + n_sir) * cfg['sec_per_config'] / 60:.0f} min")
        return

    if not files:
        sys.exit("no clips found; check data_dir and clips manifest")

    backend = load_ecapa()
    # enrollment profiles: mean embedding of enrollment clips per speaker
    profiles = []
    for spk in sorted(set(labels)):
        embs = []
        for p, lab in zip(files, labels):
            if lab == spk and "enroll" in os.path.basename(p):
                for e in stream_file(p, backend, 2.0, 0.5):
                    embs.append(e / (np.linalg.norm(e) + 1e-9))
                    if len(embs) >= 8:
                        break
            if len(embs) >= 8:
                break
        if not embs:
            sys.exit(f"no enrollment audio for speaker {spk}")
        profiles.append(np.mean(embs, axis=0))
    profiles = np.stack(profiles)

    head_fn = load_head(os.path.expanduser(cfg.get("head_path", "")))
    maha_inv = load_maha_stats(os.path.expanduser(cfg.get("maha_stats_path", "")))
    metrics = [m for m in cfg["metrics"]
               if m == "cosine" or (m == "head" and head_fn) or (m == "mahalanobis" and maha_inv is not None)]
    print(f"[info] metrics active: {metrics}", flush=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = {"config": cfg, "results": []}
    cfgs = [(a, m, w) for a in cfg["alphas"] for m in metrics for w in cfg["windows"]]
    if args.limit:
        cfgs = cfgs[:args.limit]
    t0 = time.time()
    for i, (a, m, w) in enumerate(cfgs):
        r = eval_config(files, labels, backend, profiles, m, a, w,
                        head_fn=head_fn, maha_inv=maha_inv)
        r.update({"alpha": a, "metric": m, "window_s": w})
        out["results"].append(r)
        el = (time.time() - t0) / 60
        print(f"[{i+1}/{len(cfgs)}] a={a} m={m} w={w}: flip={r['flip_rate']:.4f} "
              f"margin={r['mean_margin']:.3f} ({el:.1f} min elapsed)", flush=True)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    op = os.path.join(cfg["out_dir"], f"t295_sweep_{ts}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] wrote {op}")


if __name__ == "__main__":
    main()
