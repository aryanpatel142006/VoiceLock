"""Round-2 GPU sweep: gated tracker with switch-cue comparison (T-295/P8).

Tests whether the gated design (fixed-alpha* base + dwell boost on detected
switches) beats fixed alpha* on REAL phone-mic team audio, and — the key
Round-2 question — which switch cue clears the (precision >= 0.6,
recall >= 0.5) bar the simulator says is necessary:

  cue=margin : persistence-2 observation-vs-state margin disagreement
  cue=f0     : persistence-2 F0 discontinuity (autocorrelation, numpy only)

Turn-taking streams are SYNTHETIC CONCATENATIONS of real single-speaker
clips with known switch points (honest: the switches are constructed, the
audio and embeddings are real).

Grid (0.5 s window, phone audio):
  schedule : fixed a=0.3 | naive b=4 | gated-margin th_m in {0.12, 0.20}
             | gated-f0 th_f0=4 semitones
  metric   : cosine | head (if models/team_projection_head.pt loads)

~10 configs x ~2 min = ~20 min on the A4500 (job 447101). 30-min cap.

Usage (on ilab1, inside job 447101):
  python3 sweep_gated.py --config config_gated.json [--dry-run] [--limit N]
Output: results/t295/t295_gated_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import wave
from datetime import UTC, datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_tracking import load_ecapa, load_head

# ---------------------------------------------------------------------------
# Audio helpers (numpy only)
# ---------------------------------------------------------------------------

def read_wav_mono(path, sr=16000):
    with wave.open(path, "rb") as wf:
        n_ch, sw, fr, n_fr = (wf.getnchannels(), wf.getsampwidth(),
                              wf.getframerate(), wf.getnframes())
        raw = wf.readframes(n_fr)
    pcm = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.int32).astype(np.float32)
    pcm = pcm.reshape(-1, n_ch).mean(axis=1) / (2 ** (8 * sw - 1))
    if fr != sr:
        idx = (np.arange(int(len(pcm) * sr / fr)) * fr / sr).astype(int)
        pcm = pcm[np.clip(idx, 0, len(pcm) - 1)]
    return pcm


def f0_autocorr(pcm, sr=16000, fmin=70.0, fmax=400.0):
    """Median-lag autocorrelation F0. Returns Hz or NaN (unvoiced/weak)."""
    x = pcm - np.mean(pcm)
    if np.mean(x ** 2) < 1e-8:
        return float("nan")
    n = len(x)
    corr = np.correlate(x, x, mode="full")[n - 1:]
    corr /= (corr[0] + 1e-12)
    lo, hi = int(sr / fmax), int(sr / fmin)
    if hi >= len(corr):
        return float("nan")
    seg = corr[lo:hi]
    k = int(np.argmax(seg))
    if seg[k] < 0.35:  # weakly periodic -> abstain
        return float("nan")
    return float(sr / (lo + k))


def build_turn_streams(pcm_by_spk, sr=16000, seg_s=3.0, n_streams=4,
                       n_segs=20, seed=0):
    """Concatenate real single-speaker segments into turn-taking streams.

    Returns list of (pcm, switch_hop_idx, true_label_per_hop, hop_s).
    Window/hop fixed at 0.5 s / 0.05 s here (demo cell).
    """
    rng = np.random.default_rng(seed)
    window_s, hop_s = 0.5, 0.05
    seg_n = int(seg_s * sr)
    streams = []
    for _ in range(n_streams):
        order = [0, 1]
        rng.shuffle(order)
        seq = [order[i % 2] for i in range(n_segs)]
        parts, bounds = [], []
        for spk in seq:
            clip = pcm_by_spk[spk]
            start = int(rng.integers(0, max(1, len(clip) - seg_n)))
            parts.append(clip[start:start + seg_n])
            bounds.append(sum(len(p) for p in parts))
        pcm = np.concatenate(parts)
        # switch hop indices + per-hop labels
        n_hops = int((len(pcm) / sr - window_s) / hop_s) + 1
        labels = np.zeros(n_hops, dtype=int)
        switches = []
        hop = 0
        for i, spk in enumerate(seq):
            seg_hops = int(seg_s / hop_s)
            labels[hop:hop + seg_hops] = spk
            if i > 0:
                switches.append(hop)
            hop += seg_hops
        streams.append((pcm, switches, labels[:n_hops]))
    return streams, window_s, hop_s


# ---------------------------------------------------------------------------
# Gated tracker (mirrors experiments/theory/T-295/gated_tracking_sim.py)
# ---------------------------------------------------------------------------

class GatedTracker:
    def __init__(self, profiles, metric="cosine", head_fn=None,
                 alpha_star=0.3, schedule="fixed",
                 alpha_boost=0.7, tau_dwell=3,
                 theta_m=0.15, theta_f0_st=4.0, persist=2,
                 naive_beta=4.0):
        self.profiles = profiles / np.linalg.norm(profiles, axis=1, keepdims=True)
        self.metric = metric
        self.head_fn = head_fn
        self.a_star = alpha_star
        self.schedule = schedule          # fixed | naive | gated-margin | gated-f0
        self.alpha_boost = alpha_boost
        self.tau_dwell = tau_dwell
        self.theta_m = theta_m
        self.theta_f0_st = theta_f0_st
        self.persist = persist
        self.naive_beta = naive_beta
        if metric == "head" and head_fn is not None:
            self.mp = np.stack([head_fn(p) for p in self.profiles])
            self.mp /= np.linalg.norm(self.mp, axis=1, keepdims=True)
        else:
            self.mp = self.profiles
        self.z = None
        self.prev_margin = 0.0
        self.dwell = 0
        self._dis_n = 0
        self._prev_f0 = float("nan")
        self.detected = []

    def _space(self, e):
        if self.metric == "head" and self.head_fn is not None:
            h = self.head_fn(e)
            return h / (np.linalg.norm(h) + 1e-9)
        return e / (np.linalg.norm(e) + 1e-9)

    def step(self, emb, pcm, hop_idx, sr=16000):
        h = self._space(emb)
        sims_h = self.profiles @ (h / (np.linalg.norm(h) + 1e-9))

        if self.z is None:
            s_prev = int(np.argmax(sims_h))
            r_t = 0.0
            f0_jump = False
        else:
            zn = self.z / (np.linalg.norm(self.z) + 1e-9)
            order = np.argsort(self.mp @ zn)[::-1]
            s_prev, r_prev = int(order[0]), int(order[1])
            r_t = float(sims_h[s_prev] - sims_h[r_prev])
            f0 = f0_autocorr(pcm, sr)
            if not math.isnan(f0) and not math.isnan(self._prev_f0) and self._prev_f0 > 0:
                jump_st = abs(12.0 * math.log2(f0 / self._prev_f0))
                f0_jump = jump_st > self.theta_f0_st
            else:
                f0_jump = False
            self._prev_f0 = f0

        cue = False
        if self.schedule == "gated-margin":
            self._dis_n = self._dis_n + 1 if (r_t < -self.theta_m) else 0
            cue = self._dis_n >= self.persist
        elif self.schedule == "gated-f0":
            self._dis_n = self._dis_n + 1 if f0_jump else 0
            cue = self._dis_n >= self.persist

        if cue:
            self.detected.append(hop_idx)
            self.dwell = self.tau_dwell

        if self.schedule == "naive":
            alpha_t = self.a_star / (1.0 + math.exp(-self.naive_beta * self.prev_margin))
        elif self.dwell > 0:
            alpha_t = self.alpha_boost
        else:
            alpha_t = self.a_star  # FIXED base path (Round-2 Corollary)
        self.dwell = max(0, self.dwell - 1)

        self.z = h if self.z is None else alpha_t * h + (1.0 - alpha_t) * self.z
        zn = self.z / (np.linalg.norm(self.z) + 1e-9)
        sims = self.mp @ zn
        order = np.argsort(sims)[::-1]
        margin = float(sims[order[0]] - sims[order[1]])
        self.prev_margin = margin
        return int(order[0]), margin


def eval_schedule(streams, window_s, hop_s, backend, profiles, sr=16000, **kw):
    tr_kw = dict(kw)
    flips, margins, n = 0, 0.0, 0
    latencies, n_sw = [], 0
    all_detected, all_switches = [], []
    hop_offset = 0
    for pcm, switches, labels in streams:
        tr = GatedTracker(profiles, **tr_kw)
        w, h = int(window_s * sr), int(hop_s * sr)
        last_switch, recovered = None, True
        hop = 0
        for start in range(0, len(pcm) - w + 1, h):
            if hop >= len(labels):
                break
            emb = backend.embed(pcm[start:start + w])
            pred, margin = tr.step(emb, pcm[start:start + w], hop, sr)
            true_k = int(labels[hop])
            if hop in switches:
                last_switch, recovered = hop, False
                n_sw += 1
            if last_switch is not None and not recovered and pred == true_k:
                latencies.append(hop - last_switch)
                recovered = True
            flips += (pred != true_k)
            margins += margin
            n += 1
            hop += 1
        all_detected += [d + hop_offset for d in tr.detected]
        all_switches += [s + hop_offset for s in switches]
        hop_offset += hop
    latencies += [50] * (n_sw - len(latencies))
    det = None
    if tr_kw.get("schedule", "").startswith("gated"):
        hit = sum(any(abs(d - s) <= 3 for s in all_switches) for d in all_detected)
        hit_sw = sum(any(abs(d - s) <= 3 for d in all_detected) for s in all_switches)
        det = {"precision": hit / max(len(all_detected), 1),
               "recall": hit_sw / max(len(all_switches), 1)}
    return {"flip_rate": flips / max(n, 1),
            "mean_margin": margins / max(n, 1),
            "mean_switch_latency_hops": float(np.mean(latencies)) if latencies else 0.0,
            "n_hops": n, "n_switches": n_sw, "detection": det}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    data_dir = os.path.expanduser(cfg["data_dir"])

    n_cfg_est = len(cfg["schedules"]) * len(cfg["metrics"])
    print(f"[dry-run] {n_cfg_est} configs; "
          f"~{cfg['sec_per_config']} s/config -> ~{n_cfg_est * cfg['sec_per_config'] / 60:.0f} min")
    if args.dry_run:
        return

    pcm_by_spk = {0: [], 1: []}
    enroll_pcm = {0: [], 1: []}
    for entry in cfg["clips"]:
        p = os.path.join(data_dir, entry["file"])
        if not os.path.isfile(p):
            continue
        pcm = read_wav_mono(p)
        (enroll_pcm if "enroll" in entry["file"] else pcm_by_spk)[entry["speaker"]].append(pcm)
    pcm_by_spk = {k: (np.concatenate(v) if v else np.array([])) for k, v in pcm_by_spk.items()}
    if any(len(v) < 16000 * 10 for v in pcm_by_spk.values()):
        sys.exit("need >=10 s of test audio per speaker; check clips manifest")
    print(f"[info] test audio: {[f'{k}:{len(v)/16000:.0f}s' for k, v in pcm_by_spk.items()]}",
          flush=True)

    streams, window_s, hop_s = build_turn_streams(
        pcm_by_spk, n_streams=cfg.get("n_streams", 4),
        n_segs=cfg.get("n_segs", 20), seed=cfg.get("seed", 0))

    backend = load_ecapa()
    profiles = []
    for spk in (0, 1):
        embs = []
        for pcm in enroll_pcm[spk]:
            w, h = int(2.0 * 16000), int(0.5 * 16000)
            for start in range(0, len(pcm) - w + 1, h):
                e = backend.embed(pcm[start:start + w])
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
    metrics = [m for m in cfg["metrics"]
               if m == "cosine" or (m == "head" and head_fn)]
    print(f"[info] metrics active: {metrics}", flush=True)

    cfgs = [(s, m) for s in cfg["schedules"] for m in metrics]
    if args.limit:
        cfgs = cfgs[:args.limit]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = {"config": cfg, "results": []}
    t0 = time.time()
    for i, (sched, metric) in enumerate(cfgs):
        kw = dict(sched)
        kind = kw.pop("kind")
        r = eval_schedule(streams, window_s, hop_s, backend, profiles,
                          metric=metric, head_fn=head_fn,
                          schedule=kind, **kw)
        r.update({"schedule": kind, "params": kw, "metric": metric})
        out["results"].append(r)
        el = (time.time() - t0) / 60
        print(f"[{i+1}/{len(cfgs)}] {kind} {kw} {metric}: "
              f"flip={r['flip_rate']:.4f} lat={r['mean_switch_latency_hops']:.1f} "
              f"({el:.1f} min elapsed)", flush=True)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    op = os.path.join(cfg["out_dir"], f"t295_gated_{ts}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] wrote {op}")


if __name__ == "__main__":
    main()
