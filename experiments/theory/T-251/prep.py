"""T-251 prototype: target-conditioned extractor sketches, budgets and decision-experiment manifest.

Random weights only, CPU only, no audio, no training, no network. Everything printed is either
computed here (parameter counts, receptive field, causality check) or an ASSUMED range labelled so.

    .venv/bin/python experiments/theory/T-251/prep.py sketch   [--out DIR] [--seed N]
    .venv/bin/python experiments/theory/T-251/prep.py manifest [--out DIR] [--fixtures DIR] [--max-files N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

SR = 16000
CONDITIONING = ("none", "project", "film", "xattn")
DEFAULT_SEED = 251
MAX_FILES = 500  # input cap for the manifest scan


def _torch():
    import torch

    return torch


def build_model(mode: str, *, channels: int = 64, blocks: int = 6, emb_dim: int = 192,
                kernel: int = 3, enc_k: int = 32, enc_stride: int = 16, heads: int = 4):
    """Tiny causal TCN mask estimator; ``mode`` picks how the enrollment embedding enters.

    none     blind single-output mask (control; also the "blind stream" the ECAPA pick selects among)
    project  embedding -> Linear -> added once to the encoder output (projection-only)
    film     embedding -> per-block (gamma, beta) on the block's hidden state
    xattn    each block cross-attends to K enrollment frame features (keys/values from the past
             enrollment clip, so no causality violation on the mixture axis)
    """
    torch = _torch()
    nn = torch.nn

    class CausalBlock(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.pw1 = nn.Conv1d(channels, channels, 1)
            self.dw = nn.Conv1d(channels, channels, kernel, dilation=dilation, groups=channels)
            self.pw2 = nn.Conv1d(channels, channels, 1)
            self.act = nn.PReLU()
            self.film = nn.Linear(emb_dim, 2 * channels) if mode == "film" else None
            self.attn = nn.MultiheadAttention(channels, heads, batch_first=True) if mode == "xattn" else None
            self.kv = nn.Linear(emb_dim, channels) if mode == "xattn" else None

        def forward(self, x, emb, enroll_seq):
            h = self.act(self.pw1(x))
            h = nn.functional.pad(h, (self.pad, 0))
            h = self.act(self.dw(h))
            if self.film is not None:
                g, b = self.film(emb).chunk(2, dim=-1)
                h = h * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)
            if self.attn is not None:
                kv = self.kv(enroll_seq)
                q = h.transpose(1, 2)
                a, _ = self.attn(q, kv, kv, need_weights=False)
                h = h + a.transpose(1, 2)
            return x + self.pw2(h)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.enc = nn.Conv1d(1, channels, enc_k, stride=enc_stride)
            self.proj = nn.Linear(emb_dim, channels) if mode == "project" else None
            self.blocks = nn.ModuleList(CausalBlock(2 ** i) for i in range(blocks))
            self.mask = nn.Conv1d(channels, channels, 1)
            self.dec = nn.ConvTranspose1d(channels, 1, enc_k, stride=enc_stride)
            self.enc_k, self.enc_stride = enc_k, enc_stride

        def forward(self, mix, emb=None, enroll_seq=None):
            x = mix.unsqueeze(1)
            # left-pad so encoder frame t only sees samples <= its own window end
            w = nn.functional.pad(x, (self.enc_k - self.enc_stride, 0))
            z = torch.relu(self.enc(w))
            h = z
            if self.proj is not None:
                h = h + self.proj(emb).unsqueeze(-1)
            for blk in self.blocks:
                h = blk(h, emb, enroll_seq)
            m = torch.sigmoid(self.mask(h))
            y = self.dec(z * m)[..., self.enc_k - self.enc_stride: self.enc_k - self.enc_stride + mix.shape[-1]]
            return y.squeeze(1)

    return Net()


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def receptive_field_ms(blocks: int = 6, kernel: int = 3, enc_k: int = 32, enc_stride: int = 16) -> float:
    frames = 1 + sum((kernel - 1) * 2 ** i for i in range(blocks))
    return (frames - 1) * enc_stride / SR * 1000 + enc_k / SR * 1000


def algorithmic_latency_ms(enc_k: int = 32, enc_stride: int = 16) -> float:
    """Causal model: one encoder window (look-ahead to the end of the newest window)."""
    return enc_k / SR * 1000


def causality_leak(model, mode: str, seed: int, n: int = 4096, cut: int = 2048) -> float:
    """Max |difference| in outputs before ``cut`` when only samples >= cut change. 0 means causal.

    The prototype deliberately has no time-normalising layer (GroupNorm/global LN would leak the
    future); a trained version needs cumulative LN or none.
    """
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(1, 192, generator=g)
    seq = torch.randn(1, 20, 192, generator=g)
    a = torch.randn(1, n, generator=g)
    b = a.clone()
    b[:, cut:] = torch.randn(1, n - cut, generator=g)
    model.eval()
    with torch.no_grad():
        ya, yb = model(a, emb, seq), model(b, emb, seq)
    # one encoder window of slack before the cut is legitimately affected only via the decoder overlap
    return float((ya[:, : cut - 64] - yb[:, : cut - 64]).abs().max())


class StreamGate:
    """Lightweight adapter variant: keep the frozen blind separator, learn the pick.

    Inputs per hop: cosines of each separated stream to the target (and optional negative)
    embedding plus the previous gate. With S>=2 streams it generalises today's argmax+margin rule.
    """

    def __init__(self, streams: int = 2, hidden: int = 16, seed: int = DEFAULT_SEED):
        torch = _torch()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * streams + streams, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, streams + 1)
        )  # outputs: one logit per stream + "target absent" logit
        self.streams = streams

    def n_params(self) -> int:
        return n_params(self.net)


def sketch(seed: int) -> dict:
    torch = _torch()
    torch.manual_seed(seed)
    rows = []
    for mode in CONDITIONING:
        torch.manual_seed(seed)
        m = build_model(mode)
        leak = causality_leak(m, mode, seed)
        rows.append({
            "mode": mode,
            "params": n_params(m),
            "fp32_weights_mb": round(n_params(m) * 4 / 2**20, 3),
            "causal_leak": leak,
        })
    return {
        "seed": seed,
        "receptive_field_ms": round(receptive_field_ms(), 1),
        "algorithmic_latency_ms": algorithmic_latency_ms(),
        "stream_gate_params": {str(s): StreamGate(s).n_params() for s in (2, 3)},
        "rows": rows,
    }


def budgets() -> dict:
    """ASSUMED ranges (not measured). See docs/research/theory/target_conditioned.md §6."""
    return {
        "labelled": "assumption",
        "params_M": {"adapter_stream_gate": "<0.01", "film_on_small_tcn": "0.5-3", "full_bsrnn_or_mossformer_class": "10-60"},
        "train_hours_clean_pairs": {"adapter_stream_gate": "5-20 (can reuse labelled separator outputs)",
                                   "small_conditioned_tcn": "200-500 simulated 2-3 spk mix",
                                   "bsrnn_class": "500-2000 (REAL-TSE baselines: Libri2Mix-100, ~100 h clean)"},
        "speakers_needed": {"adapter_stream_gate": "100+", "small_conditioned_tcn": "1000+", "bsrnn_class": "1000-5000"},
        "train_vram_gb": {"adapter_stream_gate": "<2 (CPU ok)", "small_conditioned_tcn": "8-16", "bsrnn_class": "24-48"},
        "train_gpu_days": {"adapter_stream_gate": "<0.1", "small_conditioned_tcn": "1-4", "bsrnn_class": "4-14"},
        "infer_vram_gb": {"adapter_stream_gate": "0 extra", "small_conditioned_tcn": "<0.5", "bsrnn_class": "1-3"},
    }


EXPERIMENTS = [
    {"id": "E0", "name": "oracle-pick ceiling", "question": "How much of the blind+ECAPA gap is the pick, not the separator?",
     "method": "Score each window with the stream that has higher SI-SDR to the clean target vs the shipped ECAPA pick, on existing two-speaker fixtures.",
     "gpu": False, "decision": "If oracle-pick - ECAPA-pick < 1 dB SI-SDRi or < 3 points wrong-speaker time, stop: no conditioned model is worth building."},
    {"id": "E1", "name": "three-speaker leakage of the 2-output separator", "question": "How badly does a third voice contaminate both streams?",
     "method": "Mix target + 2 interferers at 0/-3/-6 dB; measure target SI-SDR of the best stream and interferer energy in the picked stream.",
     "gpu": True, "decision": "If the best stream's target SI-SDRi falls > 4 dB versus 2-speaker, a conditioned extractor (one target out, N-agnostic) is justified."},
    {"id": "E2", "name": "unseen-speaker enrollment robustness", "question": "Does the embedding pick or a conditioned model degrade for speakers outside the training distribution?",
     "method": "Enrollment 3/6/10 s, same-sex similar-voice pairs; report pick accuracy and target-words-preserved, with enrollment in a different room than the mixture.",
     "gpu": True, "decision": "Adopt only if conditioned >= ECAPA pick by 2 points wrong-speaker time on the held-out-speaker split with a CI excluding 0."},
    {"id": "E3", "name": "conditioning mechanism ablation", "question": "Is projection-only enough?",
     "method": "Train none/project/film/xattn variants of one small causal TCN (prototype in prep.py) on identical simulated mixes, same seeds.",
     "gpu": True, "decision": "Prefer the cheapest variant within 0.5 dB SI-SDRi of the best; xattn only if it wins by >= 1 dB on unseen speakers."},
    {"id": "E4", "name": "causal vs offline delay", "question": "What does streaming cost?",
     "method": "Same model, causal (enc window 2 ms) vs 20/40/100 ms look-ahead; score SI-SDRi and onset-delay.",
     "gpu": True, "decision": "Accept look-ahead only up to the latency budget in docs/ARCHITECTURE.md; otherwise stay causal."},
]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def count_inputs(fixtures: Path | None, max_files: int = MAX_FILES) -> dict:
    """Count (never read) candidate wav files; capped. Planned 'N' is separate from available."""
    out = {"fixtures_dir_set": fixtures is not None, "available_wavs": 0, "capped": False}
    if fixtures is None or not fixtures.is_dir():
        return out
    n = 0
    for _p in fixtures.rglob("*.wav"):
        n += 1
        if n >= max_files:
            out["capped"] = True
            break
    out["available_wavs"] = n
    return out


def manifest(out: Path, fixtures: Path | None, max_files: int, seed: int) -> dict:
    cfg = {"seed": seed, "experiments": EXPERIMENTS, "budgets": budgets(), "max_files": max_files,
           "cases_planned": {"smoke": 12, "cap": 300, "note": "global GPU cap300, smoke12 first; owned by Sol"}}
    result = {
        "config_sha256": sha256_text(json.dumps(cfg, sort_keys=True)),
        "inputs": count_inputs(fixtures, max_files),
        **cfg,
        "consumer_command": ".venv/bin/python experiments/theory/T-251/prep.py manifest --out $HEARING_RESULTS_DIR/overnight_prep/T-251",
        "status": "complete",
    }
    atomic_write(out / "manifest.json", json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sketch", "manifest"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--fixtures", type=Path, default=None)
    ap.add_argument("--max-files", type=int, default=MAX_FILES)
    a = ap.parse_args(argv)
    res_root = os.environ.get("HEARING_RESULTS_DIR")
    out = a.out or (Path(res_root) / "overnight_prep" / "T-251" if res_root else Path(tempfile.gettempdir()) / "T-251-out")
    if a.cmd == "sketch":
        res = sketch(a.seed)
        atomic_write(out / "sketch.json", json.dumps(res, indent=2, sort_keys=True))
    else:
        fx = a.fixtures or (Path(os.environ["HEARING_FIXTURES_DIR"]) if "HEARING_FIXTURES_DIR" in os.environ else None)
        res = manifest(out, fx, min(a.max_files, MAX_FILES), a.seed)
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
