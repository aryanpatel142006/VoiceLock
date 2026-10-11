"""Confusion eval for the identity bouncer on SIR-binned mixtures (T-413).

Renders test-split items with ``eval.sir_mixtures`` (no files written), builds simulated
extractor outputs with known ground truth and scores each window with ``IdentityGate``:

* ``target`` / ``interferer``: perfect extraction and a full wrong-speaker lock;
* ``ext20`` / ``ext10``: the target with the interferer (as loud as in the mix) suppressed by
  20 / 10 dB, the leak a real extractor leaves;
* ``wrong20``: locked onto the interferer with the target suppressed by 20 dB;
* ``mix``: no extraction (diagnostic only; left out of the error rates).

The truth for each output is "target dominant": target active level above interferer level.
Prints cosine statistics, the equal-error threshold, and false-block / false-pass rates at
the module thresholds, per SIR bin, window length and open/closed set (``--others``).
Accuracy-only: run under ``swarm-lock --shared``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from eval import sir_mixtures as sm
from hearing.tse import identity_gate as ig

CANDIDATES = ("target", "interferer", "ext20", "ext10", "wrong20", "mix")


def candidates(item: dict) -> dict[str, np.ndarray]:
    t, i, n = (item[k].astype(np.float64) for k in ("target", "interferer", "noise"))
    return {
        "target": t, "interferer": i, "mix": t + i + n,
        "ext20": t + i * 0.1, "ext10": t + i * 10 ** (-0.5), "wrong20": i + t * 0.1,
    }


def target_dominant(item: dict, name: str) -> bool:
    if name in ("target", "ext20", "ext10"):
        lead = {"target": 200.0, "ext20": 20.0, "ext10": 10.0}[name]
    elif name in ("interferer", "wrong20"):
        lead = {"interferer": -200.0, "wrong20": -20.0}[name]
    else:
        lead = 0.0
    return item["measured_sir_db"] + (0.0 if name == "mix" else lead) > 0.0


def embedder():
    from hearing.tse import enrolled

    if enrolled.available():
        from hearing import models

        return models.load(enrolled.MODEL), "ecapa"
    return enrolled.MockBackend(), "mock"


def eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """(equal error rate, threshold): accept ``score >= thr``; pos should pass, neg not."""
    best = None
    for thr in np.unique(np.concatenate([pos, neg])):
        frr, far = float(np.mean(pos < thr)), float(np.mean(neg >= thr))
        if best is None or abs(frr - far) < best[0]:
            best = (abs(frr - far), (frr + far) / 2, float(thr))
    return best[1], best[2]


def run(per_bin: int, windows: tuple[float, ...], bins: tuple[float, ...], seed: int, root: Path | None):
    corpus = sm.load_corpus(root or sm._default_root())
    emb, kind = embedder()
    enroll_emb = {}
    speakers = corpus.speakers("test")
    for s in corpus.speakers("enroll"):
        x, _ = sm._crop(corpus, corpus.of(s, "enroll"), np.random.default_rng(seed), int(sm.ENROLL_CROP_S * sm.SR))
        enroll_emb[s] = ig._unit(emb.embed(x.astype(np.float32)))
    rows = []
    t0 = time.perf_counter()
    for b in bins:
        for k in range(per_bin):
            rng = np.random.default_rng([seed, int(b * 10) + 1000, k])
            item = sm.render_item(corpus, "test", float(b), rng, target=speakers[k % len(speakers)])
            others = [v for s, v in enroll_emb.items() if s != item["target_id"]]
            # the item's own enrollment crop is random; use the fixed profile so cosines compare
            target_emb = enroll_emb[item["target_id"]]
            for name, out in candidates(item).items():
                truth = target_dominant(item, name)
                for w in windows:
                    n = int(w * sm.SR)
                    for start in range(0, len(out) - n + 1, n):
                        e = ig._unit(emb.embed(out[start : start + n].astype(np.float32)))
                        rows.append({
                            "sir": b, "cand": name, "win": w, "truth": truth,
                            "t": float(e @ target_emb), "o": max(float(e @ v) for v in others),
                        })
    elapsed = time.perf_counter() - t0
    return rows, kind, elapsed


def summarise(rows, accept=ig.ACCEPT_COS, margin=ig.MARGIN_COS) -> dict:
    out = {}
    wins = sorted({r["win"] for r in rows})
    for w in wins:
        sub = [r for r in rows if r["win"] == w]
        scored = [r for r in sub if r["cand"] != "mix"]  # the raw mix is a diagnostic, not an output
        pos = np.array([r["t"] for r in scored if r["truth"]])
        neg = np.array([r["t"] for r in scored if not r["truth"]])
        pos_m = np.array([r["t"] - r["o"] for r in scored if r["truth"]])
        neg_m = np.array([r["t"] - r["o"] for r in scored if not r["truth"]])
        e_open, thr_open = eer(pos, neg)
        e_closed, thr_closed = eer(pos_m, neg_m)
        cell = {"n": len(scored), "eer_open": round(e_open, 4), "thr_open": round(thr_open, 3),
                "eer_margin": round(e_closed, 4), "thr_margin": round(thr_closed, 3), "by": {}}
        for b in sorted({r["sir"] for r in sub}, reverse=True):
            for c in CANDIDATES:
                cs = [r for r in sub if r["sir"] == b and r["cand"] == c]
                if not cs:
                    continue
                pass_open = np.mean([r["t"] >= accept for r in cs])
                pass_closed = np.mean([r["t"] >= accept and r["t"] - r["o"] >= margin for r in cs])
                cell["by"][f"{b:+.0f}/{c}"] = {
                    "truth_target_share": round(float(np.mean([r["truth"] for r in cs])), 2),
                    "cos_t_mean": round(float(np.mean([r["t"] for r in cs])), 3),
                    "pass_open": round(float(pass_open), 3), "pass_closed": round(float(pass_closed), 3),
                }
        tp = [r for r in scored if r["truth"]]
        tn = [r for r in scored if not r["truth"]]
        def ok(r):
            return r["t"] >= accept and r["t"] - r["o"] >= margin
        cell["false_block_closed"] = round(float(np.mean([not ok(r) for r in tp])), 4)
        cell["false_pass_closed"] = round(float(np.mean([ok(r) for r in tn])), 4)
        cell["false_block_open"] = round(float(np.mean([r["t"] < accept for r in tp])), 4)
        cell["false_pass_open"] = round(float(np.mean([r["t"] >= accept for r in tn])), 4)
        out[f"{w:.2f}s"] = cell
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--windows", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--bins", type=float, nargs="+", default=list(sm.EVAL_SIR_BINS))
    ap.add_argument("--seed", type=int, default=413)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rows, kind, elapsed = run(args.per_bin, tuple(args.windows), tuple(args.bins), args.seed, args.root)
    summary = {"embedder": kind, "accept_cos": ig.ACCEPT_COS, "margin_cos": ig.MARGIN_COS,
               "windows_scored": len(rows), "seconds": round(elapsed, 1), "by_window": summarise(rows)}
    print(json.dumps(summary, indent=1))
    base = Path(os.environ.get("HEARING_RESULTS_DIR") or "eval/results")
    out = args.out or base / "T-413_identity_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
