"""Holdout eval of the causal students vs the windowed teacher and the base model (T-424).

Scenes and metrics are T-391's ``eval_sep.py`` (holdout/same takes, test cafe noise and test
LibriSpeech strangers; none of them is read by ``distill.py``): per row, wrong-speaker % (for
a fixed-slot pick this is the slot-swap %), SI-SDRi on target-active windows, leak on windows
where only the other talks, ``slot<k>_absent`` on stranger rows, each with a 95 % bootstrap CI.

- A student runs **frame-causally at its own latency**: one causal pass over the whole scene
  (equal to ``student.Streamer`` chunk by chunk, tested), so every output sample used only
  input up to one window (8-32 ms) later. Its output is then cut into the same scoring windows
  as the teacher's and put on the mixture's scale per window, as the teacher's windows are.
- The teacher (T-387 fixed-order MossFormer2) and the base MossFormer2 run non-causally per
  window (``eval_sep.evaluate_row``) at 1.0 s (hop 0.5 s) and 0.5 s (hop 0.25 s) windows;
  the student is scored on both grids. Base picks with ECAPA (it has no slot order).

    python eval_student.py --student tcn16=T-424/run_tcn16/best.pt --teacher T-374b/run_fixed/best.pt \
        --base ~/hearing-remote/models/MossFormer2_SS_16K/last_best_checkpoint.pt \
        --data T-374/data --ecapa ~/hearing-remote/models/spkrec-ecapa-voxceleb --out T-424/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "T-391")]
import eval_sep as es
import student
from train_sep import SR, load, rms

GRIDS = {"1.0s": (SR, SR // 2), "0.5s": (SR // 2, SR // 4)}
SLOT = {"aryan": 0, "joseph": 1}
SCENES = (
    ("holdout-sir0-cafe", ("holdout-01", "holdout-02"), 0.0, 5.0),
    ("holdout-sir0-none", ("holdout-01", "holdout-02"), 0.0, None),
    ("holdout-sir-6-cafe", ("holdout-01", "holdout-02"), -6.0, 5.0),
    ("same-sir0-cafe", ("same-01", "same-02"), 0.0, 5.0),
    ("holdout-sir-10-cafe", ("holdout-01", "holdout-02"), -10.0, 5.0),
    ("holdout-sir+6-cafe", ("holdout-01", "holdout-02"), 6.0, 5.0),
    ("holdout-sir0-snr0", ("holdout-01", "holdout-02"), 0.0, 0.0),
)


def build_rows(team, data):
    """(name, mix, target, other, profile, slot, absent) as eval_sep.main builds them."""
    team = Path(team)

    def take(who, names):
        return np.concatenate([load(team / who / f"{n}.wav") for n in names])

    rows = []
    for name, takes, sir, snr in SCENES:
        for tgt, oth in (("aryan", "joseph"), ("joseph", "aryan")):
            mix, t, o = es.scene(take(tgt, takes), take(oth, takes), data, snr, sir)
            rows.append((f"{tgt}:{oth} {name}", mix, t, o, load(team / tgt / "profile-01.wav"), SLOT[tgt], ()))
    for tgt in ("aryan", "joseph"):
        t_take = take(tgt, ("holdout-01", "holdout-02"))
        mix, t, o = es.scene(t_take, es.stranger_voice(data, len(t_take)), data, 5.0, -6.0)
        rows.append((f"{tgt}:stranger holdout-sir-6-cafe", mix, t, o, None, SLOT[tgt], (1 - SLOT[tgt],)))
    mix, o, _ = es.scene(es.stranger_voice(data, 60 * SR), None, data, 5.0, 0.0)
    rows.append(("stranger-only cafe", mix, np.zeros_like(mix), o, None, None, (0, 1)))
    return rows


def score_stream(est_full, mix, tgt, oth, slot, win, hop, rng, absent=()):
    """eval_sep.evaluate_row's metrics on a precomputed causal output, fixed pick."""
    out = {"wrong": [], "sdri": [], "leak": []}
    gone = {k: [] for k in absent}
    t_floor, o_floor = 0.1 * rms(tgt), 0.1 * rms(oth)
    for start in range(0, len(mix) - win + 1, hop):
        m, t, o = (x[start : start + win] for x in (mix, tgt, oth))
        est = est_full[:, start : start + win].astype(np.float64)
        total = est.sum(axis=0)
        est = est * float(m @ total) / (float(total @ total) + 1e-12)
        t_on, o_on = rms(t) > t_floor, rms(o) > o_floor
        if o_on:
            for k in absent:
                gone[k].append(10 * np.log10((est[k] @ est[k] + 1e-12) / (m @ m + 1e-12)))
        if slot is None:
            continue
        y = est[slot]
        if t_on:
            out["wrong"].append(100.0 * (es.proj_energy(y, o) > es.proj_energy(y, t)))
            out["sdri"].append(es.si_sdr(y, t) - es.si_sdr(m, t))
        elif o_on:
            out["leak"].append(10 * np.log10((y @ y + 1e-12) / (m @ m + 1e-12)))
    result = {}
    if slot is not None:
        result["fixed"] = {
            "wrong_pct": es.boot(out["wrong"], rng),
            "sdri_db": es.boot(out["sdri"], rng),
            "leak_db": es.boot(out["leak"], rng),
        }
    result.update({f"slot{k}_absent": {"absent_db": es.boot(v, rng)} for k, v in gone.items()})
    return result


def table(result):
    """Markdown: wrong % / SI-SDRi dB / leak dB means per row and column."""
    cols = sorted({c for r in result["rows"].values() for c in r if not c.split("/")[-1].startswith("slot")})
    lines = ["| row | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for row, r in result["rows"].items():
        cells = []
        for c in cols:
            s = r.get(c)
            if not s:
                cells.append("n/a")
                continue
            vals = [s.get(k, {}).get("mean") for k in ("wrong_pct", "sdri_db", "leak_db")]
            cells.append(" / ".join("-" if v is None else f"{v:+.1f}" for v in vals))
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", action="append", default=[], help="label=path to a distill.py ckpt")
    ap.add_argument("--teacher", help="T-387 fixed-order MossFormer2 best.pt")
    ap.add_argument("--base", help="shipped MossFormer2_SS_16K checkpoint")
    ap.add_argument("--team", default=str(Path.home() / "hearing-data/team"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--ecapa", help="needed for --base (ECAPA pick)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = build_rows(args.team, args.data)
    result = {"rows": {}, "grids": {g: {"win_s": w / SR, "hop_s": h / SR} for g, (w, h) in GRIDS.items()}}

    def put(row, label, score):
        for pick, s in score.items():
            result["rows"].setdefault(row, {})[f"{label}/{pick}"] = s
            print(row, label, pick, json.dumps(s), flush=True)

    for spec in args.student:
        label, path = spec.split("=", 1)
        net = student.load(path).to(dev)
        result.setdefault("students", {})[label] = {"latency_ms": net.latency_ms, "params": student.count_params(net)}
        for row, mix, t, o, _prof, slot, absent in rows:
            with torch.inference_mode():
                est = net(torch.from_numpy(mix)[None].to(dev))[0].float().cpu().numpy()
            for g, (win, hop) in GRIDS.items():
                put(row, f"{label}@{g}", score_stream(est, mix, t, o, slot, win, hop, np.random.default_rng(0), absent))
        del net
    nets = [("teacher", args.teacher, ("fixed",)), ("base", args.base, ("ecapa",))]
    ecapa = es.Ecapa(args.ecapa, dev) if args.base else None
    for label, path, picks in nets:
        if not path:
            continue
        net = es.network(path).to(dev).eval()
        for g, (win, hop) in GRIDS.items():
            es.WIN, es.HOP = win, hop
            for row, mix, t, o, prof, slot, absent in rows:
                use = picks if slot is not None and (prof is not None or "ecapa" not in picks) else ()
                score = es.evaluate_row(net, ecapa, mix, t, o, prof, use, slot, dev, np.random.default_rng(0), absent)
                put(row, f"{label}@{g}", score)
        del net
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(result, indent=1))
    Path(args.out).with_suffix(".md").write_text(table(result) + "\n")
    print(table(result))


if __name__ == "__main__":
    main()
