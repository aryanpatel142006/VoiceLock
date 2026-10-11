"""T-407: fit and score the learned continuous-gain gate (``hearing.tse.learned_gate``) offline.

Runs where T-391's dump lives (team-derived; it stays on iLab):

    PYTHONPATH=<code> ~/hearing-remote/.venv/bin/python -m eval.learned_gate_fit \
        --features ~/tmp/T-391/gate_features.npz --out ~/tmp/T-407/learned_gate.json

Data: ``experiments/gpu/T-391/dump_gate_features.py`` (v3 separator, 1.0 s windows, 0.5 s hop; train/val = 2 s v3
renders of 3 windows each from their own takes, holdout = the holdout-01/02 scenes and the stranger scenes). Each
window gives two rows, one per role (target Aryan = slot 0, target Joseph = slot 1). The heads are fitted on the
train split only, the policy constants are picked on the val split, and every number in the table is the holdout.

Arms, all from the same windows (the dump has no LDA pieces, so arm 4 and arm 5 are proxies):

- ``arm4`` (T-386 arm 4: ft + fixed + floor 20): the ECAPA contrast gate. The slot's score is its cosine contrast
  ``cos(slot, target) - cos(slot, partner)`` mapped to LDA units (target-solo slot centroid +1, partner-solo -1,
  fitted on train); score < 0 (the contrast default) -> gated at -20 dB.
- ``arm5`` (T-402 arm 5): arm 4 + ``hearing.tse.fixed_gate`` (shipped WEIGHTS, rescue only, partner-lead guard,
  swap rescue) on the same features (whole-window energies; the live gate uses the last half).
- ``learned``: :class:`hearing.tse.learned_gate.LearnedGate`.

Metrics per row (0.5 s per window, the part a hop plays): ``muted_s`` = target-active windows (reference >= -15 dB re
mix) played at <= -10 dB; ``wrong_pct`` = windows where only someone else talks (target < -25 dB, partner or stranger
>= -15 dB) and the played slot is audible (slot + gain >= -15 dB re mix), % of the row's windows; ``leak_db`` =
energy mean of the played slot (slot + gain, floored at -30) over those other-only windows, dB re mix. 95 % CIs: block
bootstrap (blocks of 4 windows, 1000 draws).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hearing.tse import fixed_gate
from hearing.tse import learned_gate as lg

HOP_S = 0.5
MUTED_DB = -10.0
AUDIBLE_DB = -15.0
LEAK_FLOOR_DB = -30.0
F = {n: i for i, n in enumerate(("mix_dbfs", "slot0_db", "slot1_db", "s0_cos_aryan", "s0_cos_joseph", "s1_cos_aryan",
                                 "s1_cos_joseph", "mix_cos_aryan", "mix_cos_joseph"))}
ROLES = ("aryan", "joseph")


def sequences(split: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Sequence id per window: train/val renders are 3 consecutive windows each; a holdout scene is one sequence."""
    seq = np.zeros(len(split), dtype=np.int64)
    base = 0
    for sp in ("train", "val"):
        idx = np.flatnonzero(split == sp)
        seq[idx] = base + np.arange(len(idx)) // 3
        base = int(seq[idx].max()) + 1 if len(idx) else base
    for kd in sorted(set(kind[split == "holdout"])):
        seq[(split == "holdout") & (kind == kd)] = base
        base += 1
    return seq


def role_view(f: np.ndarray, lab: np.ndarray, role: int) -> dict:
    """Per-window target-slot / partner-slot quantities for one role."""
    o = 1 - role
    c = lambda slot, who: f[:, F[f"s{slot}_cos_{ROLES[who]}"]]
    return {
        "mix_dbfs": f[:, F["mix_dbfs"]], "t_db": f[:, F[f"slot{role}_db"]], "p_db": f[:, F[f"slot{o}_db"]],
        "ct_t": c(role, role), "ct_p": c(role, o), "cp_t": c(o, role), "cp_p": c(o, o),
        "cm_t": f[:, F[f"mix_cos_{ROLES[role]}"]], "cm_p": f[:, F[f"mix_cos_{ROLES[o]}"]],
        "tgt": lab[:, role], "par": lab[:, o], "oth": lab[:, 2],
    }


def design(v: dict, seq: np.ndarray, idx: np.ndarray) -> np.ndarray:
    rows, prev, prev_seq = [], None, None
    for i in idx:
        p = prev if seq[i] == prev_seq else None
        x = lg.window_row(*(v[k][i] for k in ("mix_dbfs", "t_db", "p_db", "ct_t", "ct_p", "cp_t", "cp_p", "cm_t",
                                              "cm_p")), prev=p)
        rows.append(x)
        prev, prev_seq = x, seq[i]
    return np.asarray(rows)


def labels(v: dict, idx: np.ndarray) -> tuple:
    tgt, other = v["tgt"][idx], np.maximum(v["par"][idx], v["oth"][idx])
    return (tgt >= lg.ACTIVE_DB).astype(float), ((tgt < lg.SILENT_DB) & (other >= lg.ACTIVE_DB)).astype(float)


def fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Weighted L2 logistic regression by Newton steps; ``x`` already standardised, bias appended here."""
    z = np.concatenate([x, np.ones((len(x), 1))], 1)
    beta = np.zeros(z.shape[1])
    reg = np.full(z.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = lg.sigmoid(z @ beta)
        g = z.T @ (w * (p - y)) + reg * beta
        h = (z * (w * p * (1 - p))[:, None]).T @ z + np.diag(reg + 1e-9)
        step = np.linalg.solve(h, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def lda_map(views: list, idx: np.ndarray) -> list:
    """Per role: (mid, half) so that (contrast - mid) / half is +1 on target-solo slots, -1 on partner-solo slots."""
    out = []
    for v in views:
        solo_t = idx[(v["tgt"][idx] >= lg.ACTIVE_DB) & (v["par"][idx] < lg.SILENT_DB)]
        solo_p = idx[(v["par"][idx] >= lg.ACTIVE_DB) & (v["tgt"][idx] < lg.SILENT_DB)]
        m_pos = float(np.mean(v["ct_t"][solo_t] - v["ct_p"][solo_t]))  # target slot, target talking
        m_neg = float(np.mean(v["cp_t"][solo_p] - v["cp_p"][solo_p]))  # partner slot, partner talking
        out.append(((m_pos + m_neg) / 2.0, max((m_pos - m_neg) / 2.0, 1e-6)))
    return out


def run_arm(arm: str, v: dict, idx: np.ndarray, seq: np.ndarray, lda: tuple, head=None, policy=None) -> tuple:
    """Per window of ``idx``: (gain dB, played slot dB re mix)."""
    mid, half = lda
    gains, played = [], []
    fg = fixed_gate.FixedGate() if arm == "arm5" else None
    gate = lg.LearnedGate(head) if arm == "learned" else None
    if gate is not None and policy is not None:
        gate.head.policy = policy
    prev_seq = None
    for i in idx:
        if seq[i] != prev_seq:
            if fg is not None:
                fg.reset()
            if gate is not None:
                gate.reset()
            prev_seq = seq[i]
        t_db, p_db = float(v["t_db"][i]), float(v["p_db"][i])
        s_t = (v["ct_t"][i] - v["ct_p"][i] - mid) / half
        s_p = (v["cp_t"][i] - v["cp_p"][i] - mid) / half
        if arm == "learned":
            g = gate.step(*(v[k][i] for k in ("mix_dbfs", "t_db", "p_db", "ct_t", "ct_p", "cp_t", "cp_p", "cm_t",
                                              "cm_p")))
            gains.append(g)
            played.append(t_db)
            continue
        gated = s_t < 0.0
        slot_db = t_db
        if fg is not None:
            # fixed_gate.features works on waveforms; feed the same quantities through its row layout
            x = np.array([np.clip(t_db - p_db, -40, 40), np.clip(t_db, -40, 40), np.clip(p_db, -40, 40),
                          v["ct_t"][i], v["ct_p"][i], 1.0])
            p = fixed_gate.softmax_probs(fg.weights, x)
            fg.x, fg.swapped, fg.probs = x, False, [float(u) for u in p]
            pt = float(p[0])
            fg.stranger = bool(x[1] >= fixed_gate.STRANGER_ACTIVE_DB and max(v["ct_t"][i], v["ct_p"][i])
                               < fixed_gate.STRANGER_COS)
            if fg.stranger:
                fg.open, fg._below = False, 0
            elif fg.open:
                fg._below = 0 if pt >= fixed_gate.CLOSE_P else fg._below + 1
                fg.open = fg._below <= fixed_gate.HOLD_WINDOWS
            else:
                fg.open, fg._below = pt >= fixed_gate.OPEN_P, 0
            if gated and fg.open and s_p - s_t <= fixed_gate.RESCUE_MAX_PARTNER_LEAD:
                gated = False
            elif gated and fg.swap(s_p):
                gated, slot_db = False, p_db
        gains.append(lg.FLOOR_DB if gated else 0.0)
        played.append(slot_db)
    return np.asarray(gains), np.asarray(played)


def window_scores(v: dict, idx: np.ndarray, gains: np.ndarray, played: np.ndarray) -> tuple:
    y_t, y_l = labels(v, idx)
    out = played + gains
    muted = (y_t > 0) & (gains <= MUTED_DB)
    wrong = (y_l > 0) & (out >= AUDIBLE_DB)
    leak = np.where(y_l > 0, 10.0 ** (np.maximum(out, LEAK_FLOOR_DB) / 10.0), np.nan)
    return muted.astype(float), wrong.astype(float), leak


def summarize(muted, wrong, leak) -> dict:
    lk = leak[~np.isnan(leak)]
    return {"muted_s": float(muted.sum() * HOP_S), "wrong_pct": float(100.0 * wrong.mean()),
            "leak_db": float(10 * np.log10(lk.mean())) if len(lk) else None}


def bootstrap(muted, wrong, leak, rng, draws=1000, block=4) -> dict:
    n = len(muted)
    starts = np.arange(0, max(n - block + 1, 1))
    nb = int(np.ceil(n / block))
    vals = {"muted_s": [], "wrong_pct": [], "leak_db": []}
    for _ in range(draws):
        ii = np.concatenate([np.arange(s, min(s + block, n)) for s in rng.choice(starts, nb)])[:n]
        s = summarize(muted[ii], wrong[ii], leak[ii])
        for k, u in vals.items():
            if s[k] is not None:
                u.append(s[k])
    return {k: [float(np.percentile(u, 2.5)), float(np.percentile(u, 97.5))] if u else None for k, u in vals.items()}


def tune_policy(views, seqs_idx, seq, ldas, head, arm4_cost, wrong_cost=4.0) -> dict:
    """Grid on the val split: fewest muted windows with wrong windows at most arm 4's; cost = muted + ``wrong_cost`` x wrong."""
    best = None
    for lw in (0.5, 1.0, 2.0):
        for lo in (0.0, 0.1, 0.2, 0.3, 0.4):
            for span in (0.2, 0.4):
                for hyst in (0.0, 0.1, 0.2):
                    for lc in (0.3, 0.4, 0.5, 0.7, 0.9):
                        pol = {"leak_weight": lw, "lo": lo, "hi": lo + span, "hyst": hyst, "leak_close": lc}
                        m = w = 0.0
                        for r, v in enumerate(views):
                            g, pl = run_arm("learned", v, seqs_idx, seq, ldas[r], head, pol)
                            mu, wr, _ = window_scores(v, seqs_idx, g, pl)
                            m, w = m + mu.sum(), w + wr.sum()
                        if w > arm4_cost[1]:
                            continue
                        cost = m + wrong_cost * w
                        if best is None or cost < best[0]:
                            best = (cost, pol, m, w)
    if best is None:
        return {"policy": dict(lg.POLICY), "val_muted_windows": None, "val_wrong_windows": None}
    return {"policy": best[1], "val_muted_windows": best[2], "val_wrong_windows": best[3]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leak-cost", type=float, default=3.0, help="class weight of leak positives (partner-in cost)")
    ap.add_argument("--wrong-cost", type=float, default=4.0, help="val tuning: one wrong window costs this many muted")
    ap.add_argument("--seed", type=int, default=407)
    args = ap.parse_args(argv)
    d = np.load(Path(args.features).expanduser())
    f, lab = d["features"].astype(np.float64), d["labels"].astype(np.float64)
    split, kind = d["split"].astype(str), d["kind"].astype(str)
    seq = sequences(split, kind)
    views = [role_view(f, lab, r) for r in range(2)]
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    ldas = lda_map(views, tr)
    x = np.concatenate([design(v, seq, tr) for v in views])
    yt, yl = (np.concatenate(z) for z in zip(*(labels(v, tr) for v in views), strict=True))
    mean, std = x.mean(0), x.std(0) + 1e-6
    std[-1] = 1.0
    xs = (x - mean) / std
    w_t = fit_logistic(xs, yt, np.ones(len(yt)))
    w_l = fit_logistic(xs, yl, np.where(yl > 0, args.leak_cost, 1.0))
    head = lg.LearnedHead(mean, std, w_t, w_l)
    pt, pl = head.probs(x)
    fit = {"train_rows": len(yt), "target_pos": int(yt.sum()), "leak_pos": int(yl.sum()),
           "target_acc": float(np.mean((pt >= 0.5) == (yt > 0))), "leak_acc": float(np.mean((pl >= 0.5) == (yl > 0)))}
    arm4_val = [0.0, 0.0]
    for r, v in enumerate(views):
        g, p = run_arm("arm4", v, va, seq, ldas[r])
        mu, wr, _ = window_scores(v, va, g, p)
        arm4_val[0] += mu.sum()
        arm4_val[1] += wr.sum()
    tuned = tune_policy(views, va, seq, ldas, head, arm4_val, args.wrong_cost)
    head.policy = tuned["policy"]
    rng = np.random.default_rng(args.seed)
    rows = []
    ho_kinds = sorted({k for k in kind[split == "holdout"]})
    for kd in ho_kinds:
        roles = [ROLES.index(kd.split(":")[0])] if ":" in kd else [0, 1]
        idx = np.flatnonzero((split == "holdout") & (kind == kd))
        for r in roles:
            row = {"scene": kd.split(":")[-1], "target": ROLES[r], "windows": len(idx)}
            for arm in ("arm4", "arm5", "learned"):
                g, p = run_arm(arm, views[r], idx, seq, ldas[r], head if arm == "learned" else None,
                               head.policy if arm == "learned" else None)
                sc = window_scores(views[r], idx, g, p)
                row[arm] = {**summarize(*sc), "ci95": bootstrap(*sc, rng)}
            rows.append(row)
    arm4_val_dict = {"muted_windows": arm4_val[0], "wrong_windows": arm4_val[1]}
    res = {"features": str(args.features), "ckpt": str(d["ckpt"]), "fit": fit, "lda_proxy": ldas,
           "val": {"arm4": arm4_val_dict, **tuned}, "head": head.to_dict(), "rows": rows}
    Path(args.out).expanduser().write_text(json.dumps(res, indent=1))
    print(json.dumps({"fit": fit, "val": res["val"]}, indent=1))
    print(table(rows))


def fmt(a: dict, k: str) -> str:
    v, ci = a[k], a["ci95"][k]
    if v is None:
        return "-"
    return f"{v:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else f"{v:.1f}"


def table(rows: list) -> str:
    head = ("| scene | target | arm 4 muted s | arm 5 muted s | learned muted s | arm 4 wrong % | arm 5 wrong % | "
            "learned wrong % | arm 4 leak dB | arm 5 leak dB | learned leak dB |")
    out = [head, "|" + "---|" * 11]
    for r in rows:
        cells = [r["scene"], r["target"]] + [fmt(r[a], k) for k in ("muted_s", "wrong_pct", "leak_db")
                                              for a in ("arm4", "arm5", "learned")]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
