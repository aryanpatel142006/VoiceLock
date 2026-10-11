"""T-249 offline prototype: calibrated target-selection confidence + accept/hold/abstain policy.

Synthetic score traces only (no audio, no weights, no production thresholds). numpy only.
CLI: python experiments/theory/T-249/selection_calibration.py --out DIR [--seed 249] [--max-cases 400]
Writes DIR/result.json atomically; a verified result with the same config hash is reused (resume).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SCHEMA = 1
HOP_S = 0.1
SCENARIOS = ("target_present", "target_absent", "similar_voice_absent", "pause_hijack", "louder_competitor")
# (mean, std) of the raw cosine to the target profile per frame class. Assumed, not measured.
CLASSES = {
    "target": (0.60, 0.12),
    "target_masked": (0.42, 0.15),   # louder competitor dilutes the target embedding
    "other": (0.18, 0.12),
    "similar_other": (0.38, 0.12),   # hard negative (saved negative semantics: old clip)
    "silence": (0.05, 0.20),
}
DEFAULT = dict(n_groups=24, frames=120, dev_frac=0.5, seed=249, max_cases=400,
               policy=dict(accept=0.7, release=0.4, dwell_frames=5, hold_frames=20, switch_cost=2.0))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# ---------------- trace generation ----------------
def make_trace(rng, scenario: str, frames: int, group_bias: float):
    """Return (scores, is_target_speaking) per frame. group_bias shifts a source group's scores."""
    cls = []
    if scenario == "target_present":
        cls = ["target"] * frames
    elif scenario == "target_absent":
        cls = ["other"] * frames
    elif scenario == "similar_voice_absent":
        cls = ["similar_other"] * frames
    elif scenario == "pause_hijack":  # target talks, pauses (silence then competitor), resumes
        a, b = frames // 3, 2 * frames // 3
        cls = ["target"] * a + ["silence"] * 10 + ["similar_other"] * (b - a - 10) + ["target"] * (frames - b)
    elif scenario == "louder_competitor":  # target speaks masked while a louder other talks
        cls = ["target_masked"] * frames
    truth = np.array([c.startswith("target") for c in cls])
    mu = np.array([CLASSES[c][0] for c in cls])
    sd = np.array([CLASSES[c][1] for c in cls])
    return np.clip(mu + group_bias + sd * rng.standard_normal(len(cls)), -1, 1), truth, cls


def make_dataset(cfg):
    rng = np.random.default_rng(cfg["seed"])
    per = 1  # one trace per (group, scenario); the case cap bounds n_groups below
    n_groups = min(cfg["n_groups"], max(2, cfg["max_cases"] // len(SCENARIOS)))
    rows = []
    n_dev = int(n_groups * cfg["dev_frac"])
    for g in range(n_groups):
        bias = float(rng.normal(0, 0.04))
        for sc in SCENARIOS:
            for _ in range(per):
                s, t, c = make_trace(rng, sc, cfg["frames"], bias)
                rows.append(dict(group=g, split="dev" if g < n_dev else "holdout", scenario=sc, s=s, truth=t, cls=c))
    return rows


# ---------------- calibrators (fit on dev only) ----------------
def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def fit_platt(s, y, iters=300):
    a, b = 8.0, -4.0
    for _ in range(iters):  # Newton on logistic loss
        p = _sig(a * s + b)
        w = np.maximum(p * (1 - p), 1e-6)
        X = np.stack([s, np.ones_like(s)], 1)
        g = X.T @ (p - y) + 1e-4 * np.array([a, b])
        H = (X * w[:, None]).T @ X + 1e-4 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return lambda x: _sig(a * x + b), dict(a=float(a), b=float(b))


def fit_temperature(s, y, center=0.4):
    """One-parameter: p = sigmoid((s-center)/T). Grid-searched T keeps the margin semantics fixed."""
    best = min(np.geomspace(0.02, 1.0, 200), key=lambda T: _nll(_sig((s - center) / T), y))
    return lambda x: _sig((x - center) / best), dict(T=float(best), center=center)


def _nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_isotonic(s, y):
    order = np.argsort(s)
    xs, ys = s[order], y[order].astype(float)
    blocks = [[v, 1] for v in ys]  # PAV: [sum, count]
    out = []
    for blk in blocks:
        out.append(blk)
        while len(out) > 1 and out[-2][0] / out[-2][1] >= out[-1][0] / out[-1][1]:
            b2 = out.pop()
            out[-1][0] += b2[0]
            out[-1][1] += b2[1]
    fitted = np.concatenate([[b[0] / b[1]] * b[1] for b in out])
    return (lambda x: np.interp(x, xs, fitted)), dict(n_knots=len(out))


def fit_identity(s, y):
    return (lambda x: np.clip(x, 0, 1)), {}


CALIBRATORS = dict(raw_cosine=fit_identity, temperature=fit_temperature, platt=fit_platt, isotonic=fit_isotonic)


# ---------------- metrics ----------------
def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    curve = []
    for k in range(bins):
        m = idx == k
        if m.any():
            gap = abs(p[m].mean() - y[m].mean())
            tot += m.mean() * gap
            curve.append([k / bins, int(m.sum()), float(p[m].mean()), float(y[m].mean())])
    return float(tot), curve


def brier(p, y):
    return float(((p - y) ** 2).mean())


def coverage_risk(p, y, n=11):
    """Accept frames with p >= t; coverage = accepted share of target frames' ... of all frames; risk = false share of accepted."""
    out = []
    for t in np.linspace(0, 1, n):
        a = p >= t
        out.append(dict(thr=float(t), coverage=float(a.mean()), risk=float((~y[a]).mean()) if a.any() else 0.0))
    return out


# ---------------- policy: accept / hold / abstain ----------------
def run_policy(p, pol, hmm=True):
    """Return per-frame passed(bool). States: 0 reject, 1 pass. Hysteresis accept/release;
    pass opens after dwell_frames consecutive p>=accept; once open it is held hold_frames through p<release
    (pause bridging) so word onsets after a pause are kept. hmm=True: forward filter with switch cost
    (log-odds evidence, symmetric penalty) instead of hard dwell."""
    n = len(p)
    out = np.zeros(n, bool)
    if not hmm:
        run = hold = 0
        st = False
        for i, v in enumerate(p):
            if not st:
                run = run + 1 if v >= pol["accept"] else 0
                if run >= pol["dwell_frames"]:
                    st, hold = True, 0
            else:
                hold = 0 if v >= pol["release"] else hold + 1
                if hold > pol["hold_frames"]:
                    st, run = False, 0
            out[i] = st
        return out
    # Viterbi-style online filter, 2 states, switching cost in nats; dwell cost emerges from evidence accumulation.
    llr = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    llr = np.clip(llr, -2.0, 2.0)  # bound so one frame (e.g. a pause hijack) can't dominate
    score = np.zeros(2)
    c = pol["switch_cost"] * 2
    for i, l in enumerate(llr):
        score = np.array([max(score[0], score[1] - c) - l / 2, max(score[1], score[0] - c) + l / 2])
        score -= score.max()
        out[i] = score[1] >= score[0]
    return out


def policy_eval(rows, cal, pol, hmm):
    fa_frames = fa_total = 0
    keep = keep_total = 0
    hij = hij_total = 0
    per = {}
    for r in rows:
        passed = run_policy(cal(r["s"]), pol, hmm)
        truth = r["truth"]
        keep += int((passed & truth).sum())
        keep_total += int(truth.sum())
        wrong = ~truth & np.array([c in ("other", "similar_other") for c in r["cls"]])
        if r["scenario"] in ("target_absent", "similar_voice_absent"):
            fa_frames += int(passed.sum())
            fa_total += len(passed)
        if r["scenario"] == "pause_hijack":
            hij += int((passed & wrong).sum())
            hij_total += int(wrong.sum())
        per.setdefault(r["scenario"], []).append(float(passed.mean()))
    return dict(
        target_absent_false_accept=fa_frames / max(fa_total, 1),
        target_retention=keep / max(keep_total, 1),
        pause_hijack_wrong_speaker=hij / max(hij_total, 1),
        pass_rate_by_scenario={k: float(np.mean(v)) for k, v in per.items()},
    )


GATES = dict(  # proposed, measurable; apply on holdout, never tuned there
    ece_max=0.05, brier_improve_vs_raw=0.0, target_absent_false_accept_max=0.02,
    target_retention_min=0.85, pause_hijack_wrong_speaker_max=0.15, holdout_dev_ece_gap_max=0.03,
)


def run(cfg: dict) -> dict:
    rows = make_dataset(cfg)
    dev = [r for r in rows if r["split"] == "dev"]
    hold = [r for r in rows if r["split"] == "holdout"]
    assert not ({r["group"] for r in dev} & {r["group"] for r in hold}), "source-group leak"
    sd = np.concatenate([r["s"] for r in dev])
    yd = np.concatenate([r["truth"] for r in dev]).astype(float)
    sh = np.concatenate([r["s"] for r in hold])
    yh = np.concatenate([r["truth"] for r in hold]).astype(float)
    res = dict(schema=SCHEMA, config=cfg, config_hash=config_hash(cfg), counts=dict(
        traces_dev=len(dev), traces_holdout=len(hold), frames_dev=len(sd), frames_holdout=len(sh),
        groups_dev=len({r["group"] for r in dev}), groups_holdout=len({r["group"] for r in hold})),
        calibrators={}, gates=GATES, counterexamples=[])
    pol = cfg["policy"]
    for name, fit in CALIBRATORS.items():
        cal, params = fit(sd, yd)
        e_d, _ = ece(cal(sd), yd)
        e_h, curve = ece(cal(sh), yh)
        entry = dict(params=params, dev=dict(ece=e_d, brier=brier(cal(sd), yd)),
                     holdout=dict(ece=e_h, brier=brier(cal(sh), yh), calibration_curve=curve,
                                  coverage_risk=coverage_risk(cal(sh), yh.astype(bool))))
        # policy thresholds are in probability space (the calibrated value), identical for every calibrator
        entry["policy_hard_dwell"] = policy_eval(hold, cal, pol, hmm=False)
        entry["policy_hmm"] = policy_eval(hold, cal, pol, hmm=True)
        res["calibrators"][name] = entry
    # dwell sweep (dev only) for the platt calibrator
    cal, _ = fit_platt(sd, yd)
    res["dwell_sweep_dev"] = [dict(dwell_frames=d, **policy_eval(dev, cal, {**pol, "dwell_frames": d}, hmm=False))
                              for d in (1, 3, 5, 10)]
    # gate verdicts on holdout, best-ECE calibrator
    best = min((k for k in res["calibrators"] if k != "raw_cosine"), key=lambda k: res["calibrators"][k]["dev"]["ece"])
    c = res["calibrators"][best]
    pm = c["policy_hmm"]
    res["selected_by_dev_ece"] = best
    res["gate_verdicts_holdout"] = dict(
        ece=c["holdout"]["ece"] <= GATES["ece_max"],
        brier_vs_raw=c["holdout"]["brier"] <= res["calibrators"]["raw_cosine"]["holdout"]["brier"] - GATES["brier_improve_vs_raw"],
        target_absent_false_accept=pm["target_absent_false_accept"] <= GATES["target_absent_false_accept_max"],
        target_retention=pm["target_retention"] >= GATES["target_retention_min"],
        pause_hijack=pm["pause_hijack_wrong_speaker"] <= GATES["pause_hijack_wrong_speaker_max"],
        generalisation=abs(c["holdout"]["ece"] - c["dev"]["ece"]) <= GATES["holdout_dev_ece_gap_max"])
    # counterexamples: explicit failures of this model under its own assumptions
    if c["holdout"]["ece"] > c["dev"]["ece"]:
        res["counterexamples"].append("holdout ECE above dev ECE: group bias shifts scores")
    sim = pm["pass_rate_by_scenario"].get("similar_voice_absent", 0)
    if sim > 0.02:
        res["counterexamples"].append(f"similar-voice absent pass rate {sim:.3f}: cosine cannot separate close voices")
    if pm["pause_hijack_wrong_speaker"] > 0:
        res["counterexamples"].append(
            f"pause hijack: wrong speaker passed {pm['pause_hijack_wrong_speaker']:.3f} of competitor frames (hold window bridges the pause)")
    lc = pm["pass_rate_by_scenario"].get("louder_competitor", 1)
    if lc < GATES["target_retention_min"]:
        res["counterexamples"].append(f"louder competitor: target kept {lc:.3f} < gate; masked cosine below release")
    return res


def write_atomic(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=DEFAULT["seed"])
    ap.add_argument("--max-cases", type=int, default=DEFAULT["max_cases"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = {**DEFAULT, "seed": a.seed, "max_cases": a.max_cases}
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / "result.json"
    if f.exists() and not a.force:
        try:
            old = json.loads(f.read_text())
            if old.get("config_hash") == config_hash(cfg) and old.get("status") == "complete":
                print(f"resume: complete result for config {old['config_hash']} kept")
                return 0
        except json.JSONDecodeError:
            pass
    try:
        res = run(cfg)
        res["status"] = "complete"
    except Exception as e:  # deterministic failure status, no retries
        write_atomic(f, dict(schema=SCHEMA, status="failed", error=repr(e), config_hash=config_hash(cfg)))
        return 1
    write_atomic(f, res)
    print(f"wrote {f} hash={res['config_hash']} gates={res['gate_verdicts_holdout']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
