"""Forensic audit of the T-273 TeamProjectionHead ("Gemini's 97.5%"), T-341.

Two parts, both CPU:

1. ``pilot``: Gemini's own split (``eval/projection_head_pilot.py``): train on 0.5 s crops of
   ``profiles/<name>/profile-01.wav``, test on 1.0 s crops of ``team/<name>/fixture-01.wav``,
   2-way nearest prototype (mean of the first 10 train crops). The fixture is the same voice memo as
   the profile (Aryan: seconds 50-108.7 of it), so this split is same-session, same-mic.
2. ``scenes``: the t141 team scenes (``$HEARING_FIXTURES_DIR/t141/<target>/demo-team-{easy,hard}``),
   the mix separated per 1.0 s window (0.5 s hop, 59 windows, as T-336's offline test) by the live
   ``BlindSeparator``; every scorer ranks the same two streams per window. Target-absent = the same
   scene without the target stem (other teammate + babble); unknown = babble only. Thresholds are set
   on ``demo-team-same`` (dev) at <= 5 % target-absent false accepts, never on test.

    .venv/bin/python -m eval.team_head_forensics extract --out $TMPDIR/T-341-cache.npz
    .venv/bin/python -m eval.team_head_forensics score --cache $TMPDIR/T-341-cache.npz --out eval/results/T-341_forensics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
WIN_S, HOP_S = 1.0, 0.5
NAMES = ("aryan", "joseph")
TEST_SCENES = ("demo-team-easy", "demo-team-hard")
DEV_SCENE = "demo-team-same"
GEMINI_PT = Path(os.environ.get("HEARING_MODELS_DIR", "models")) / "team_projection_head.pt"
DEV_FA = 0.05


def fixtures() -> Path:
    return Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def rms_db(x: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-9)


# ---------------------------------------------------------------- extraction (ECAPA + separator)


def extract(out: Path) -> None:
    from eval.projection_head_pilot import slice_audio
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enrolled import _load_encoder, _unit_embedding

    enc = _load_encoder()
    sep = BlindSeparator()
    sep.load()
    fx = fixtures()
    data: dict[str, np.ndarray] = {}
    manifest: dict[str, str] = {}

    def emb(a):
        return _unit_embedding(enc, np.ascontiguousarray(a, dtype=np.float32))

    for name in NAMES:
        prof = fx / "profiles" / name / "profile-01.wav"
        fixt = fx / "team" / name / "fixture-01.wav"
        manifest[str(prof)], manifest[str(fixt)] = sha(prof), sha(fixt)
        p, _ = sf.read(prof, dtype="float32")
        f, _ = sf.read(fixt, dtype="float32")
        data[f"train/{name}"] = np.array([emb(c) for c in slice_audio(p, chunk_s=0.5, step_s=0.15)[:120]])
        data[f"pilot_test/{name}"] = np.array([emb(c) for c in slice_audio(f, chunk_s=1.0, step_s=1.0)[:60]])
        data[f"enroll/{name}"] = emb(p)[None]
        print(name, "train", len(data[f"train/{name}"]), "pilot test", len(data[f"pilot_test/{name}"]), flush=True)

    n, hop = int(WIN_S * SR), int(HOP_S * SR)
    for target in NAMES:
        for scene in (*TEST_SCENES, DEV_SCENE):
            d = fx / "t141" / target / scene / "stems"
            st = {k: sf.read(d / f"{k}.wav", dtype="float32")[0] for k in ("mix", "target", "other", "noise")}
            for k in st:
                manifest[str(d / f"{k}.wav")] = sha(d / f"{k}.wav")
            variants = {"present": st["mix"], "absent": st["other"] + st["noise"], "unknown": st["noise"]}
            for var, x in variants.items():
                embs, lab, tdb, odb = [], [], [], []
                t0 = time.perf_counter()
                for s in range(0, len(x) - n + 1, hop):
                    streams = sep.separate(x[s : s + n], SR)
                    embs.append([emb(z) for z in streams[:2]])
                    tw, ow = st["target"][s : s + n], st["other"][s : s + n]
                    corr = [abs(float(np.corrcoef(z[:n], tw)[0, 1])) if np.std(tw) > 1e-6 else 0.0 for z in streams[:2]]
                    lab.append(int(np.argmax(corr)))
                    tdb.append(rms_db(tw))
                    odb.append(rms_db(ow))
                key = f"scene/{target}/{scene}/{var}"
                data[key + "/emb"] = np.array(embs)
                data[key + "/target_stream"] = np.array(lab)
                data[key + "/target_db"] = np.array(tdb)
                data[key + "/other_db"] = np.array(odb)
                print(key, len(embs), f"{time.perf_counter() - t0:.1f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out)


# ---------------------------------------------------------------- scorers


def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def head_fn(state_dict=None, model=None):
    import torch

    from eval.projection_head_pilot import TeamProjectionHead

    if model is None:
        model = TeamProjectionHead()
        model.load_state_dict(state_dict)
    model.eval()

    def f(e):
        with torch.no_grad():
            return model(torch.from_numpy(np.asarray(e, np.float32))).numpy().astype(np.float64)

    return f, model


def train_linear_head(xa, xj, seed):
    """Ablation D1: the T-273 contrastive objective on a single Linear(192, 32) (no LayerNorm/GELU)."""
    import torch
    from torch import nn, optim

    from eval.projection_head_pilot import ContrastiveProjectionLoss

    torch.manual_seed(seed)
    lin = nn.Linear(192, 32)
    x = torch.from_numpy(np.concatenate([xa, xj]).astype(np.float32))
    y = torch.cat([torch.zeros(len(xa), dtype=torch.long), torch.ones(len(xj), dtype=torch.long)])
    crit, opt = ContrastiveProjectionLoss(), optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(120):
        opt.zero_grad()
        loss = crit(nn.functional.normalize(lin(x), dim=-1), y)
        loss.backward()
        opt.step()

    def f(e):
        with torch.no_grad():
            return nn.functional.normalize(lin(torch.from_numpy(np.asarray(e, np.float32))), dim=-1).numpy().astype(np.float64)

    return f


def build_scorers(c: dict, seeds=(0, 1, 2, 3, 4)) -> dict:
    """name -> score(emb[..., 192], target) -> float array; higher = more like ``target``."""
    import torch

    from eval.projection_head_pilot import train_projection_head
    from hearing.tse.enrolled import fit_lda

    tr = {k: c[f"train/{k}"] for k in NAMES}
    enroll = {k: c[f"enroll/{k}"][0] for k in NAMES}
    proto10 = {k: _norm(tr[k][:10].mean(0)) for k in NAMES}  # T-273 scoring prototype
    other = {"aryan": "joseph", "joseph": "aryan"}
    sc: dict = {}

    sc["A raw ECAPA cos(enroll)"] = lambda e, t: e @ enroll[t]
    sc["A2 raw ECAPA cos(proto10)"] = lambda e, t: e @ proto10[t]
    sc["A3 raw ECAPA closed-set cos(t)-cos(o)"] = lambda e, t: e @ enroll[t] - e @ enroll[other[t]]
    mu = np.concatenate([tr["aryan"], tr["joseph"]]).mean(0)
    sc["D3 centred ECAPA cos(enroll)"] = lambda e, t: _norm(e - mu) @ _norm(enroll[t] - mu)

    lda = fit_lda(tr["aryan"], tr["joseph"])
    if lda is not None:
        w, b = lda
        sc["D4 LDA (prod fit_lda, train crops)"] = lambda e, t: (e @ w - b) * (1 if t == "aryan" else -1)

    def head_scorers(label, fns):
        def s(e, t, fns=fns, proto=True):
            vals = []
            for f in fns:
                ref = f(proto10[t][None])[0] if proto else f(enroll[t][None])[0]
                vals.append(f(e.reshape(-1, 192)).reshape(*e.shape[:-1], -1) @ ref)
            return np.mean(vals, axis=0)

        sc[f"{label} cos(proto10)"] = lambda e, t: s(e, t, proto=True)
        sc[f"{label} cos(enroll)"] = lambda e, t: s(e, t, proto=False)

    if GEMINI_PT.exists():
        f, _ = head_fn(state_dict=torch.load(GEMINI_PT, map_location="cpu"))
        head_scorers("B on-disk team_projection_head.pt", [f])
    for loss in ("contrastive", "arcface"):
        fns = []
        for s in seeds:
            torch.manual_seed(s)
            m, _ = train_projection_head(tr["aryan"].astype(np.float32), tr["joseph"].astype(np.float32), loss_type=loss)
            fns.append(head_fn(model=m)[0])
        head_scorers(f"B' T-273 head {loss} (5 seeds, mean score)", fns)
        if loss == "contrastive":
            head_scorers("B' T-273 head contrastive seed0", fns[:1])
    lin = [train_linear_head(tr["aryan"], tr["joseph"], s) for s in seeds]
    head_scorers("D1 linear 192->32, contrastive (no LN/GELU)", lin)
    return sc


# ---------------------------------------------------------------- evaluation


def pilot_eval(c: dict, sc: dict) -> dict:
    """Gemini's own number: 2-way nearest prototype on the same-session fixture crops."""
    out = {}
    for name, f in sc.items():
        if "proto10" not in name and not name.startswith("D4") and not name.startswith("A3"):
            continue
        err = n = 0
        for t in NAMES:
            x = c[f"pilot_test/{t}"]
            o = "joseph" if t == "aryan" else "aryan"
            st, so = f(x, t), f(x, o)
            if name.startswith(("D4", "A3")):
                err += int(np.sum(st <= 0))
            else:
                err += int(np.sum(so > st))
            n += len(x)
        out[name] = {"acc": round(1 - err / n, 4), "errors": err, "n": n, "ci95": wilson(n - err, n)}
    return out


def scene_rows(c, f, target, scene):
    """Per window: score of the target stream, of the other stream, and the windows' labels."""
    k = f"scene/{target}/{scene}"
    pres = c[k + "/present/emb"]
    s = f(pres, target)  # (W, 2)
    lab = c[k + "/present/target_stream"]
    tdb, odb = c[k + "/present/target_db"], c[k + "/present/other_db"]
    active = tdb > -45.0
    absent = f(c[k + "/absent/emb"], target).max(1)
    unknown = f(c[k + "/unknown/emb"], target).max(1)
    return s, lab, active, tdb - odb, absent, unknown


def evaluate(c: dict, sc: dict) -> dict:
    res = {}
    for name, f in sc.items():
        # dev threshold: <= 5 % target-absent windows accepted on demo-team-same (both targets)
        dev_abs = np.concatenate([scene_rows(c, f, t, DEV_SCENE)[4] for t in NAMES])
        thr = float(np.quantile(dev_abs, 1 - DEV_FA))
        agg = {"top1": [0, 0], "ta": [0, 0], "wrong": [0, 0], "fa_absent": [0, 0], "fa_unknown": [0, 0],
               "quiet_top1": [0, 0]}
        margins, per = [], {}
        t0 = time.perf_counter()
        for t in NAMES:
            for scene in TEST_SCENES:
                s, lab, act, rel, absent, unknown = scene_rows(c, f, t, scene)
                idx = np.arange(len(lab))
                st, so = s[idx, lab], s[idx, 1 - lab]
                pick = s.argmax(1)
                right = (pick == lab) & act
                acc = s.max(1) >= thr
                quiet = act & (rel < 0)
                row = {
                    "windows": len(lab), "target_windows": int(act.sum()),
                    "top1": int(right.sum()), "accept_right": int((right & acc).sum()),
                    "accept_wrong": int(((pick != lab) & act & acc).sum()),
                    "absent_accept": int((absent >= thr).sum()), "absent_windows": len(absent),
                    "unknown_accept": int((unknown >= thr).sum()),
                }
                per[f"{t}/{scene}"] = row
                agg["top1"][0] += row["top1"]
                agg["top1"][1] += row["target_windows"]
                agg["ta"][0] += row["accept_right"]
                agg["ta"][1] += row["target_windows"]
                agg["wrong"][0] += row["accept_wrong"]
                agg["wrong"][1] += row["target_windows"]
                agg["fa_absent"][0] += row["absent_accept"]
                agg["fa_absent"][1] += len(absent)
                agg["fa_unknown"][0] += row["unknown_accept"]
                agg["fa_unknown"][1] += len(unknown)
                agg["quiet_top1"][0] += int((right & quiet).sum())
                agg["quiet_top1"][1] += int(quiet.sum())
                margins.append((st - so)[act])
        m = np.concatenate(margins)
        res[name] = {
            "threshold_dev": round(thr, 4),
            **{k: {"k": v[0], "n": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None, "ci95": wilson(*v)}
               for k, v in agg.items()},
            "margin_mean": round(float(m.mean()), 4), "margin_std": round(float(m.std()), 4),
            "margin_dprime": round(float(m.mean() / (m.std() + 1e-9)), 3),
            "margin_ci95": [round(float(m.mean() - 1.96 * m.std() / math.sqrt(len(m))), 4),
                            round(float(m.mean() + 1.96 * m.std() / math.sqrt(len(m))), 4)],
            "score_s": round(time.perf_counter() - t0, 3),
            "per_scene": per,
        }
    return res


def head_latency_us(iters: int = 2000) -> float:
    import torch

    from eval.projection_head_pilot import TeamProjectionHead

    m = TeamProjectionHead().eval()
    x = torch.randn(2, 192)
    with torch.no_grad():
        for _ in range(50):
            m(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x)
    return (time.perf_counter() - t0) / iters * 1e6


def score(cache: Path, out: Path) -> None:
    c = dict(np.load(cache))
    sc = build_scorers(c)
    report = {
        "task": "T-341",
        "cache": str(cache), "cache_sha256": sha(cache),
        "gemini_pt": str(GEMINI_PT), "gemini_pt_sha256": sha(GEMINI_PT) if GEMINI_PT.exists() else None,
        "windowing": {"win_s": WIN_S, "hop_s": HOP_S, "target_active_db": -45.0, "dev_scene": DEV_SCENE,
                      "dev_fa": DEV_FA},
        "pilot_same_session": pilot_eval(c, sc),
        "scenes": evaluate(c, sc),
        "head_latency_us_2streams_cpu": round(head_latency_us(), 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"{'scorer':58s} {'pilot':>6s} | {'top1':>12s} {'TA':>6s} {'wrong':>6s} {'FAabs':>6s} {'FAunk':>6s} {'quiet':>9s} {'dprime':>6s}")
    for name, r in report["scenes"].items():
        p = report["pilot_same_session"].get(name, {}).get("acc")
        print(f"{name:58s} {p if p is not None else '-':>6} | {r['top1']['k']:>3}/{r['top1']['n']:<3} {r['top1']['rate']:.3f} "
              f"{r['ta']['rate']:.3f} {r['wrong']['rate']:.3f} {r['fa_absent']['rate']:.3f} {r['fa_unknown']['rate']:.3f} "
              f"{r['quiet_top1']['k']:>3}/{r['quiet_top1']['n']:<3} {r['margin_dprime']:>6}")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["extract", "score"])
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "T-341-cache.npz")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.out or a.cache)
    else:
        score(a.cache, a.out or Path("eval/results/T-341_forensics.json"))


if __name__ == "__main__":
    main()
