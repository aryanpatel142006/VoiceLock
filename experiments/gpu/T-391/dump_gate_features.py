"""Dump per-window gate features of a fine-tuned fixed-order separator (T-391, accuracy only).

For a later gate (does slot 0 / slot 1 hold Aryan / Joseph right now, or a stranger, or
nothing?), every 1.0 s window (0.5 s hop, as the live enrolled tier) is separated alone, put on
the mixture's scale (shared gain) and described by:

    features[:, 0]    mixture level, dBFS
    features[:, 1:3]  slot 0 / slot 1 energy over mixture energy, dB
    features[:, 3:5]  ECAPA cosine of slot 0 to the Aryan / Joseph profile (profile-01)
    features[:, 5:7]  ECAPA cosine of slot 1 to the Aryan / Joseph profile
    features[:, 7:9]  ECAPA cosine of the mixture to the Aryan / Joseph profile

with labels ``labels[:, 0:3]`` = Aryan / Joseph / other-talker (stranger or babble-free third
voice) reference energy over the mixture energy in dB (-100 when silent). ``split`` is
train / val / holdout; ``kind`` the render kind (train/val, ``train_sep.render`` with the v3
recipe) or the eval scene name (holdout, ``eval_sep.scene``). Train and val read only their
own takes and voices; holdout rows read holdout-01/02 and the test voices, so a gate trained
on train/val can be scored on them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_sep
from eval_sep import HOP, WIN, Ecapa, scene, stranger_voice
from train_sep import CROP, SR, Pool, load, network, render, rms, separate

SILENT_DB = -100.0


def db(x, ref):
    e = float(x @ x)
    return SILENT_DB if e < 1e-10 else 10 * np.log10(e / (float(ref @ ref) + 1e-12))


def window_features(net, ecapa, profs, m, dev):
    with torch.inference_mode():
        est = (
            separate(net, torch.from_numpy(m)[None].to(dev))[0]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    total = est.sum(axis=0)
    est *= float(m @ total) / (float(total @ total) + 1e-12)
    cos = [[float(ecapa(x.astype(np.float32)) @ p) for p in profs] for x in (*est, m)]
    return [
        20 * np.log10(rms(m) + 1e-12),
        db(est[0], m),
        db(est[1], m),
        *cos[0],
        *cos[1],
        *cos[2],
    ]


def windows(n):
    return range(0, n - WIN + 1, HOP)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--team", default=str(Path.home() / "hearing-data/team"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--ecapa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-items", type=int, default=2000)
    ap.add_argument("--val-items", type=int, default=400)
    ap.add_argument("--sir", type=float, nargs=2, default=(-10.0, 10.0))
    ap.add_argument("--seed", type=int, default=391)
    args = ap.parse_args(argv)
    train_sep.R = train_sep.RECIPES["v3"]
    train_sep.SIR_RANGE = tuple(args.sir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = network(args.ckpt).to(dev).eval()
    ecapa = Ecapa(args.ecapa, dev)
    team = Path(args.team)
    profs = [ecapa(load(team / who / "profile-01.wav")) for who in ("aryan", "joseph")]
    feats, labels, split, kind = [], [], [], []

    def add(m, a, j, o, sp, kd):
        feats.append(window_features(net, ecapa, profs, m, dev))
        labels.append([db(a, m), db(j, m), db(o, m)])
        split.append(sp)
        kind.append(kd)

    rng = np.random.default_rng(args.seed)
    for sp, n_items in (("train", args.train_items), ("val", args.val_items)):
        pool = Pool(args.team, args.data, sp)
        for i in range(n_items):
            parts = {}
            mix, refs, kd = render(pool, rng, parts=parts)
            for s in windows(CROP):
                w = slice(s, s + WIN)
                add(mix[w], refs[0][w], refs[1][w], parts["talker"][w], sp, kd)
            if i % 500 == 0:
                print(sp, i, len(feats), flush=True)
    take = lambda who: np.concatenate(
        [load(team / who / f"{n}.wav") for n in ("holdout-01", "holdout-02")]
    )
    ar, jo = take("aryan"), take("joseph")
    for name, sir, snr in (
        ("sir0-cafe", 0.0, 5.0),
        ("sir-6-cafe", -6.0, 5.0),
        ("sir0-none", 0.0, None),
    ):
        for tgt_name in ("aryan", "joseph"):
            t_in, o_in = (ar, jo) if tgt_name == "aryan" else (jo, ar)
            mix, t, o = scene(t_in, o_in, args.data, snr, sir)
            a, j = (t, o) if tgt_name == "aryan" else (o, t)
            for s in windows(len(mix)):
                w = slice(s, s + WIN)
                add(
                    mix[w],
                    a[w],
                    j[w],
                    np.zeros(WIN, np.float32),
                    "holdout",
                    f"{tgt_name}:{name}",
                )
    for tgt_name, t_in in (("aryan", ar), ("joseph", jo)):
        mix, t, o = scene(
            t_in, stranger_voice(args.data, len(t_in)), args.data, 5.0, -6.0
        )
        z = np.zeros_like(t)
        a, j = (t, z) if tgt_name == "aryan" else (z, t)
        for s in windows(len(mix)):
            w = slice(s, s + WIN)
            add(mix[w], a[w], j[w], o[w], "holdout", f"{tgt_name}:stranger-sir-6-cafe")
    st = stranger_voice(args.data, 60 * SR)
    mix, o, _ = scene(st, None, args.data, 5.0, 0.0)
    z = np.zeros_like(mix)
    for s in windows(len(mix)):
        w = slice(s, s + WIN)
        add(mix[w], z[w], z[w], o[w], "holdout", "stranger-only-cafe")
    np.savez_compressed(
        args.out,
        features=np.asarray(feats, np.float32),
        labels=np.asarray(labels, np.float32),
        split=np.asarray(split),
        kind=np.asarray(kind),
        feature_names=np.asarray(
            [
                "mix_dbfs",
                "slot0_db",
                "slot1_db",
                "s0_cos_aryan",
                "s0_cos_joseph",
                "s1_cos_aryan",
                "s1_cos_joseph",
                "mix_cos_aryan",
                "mix_cos_joseph",
            ]
        ),
        label_names=np.asarray(["aryan_db", "joseph_db", "other_db"]),
        ckpt=np.asarray(args.ckpt),
    )
    print("wrote", args.out, len(feats), "windows", flush=True)


if __name__ == "__main__":
    main()
