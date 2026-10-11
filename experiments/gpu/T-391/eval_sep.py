"""Held-out window eval for T-374: base vs fine-tuned MossFormer2 checkpoints (accuracy only).

Rows: target in {aryan, joseph} x scene: holdout-01+02 at SIR 0 dB with cafe / without noise
(the four real-pair rows), at SIR -6 dB with cafe, and same-01+02 at SIR 0 dB with cafe. The
target take is mixed with the other person's take of the same name; "cafe" is the test cafe
noise (PCAFETER ch01 240-300 s) plus the test babble voices at SNR 5 dB. None of these
takes, noise seconds or voices is read by train_sep.py. Generality rows use GRID s1/s4.

Live windowing (1.0 s window, 0.5 s hop). Each window is separated, put on the mixture's scale
(shared gain, as ``hearing.tse.blind.rescale_to_mix``), and one stream is picked: ``ecapa``
(the stream with the higher ECAPA cosine to the target's profile, as the enrolled tier picks)
or ``fixed`` (slot 0 = Aryan, 1 = Joseph; team rows only). Per row, with 95 % bootstrap
intervals over windows and window counts:
- wrong %: target-active windows whose pick holds more interferer than target energy
  (projection onto each reference);
- SI-SDRi: picked stream vs target, minus the mixture's, on target-active windows;
- leak dB: picked-stream energy over mixture energy on windows where only the other talks.

T-391 rows: holdout at SIR -10 and +6 dB (cafe), holdout SIR 0 at SNR 0 dB, "stranger" rows
(the target's holdout takes with a test-split LibriSpeech voice 6 dB louder as the other
talker, cafe at 5 dB) and a "stranger-only" row (that voice + cafe, no team member). On those
rows ``<label>/slot<k>_absent`` is the energy of fixed slot k (the team member who is not
there) over the mixture energy on windows where the stranger talks, in dB: the leak of a
stranger into a fixed slot (lower is better; the base net has no slot meaning, so read it for
the fine-tuned ones).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_sep import BABBLE_SPK, CAFE, SR, load, network, rms, separate

WIN, HOP = SR, SR // 2
SLOT = {"aryan": 0, "joseph": 1}


def fit(x, n):
    return np.resize(x, n) if len(x) < n else x[:n]


def scene(target, other, noise_dir, snr_db=5.0, sir_db=0.0, seed=0):
    rng = np.random.default_rng(seed)
    if other is None:  # T-391: one talker only
        other = np.zeros_like(target)
    n = min(len(target), len(other))
    t = target[:n] / rms(target[:n])
    o = other[:n] / rms(other[:n]) * 10 ** (-sir_db / 20) if other.any() else other[:n]
    cafe = load(Path(noise_dir) / "cafe" / "ch01.wav")[
        CAFE["test"][0] * SR : CAFE["test"][1] * SR
    ]
    noise = fit(cafe, n) / rms(cafe)
    for p in sorted((Path(noise_dir) / "babble").glob("*.wav")):
        if p.name.split("-")[0] in BABBLE_SPK["test"] and rng.random() < 0.5:
            v = np.concatenate([load(p), np.zeros(SR // 2, np.float32)])
            noise += fit(np.roll(v, rng.integers(len(v))), n) / rms(v) * 10 ** (-6 / 20)
    noise *= rms(t) / rms(noise) * 10 ** (-snr_db / 20) if snr_db is not None else 0.0
    mix = t + o + noise
    g = 0.1 / rms(mix)
    return (
        (mix * g).astype(np.float32),
        (t * g).astype(np.float32),
        (o * g).astype(np.float32),
    )


def si_sdr(est, ref):
    ref, est = ref - ref.mean(), est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = a * ref
    return 10 * np.log10((proj @ proj + 1e-12) / ((est - proj) @ (est - proj) + 1e-12))


def proj_energy(est, ref):
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    return a * a * float(ref @ ref)


def boot(values, rng, n=1000):
    v = np.asarray(values, np.float64)
    if not v.size:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    means = v[rng.integers(0, v.size, (n, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": round(float(v.mean()), 2),
        "lo": round(float(lo), 2),
        "hi": round(float(hi), 2),
        "n": int(v.size),
    }


class Ecapa:
    def __init__(self, root, dev):
        from speechbrain.inference.speaker import EncoderClassifier

        self.enc = EncoderClassifier.from_hparams(
            source=str(root),
            savedir=str(root),
            overrides={"pretrained_path": str(root)},
            run_opts={"device": str(dev)},
        )
        self.dev = dev

    def __call__(self, x):
        with torch.inference_mode():
            e = self.enc.encode_batch(
                torch.from_numpy(np.ascontiguousarray(x))[None].to(self.dev)
            )
        e = e.reshape(-1).float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-12)


def stranger_voice(data, n):
    """The test-split LibriSpeech speakers' utterances back to back (0.3 s gaps), cut or looped to n."""
    parts = []
    for p in sorted((Path(data) / "babble").glob("*.wav")):
        if p.name.split("-")[0] in BABBLE_SPK["test"]:
            parts += [load(p), np.zeros(int(0.3 * SR), np.float32)]
    return fit(np.concatenate(parts), n)


def evaluate_row(net, ecapa, mix, tgt, oth, profile, picks, slot, dev, rng, absent=()):
    prof = ecapa(profile) if picks and "ecapa" in picks else None
    out = {p: {"wrong": [], "sdri": [], "leak": []} for p in picks}
    gone = {k: [] for k in absent}
    t_floor, o_floor = 0.1 * rms(tgt), 0.1 * rms(oth)
    for start in range(0, len(mix) - WIN + 1, HOP):
        m, t, o = (x[start : start + WIN] for x in (mix, tgt, oth))
        with torch.inference_mode():
            est = (
                separate(net, torch.from_numpy(m)[None].to(dev))[0]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        total = est.sum(axis=0)
        est *= float(m @ total) / (float(total @ total) + 1e-12)
        t_on, o_on = rms(t) > t_floor, rms(o) > o_floor
        if o_on:
            for k in absent:
                gone[k].append(
                    10 * np.log10((est[k] @ est[k] + 1e-12) / (m @ m + 1e-12))
                )
        for pick in picks:
            if pick == "fixed":
                k = slot
            else:
                k = int(
                    np.argmax([float(ecapa(s.astype(np.float32)) @ prof) for s in est])
                )
            y = est[k]
            if t_on:
                out[pick]["wrong"].append(
                    100.0 * (proj_energy(y, o) > proj_energy(y, t))
                )
                out[pick]["sdri"].append(si_sdr(y, t) - si_sdr(m, t))
            elif o_on:
                out[pick]["leak"].append(
                    10 * np.log10((y @ y + 1e-12) / (m @ m + 1e-12))
                )
    result = {
        pick: {
            "wrong_pct": boot(v["wrong"], rng),
            "sdri_db": boot(v["sdri"], rng),
            "leak_db": boot(v["leak"], rng),
        }
        for pick, v in out.items()
    }
    result.update(
        {f"slot{k}_absent": {"absent_db": boot(v, rng)} for k, v in gone.items()}
    )
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="label=path")
    ap.add_argument("--team", default=str(Path.home() / "hearing-data/team"))
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--grid",
        help="dir with s1.wav, s4.wav (test) and s1_profile.wav, s4_profile.wav",
    )
    ap.add_argument("--ecapa", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecapa = Ecapa(args.ecapa, dev)
    team = Path(args.team)
    take = lambda who, names: np.concatenate(
        [load(team / who / f"{n}.wav") for n in names]
    )
    rows = []
    scenes = (  # the four real-pair rows (holdout sir0 cafe/none), sir -6, and the same-sentence takes
        ("holdout-sir0-cafe", ("holdout-01", "holdout-02"), 0.0, 5.0),
        ("holdout-sir0-none", ("holdout-01", "holdout-02"), 0.0, None),
        ("holdout-sir-6-cafe", ("holdout-01", "holdout-02"), -6.0, 5.0),
        ("same-sir0-cafe", ("same-01", "same-02"), 0.0, 5.0),
        # T-391
        ("holdout-sir-10-cafe", ("holdout-01", "holdout-02"), -10.0, 5.0),
        ("holdout-sir+6-cafe", ("holdout-01", "holdout-02"), 6.0, 5.0),
        ("holdout-sir0-snr0", ("holdout-01", "holdout-02"), 0.0, 0.0),
    )
    for name, takes, sir, snr in scenes:
        for tgt_name, oth_name in (("aryan", "joseph"), ("joseph", "aryan")):
            mix, t, o = scene(
                take(tgt_name, takes), take(oth_name, takes), args.data, snr, sir
            )
            rows.append(
                (
                    f"{tgt_name}:{oth_name} {name}",
                    mix,
                    t,
                    o,
                    load(team / tgt_name / "profile-01.wav"),
                    SLOT[tgt_name],
                    ("ecapa", "fixed"),
                    (),
                )
            )
    holdout = ("holdout-01", "holdout-02")
    for tgt_name in (
        "aryan",
        "joseph",
    ):  # T-391: a stranger 6 dB louder than the target, cafe 5 dB
        tgt = take(tgt_name, holdout)
        mix, t, o = scene(
            tgt, stranger_voice(args.data, len(tgt)), args.data, 5.0, -6.0
        )
        rows.append(
            (
                f"{tgt_name}:stranger holdout-sir-6-cafe",
                mix,
                t,
                o,
                load(team / tgt_name / "profile-01.wav"),
                SLOT[tgt_name],
                ("ecapa", "fixed"),
                (1 - SLOT[tgt_name],),
            )
        )
    # T-391: the stranger alone in cafe: both fixed slots should stay silent
    st = stranger_voice(args.data, 60 * SR)
    mix, o, _ = scene(st, None, args.data, 5.0, 0.0)
    rows.append(
        ("stranger-only cafe", mix, np.zeros_like(mix), o, None, None, (), (0, 1))
    )
    if args.grid:
        g = Path(args.grid)
        for a, b in (("s1", "s4"), ("s4", "s1")):
            mix, t, o = scene(load(g / f"{a}.wav"), load(g / f"{b}.wav"), args.data)
            rows.append(
                (
                    f"grid-{a}:grid-{b} sir0-cafe",
                    mix,
                    t,
                    o,
                    load(g / f"{a}_profile.wav"),
                    None,
                    ("ecapa",),
                    (),
                )
            )
    result = {"rows": {}, "windows": {"win_s": WIN / SR, "hop_s": HOP / SR}}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        net = network(path).to(dev).eval()
        for row, mix, t, o, prof, slot, picks, absent in rows:
            rng = np.random.default_rng(0)
            score = evaluate_row(
                net, ecapa, mix, t, o, prof, picks, slot, dev, rng, absent
            )
            for pick, s in score.items():
                result["rows"].setdefault(row, {})[f"{label}/{pick}"] = s
                print(row, label, pick, json.dumps(s), flush=True)
        del net
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
