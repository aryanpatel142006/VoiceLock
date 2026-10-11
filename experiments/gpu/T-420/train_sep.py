"""Fine-tune MossFormer2_SS_16K on Aryan+Joseph mixtures rendered on the fly (T-374, T-387, T-391, T-420).

T-420 adds ``--crop-s`` (default 2.0, the T-374/T-387/T-391 crop): train and validate on shorter
crops for the low-delay live path (0.5 / 0.75 / 1.0 s windows). The turn-taking hand-over point
scales with the crop and its overlap is capped at 40 % of the crop; at 2.0 s every draw is the
T-391 one bit for bit.

T-391 adds ``--recipe v3`` (``--recipe v1`` reproduces T-374/T-387 exactly): profile-07 joins
the train takes; noise SNR 0-20 dB with 20 % clean items; two-talker overlap 5-100 % plus
turn-taking items (0.2-0.6 s overlap at the hand-over); level jitter -35..-5 dBFS; 10 %
"stranger" items, where a LibriSpeech voice of the split's own speakers is the loud talker
(0-10 dB above the team voice) next to one team member or alone, so its refs stay silent and
the net learns to route a stranger to neither slot; cosine learning-rate decay over the run's
wall-clock budget. Stranger voices are the split's babble speakers (train for train, val for
val); the test speakers stay reserved for eval_sep.py's stranger rows.

Two runs differ only in the permutation: ``--order fixed`` scores slot 0 against Aryan and
slot 1 against Joseph; ``--order pit`` takes the better of the two assignments. Both use the
same per-slot loss: the streams get the one shared least-squares gain to the mixture that
inference applies (``hearing.tse.blind.rescale_to_mix``), then MixIT's thresholded SNR
(tau = 1e-3, 30 dB) against an active reference, or its L0 term
``10log10(|y_hat|^2 + tau |x|^2)`` against a silent one.

Data follows docs/research/RESEARCH_oct10_adapt.md "Recipe for #1", split by take: train
profile-02..05, val profile-06 per person; holdout-01/02, same-01/02, profile-01/07 and the
fixture (cafe) takes are never read here. Noise: DEMAND PCAFETER ch01 (train 0-180 s, val
180-240 s; 240-300 s is the eval's) plus 0-2 LibriSpeech voices (speakers split by id), so a
third talker belongs to neither slot. Self-contained: torch + clearvoice + soundfile only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

SR = 16000
CROP = 2 * SR
TAU = 1e-3
TRAIN_TAKES = ("profile-02", "profile-03", "profile-04", "profile-05")
TRAIN_TAKES_V3 = (*TRAIN_TAKES, "profile-07")
VAL_TAKES = ("profile-06",)
CAFE = {"train": (0, 180), "val": (180, 240), "test": (240, 300)}
BABBLE_SPK = {
    "train": ("1673", "1988", "2035", "2086", "2277", "5536", "6295", "6345"),
    "val": ("777", "7976"),
    "test": ("8297", "84"),
}
KINDS = (("aj", 0.55), ("a", 0.15), ("j", 0.15), ("n", 0.05), ("aj_clean", 0.10))
# v3: "turn" = hand-over with a 0.2-0.6 s overlap; "sa"/"sj" = stranger + one team member;
# "s" = stranger alone. Clean (no noise) is drawn separately with CLEAN_FRAC.
KINDS_V3 = (
    ("aj", 0.45),
    ("turn", 0.12),
    ("a", 0.13),
    ("j", 0.13),
    ("n", 0.07),
    ("sa", 0.04),
    ("sj", 0.04),
    ("s", 0.02),
)
RECIPES = {
    "v1": {
        "kinds": KINDS,
        "takes": TRAIN_TAKES,
        "snr": (0.0, 15.0),
        "clean": 0.0,
        "overlap": (0.25, 1.0),
        "level": (-25.0, -5.0),
    },
    "v3": {
        "kinds": KINDS_V3,
        "takes": TRAIN_TAKES_V3,
        "snr": (0.0, 20.0),
        "clean": 0.2,
        "overlap": (0.05, 1.0),
        "level": (-35.0, -5.0),
    },
}
R = RECIPES["v1"]  # set by main()


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return x.mean(axis=1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


class Pool:
    """Clean takes, cafe noise and babble voices for one split."""

    def __init__(self, team, data, split):
        takes = R["takes"] if split == "train" else VAL_TAKES
        self.speech = [
            [load(Path(team) / who / f"{t}.wav") for t in takes]
            for who in ("aryan", "joseph")
        ]
        cafe = load(Path(data) / "cafe" / "ch01.wav")
        lo, hi = CAFE[split]
        self.cafe = cafe[lo * SR : hi * SR]
        self.babble = [
            load(p)
            for p in sorted((Path(data) / "babble").glob("*.wav"))
            if p.name.split("-")[0] in BABBLE_SPK[split]
        ]
        self.take_rms = [[rms(t) for t in person] for person in self.speech]

    def speech_crop(self, rng, who, length):
        for _ in range(20):
            k = rng.integers(len(self.speech[who]))
            take = self.speech[who][k]
            start = rng.integers(0, len(take) - length)
            crop = take[start : start + length]
            if (
                rms(crop) > 0.3 * self.take_rms[who][k]
            ):  # skip crops that are mostly pause
                return crop.copy()
        return crop.copy()

    def stranger(self, rng, length):
        """A babble voice of this split as a talker (pauses kept, looped when short)."""
        for _ in range(20):
            voice = self.babble[rng.integers(len(self.babble))]
            if len(voice) < length:
                voice = np.tile(voice, length // len(voice) + 1)
            start = rng.integers(0, len(voice) - length + 1)
            seg = voice[start : start + length]
            if rms(seg) > 0.3 * rms(voice):
                return seg.copy()
        return seg.copy()

    def noise(self, rng):
        start = rng.integers(0, len(self.cafe) - CROP)
        noise = self.cafe[start : start + CROP].copy()
        noise /= rms(noise)
        for _ in range(rng.integers(0, 3)):
            voice = self.babble[rng.integers(len(self.babble))]
            if len(voice) < CROP:
                voice = np.pad(voice, (0, CROP - len(voice)))
            start = rng.integers(0, len(voice) - CROP + 1)
            seg = voice[start : start + CROP]
            noise += seg / rms(seg) * 10 ** (rng.uniform(-5, 3) / 20)
        return noise / rms(noise)


def render(pool, rng, kind=None, parts=None):
    """One 2 s item: mix, refs [2, T] (slot 0 = Aryan, 1 = Joseph; silent slot all zeros)."""
    if kind is None:
        kinds = R["kinds"]
        kind = kinds[rng.choice(len(kinds), p=[p for _, p in kinds])][0]
        if R["clean"] and kind != "n" and rng.random() < R["clean"]:
            kind += "_clean"
    refs = np.zeros((2, CROP), np.float32)
    talker = np.zeros(CROP, np.float32)  # in the mix, in neither slot (v3 stranger)
    if kind.startswith("turn"):
        first = rng.integers(2)
        scale = CROP / (2 * SR)  # T-420: 1.0 at the 2 s crop, so the draws match T-391
        ov = int(min(rng.uniform(0.2, 0.6), 0.4 * CROP / SR) * SR)
        cut = int(rng.uniform(0.6, 1.4) * scale * SR)
        a, b = cut - ov // 2, cut + (ov - ov // 2)
        refs[first, :b] = pool.speech_crop(rng, first, b)
        refs[1 - first, a:] = pool.speech_crop(rng, 1 - first, CROP - a)
        refs[0] /= rms(refs[0])
        refs[1] *= 10 ** (rng.uniform(*SIR_RANGE) / 20) / rms(refs[1])
    elif kind.startswith("s"):
        if kind[:2] in ("sa", "sj"):
            who = 0 if kind[:2] == "sa" else 1
            refs[who] = pool.speech_crop(rng, who, CROP)
            refs[who] /= rms(refs[who])
        talker = pool.stranger(rng, CROP)
        talker *= 10 ** (rng.uniform(0, 10) / 20) / rms(talker)  # the loud talker
    elif kind.startswith("aj"):
        overlap = rng.uniform(*R["overlap"])
        partial = rng.integers(2)
        refs[1 - partial] = pool.speech_crop(rng, 1 - partial, CROP)
        n = int(overlap * CROP)
        offset = rng.integers(0, CROP - n + 1)
        refs[partial, offset : offset + n] = pool.speech_crop(rng, partial, n)
        refs[0] /= rms(refs[0])
        refs[1] *= 10 ** (rng.uniform(*SIR_RANGE) / 20) / rms(refs[1])
    elif kind.removesuffix("_clean") in ("a", "j"):
        who = 0 if kind.startswith("a") else 1
        refs[who] = pool.speech_crop(rng, who, CROP)
        refs[who] /= rms(refs[who])
    speech = refs.sum(axis=0) + talker
    mix = speech.copy()
    if not kind.endswith("_clean"):
        noise = pool.noise(rng)
        if kind == "n":
            mix = noise
        else:
            mix += noise * rms(speech) * 10 ** (-rng.uniform(*R["snr"]) / 20)
    gain = 10 ** (rng.uniform(*R["level"]) / 20) / rms(mix)
    mix, refs = mix * gain, refs * gain
    peak = np.abs(mix).max()
    if peak > 0.99:
        gain *= 0.99 / peak
        mix, refs = mix * 0.99 / peak, refs * 0.99 / peak
    if parts is not None:  # T-391 gate features: the stranger as it sits in the mix
        parts["talker"] = (talker * gain).astype(np.float32)
    return mix.astype(np.float32), refs.astype(np.float32), kind


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, team, data, seed):
        self.team, self.data, self.seed = team, data, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 7919)
        pool = Pool(self.team, self.data, "train")
        while True:
            mix, refs, _ = render(pool, rng)
            yield torch.from_numpy(mix), torch.from_numpy(refs)


SIR_RANGE = (-6.0, 6.0)  # T-387: --sir LO HI widens the dB range of the second talker


def network(checkpoint):
    from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K

    args = SimpleNamespace(
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        encoder_kernel_size=16,
        num_spks=2,
    )
    net = MossFormer2_SS_16K(args).model
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    net.load_state_dict(
        {k.removeprefix("module."): v for k, v in state.items()}, strict=True
    )
    return net


def separate(net, mix):
    out = net(mix)
    out = torch.stack(list(out), dim=1) if isinstance(out, (list, tuple)) else out
    return out[..., : mix.shape[-1]].float()  # [B, 2, T]


def slot_losses(est, refs, mix):
    """[B, 2 perms, 2 slots] loss in dB (lower is better); perm 0 = identity."""
    total = est.sum(dim=1)
    gain = (mix * total).sum(-1) / (
        total.square().sum(-1) + 1e-8
    )  # shared, as at inference
    est = est * gain[:, None, None]
    ref_e = refs.square().sum(-1)  # [B, 2]
    mix_e = mix.square().sum(-1)  # [B]
    perms = []
    for order in ((0, 1), (1, 0)):
        e = est[:, list(order)]
        err = (refs - e).square().sum(-1)
        active = 10 * torch.log10(err + TAU * ref_e + 1e-8) - 10 * torch.log10(
            ref_e + 1e-8
        )
        silent = 10 * torch.log10(
            e.square().sum(-1) + TAU * mix_e[:, None] + 1e-8
        ) - 10 * torch.log10(mix_e[:, None] + 1e-8)
        perms.append(torch.where(ref_e > 1e-8, active, silent))
    return torch.stack(perms, dim=1)


def objective(losses, order):
    per_perm = losses.mean(-1)  # [B, 2]
    if order == "fixed":
        return per_perm[:, 0].mean()
    return per_perm.min(dim=1).values.mean()


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    proj = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    return 10 * math.log10(
        (np.dot(proj, proj) + 1e-12) / (np.dot(est - proj, est - proj) + 1e-12)
    )


@torch.no_grad()
def validate(net, val, dev, order):
    net.eval()
    losses, fixed_sdri, swapped, stranger_db = [], [], 0, []
    for mix, refs, kind in val:
        m = torch.from_numpy(mix)[None].to(dev)
        r = torch.from_numpy(refs)[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            est = separate(net, m)
        sl = slot_losses(est, r, m)
        losses.append(float(objective(sl, order)))
        if kind.startswith(
            "s"
        ):  # energy in the slot(s) that should be silent, rel. the mixture
            e = est[0].float().cpu().numpy().astype(np.float64)
            total = e.sum(axis=0)
            e *= float(mix @ total) / (
                float(total @ total) + 1e-12
            )  # shared gain, as at inference
            for k in range(2):
                if not refs[k].any():
                    stranger_db.append(
                        10 * math.log10((e[k] @ e[k] + 1e-12) / (mix @ mix + 1e-12))
                    )
        if kind.startswith(("aj", "turn")):
            e = est[0].float().cpu().numpy()
            ident = (
                sum(si_sdr(e[i], refs[i]) - si_sdr(mix, refs[i]) for i in range(2)) / 2
            )
            swap = (
                sum(si_sdr(e[1 - i], refs[i]) - si_sdr(mix, refs[i]) for i in range(2))
                / 2
            )
            fixed_sdri.append(ident)
            swapped += swap > ident
    net.train()
    n_aj = len(fixed_sdri)
    return {
        "loss": float(np.mean(losses)),
        "fixed_sdri_db": float(np.mean(fixed_sdri)),
        "slot_swapped_pct": 100.0 * swapped / max(n_aj, 1),
        "n_two_speaker": n_aj,
        "stranger_slot_db": float(np.mean(stranger_db)) if stranger_db else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=("fixed", "pit"), required=True)
    ap.add_argument("--team", default=str(Path.home() / "hearing-data/team"))
    ap.add_argument(
        "--data", required=True, help="dir with cafe/ch01.wav and babble/*.wav"
    )
    ap.add_argument(
        "--init", required=True, help="shipped MossFormer2_SS_16K checkpoint"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument(
        "--accum", type=int, default=1, help="micro-batches per optimiser step"
    )
    ap.add_argument(
        "--minutes", type=float, default=50.0, help="wall-clock cap for training"
    )
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-items", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--sir", type=float, nargs=2, default=(-6.0, 6.0), metavar=("LO", "HI")
    )
    ap.add_argument("--seed", type=int, default=374)
    ap.add_argument(
        "--crop-s",
        type=float,
        default=2.0,
        help="training/validation crop in seconds (T-420; 2.0 = T-374..T-391)",
    )
    ap.add_argument(
        "--recipe", choices=tuple(RECIPES), default="v1", help="data recipe (T-391: v3)"
    )
    ap.add_argument(
        "--lr-sched",
        choices=("const", "cosine"),
        default="const",
        help="cosine: decay to --lr-min-frac x lr over --minutes (or --steps, whichever ends first)",
    )
    ap.add_argument("--lr-min-frac", type=float, default=0.05)
    ap.add_argument(
        "--mem-fraction", type=float, default=0.0, help="cap this process's GPU share"
    )
    args = ap.parse_args(argv)

    global SIR_RANGE, R, CROP
    SIR_RANGE = tuple(args.sir)
    CROP = round(args.crop_s * SR)
    R = RECIPES[args.recipe]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    if (
        args.mem_fraction and dev.type == "cuda"
    ):  # a shared GPU: hearing-remote needs >= 6 GB free
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    torch.manual_seed(args.seed)
    net = network(args.init).to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    vpool = Pool(args.team, args.data, "val")
    val = [render(vpool, rng) for _ in range(args.val_items)]
    loader = torch.utils.data.DataLoader(
        Stream(args.team, args.data, args.seed + 2),
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    log = open(out / "log.jsonl", "a")  # noqa: SIM115 - held for the whole run

    def record(entry):
        entry["t_s"] = round(time.time() - t0, 1)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)

    t0 = time.time()
    best = validate(net, val, dev, args.order)
    record(
        {
            "step": 0,
            "val": best,
            "order": args.order,
            "lr": args.lr,
            "batch": args.batch,
            "accum": args.accum,
            "recipe": args.recipe,
            "lr_sched": args.lr_sched,
            "sir": list(SIR_RANGE),
            "seed": args.seed,
            "crop_s": CROP / SR,
            "init": args.init,
        }
    )
    torch.save({"model": net.state_dict(), "step": 0, "val": best}, out / "best.pt")
    running, step, micro = [], 0, 0
    for mix, refs in loader:
        mix, refs = mix.to(dev), refs.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            est = separate(net, mix)
        loss = objective(slot_losses(est, refs, mix), args.order)
        (loss / args.accum).backward()
        micro += 1
        if micro % args.accum:
            continue
        step += 1
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        if args.lr_sched == "cosine":
            frac = min(
                1.0, max((time.time() - t0) / (args.minutes * 60), step / args.steps)
            )
            lr = args.lr * (
                args.lr_min_frac
                + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac))
            )
            for group in opt.param_groups:
                group["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        running.append(float(loss))
        done = step >= args.steps or time.time() - t0 > args.minutes * 60
        if step % 50 == 0:
            record(
                {
                    "step": step,
                    "train_loss": float(np.mean(running[-50:])),
                    "lr": opt.param_groups[0]["lr"],
                }
            )
        if step % args.val_every == 0 or done:
            score = validate(net, val, dev, args.order)
            record({"step": step, "val": score})
            if score["loss"] < best["loss"]:
                best = score
                torch.save(
                    {"model": net.state_dict(), "step": step, "val": score},
                    out / "best.pt",
                )
        if done:
            break
    record(
        {
            "done": True,
            "steps": step,
            "best": best,
            "gpu_minutes": round((time.time() - t0) / 60, 1),
        }
    )


if __name__ == "__main__":
    main()
