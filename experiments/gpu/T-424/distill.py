"""Distil the fine-tuned fixed-order MossFormer2 (teacher) into a small causal student (T-424).

Data: T-391's ``--recipe v3`` renderer (``../T-391/train_sep.py``), split by take exactly like
T-374/T-391 (train profile-02..05 + 07, val profile-06; holdout/same/profile-01 and the cafe
fixture takes are never read), rendered on the fly as ``--crop-s`` second items.
Teacher: the T-387 checkpoint (``~/tmp/T-374b/run_fixed/best.pt``), run non-causally on the
whole item (bf16, no grad). Student: ``student.py`` (causal, fixed order, slot 0 = Aryan).

Loss on the student's raw output (no shared gain), fixed order: T-374's per-slot thresholded
SNR (tau 1e-3; L0 term against a silent reference) over three slots, Aryan, Joseph and the
residual ``mix - aryan - joseph`` (noise + strangers), + ``--alpha`` x a distillation term per
team slot, ``10 log10(|s_k - t_k|^2 / |mix|^2 + 1e-3)`` with the teacher on its shared gain to
the mixture. The residual slot and the student's softmax masks keep the slots summing to the
mixture. Two 2-slot losses collapsed from scratch (they are fine for the fine-tunes, which
start from a separator): T-374's shared-gain loss -> an output orthogonal to the mixture
(gain -> 0, silent slots at the floor); the same loss without the gain -> all-zero sigmoid
masks (0 dB on active slots beats an early crude mask). Both froze at val loss -8.0 with
fixed-slot SI-SDRi -58 / -27 dB. Validation (``ts.validate``) scores slots 0-1 with
``raw_slot_losses``; ``best.pt`` is chosen by that.
``--alpha 0`` skips the teacher (ablation: is the teacher worth it?).

The teacher costs ~3 s per 8 x 3 s batch on an A4000 while a student step takes ~0.05 s, so
teacher-labelled items go into a GPU replay buffer (``--buffer`` items, FIFO) and the student
takes ``--k`` steps per teacher batch: the fresh batch, then ``k - 1`` batches drawn from the
buffer with a random gain (-6..+6 dB, every loss here is scale-invariant up to the shared gain).

Checkpoints: ``best.pt`` by val loss (clean-stem loss on 300 profile-06 items), ``last.pt`` and
``ckpt_<min>m.pt`` every ``--ckpt-every`` minutes; a job restarted with the same ``--out``
resumes from ``last.pt``. Cosine lr decay over ``--minutes``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "T-391")]
import student
import train_sep as ts

DISTILL_FLOOR = 1e-3


def raw_slot_losses(est, refs, mix):
    """ts.slot_losses on the raw output (no shared gain), any slot count; [B, 2 perms, S], perm 0 = identity."""
    ref_e = refs.square().sum(-1)
    mix_e = mix.square().sum(-1)[:, None]
    perms = []
    for order in ((0, 1), (1, 0)):
        e = est[:, [*order, *range(2, est.shape[1])]]
        err = (refs - e).square().sum(-1)
        active = 10 * torch.log10(err + ts.TAU * ref_e + 1e-8) - 10 * torch.log10(ref_e + 1e-8)
        silent = 10 * torch.log10(e.square().sum(-1) + ts.TAU * mix_e + 1e-8) - 10 * torch.log10(mix_e + 1e-8)
        perms.append(torch.where(ref_e > 1e-8, active, silent))
    return torch.stack(perms, dim=1)


def shared_gain(est, mix):
    total = est.sum(dim=1)
    gain = (mix * total).sum(-1) / (total.square().sum(-1) + 1e-8)
    return est * gain[:, None, None]


def distill_loss(est, teacher, mix):
    """Per team slot 10 log10(|s_k - t_k|^2 / |mix|^2 + 1e-3), teacher on its shared gain."""
    t = shared_gain(teacher, mix)
    mix_e = mix.square().sum(-1)[:, None] + 1e-8
    return (10 * torch.log10((est[:, :2] - t).square().sum(-1) / mix_e + DISTILL_FLOOR)).mean()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=tuple(student.VARIANTS), required=True)
    ap.add_argument("--teacher", required=True, help="T-387 fixed-order MossFormer2 best.pt")
    ap.add_argument("--team", default=str(Path.home() / "hearing-data/team"))
    ap.add_argument("--data", required=True, help="dir with cafe/ch01.wav and babble/*.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0, help="distillation weight (0 = no teacher)")
    ap.add_argument("--crop-s", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min-frac", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--minutes", type=float, default=270.0)
    ap.add_argument("--ckpt-every", type=float, default=30.0, help="minutes")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-items", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sir", type=float, nargs=2, default=(-10.0, 10.0))
    ap.add_argument("--seed", type=int, default=424)
    ap.add_argument("--k", type=int, default=8, help="student steps per teacher batch")
    ap.add_argument("--buffer", type=int, default=2048, help="replay buffer size in items")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ts.R = ts.RECIPES["v3"]
    ts.slot_losses = raw_slot_losses  # ts.validate scores the student with it too
    ts.SIR_RANGE = tuple(args.sir)
    ts.CROP = int(args.crop_s * ts.SR)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    torch.manual_seed(args.seed)
    net = student.build(args.variant).to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    teacher = ts.network(args.teacher).to(dev).eval() if args.alpha > 0 else None
    if teacher is not None:
        for p in teacher.parameters():
            p.requires_grad_(False)
    rng = np.random.default_rng(args.seed + 1)
    val = [ts.render(ts.Pool(args.team, args.data, "val"), rng) for _ in range(args.val_items)]
    loader = torch.utils.data.DataLoader(
        ts.Stream(args.team, args.data, args.seed + 2),
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    step, spent, best = 0, 0.0, None
    if (out / "last.pt").exists():  # resume after a requeue / time limit
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, spent, best = state["step"], state["spent_s"], state.get("best")
    log = open(out / "log.jsonl", "a")  # noqa: SIM115 - held for the whole run
    t0 = time.time() - spent

    def record(entry):
        entry["t_s"] = round(time.time() - t0, 1)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)

    def save(name, **extra):
        torch.save(
            {
                "model": net.state_dict(),
                "variant": args.variant,
                "step": step,
                "latency_ms": net.latency_ms,
                "params": student.count_params(net),
                "args": vars(args),
                **extra,
            },
            out / name,
        )

    record(
        {
            "start": True,
            "step": step,
            "variant": args.variant,
            "params": student.count_params(net),
            "latency_ms": net.latency_ms,
            "receptive_s": round(net.receptive_s, 2),
            "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
            "args": vars(args),
        }
    )
    next_ckpt = (math.floor((time.time() - t0) / 60 / args.ckpt_every) + 1) * args.ckpt_every
    running = []
    buf, fill, head = None, 0, 0
    gen = torch.Generator(device=dev).manual_seed(args.seed + 3)

    def batches():
        nonlocal buf, fill, head
        for mix, refs in loader:
            mix, refs = mix.to(dev), refs.to(dev)
            if teacher is None:
                yield mix, refs, None
                continue
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                teach = ts.separate(teacher, mix)
            item = torch.cat([mix[:, None], refs, teach], dim=1)  # [B, 5, T]
            if buf is None:
                buf = item.new_zeros(args.buffer, *item.shape[1:])
            idx = (head + torch.arange(len(item), device=dev)) % args.buffer
            buf[idx] = item
            head, fill = (head + len(item)) % args.buffer, min(fill + len(item), args.buffer)
            yield mix, refs, teach
            for _ in range(args.k - 1):
                pick = torch.randint(0, fill, (len(item),), device=dev, generator=gen)
                g = 10 ** ((torch.rand(len(item), 1, 1, device=dev, generator=gen) * 12 - 6) / 20)
                b = buf[pick] * g
                yield b[:, 0], b[:, 1:3], b[:, 3:5]

    for mix, refs, teach in batches():
        est = net(mix, residual=True)
        refs3 = torch.cat([refs, (mix - refs.sum(1))[:, None]], dim=1)
        clean = ts.objective(raw_slot_losses(est, refs3, mix), "fixed")
        loss = clean
        dist = None
        if teach is not None:
            dist = distill_loss(est, teach, mix)
            loss = clean + args.alpha * dist
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        frac = min(1.0, max((time.time() - t0) / (args.minutes * 60), step / args.steps))
        lr = args.lr * (args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        step += 1
        running.append((float(clean), float(dist) if dist is not None else 0.0))
        minutes = (time.time() - t0) / 60
        done = step >= args.steps or minutes > args.minutes
        if step % 100 == 0:
            r = np.mean(running[-100:], axis=0)
            record({"step": step, "clean": round(r[0], 3), "distill": round(r[1], 3), "lr": lr})
        if step % args.val_every == 0 or done:
            score = ts.validate(net, val, dev, "fixed")
            record({"step": step, "val": score})
            if best is None or score["loss"] < best["loss"]:
                best = score
                save("best.pt", val=score)
        if minutes >= next_ckpt or done:
            save("last.pt", opt=opt.state_dict(), spent_s=time.time() - t0, best=best)
            if not done:
                save(f"ckpt_{int(next_ckpt)}m.pt")
            next_ckpt += args.ckpt_every
        if done:
            break
    record({"done": True, "steps": step, "best": best, "gpu_minutes": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
