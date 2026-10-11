"""The tier ladder on the association scenes: does the same-gender guard keep the other person out? (T-070)

``eval/association_eval.py`` drives the Enrolled ``tse`` stage alone. This harness runs the whole
``tiers`` stage (``config/tiers.yaml``: enrolled -> AV Focus HQ -> gate -> audio-only) over the same
rendered two-face scenes, live-style, one pass per scene:

* every frame goes through the session's ``VisionEngine`` (MediaPipe + tracker + Light-ASD, **with lip
  crops** into a ``LipFeed`` for the AV tier, and the apparent-face hook), the audio in 10 ms chunks
  through the ladder with the context one video frame behind, as live;
* the wearer's lock is the same oracle as in ``association_eval`` (the track on the chosen person's side);
* ``same_bucket`` comes from :class:`hearing.synth.voice_match.SameBucket` over per-track buckets from the
  real apparent-face estimator (``--buckets face``, the live path) or from the corpus sex of each talker
  (``--buckets oracle``);
* ``ctx.suppression = 1.0`` (slider 100: only the target), so a tier's output is its target estimate
  (the T-061 numbers score the extracted stream the same way);
* tse stages run ``threaded=false`` (deterministic), so each plays ``window`` behind instead of the
  live ``window + hop``; delay numbers are reported with the live values (``live_delay_ms``).

Every tier runs on every chunk (as in the ladder), so one pass also gives the ladder **without** the
guard: the policy is replayed (:class:`hearing.fusion.tiers.TierPolicy`, same hysteresis) with the
Enrolled tier's eligibility ignoring ``same_bucket``. Both are scored on the same tier outputs:

* ``wrong/10s``: 40 ms output frames (one tier, at least ``crossfade_ms`` after a switch, mapped back by
  that tier's delay) in which the other talker dominates the output (its least-squares share beats the
  target's and is suppressed by less than 10 dB vs the mix, as ``association_eval``), per 10 s of
  scored output, over **all** tiers (the gate passes both people while both talk; that counts).
* ``acc``: 1 s output spans every 0.5 s (same rules) in which the target talks: correct when
  SI-SDR(out, target) > SI-SDR(out, other) while the other talks, else when SI-SDR(out, target) > 0 dB
  (a muted span is a miss: a dropout). Not window-for-window the same as ``association_eval``'s
  ``acc`` (which scores enrolled windows only, against the oracle stream); compare guard vs no-guard here.

    .venv/bin/python -m eval.association_ladder --pair mm             # 20 scenes, ~5-8 min on laptop-a
    .venv/bin/python -m eval.association_ladder --pair mf,mm,ff --buckets oracle
    .venv/bin/python -m eval.association_ladder --pair mm --all-devices   # with the AV tier (GPU ladder)

Writes ``$HEARING_RESULTS_DIR/association_ladder_<pairs>_<buckets>_<stamp>.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.association_eval import (
    FAST_SCENES,
    FPS,
    HOP,
    LEAK_DB,
    PAIRS,
    SPF,
    SR,
    WRONG_FRAME,
    OracleLock,
    _energy,
    _results_dir,
    _si_sdr,
    _side,
    active,
    load_scenes,
    load_talker,
    ls_shares,
    render,
    speakers,
    target_segments,
)
from hearing.types import AudioChunk, StageContext, TargetState

SEX = {"s1": "male", "s2": "male", "s4": "female", "s7": "female"}  # GRID corpus docs (T-069)
SPAN = SR  # 1 s accuracy spans
SPAN_STEP = SR // 2
CHUNKS_PER_FRAME = SPF // HOP


def ladder_config(strict: bool = False, all_devices: bool = False) -> dict:
    """``config/tiers.yaml`` with every tse stage ``threaded=false`` (and the strict pause preset).

    ``all_devices`` drops each tier's ``devices`` list, so the CUDA-only AV tier (T-072) is built on
    this host too: the ladder the GPU box plays. Without it the host's own ladder runs (on the Mac:
    enrolled -> gate -> audio-only).
    """
    import yaml

    from hearing.fusion.tiers import DEFAULT_CONFIG

    data = yaml.safe_load(DEFAULT_CONFIG.read_text())

    def fix(spec: str) -> str:
        spec = re.sub(r"\btse\(", "tse(threaded=false, ", spec)
        if strict and 'candidate="enrolled"' in spec:
            spec = spec.replace("tse(", 'tse(pause_floor=0.25, pause_p=0.5, pause_action="duck", ', 1)
        return spec

    for tier in data["tiers"]:
        if all_devices:
            tier.pop("devices", None)
        specs = tier["pipeline"]
        tier["pipeline"] = fix(specs) if isinstance(specs, str) else [fix(s) for s in specs]
    data.pop("remote", None)
    return data


def live_delays() -> dict[str, float]:
    """Each tier's delay as the live session plays it (threaded tse: window + hop)."""
    from hearing import registry
    from hearing.fusion.tiers import LadderConfig

    out = {}
    for spec in LadderConfig.load().tiers:
        for text in spec.pipelines:
            try:
                out[spec.name] = float(registry.build_pipeline(text).latency_ms)
                break
            except Exception as e:  # noqa: BLE001  (unavailable spec: try the next, as the ladder does)
                print(f"[live delay] {spec.name}: {text!r} unavailable: {e}", file=sys.stderr)
    return out


# ------------------------------------------------------------------------------------- run
@dataclass
class LadderRun:
    order: tuple[str, ...]
    outs: dict[str, np.ndarray]  # per tier, the whole output timeline
    delays: dict[str, int]  # per tier, samples (as run: threaded=false)
    ok_guard: np.ndarray  # chunks x tiers, eligibility as the ladder saw it
    ok_free: np.ndarray  # same, Enrolled ignoring same_bucket
    same: np.ndarray  # chunks: same_bucket flag
    played: list[str]  # the ladder's own choice per chunk
    upgrade_after_s: float
    crossfade_ms: float
    compute_s: float
    reasons: list[str] = field(default_factory=list)


def run_ladder(r, buckets: str = "face", faces: str = "mediapipe", asd: str = "light_asd",
               strict: bool = False, pair: str = "mf", all_devices: bool = False) -> LadderRun:
    from eval.association_eval import _jpeg
    from hearing.fusion.tiers import TierLadder, eligible
    from hearing.server.vision import VisionEngine
    from hearing.synth import voice_match
    from hearing.tse.lips import FEED_KEY, LipFeed

    matcher = voice_match.VoiceMatcher()
    estimator = voice_match.get_face_estimator() if buckets == "face" else None
    if buckets == "face" and estimator is None:
        raise RuntimeError("no apparent-face estimator (weights missing); use --buckets oracle")

    def on_face(tid, image, box):
        est = estimator.estimate(image, box) if estimator is not None else None
        if est is not None:
            matcher.update(tid, face=est)
        return []

    spk = dict(zip((r.side_of("A"), r.side_of("B")), speakers(r.variant, pair), strict=True))
    sides: dict[int, str] = {}

    def bucket_of(tid: int) -> str:
        if buckets == "oracle":
            return f"{SEX[spk[sides[tid]]]}_adult" if tid in sides else "neutral"
        return matcher.choice(tid).bucket

    lad = TierLadder(ladder_config(strict, all_devices))
    engine = VisionEngine(faces, asd, threaded=False, lips=True, on_face=on_face if estimator else None)
    same_book = voice_match.SameBucket()
    ctx = StageContext.default()
    ctx.suppression = 1.0
    ctx.telemetry.extras[FEED_KEY] = LipFeed()
    lock = OracleLock()
    seg = target_segments(r.scene)
    order = lad.order
    enrolled = order.index("enrolled") if "enrolled" in order else None
    outs: dict[str, list[np.ndarray]] = {n: [] for n in order}
    for name, tier in lad.built.items():  # capture every tier's output (the ladder runs them all)
        inner = tier.run

        def run(chunk, c, _inner=inner, _name=name):
            y = _inner(chunk, c)
            outs[_name].append(y.copy())
            return y

        tier.run = run
    ok_guard, ok_free, same, played = [], [], [], []
    t0 = time.perf_counter()
    try:
        for i in range(r.scene.n_frames):
            for c in range(CHUNKS_PER_FRAME):
                pos = i * SPF + c * HOP
                x = r.mix[pos : pos + HOP]
                engine.push_audio(x, pos / SR)
                lad.process(AudioChunk(x, SR, pos // HOP, pos / SR), ctx)
                verdicts = [lad._eligibility[n][0] for n in order]
                ok_guard.append(verdicts)
                free = list(verdicts)
                if enrolled is not None:
                    spec = lad.built["enrolled"].spec
                    healthy = all(h.get("ok", False) for h in lad._health["enrolled"])
                    free[enrolled] = (lad.built["enrolled"].error is None and eligible(
                        replace(spec, avoid=frozenset()), ctx, (healthy, ""), lad.config.min_target_confidence)[0])
                ok_free.append(free)
                same.append(bool(ctx.telemetry.extras.get(voice_match.SAME_BUCKET_KEY)))
                played.append(lad.active)
            ts = (i + 1) / FPS - 0.005
            engine.submit(i, ts, _jpeg(r.frame(i)))
            for ev in engine.poll(ctx):
                if ev.get("degraded_to"):
                    raise RuntimeError(f"vision degraded: {ev}")
            sides.update({t.track_id: _side(t) for t in ctx.tracks})
            who = next(p for a, b, p in seg if a <= i / FPS < b)
            tid = lock.update(ctx.tracks, r.side_of(who), ts)
            ctx.target = TargetState(tid, 1.0 if tid is not None else 0.0, tid is not None, ts)
            same_book.update(ctx.tracks, bucket_of, ts)
            same_book.publish(ctx)
    finally:
        engine.close()
        lad.close()
    return LadderRun(order, {n: np.concatenate(v) for n, v in outs.items()},
                     {n: round(lad.built[n].delay_ms * SR / 1000) for n in order},
                     np.array(ok_guard, bool), np.array(ok_free, bool), np.array(same, bool), played,
                     lad.config.upgrade_after_s, lad.config.crossfade_ms, time.perf_counter() - t0,
                     [h[2] for h in lad.history])


def replay(ok: np.ndarray, order: tuple[str, ...], upgrade_after_s: float) -> list[str]:
    """The tier per chunk that :class:`TierPolicy` picks from per-chunk eligibility."""
    from hearing.fusion.tiers import TierPolicy

    policy = TierPolicy(order, upgrade_after_s)
    return [order[policy.choose(list(row), k * HOP / SR)] for k, row in enumerate(ok)]


# ------------------------------------------------------------------------------------- score
def _steady(tiers: list[str], k0: int, k1: int, settle: int) -> str | None:
    """The tier of chunks ``[k0, k1)`` if it is one tier and no switch happened ``settle`` chunks before
    (the start of the run is not a switch: there is no crossfade)."""
    a = max(0, k0 - settle)
    seq = tiers[a:k1]
    return seq[0] if seq and all(t == seq[0] for t in seq) else None


def score_policy(r, run: LadderRun, tiers: list[str], live: dict[str, float]) -> dict:
    stems = (r.stems["A"], r.stems["B"])
    tgt_at = r.target_at()
    settle = round(run.crossfade_ms * SR / 1000 / HOP)
    n = len(tiers) * HOP
    wrong = frames = 0
    per_tier: dict[str, list[int]] = {name: [0, 0] for name in run.order}  # wrong, frames
    for f0 in range(0, n - WRONG_FRAME + 1, WRONG_FRAME):
        tier = _steady(tiers, f0 // HOP, (f0 + WRONG_FRAME) // HOP, settle)
        s0 = f0 - run.delays[tier] if tier else -1
        if tier is None or s0 < 0:
            continue
        frames += 1
        per_tier[tier][1] += 1
        sl = slice(s0, s0 + WRONG_FRAME)
        who = int(tgt_at[s0 + WRONG_FRAME // 2])
        tgt, oth = stems[who][sl], stems[1 - who][sl]
        if not active(oth):
            continue
        e_t, e_o, _ = ls_shares(run.outs[tier][f0 : f0 + WRONG_FRAME], [tgt, oth, r.stems["noise"][sl]])
        if e_o > e_t and e_o > _energy(oth) * 10 ** (-LEAK_DB / 10):
            wrong += 1
            per_tier[tier][0] += 1
    spans = correct = 0
    for f0 in range(0, n - SPAN + 1, SPAN_STEP):
        tier = _steady(tiers, f0 // HOP, (f0 + SPAN) // HOP, settle)
        s0 = f0 - run.delays[tier] if tier else -1
        if tier is None or s0 < 0:
            continue
        sl = slice(s0, s0 + SPAN)
        who = int(tgt_at[s0 + SPAN // 2])
        tgt, oth = stems[who][sl], stems[1 - who][sl]
        if not active(tgt):
            continue
        y = run.outs[tier][f0 : f0 + SPAN]
        spans += 1
        if not np.any(y):
            continue
        correct += (_si_sdr(y, tgt) > _si_sdr(y, oth)) if active(oth) else (_si_sdr(y, tgt) > 0.0)
    share = {name: tiers.count(name) / len(tiers) for name in run.order}
    switches = sum(1 for a, b in itertools.pairwise(tiers) if a != b)
    return {
        "wrong_frames": wrong, "frames": frames, "spans": spans, "correct": correct,
        "per_tier": per_tier, "share": share, "switches": switches, "seconds": len(tiers) * HOP / SR,
        "live_delay_ms": float(np.mean([live.get(t, 0.0) for t in tiers])),
    }


def summarize(per: list[dict]) -> dict:
    frames = sum(p["frames"] for p in per)
    spans = sum(p["spans"] for p in per)
    secs = sum(p["seconds"] for p in per)
    wrong = sum(p["wrong_frames"] for p in per) * WRONG_FRAME / SR
    names = per[0]["share"] if per else {}
    out = {
        "acc": sum(p["correct"] for p in per) / spans if spans else None, "spans": spans,
        "wrong_per_10s": wrong / (frames * WRONG_FRAME / SR) * 10 if frames else None,
        "wrong_s": round(wrong, 2), "scored_s": round(frames * WRONG_FRAME / SR, 1),
        "share": {k: round(sum(p["share"][k] * p["seconds"] for p in per) / secs, 3) for k in names},
        "switches_per_min": sum(p["switches"] for p in per) / (secs / 60) if secs else 0.0,
        "live_delay_ms": sum(p["live_delay_ms"] * p["seconds"] for p in per) / secs if secs else None,
        "wrong_per_10s_by_tier": {},
    }
    for k in names:
        w = sum(p["per_tier"][k][0] for p in per)
        f = sum(p["per_tier"][k][1] for p in per)
        out["wrong_per_10s_by_tier"][k] = round(w / f * 10, 2) if f else None
    return out


# ------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="mf", help="comma-separated, from " + ", ".join(PAIRS))
    ap.add_argument("--fast", action="store_true", help=f"scenes {', '.join(FAST_SCENES)}, variant ab")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--buckets", choices=("face", "oracle"), default="face")
    ap.add_argument("--strict", action="store_true", help="enrolled tier with the strict pause preset")
    ap.add_argument("--all-devices", action="store_true",
                    help="build every tier on this host (the CUDA-only AV tier too): the GPU box's ladder")
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--noise", choices=("babble", "pink"), default="babble")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    pairs = [p for p in args.pair.split(",") if p]
    if set(pairs) - set(PAIRS):
        ap.error(f"unknown pairs {sorted(set(pairs) - set(PAIRS))}")

    from hearing import device
    from hearing.server.session import load_stage_modules

    load_stage_modules()  # every stage the tiers name (dfn_ll, gate, mixer, tse, ...), as the server does

    scenes = load_scenes()
    wanted = [s for s in args.scenes.split(",") if s] or (list(FAST_SCENES) if args.fast else [])
    if wanted:
        scenes = [s for s in scenes if s.name in wanted]
    variants = [v for v in args.variants.split(",") if v] or (["ab"] if args.fast else ["ab", "ba"])
    live = live_delays()
    out = args.out or _results_dir() / (f"association_ladder_{'-'.join(pairs)}_{args.buckets}"
                                        f"{'_strict' if args.strict else ''}{'_alldev' if args.all_devices else ''}_{datetime.now(UTC):%Y%m%d-%H%M%S}.json")
    wall = time.perf_counter()
    results: dict[str, dict[str, list[dict]]] = {}
    for pair in pairs:
        talkers = {s: load_talker(s) for v in variants for s in speakers(v, pair)}
        results[pair] = {"guard": [], "noguard": []}
        for si, scene in enumerate(scenes):
            for variant in variants:
                r = render(scene, variant, talkers, args.snr, args.noise, args.seed + si, pair)
                run = run_ladder(r, args.buckets, strict=args.strict, pair=pair,
                                 all_devices=args.all_devices)
                guard = replay(run.ok_guard, run.order, run.upgrade_after_s)
                if guard != run.played:
                    raise AssertionError("replayed policy differs from the ladder's own choices")
                free = replay(run.ok_free, run.order, run.upgrade_after_s)
                for name, tiers in (("guard", guard), ("noguard", free)):
                    p = score_policy(r, run, tiers, live)
                    p.update(scene=scene.name, variant=variant, same_share=float(run.same.mean()),
                             compute_s=round(run.compute_s, 1))
                    results[pair][name].append(p)
                g, f = results[pair]["guard"][-1], results[pair]["noguard"][-1]
                print(f"[{pair}] {scene.name}/{variant}: same {g['same_share']:.2f} | guard "
                      f"{g['correct']}/{g['spans']} wrong {g['wrong_frames']} share "
                      f"{ {k: round(v, 2) for k, v in g['share'].items() if v} } | noguard "
                      f"{f['correct']}/{f['spans']} wrong {f['wrong_frames']} ({run.compute_s:.0f}s)",
                      file=sys.stderr, flush=True)
                _write(out, args, results, live, wall, partial=True)
    _write(out, args, results, live, wall)
    print(f"\ndevice {device.pick()}, buckets {args.buckets}, snr {args.snr:g} dB {args.noise}, "
          f"live delays {live}")
    print("| pair | ladder | acc | wrong s / 10 s | same_bucket time | enrolled / focus_hq / gate | mean delay (live) "
          "| switches / min |")
    print("|---|---|---|---|---|---|---|---|")
    for pair, per in results.items():
        for name, rows in per.items():
            s = summarize(rows)
            same = np.mean([p["same_share"] for p in rows])
            sh = s["share"]
            print(f"| {pair} | {name} | {s['acc']:.3f} | {s['wrong_per_10s']:.2f} | {same:.2f} | "
                  f"{sh.get('enrolled', 0):.2f} / {sh.get('focus_hq', 0):.2f} / {sh.get('gate', 0):.2f} | "
                  f"{s['live_delay_ms'] / 1000:.2f} s | {s['switches_per_min']:.1f} |")
    print(f"wrote {out}")
    return 0


def _write(out: Path, args, results: dict, live: dict, wall: float, partial: bool = False) -> None:
    from hearing import device

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": device.describe(), "args": {k: str(v) for k, v in vars(args).items()}, "partial": partial,
        "live_delay_ms": live, "wall_s": round(time.perf_counter() - wall, 1),
        "summary": {pair: {name: summarize(rows) for name, rows in per.items() if rows}
                    for pair, per in results.items()},
        "per_scene": results,
    }
    tmp = out.with_suffix(".part")
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.replace(out)


if __name__ == "__main__":
    sys.exit(main())
