"""Offline live-path follow-up to the T-200 oracle diagnostic (T-206).

T-200 scored its final stage with a PROXY mixer (``est + (1-s)*(mix-est)``). This adapter reuses
T-200's replay/classifier unchanged and adds, on identical recorded input, the stages the live
path really runs, each labelled with its class and whether it is real or a proxy:

* ``raw_mix``              the input (baseline for SI-SDRi)
* ``oracle_candidate``     overlap-added oracle-picked separator stream (clean target used; NOT deployable)
* ``actual_selected``      overlap-added stream the real ``EnrolledCandidate`` chose
* ``proxy_mixer``          T-200's ``mix_final`` (PROXY, never called the real mixer)
* ``real_mixer``           ``hearing.fusion.mixer.Mixer`` streamed in 10 ms chunks, residual = mix - selected,
                           per-window confidence through ``extras["tse_confidence"]`` (confidence floor 6 + 34c dB)
* ``real_mixer_agc``       ``real_mixer`` then ``hearing.audio.agc.Agc`` (the judge-profile output chain)
* ``cf_floor24_mixer``     COUNTERFACTUAL: real Mixer, confidence ignored, ``max_duck_db=24`` (not a default)
* ``live_wrapper``         ``hearing.tse.stage.FocusHQ(threaded=False)`` + real EnrolledCandidate + a profile-enrolled
                           ``AutoEnroller`` + real Mixer, delay-compensated (skipped with the exact error if it cannot run)

SI-SDR columns are ABSOLUTE (dB vs the clean target, per window, mean); ``sisdri`` = absolute minus the
raw mix's SI-SDR over the same windows (improvement). Target-absent windows report leak (output
energy relative to the mix), never SI-SDR. Wrong-speaker windows: SI-SDR(out, other) > SI-SDR(out, target).

Not exercised here, reported as gaps in the manifest rather than inferred: strict pause/gate session hooks,
same_bucket/active tier, negative-active routing (needs the live session), and independent target-word
checks (no transcripts supplied: ``word_check.status = "missing_transcripts"``). A digital mix of one
session is preliminary: a room conclusion needs independent same-mic recordings.

    python -m eval.isolation_live_path --dry-run
    python -m eval.isolation_live_path --scene <scene dir> --profiles <dir> --noise
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval import isolation_diagnostic as d
from eval import metrics

SR = d.SR
HOP = 160  # 10 ms
STAGES = ("raw_mix", "oracle_candidate", "actual_selected", "proxy_mixer", "real_mixer", "real_mixer_agc",
          "cf_floor24_mixer", "live_wrapper")
PROVENANCE = {
    "raw_mix": {"class": "input", "real": True, "proxy": False, "deployable": True},
    "oracle_candidate": {"class": "hearing.tse.blind.BlindSeparator (oracle pick)", "real": True, "proxy": False,
                         "deployable": False},
    "actual_selected": {"class": "hearing.tse.enrolled.EnrolledCandidate", "real": True, "proxy": False,
                        "deployable": True},
    "proxy_mixer": {"class": "eval.isolation_diagnostic.mix_final", "real": False, "proxy": True, "deployable": False},
    "real_mixer": {"class": "hearing.fusion.mixer.Mixer", "real": True, "proxy": False, "deployable": True},
    "real_mixer_agc": {"class": "hearing.fusion.mixer.Mixer + hearing.audio.agc.Agc", "real": True, "proxy": False,
                       "deployable": True},
    "cf_floor24_mixer": {"class": "hearing.fusion.mixer.Mixer(max_duck_db=24), confidence ignored",
                         "real": True, "proxy": False, "deployable": False, "counterfactual": True},
    "live_wrapper": {"class": "hearing.tse.stage.FocusHQ(threaded=False) + Mixer", "real": True, "proxy": False,
                     "deployable": True},
}
FIX_OWNERS = {
    "separation": "conditional model trial (no default edit); existing owners T-178/T-179/T-180 untouched",
    "selector": "existing profile/contrast owner (T-178)",
    "mix_back": "existing floor/AGC owner (T-180)",
    "deadline": "scheduler: hop/compute budget (existing owner), not a model change",
    "none": "no failing branch in this run",
}


def conf_per_sample(recs, n, hop):
    """Per-sample pick confidence: window i's confidence covers samples ``[i*hop, (i+1)*hop)``
    (the part of the stream only that window owns). Silenced / missing confidence -> None."""
    out = np.full(n, np.nan, np.float32)
    for r in recs:
        c = r["extra"].get("confidence") if isinstance(r.get("extra"), dict) else None
        if r["skipped"]:
            c = 0.0
        if c is not None:
            out[int(r["start_s"] * SR):int(r["start_s"] * SR) + hop] = float(c)
    return out


def run_real_mixer(est, mix, conf, suppression=1.0, max_duck_db=None, use_conf=True):
    """Stream ``est`` through the real ``Mixer`` in 10 ms chunks with residual = mix - est."""
    from hearing.fusion.mixer import CONFIDENCE_KEY, RESIDUAL_KEY, Mixer
    from hearing.types import AudioChunk, StageContext

    m = Mixer(max_duck_db=max_duck_db)
    ctx = StageContext.default()
    ctx.suppression = suppression
    out = np.zeros(len(est), np.float32)
    floors = []
    for i in range(0, len(est), HOP):
        seg = est[i:i + HOP].astype(np.float32)
        ctx.telemetry.extras[RESIDUAL_KEY] = (mix[i:i + HOP] - seg).astype(np.float32)
        c = conf[i] if use_conf and conf is not None else np.nan
        if np.isfinite(c):
            ctx.telemetry.extras[CONFIDENCE_KEY] = float(c)
        res = m.process(AudioChunk(seg, SR, i // HOP, i / SR), ctx)
        out[i:i + len(seg)] = res.samples
        f = ctx.telemetry.extras.get("mixer", {}).get("floor_db")
        if f is not None:
            floors.append(f)
    return out, {"residual_floor_db_mean": float(np.mean(floors)) if floors else None,
                 "residual_floor_db_min": float(np.min(floors)) if floors else None}


def apply_agc(x):
    from hearing.audio.agc import Agc

    return Agc().process_array(np.asarray(x, np.float32)).astype(np.float32)


def stage_metrics(out, mix, target, other, recs, win):
    """Absolute SI-SDR, SI-SDRi vs raw, wrong-speaker windows and absent-window leak for one output."""
    abs_, base, leaks, n_wrong = [], [], [], 0
    for r in recs:
        st = int(r["start_s"] * SR)
        sl = slice(st, st + win)
        if r["target_active"]:
            s_t = metrics.si_sdr(out[sl], target[sl])
            abs_.append(s_t)
            base.append(metrics.si_sdr(mix[sl], target[sl]))
            if r["other_active"] and metrics.si_sdr(out[sl], other[sl]) > s_t:
                n_wrong += 1
        else:
            leaks.append(d._db(d._rms(out[sl])) - d._db(max(d._rms(mix[sl]), 1e-9)))
    n_act = len(abs_)
    act = d._mean(abs_)
    return {
        "n_target_active_windows": n_act, "n_target_absent_windows": len(leaks),
        "sisdr_absolute_db": act,
        "sisdri_db": None if act is None else act - d._mean(base),
        "raw_sisdr_absolute_db": d._mean(base),
        "wrong_speaker_windows": n_wrong, "target_absent_leak_db": d._mean(leaks), "sample_rate_hz": SR,
    }


def live_wrapper_replay(make_candidate, enroll, other, mix, window_s, hop_s):
    """Replay through the real ``FocusHQ`` stage (unthreaded) fed 10 ms chunks, enrolled from the
    profile exactly as a session does (``AutoEnroller.enroll_profile`` is bypassed with ``enroll_clip``
    on the profile clips: same calibrated path). Returns (output aligned to the input, counters)."""
    from hearing.fusion.mixer import Mixer
    from hearing.tse.enroll import ENROLLER_KEY, AutoEnroller
    from hearing.tse.stage import FocusHQ
    from hearing.types import AudioChunk, StageContext

    stage = FocusHQ(window_s=window_s, hop_s=hop_s, threaded=False, adapt_hop=False,
                    _candidate=make_candidate())
    stage.load()
    enr = AutoEnroller()
    enr.enroll_clip(enroll, other)
    ctx = StageContext.default()
    ctx.suppression = 1.0
    ctx.telemetry.extras[ENROLLER_KEY] = enr
    mixer = Mixer()
    outs = []
    n = len(mix)
    padded = np.concatenate([mix, np.zeros(stage.delay + HOP, np.float32)])
    t0 = time.perf_counter()
    for i in range(0, len(padded) - HOP + 1, HOP):
        ch = stage.process(AudioChunk(padded[i:i + HOP], SR, i // HOP, i / SR), ctx)
        outs.append(np.asarray(mixer.process(ch, ctx).samples, np.float32))
    wall = time.perf_counter() - t0
    y = np.concatenate(outs)
    delay = round(stage.latency_ms * SR / 1000)
    y = y[delay:delay + n]
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y), np.float32)])
    health = stage.health()
    counters = {k: health.get(k) for k in ("ok", "reason", "hop_changes", "delay_ms", "compute_p50_ms",
                                           "compute_p95_ms", "overruns", "late", "skipped", "utilisation")}
    counters.update({"wall_s": wall, "enroller_source": getattr(enr, "source", None),
                     "enroller_ready": bool(getattr(enr, "ready", False)), "delay_samples": delay})
    return y, counters


def condition_report(runner_factory, t, o, mix, args, live_factory=None, enroll=None, other_clip=None):
    runner = runner_factory()
    rep = d.replay(runner, mix, t, o, window_s=args.window_s, hop_s=args.hop_s,
                   sep_min_db=args.sep_min_db, drop_db=args.drop_db)
    win, hop = int(args.window_s * SR), int(args.hop_s * SR)
    recs = rep.windows[len(rep.windows) // 2:] if args.split == "heldout" else rep.windows
    conf = conf_per_sample(rep.windows, len(mix), hop)
    outs, info = {}, {}
    outs["raw_mix"] = mix
    outs["oracle_candidate"], outs["actual_selected"], outs["proxy_mixer"] = rep.oracle_est, rep.estimate, rep.final
    info["real_mixer"] = {}
    outs["real_mixer"], info["real_mixer"] = run_real_mixer(rep.estimate, mix, conf, args.suppression)
    outs["real_mixer_agc"] = apply_agc(outs["real_mixer"])
    outs["cf_floor24_mixer"], info["cf_floor24_mixer"] = run_real_mixer(
        rep.estimate, mix, None, args.suppression, max_duck_db=24.0, use_conf=False)
    if live_factory is not None:
        try:
            outs["live_wrapper"], info["live_wrapper"] = live_wrapper_replay(
                live_factory, enroll, other_clip, mix, args.window_s, args.hop_s)
        except Exception as e:  # noqa: BLE001  report the exact blocker, do not infer
            info["live_wrapper"] = {"error": f"{type(e).__name__}: {e}"}
    rows = {}
    for name in STAGES:
        if name in outs:
            rows[name] = {**PROVENANCE[name], **stage_metrics(outs[name], mix, t, o, recs, win), **info.get(name, {})}
        else:
            rows[name] = {**PROVENANCE[name], "status": "not_run", **info.get(name, {})}
    base = d.summarize(rep, mix, t, o, args.split)
    sel = [r for r in rep.windows if r["extra"]]
    counters = {
        "n_deadline_missed": base["n_deadline_missed"], "n_raw_fallback": base["n_raw_fallback"],
        "n_skipped_gated": base["n_skipped"], "compute_ms_mean": base["compute_ms_mean"],
        "compute_ms_max": base["compute_ms_max"], "window_s": args.window_s, "hop_s": args.hop_s,
        "n_held": sum(1 for r in sel if r["extra"].get("held")), "n_paused": sum(1 for r in sel if r["extra"].get("paused")),
        "n_vetoed": sum(1 for r in sel if r["extra"].get("vetoed")),
    }
    return {"stages": rows, "categories": base["categories"], "dominant_failure": base["dominant_failure"],
            "fix_owner": FIX_OWNERS[base["dominant_failure"]], "counters": counters,
            "selector_agrees_with_oracle": base["selector_agrees_with_oracle"],
            "failure_attribution": attribute(rows, base)}


def attribute(rows, base):
    """Name the first failing link: model target loss, wrong selection, output leak, scheduler."""
    if base["n_deadline_missed"]:
        return "scheduler_failure"
    c = base["categories"]
    if c.get("separation"):
        return "model_target_loss"
    if c.get("selector"):
        return "wrong_selection"
    rm, act = rows["real_mixer"].get("sisdr_absolute_db"), rows["actual_selected"].get("sisdr_absolute_db")
    if c.get("mix_back") or (rm is not None and act is not None and rm < act - 3.0):
        return "output_leak"
    return "none"


def dry_run(args):
    t, o = d.fake_stems(args.fake_seconds)
    mix = (t + o).astype(np.float32)
    results = []
    for case in d.FakeRunner.CASES:
        rep = condition_report(lambda c=case: d.FakeRunner(c, t, o, args.hop_s), t, o, mix, args)
        results.append({"condition": f"FAKE:{case}", **rep})
    return {"mode": "dry_run", "fake": True, "label": "FAKE models and constructed stems: adapter check, NOT measured separation",
            "results": results, "inputs": {}, "device": "fake"}


def gaps(args, has_transcripts=False):
    return {
        "strict_pause_gate_session_hooks": "not_exercised: needs the live session (control hooks); no offline arithmetic substituted",
        "same_bucket_active_tier": "not_exercised: needs the live tier ladder",
        "negative_active": "partial: the negative profile is passed to the real EnrolledCandidate; no live session flag",
        "word_check": {"status": "present" if has_transcripts else "missing_transcripts",
                       "note": "no independent target-word transcripts supplied; SI-SDR/leak only"},
        "room_conclusion": "preliminary: digital mix of one scene; needs independent same-mic/session data",
    }


def real_run(args):
    scene = Path(args.scene)
    man = json.loads((scene / "manifest.json").read_text())
    stems = {k: scene / v for k, v in man["stems"].items()}
    prof = {k: sorted((Path(args.profiles) / n).glob("profile-*.wav")) for k, n in
            (("target", args.target_name), ("other", args.other_name))}
    missing = [str(p) for p in stems.values() if not p.exists()] + [f"profile wavs for {k}" for k, v in prof.items() if not v]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))
    t0_, o0_ = d._read(stems["target"]), d._read(stems["other"])
    noise = d._read(stems["noise"]) if args.noise and "noise" in stems else None
    pt, po = d._read(prof["target"][0]), d._read(prof["other"][0])
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enrolled import EnrolledCandidate

    sep = BlindSeparator()
    probe = EnrolledCandidate(separator=sep)
    probe.load()
    results = []
    roles = ["A", "B"] if args.roles == "both" else [args.roles]
    for role in roles:
        a, b = (t0_, o0_) if role == "A" else (o0_, t0_)
        ea, eo = (pt, po) if role == "A" else (po, pt)
        for ratio in args.ratios:
            t, o = d.scale_target(a, b, ratio)
            n = min(len(t), len(o))
            t, o = t[:n], o[:n]
            mix = (t + o + (noise[:n] if noise is not None else 0)).astype(np.float32)
            rep = condition_report(
                lambda ea=ea, eo=eo: d.EnrolledRunner(ea, eo, separator=sep, embedder=probe.embedder), t, o, mix, args,
                live_factory=(lambda: EnrolledCandidate(separator=sep, embedder=probe.embedder)) if args.live else None,
                enroll=ea, other_clip=eo)
            results.append({"condition": f"role{role} target{ratio:+.0f}dB", "role": role, "ratio_db": ratio, **rep})
    inputs = {"scene": str(scene), "scene_manifest_sha": d._sha(scene / "manifest.json"),
              **{f"stem_{k}": d._sha(p) for k, p in stems.items()},
              "profile_target": d._sha(prof["target"][0]), "profile_other": d._sha(prof["other"][0])}
    return {"mode": "real", "fake": False, "results": results, "inputs": inputs, "device": str(getattr(sep, "device", ""))}


def table(results):
    def f(x, nd=1):
        return "-" if x is None else f"{x:.{nd}f}"
    head = "| condition | stage | real? | SI-SDR abs dB | SI-SDRi dB | wrong-spk win | absent leak dB |\n|---|---|---|---|---|---|---|"
    lines = [head]
    for r in results:
        for name, s in r["stages"].items():
            if s.get("status") == "not_run":
                lines.append(f"| {r['condition']} | {name} | - | not run {s.get('error', '')} | | | |")
                continue
            tag = "PROXY" if s["proxy"] else ("counterfactual" if s.get("counterfactual") else
                                              ("oracle" if not s["deployable"] else "real"))
            lines.append(f"| {r['condition']} | {name} | {tag} | {f(s['sisdr_absolute_db'])} | {f(s['sisdri_db'])} | "
                         f"{s['wrong_speaker_windows']} | {f(s['target_absent_leak_db'])} |")
        lines.append(f"| {r['condition']} | **attribution** | {r['failure_attribution']} | fix: {r['fix_owner']} | | | |")
    return "\n".join(lines)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--scene")
    p.add_argument("--profiles")
    p.add_argument("--target-name", default="aryan")
    p.add_argument("--other-name", default="joseph")
    p.add_argument("--ratios", type=float, nargs="+", default=list(d.DEFAULT_RATIOS))
    p.add_argument("--roles", choices=("A", "B", "both"), default="both")
    p.add_argument("--split", choices=("all", "heldout"), default="heldout")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--hop-s", type=float, default=1.0)
    p.add_argument("--sep-min-db", type=float, default=4.0)
    p.add_argument("--drop-db", type=float, default=3.0)
    p.add_argument("--suppression", type=float, default=1.0, help="slider for the real Mixer (1 = target only)")
    p.add_argument("--live", action="store_true", help="also replay through the real FocusHQ stage")
    p.add_argument("--fake-seconds", type=float, default=8.0)
    p.add_argument("--out")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    if not args.dry_run and not (args.scene and args.profiles):
        print("BLOCKED: real run needs --scene and --profiles (or --dry-run)", file=sys.stderr)
        return 2
    try:
        data = dry_run(args) if args.dry_run else real_run(args)
    except FileNotFoundError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    data.update({"stamp": datetime.now(UTC).isoformat(timespec="seconds"), "host": platform.node(),
                 "config": vars(args), "gaps": gaps(args)})
    print(("** " + data["label"] + " **\n") if data["fake"] else "", end="")
    print(table(data["results"]))
    out = Path(args.out) if args.out else Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / "isolation_live_path" / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + (".dryrun" if args.dry_run else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, default=float))
    print("manifest:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
