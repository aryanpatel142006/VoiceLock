"""T-248 failure-localizing benchmark: metric prototype on synthetic arrays (no model, no GPU, no audio files).

Subcommands: plan | run | report.  See docs/research/theory/benchmark_design.md.
Consumer: .venv/bin/python experiments/theory/T-248/bench_design.py run --max-cases 24 --seed 248
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
HOP = 160  # 10 ms
DEADLINE_MS = 10.0  # one hop, per docs/ARCHITECTURE.md latency budget
NONE, TGT, OTH = 0, 1, 2  # selector choices per frame

# ---------------------------------------------------------------- case matrix
ROLES = ("A", "B")
TARGET_DB = (0, -10, -15)
ROOMS = ("matched", "mismatched")
SCENARIOS = ("overlap", "target_absent", "pause_swap", "switch_reacq")
SYNTH_GROUPS = tuple(f"g{i:02d}" for i in range(12))


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def split_of(group: str, room_cond: str, seed: int) -> str:
    """Heldout by source group AND by room condition; a case is heldout only if both are heldout-side,
    and any case touching exactly one heldout side is DROPPED (disjointness)."""
    g = int(_h("grp", seed, group)[:8], 16) % 4 == 0
    room_cond.startswith("mismatched") and int(
        _h("room", seed, group[:1])[:8], 16
    ) % 4 == 0
    return "heldout" if g else "dev"


def build_matrix(seed: int) -> list[dict]:
    cases = []
    for gi, g in enumerate(SYNTH_GROUPS):
        for role in ROLES:
            for tdb in TARGET_DB:
                for room in ROOMS:
                    for sc in SCENARIOS:
                        if sc == "target_absent" and tdb != 0:
                            continue  # level irrelevant when the target is absent
                        cid = f"{g}-{role}-{tdb}-{room}-{sc}"
                        cases.append(
                            {
                                "case_id": cid,
                                "group": g,
                                "role": role,
                                "target_db": tdb,
                                "room": room,
                                "scenario": sc,
                                "voice_bucket": "same",
                                "split": split_of(g, room, seed),
                                "case_seed": int(_h(seed, cid)[:8], 16),
                            }
                        )
    return cases


def select_cases(matrix: list[dict], max_cases: int, split: str) -> list[dict]:
    pool = sorted(
        (c for c in matrix if c["split"] == split),
        key=lambda c: _h(c["case_seed"], c["case_id"]),
    )
    return pool[:max_cases]  # deterministic hash order = stratified-enough, no repeats


# ---------------------------------------------------------------- synthetic signals
def _speech(rng, n, spans):
    x = np.zeros(n, np.float32)
    for a, b in spans:
        seg = rng.standard_normal(b - a).astype(np.float32)
        env = np.hanning(b - a).astype(np.float32) ** 0.3
        x[a:b] = seg * env
    return x


def synth_case(case: dict, secs: float = 6.0):
    """Returns dict of arrays + word spans (samples). Scenario decides timing; scenario is the failure probe."""
    rng = np.random.default_rng(case["case_seed"])
    n = int(secs * SR)
    f = lambda s: int(s * SR)
    sc = case["scenario"]
    if sc == "overlap":
        tw = [(f(0.5), f(1.4)), (f(1.8), f(2.8)), (f(3.2), f(4.2)), (f(4.6), f(5.6))]
        ow = [(f(1.0), f(2.2)), (f(3.8), f(5.0))]
    elif sc == "target_absent":
        tw, ow = [], [(f(0.5), f(1.6)), (f(2.2), f(3.4)), (f(4.0), f(5.2))]
    elif sc == "pause_swap":
        tw = [(f(0.5), f(1.3)), (f(1.6), f(2.4)), (f(3.2), f(4.0)), (f(4.3), f(5.1))]
        ow = [
            (f(1.3), f(1.6)),
            (f(2.4), f(3.2)),
            (f(4.0), f(4.3)),
        ]  # interferer fills target pauses
    else:  # switch_reacq: interferer-only stretch, then target returns
        tw = [(f(0.4), f(1.2)), (f(3.0), f(3.9)), (f(4.2), f(5.2))]
        ow = [(f(1.3), f(2.9)), (f(3.0), f(3.6))]
    gain = 10 ** (case["target_db"] / 20)
    room_leak = (
        0.25 if case["room"] == "mismatched" else 0.08
    )  # separator leak of the other talker
    t = _speech(rng, n, tw) * gain
    i = _speech(rng, n, ow)
    sep_err = 0.12 if case["room"] == "matched" else 0.2
    s_t = t + sep_err * gain * rng.standard_normal(n).astype(np.float32) * (
        np.abs(t) > 0
    )
    s_t = s_t + room_leak * i
    s_o = i + room_leak * t
    return {"t": t, "i": i, "s_t": s_t, "s_o": s_o, "tw": tw, "ow": ow, "n": n}


def frame_mask(spans, n_frames):
    m = np.zeros(n_frames, bool)
    for a, b in spans:
        m[a // HOP : b // HOP] = True
    return m


def oracle_choice(sig, n_frames):
    tm, _om = frame_mask(sig["tw"], n_frames), frame_mask(sig["ow"], n_frames)
    c = np.full(
        n_frames, NONE, np.int8
    )  # oracle output = target only; everything else silent
    c[tm] = TGT
    return c


def selector_model(sig, n_frames, case, rng, onset_delay=6, hold=0, p_false=0.0):
    """Synthetic selector over the oracle: delayed target onset (reacquisition), `hold` frames of hysteresis
    after target offset, and per-interferer-segment false accept (the whole segment is passed as TGT-stream
    choice OTH, which covers pause swaps when the segment sits in a target pause).
    A real selector's decisions replace this array; metrics below do not care where it came from."""
    orc = oracle_choice(sig, n_frames)
    c = np.full(n_frames, NONE, np.int8)
    for a, b in runs(orc == TGT):
        c[a + onset_delay : b + hold] = TGT
    om = frame_mask(sig["ow"], n_frames)
    for a, b in runs(om):
        if p_false and rng.random() < p_false:
            seg = slice(a + onset_delay, b)
            c[seg] = np.where(c[seg] == TGT, TGT, OTH)
    return c


# ---------------------------------------------------------------- rendering stages
def render(sig, choice, post_gain=None, n_frames=None, dropped=None):
    n_frames = n_frames or len(choice)
    y = np.zeros(sig["n"], np.float32)
    for k in range(n_frames):
        a, b = k * HOP, (k + 1) * HOP
        if dropped is not None and dropped[k]:
            continue  # deadline miss without fallback: the frame is lost
        if choice[k] == TGT:
            y[a:b] = sig["s_t"][a:b]
        elif choice[k] == OTH:
            y[a:b] = sig["s_o"][a:b]
    if post_gain is not None:
        y = y * np.repeat(post_gain, HOP)[: len(y)]
    return y


def agc_gain(y, target_rms=0.1, tau_frames=30, max_db=6.0):
    """Toy AGC: RMS tracker updated on active frames only, gain held in silence; a measurable post-stage effect."""
    nf = len(y) // HOP
    g = np.ones(nf, np.float32)
    env, lim = 0.05, 10 ** (max_db / 20)
    for k in range(nf):
        r = float(np.sqrt(np.mean(y[k * HOP : (k + 1) * HOP] ** 2)))
        if r > 1e-2:
            env += (r - env) / tau_frames
        g[k] = np.clip(target_rms / max(env, 1e-4), 1 / lim, lim)
    return g


# ---------------------------------------------------------------- metrics
def si_sdr(est, ref, mask=None, eps=1e-9):
    if mask is not None:
        est, ref = est[mask], ref[mask]
    if ref.size == 0 or float(np.sum(ref**2)) < eps:
        return float("nan")
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    tgt = a * ref
    return float(
        10 * np.log10((np.sum(tgt**2) + eps) / (np.sum((est - tgt) ** 2) + eps))
    )


def sample_mask(spans, n):
    m = np.zeros(n, bool)
    for a, b in spans:
        m[a:b] = True
    return m


def speech_frames(y, thr_db=-40.0):
    nf = len(y) // HOP
    r = np.sqrt(np.mean(y[: nf * HOP].reshape(nf, HOP) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(r) > thr_db


def runs(mask):
    out, s = [], None
    for k, v in enumerate(mask):
        if v and s is None:
            s = k
        if not v and s is not None:
            out.append((s, k))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def selector_metrics(choice, orc, tw, ow, nf):
    tm = frame_mask(tw, nf)
    om = frame_mask(ow, nf)
    absent = ~tm
    out = {}
    out["target_miss_rate"] = (
        float(np.mean(choice[tm] != TGT)) if tm.any() else float("nan")
    )
    out["wrong_speaker_rate"] = (
        float(np.mean(choice[tm] == OTH)) if tm.any() else float("nan")
    )
    out["false_accept_rate"] = (
        float(np.mean(choice[absent & om] == OTH))
        if (absent & om).any()
        else float("nan")
    )
    out["target_absent_output_rate"] = (
        float(np.mean(choice[absent] != NONE)) if absent.any() else float("nan")
    )
    out["frame_agreement"] = float(np.mean(choice == orc))
    # hysteresis / pause swaps: gaps between consecutive target words of 100..800 ms
    gaps = [(tw[j][1] // HOP, tw[j + 1][0] // HOP) for j in range(len(tw) - 1)]
    gaps = [(a, b) for a, b in gaps if 10 <= b - a <= 80]
    swaps = sum(1 for a, b in gaps if (choice[a:b] == OTH).any())
    out["pause_swap_count"], out["pause_count"] = int(swaps), len(gaps)
    out["pause_swap_rate"] = swaps / len(gaps) if gaps else float("nan")
    # switching / reacquisition: target onset after >=500 ms without target
    lat = []
    for j, (a, _) in enumerate(tw):
        prev_end = tw[j - 1][1] if j else -10 * SR
        if a - prev_end >= SR // 2:
            fa = a // HOP
            k = fa
            while k < nf and choice[k] != TGT:
                k += 1
            lat.append((k - fa) * 10.0 if k < nf else float("inf"))
    out["reacq_latency_ms"] = lat
    return out


def word_boundary_metrics(sig, choice, y_pre, edge_ms=50):
    """Per word: energy kept (output vs target reference) in first/last edge_ms and whole word, in dB re ref."""
    edge = edge_ms * SR // 1000
    res = []
    for a, b in sig["tw"]:

        def kept(x0, x1):
            ref = float(np.sum(sig["t"][x0:x1] ** 2)) + 1e-12
            got = float(np.sum(y_pre[x0:x1] ** 2)) + 1e-12
            return 10 * np.log10(got / ref)

        # onset clip: ms from word start until target chosen
        fa, fb = a // HOP, b // HOP
        k = fa
        while k < fb and choice[k] != TGT:
            k += 1
        kt = fb
        while kt > fa and choice[kt - 1] != TGT:
            kt -= 1
        res.append(
            {
                "onset_clip_ms": (k - fa) * 10.0,
                "tail_clip_ms": (fb - kt) * 10.0,
                "head_db": kept(a, a + edge),
                "tail_db": kept(b - edge, b),
                "word_db": kept(a, b),
            }
        )
    return res


def target_absent_false_speech(y, sig, thr_db=-40.0):
    """Seconds of output speech-active frames where the reference target is silent (any source)."""
    nf = len(y) // HOP
    act = speech_frames(y, thr_db)[:nf]
    absent = ~frame_mask(sig["tw"], nf)
    other_src = frame_mask(sig["ow"], nf)
    return {
        "false_speech_s": float(np.sum(act & absent) * HOP / SR),
        "false_speech_interferer_s": float(np.sum(act & absent & other_src) * HOP / SR),
        "absent_s": float(np.sum(absent) * HOP / SR),
    }


def streaming_metrics(frame_ms, deadline=DEADLINE_MS):
    miss = np.asarray(frame_ms) > deadline
    bursts = [b - a for a, b in runs(miss)]
    return {
        "deadline_miss_rate": float(miss.mean()),
        "n_miss": int(miss.sum()),
        "max_miss_burst_frames": int(max(bursts) if bursts else 0),
        "p99_ms": float(np.percentile(frame_ms, 99)),
    }


def apply_fallback(choice, frame_ms, deadline=DEADLINE_MS):
    """Fallback policy under test: on a missed frame hold the previous frame's choice (never swap on a miss)."""
    c = choice.copy()
    miss = np.asarray(frame_ms) > deadline
    for k in range(1, len(c)):
        if miss[k]:
            c[k] = c[k - 1]
    return c, miss


def _post_sdr(y, sig, tm_s):
    if not sig["tw"]:
        return float("nan")
    return si_sdr(y * np.repeat(agc_gain(y), HOP)[: len(y)], sig["t"], tm_s)


# ---------------------------------------------------------------- one case, all stages
def eval_case(case: dict, knobs: dict) -> dict:
    sig = synth_case(case)
    nf = sig["n"] // HOP
    rng = np.random.default_rng(case["case_seed"] + 1)
    orc = oracle_choice(sig, nf)
    sel = selector_model(sig, nf, case, rng, **knobs["selector"])
    ms = np.abs(rng.normal(knobs["stream"]["mean_ms"], knobs["stream"]["sd_ms"], nf))
    sel_fb, missed = apply_fallback(sel, ms)
    tm_s = sample_mask(sig["tw"], sig["n"])
    # stage 0: preservation BEFORE selection: separator stream vs target on target words, gate fully open
    pres = si_sdr(sig["s_t"], sig["t"], tm_s) if sig["tw"] else float("nan")
    y_orc = render(sig, orc, None)
    y_sel = render(sig, sel_fb, None)
    g = agc_gain(y_sel)
    y_post = y_sel * np.repeat(g, HOP)[: len(y_sel)]
    y_orc_post = y_orc * np.repeat(agc_gain(y_orc), HOP)[: len(y_orc)]
    # ablations: replace each stage with its oracle (one at a time), full pipeline = selector+post
    sep_orc = dict(sig, s_t=sig["t"], s_o=sig["i"])
    abl = {
        "full": si_sdr(y_post, sig["t"], tm_s) if sig["tw"] else float("nan"),
        "oracle_selector": si_sdr(y_orc_post, sig["t"], tm_s)
        if sig["tw"]
        else float("nan"),
        "no_post": si_sdr(y_sel, sig["t"], tm_s) if sig["tw"] else float("nan"),
        "oracle_separator": si_sdr(
            render(sep_orc, sel_fb) * np.repeat(g, HOP)[: sig["n"]], sig["t"], tm_s
        )
        if sig["tw"]
        else float("nan"),
        "no_fallback": _post_sdr(render(sig, sel, dropped=ms > DEADLINE_MS), sig, tm_s),
    }
    gdb = 20 * np.log10(g)
    sp = frame_mask(sig["tw"], nf)
    post = {
        "gain_mean_db": float(gdb.mean()),
        "gain_std_db": float(gdb.std()),
        "gain_speech_swing_db": float(np.ptp(gdb[sp])) if sp.any() else float("nan"),
    }
    return {
        "case": case,
        "preservation_si_sdr_db": pres,
        "ablation_si_sdr_db": abl,
        "selector": selector_metrics(sel_fb, orc, sig["tw"], sig["ow"], nf),
        "post": post,
        "words": word_boundary_metrics(sig, sel_fb, y_sel),
        "false_speech": target_absent_false_speech(y_post, sig),
        "streaming": streaming_metrics(ms) | {"fallback_frames": int(missed.sum())},
    }


# ---------------------------------------------------------------- statistics
def cluster_bootstrap(values_by_group: dict, n_boot=1000, seed=0, stat=np.nanmean):
    """Resample SOURCE GROUPS (not cases): cases of one group are correlated. Returns (est, lo, hi)."""
    groups = sorted(values_by_group)
    rng = np.random.default_rng(seed)
    vals = [np.asarray(values_by_group[g], float) for g in groups]
    pooled = np.concatenate(vals) if vals else np.array([np.nan])
    est = float(stat(pooled)) if pooled.size else float("nan")
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        c = np.concatenate([vals[k] for k in pick]) if len(pick) else np.array([np.nan])
        bs.append(stat(c))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
    return est, float(lo), float(hi)


def collect(results, getter):
    by = {}
    for r in results:
        v = getter(r)
        vs = [
            x
            for x in (v if isinstance(v, list) else [v])
            if x is not None and np.isfinite(x)
        ]
        by.setdefault(r["case"]["group"], []).extend(vs)
    return {g: v for g, v in by.items() if v}


def summarize(results, seed=0, n_boot=500):
    def ci(getter):
        e, lo, hi = cluster_bootstrap(collect(results, getter), n_boot, seed)
        return {"est": e, "lo": lo, "hi": hi}

    out = {
        "n_cases": len(results),
        "n_groups": len({r["case"]["group"] for r in results}),
    }
    out["preservation_si_sdr_db"] = ci(lambda r: r["preservation_si_sdr_db"])
    for k in ("oracle_selector", "full", "no_post", "oracle_separator", "no_fallback"):
        out[f"ablation_{k}"] = ci(lambda r, k=k: r["ablation_si_sdr_db"][k])
    # causal attribution = paired difference (same case) vs full pipeline
    for k in ("oracle_selector", "no_post", "oracle_separator", "no_fallback"):
        out[f"delta_{k}_minus_full"] = ci(
            lambda r, k=k: r["ablation_si_sdr_db"][k] - r["ablation_si_sdr_db"]["full"]
        )
    for k in (
        "target_miss_rate",
        "wrong_speaker_rate",
        "false_accept_rate",
        "pause_swap_rate",
    ):
        out[f"selector_{k}"] = ci(lambda r, k=k: r["selector"][k])
    out["reacq_latency_ms"] = ci(
        lambda r: [x for x in r["selector"]["reacq_latency_ms"] if np.isfinite(x)]
    )
    out["onset_clip_ms"] = ci(lambda r: [w["onset_clip_ms"] for w in r["words"]])
    out["false_speech_s"] = ci(
        lambda r: (
            r["false_speech"]["false_speech_s"]
            if r["case"]["scenario"] == "target_absent"
            else None
        )
    )
    out["gain_speech_swing_db"] = ci(lambda r: r["post"]["gain_speech_swing_db"])
    out["deadline_miss_rate"] = ci(lambda r: r["streaming"]["deadline_miss_rate"])
    # failure localization: which stage owns the largest paired loss, per scenario
    loc = {}
    for sc in SCENARIOS:
        rs = [
            r
            for r in results
            if r["case"]["scenario"] == sc
            and np.isfinite(r["ablation_si_sdr_db"]["full"])
        ]
        if rs:
            loc[sc] = {
                k: float(
                    np.mean(
                        [
                            r["ablation_si_sdr_db"][k] - r["ablation_si_sdr_db"]["full"]
                            for r in rs
                        ]
                    )
                )
                for k in (
                    "oracle_selector",
                    "no_post",
                    "oracle_separator",
                    "no_fallback",
                )
            }
    out["gain_by_scenario_if_stage_were_oracle_or_removed_db"] = loc
    return out


# ---------------------------------------------------------------- CLI
def out_dir() -> Path:
    base = os.environ.get("HEARING_RESULTS_DIR") or (
        Path(os.environ.get("TMPDIR", "/tmp")) / "T-248-results"
    )
    p = Path(base) / "overnight_prep" / "T-248"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_knobs():
    return {
        "selector": {"onset_delay": 6, "hold": 3, "p_false": 0.4},
        "stream": {"mean_ms": 4.0, "sd_ms": 2.5},
    }


def config_hash(knobs, seed, split):
    return _h(json.dumps(knobs, sort_keys=True), seed, split, "v1")


def run(args) -> dict:
    knobs = default_knobs()
    matrix = build_matrix(args.seed)
    cases = select_cases(matrix, min(args.max_cases, 300), args.split)
    od = out_dir() / f"{args.split}-seed{args.seed}"
    (od / "cases").mkdir(parents=True, exist_ok=True)
    chash = config_hash(knobs, args.seed, args.split)
    t0, done, skipped, byte_total, status = time.time(), 0, 0, 0, "complete"
    for c in cases:
        p = od / "cases" / f"{c['case_id']}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if (
                    d.get("config_hash") == chash
                    and d.get("case", {}).get("case_id") == c["case_id"]
                ):
                    skipped += 1
                    continue
            except json.JSONDecodeError:
                pass  # torn/invalid checkpoint: recompute
        if time.time() - t0 > args.time_cap_s or byte_total > args.byte_cap:
            status = "stopped_cap"
            break
        try:
            r = eval_case(c, knobs)
        except Exception as e:  # noqa: BLE001  failure status per case; deterministic stop, no retries
            status = f"failed:{c['case_id']}:{type(e).__name__}"
            break
        r["config_hash"] = chash
        tmp = p.with_suffix(".tmp")
        s = json.dumps(
            r,
            default=lambda o: (
                None if o is None else float(o) if np.isscalar(o) else list(o)
            ),
        )
        tmp.write_text(s)
        os.replace(tmp, p)  # atomic
        byte_total += len(s)
        done += 1
    manifest = {
        "task": "T-248",
        "seed": args.seed,
        "split": args.split,
        "config_hash": chash,
        "knobs": knobs,
        "matrix_planned": len(matrix),
        "matrix_split_planned": sum(c["split"] == args.split for c in matrix),
        "selected": len(cases),
        "computed": done,
        "resumed_skipped": skipped,
        "status": status,
        "real_inputs_available": 0,
        "real_inputs_used": 0,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (od / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_results(od: Path):
    return [json.loads(p.read_text()) for p in sorted((od / "cases").glob("*.json"))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, default=248)
        sp.add_argument("--split", choices=("dev", "heldout"), default="dev")
        sp.add_argument("--max-cases", type=int, default=24)
        sp.add_argument("--time-cap-s", type=float, default=300)
        sp.add_argument("--byte-cap", type=int, default=64 * 2**20)
        sp.add_argument(
            "--final-heldout",
            action="store_true",
            help="required to touch the heldout split; one use, no tuning afterwards",
        )
    a = ap.parse_args(argv)
    if a.split == "heldout" and not a.final_heldout:
        print("refusing heldout split without --final-heldout", file=sys.stderr)
        return 2
    if a.cmd == "plan":
        m = build_matrix(a.seed)
        print(
            json.dumps(
                {
                    "planned": len(m),
                    "dev": sum(c["split"] == "dev" for c in m),
                    "heldout": sum(c["split"] == "heldout" for c in m),
                },
                indent=1,
            )
        )
        return 0
    man = run(a) if a.cmd == "run" else None
    if man:
        print(json.dumps(man, indent=1))
    od = out_dir() / f"{a.split}-seed{a.seed}"
    res = load_results(od)
    s = summarize(res, a.seed)
    (od / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    return 0 if (man is None or man["status"] in ("complete", "stopped_cap")) else 1


if __name__ == "__main__":
    sys.exit(main())
