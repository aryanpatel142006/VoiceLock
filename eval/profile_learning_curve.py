"""Offline learning curve for optional voice-profile personalization (T-201).

Question: do 1, 2, 4 and 8 user-confirmed clean examples recognise the target better than the first
clean example alone, without accepting the wrong speaker or noise more often? ECAPA stays frozen,
no separator runs, no live code is touched, nothing is promoted.

* The enrollment pool and the evaluation recordings are **different audio**: ``check_disjoint`` fails
  the run if any evaluation block also occurs in the pool (exact int16 block match at any offset).
* Examples are nested and time-spread (``spread_order``): n=1 is the first clean example, n=2 adds the
  last one, n=4 and n=8 fill the gaps, so a larger n always contains the smaller one.
* Methods: ``first`` (the first example only), ``mean`` (normalised mean of unit embeddings, which is
  what ``hearing.tse.enrolled`` already does for a saved profile), ``multi`` (max cosine over the
  example prototypes). ``reference_full`` = the existing whole saved profile, a separate row.
* The evaluation recordings are cut into windows; the first half (time) calibrates one accept
  threshold per profile (95th percentile of the dev negative scores), the second half is scored.
  Conditions: target present clean, target present quiet (-12 dB plus a noise floor), wrong speaker
  clean/quiet, and target absent (noise only).
* ``promotion_verdict`` never says "promote": the best it can say is ``embedding_gain_only``
  (needs a separator/word-retention/delay check on top), otherwise ``no_win``.

    python -m eval.profile_learning_curve --dry-run
    python -m eval.profile_learning_curve            # existing team recordings, real ECAPA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SR = 16000
NS = (1, 2, 4, 8)
METHODS = ("first", "mean", "multi")
EX_S = 1.5  # one confirmed example (hearing.tse.enroll.PROFILE_MIN_PIECE_S: the shortest profile piece)
CROPS_S = (1.0, 2.0)  # ECAPA crop duration under test (T-200: margin falls from 0.77 at 4 s to 0.42 at 0.5 s)
HOP_S = 1.0
QUIET_DB = -12.0
NOISE_DBFS = -45.0  # noise floor added to the quiet-target condition
ABSENT_NOISE_DBFS = -35.0  # the target-absent noise-only windows
TARGET_FAR = 0.05  # dev-set false-accept rate the per-profile threshold is set for
MIN_GAIN = 0.05  # mean absolute TPR gain (clean and quiet) that counts as a recognition gain
VOICED_FRAC = 0.6  # share of 20 ms frames above the voiced level for a clean example / eval window
VOICED_DB = -45.0  # a frame is voiced at/above this level (dBFS rms)
CLIP_FRAC = 0.001  # more samples than this at full scale: clipped, rejected
N_SUBSETS = 20  # random k-subsets for the order-sensitivity spread
SAME_VOICE = 0.7  # hearing.tse.enrolled.SAME_VOICE: target/negative profile closer than this = same person
DEFAULT_SEED = 201


def _main_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _generated() -> Path:
    env = os.environ.get("HEARING_FIXTURES_DIR")
    return Path(env) if env else _main_root() / "eval" / "fixtures" / "generated"


def default_inputs(gen: Path | None = None) -> dict:
    """The existing recordings (Oct 9 Voice Memos): pool = saved enrollment, holdout = different sentences."""
    gen = gen or _generated()
    return {
        "aryan": {"pool": gen / "team/aryan/profile-01.wav", "holdout": gen / "team/aryan/fixture-01.wav"},
        "joseph": {"pool": gen / "team/joseph/profile-01.wav", "holdout": gen / "team/joseph/fixture-01.wav"},
    }


# ---------------------------------------------------------------- audio helpers

def _frame_db(x: np.ndarray, frame: int = 320) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([-120.0])
    f = x[: n * frame].reshape(n, frame).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt((f * f).mean(axis=1)), 1e-9))


def is_clean(x: np.ndarray) -> bool:
    """Voiced enough, no clipping (the quality gate VOICE_EMBEDDINGS.md asks for; overlap is not checkable
    on a single-speaker recording and is declared, not tested)."""
    x = np.asarray(x, np.float32)
    if len(x) == 0 or not np.isfinite(x).all():
        return False
    if float(np.mean(np.abs(x) >= 0.999)) > CLIP_FRAC:
        return False
    return float(np.mean(_frame_db(x) >= VOICED_DB)) >= VOICED_FRAC


def cut(x: np.ndarray, length_s: float, hop_s: float | None = None, sr: int = SR) -> list[tuple[int, np.ndarray]]:
    n, h = round(length_s * sr), round((hop_s or length_s) * sr)
    return [(i, x[i : i + n]) for i in range(0, len(x) - n + 1, h)]


def clean_examples(pool: np.ndarray, ex_s: float = EX_S) -> list[tuple[int, np.ndarray]]:
    """Non-overlapping clean examples of ``ex_s`` from the pool, in time order: ``(start sample, audio)``."""
    return [(i, w) for i, w in cut(pool, ex_s) if is_clean(w)]


def spread_order(m: int) -> list[int]:
    """Indices 0..m-1 ordered so every prefix is time-spread: 0, last, middle, then the gaps."""
    if m <= 0:
        return []
    order, seen = [0], {0}
    queue = [(0, m - 1)] if m > 1 else []
    if m > 1:
        order.append(m - 1)
        seen.add(m - 1)
    while queue:
        nxt = []
        for a, b in queue:
            mid = (a + b) // 2
            if mid not in seen:
                order.append(mid)
                seen.add(mid)
            if mid - a > 1:
                nxt.append((a, mid))
            if b - mid > 1:
                nxt.append((mid, b))
        queue = nxt
    order += [i for i in range(m) if i not in seen]
    return order


def check_disjoint(pool: np.ndarray, holdout: np.ndarray, block: int = 4000) -> bool:
    """True when no ``block``-sample int16 slice of ``holdout`` (aligned) occurs anywhere in ``pool``."""
    p = np.round(np.asarray(pool, np.float64) * 32767).astype(np.int16)
    h = np.round(np.asarray(holdout, np.float64) * 32767).astype(np.int16)
    if len(p) < block or len(h) < block:
        return True
    probe = 64
    seen = {p[i : i + probe].tobytes() for i in range(len(p) - block + 1)}
    for i in range(0, len(h) - block + 1, block):
        if h[i : i + probe].tobytes() in seen and np.any(h[i : i + probe] != 0):
            return False
    return True


def split_dev_test(windows: list, gap: int = 1) -> tuple[list, list]:
    """Contiguous time halves with ``gap`` windows dropped between them (dev calibrates, test scores)."""
    h = len(windows) // 2
    return windows[:h], windows[h + gap :]


def _noise(n: int, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n + 16)
    x = np.convolve(w, np.ones(8) / 8.0, mode="valid")[:n]  # low-passed: closer to room noise than white
    return (x / max(float(np.sqrt(np.mean(x * x))), 1e-9) * 10 ** (dbfs / 20)).astype(np.float32)


def quiet_version(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (w * 10 ** (QUIET_DB / 20) + _noise(len(w), NOISE_DBFS, rng)).astype(np.float32)


# ---------------------------------------------------------------- profile methods

def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def build_profile(units: np.ndarray) -> dict:
    """The three candidate profile shapes from ``(n, d)`` unit embeddings."""
    units = np.atleast_2d(units)
    return {"first": units[:1], "mean": unit(units.mean(axis=0))[None], "multi": units}


def score(profile: np.ndarray, e: np.ndarray) -> float:
    """Max cosine over the profile's prototypes (one prototype for first/mean)."""
    return float(np.max(profile @ unit(e)))


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, q = np.asarray(pos), np.asarray(neg)
    return float((p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean())


def calibrate_threshold(neg_scores: list[float], far: float = TARGET_FAR) -> float:
    if not neg_scores:
        return float("inf")
    return float(np.quantile(np.asarray(neg_scores), 1.0 - far, method="higher"))


def _rate(scores: list[float], thr: float) -> float | None:
    return float(np.mean(np.asarray(scores) >= thr)) if scores else None


def evaluate(profile: np.ndarray, other_profile: np.ndarray, dev: dict, test: dict, hop_s: float = HOP_S) -> dict:
    """Metrics for one profile. ``dev``/``test`` map condition -> list of unit embeddings.

    Conditions: present_clean, present_quiet, wrong_clean, wrong_quiet, absent_noise.
    """
    neg = [score(profile, e) for c in ("wrong_clean", "wrong_quiet", "absent_noise") for e in dev[c]]
    thr = calibrate_threshold(neg)
    s = {c: [score(profile, e) for e in v] for c, v in test.items()}
    wrong_fa = sum(x >= thr for c in ("wrong_clean", "wrong_quiet") for x in s[c])
    wrong_n = len(s["wrong_clean"]) + len(s["wrong_quiet"])
    # closed set (the repo's deployable pick between target and negative profile): margin > 0 = target
    margin = {c: [score(profile, e) - score(other_profile, e) for e in v] for c, v in test.items()}
    cs_pos = [m > 0 for c in ("present_clean", "present_quiet") for m in margin[c]]
    cs_neg = [m <= 0 for c in ("wrong_clean", "wrong_quiet") for m in margin[c]]
    return {
        "threshold": thr,
        "tpr_clean": _rate(s["present_clean"], thr),
        "tpr_quiet": _rate(s["present_quiet"], thr),
        "far_wrong": (wrong_fa / wrong_n) if wrong_n else None,
        "far_absent": _rate(s["absent_noise"], thr),
        "wrong_speaker_s": wrong_fa * hop_s,
        "wrong_total_s": wrong_n * hop_s,
        "auc_clean": auc(s["present_clean"], s["wrong_clean"] + s["absent_noise"]),
        "auc_quiet": auc(s["present_quiet"], s["wrong_clean"] + s["wrong_quiet"] + s["absent_noise"]),
        "closed_set_acc": float(np.mean(cs_pos + cs_neg)) if cs_pos + cs_neg else None,
        "n_test": {c: len(v) for c, v in test.items()},
    }


def _mean_tpr(m: dict) -> float:
    return float(np.mean([m["tpr_clean"], m["tpr_quiet"]]))


def promotion_verdict(base: dict, cand: dict) -> dict:
    """Never ``promote``. ``embedding_gain_only`` = recognition up with no worse wrong/absent acceptance;
    word retention, dropouts, intelligibility and delay are separate checks this offline eval cannot run."""
    reasons = []
    gain = _mean_tpr(cand) - _mean_tpr(base)
    if gain < MIN_GAIN:
        reasons.append(f"target recognition gain {gain:+.3f} < {MIN_GAIN}")
    for k in ("far_wrong", "far_absent"):
        if cand[k] is not None and base[k] is not None and cand[k] > base[k] + 1e-12:
            reasons.append(f"{k} worse: {cand[k]:.3f} > {base[k]:.3f}")
    if cand["wrong_speaker_s"] > base["wrong_speaker_s"] + 1e-12:
        reasons.append(f"wrong-speaker seconds worse: {cand['wrong_speaker_s']:.1f} > {base['wrong_speaker_s']:.1f}")
    if cand["closed_set_acc"] is not None and base["closed_set_acc"] is not None \
            and cand["closed_set_acc"] + 1e-12 < base["closed_set_acc"]:
        reasons.append("closed-set accuracy worse")
    return {"verdict": "no_win" if reasons else "embedding_gain_only", "tpr_gain": gain, "reasons": reasons}


# ---------------------------------------------------------------- experiment

class Embedder:
    """Frozen ECAPA through the repo loader (hearing.tse.enrolled); injectable for tests."""

    def __init__(self, backend=None):
        if backend is None:
            from hearing import models
            from hearing.tse import enrolled

            backend = models.load(enrolled.MODEL)
        self.backend = backend

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return unit(self.backend.embed(np.ascontiguousarray(audio, dtype=np.float32)))


def eval_windows(x: np.ndarray, crop_s: float) -> list[np.ndarray]:
    return [w for _, w in cut(x, crop_s, HOP_S) if is_clean(w)]


def condition_embeddings(own: np.ndarray, other: np.ndarray, crop_s: float, embed, rng) -> dict:
    """Embeddings per condition for one target; windows are split in time into dev and test halves."""
    ow, xw = eval_windows(own, crop_s), eval_windows(other, crop_s)
    od, ot = split_dev_test(ow)
    xd, xt = split_dev_test(xw)
    n = round(crop_s * SR)

    def absent(k):
        return [_noise(n, ABSENT_NOISE_DBFS, rng) for _ in range(k)]

    def emb(ws):
        return [embed(w) for w in ws]

    def side(o, x, a):
        return {"present_clean": emb(o), "present_quiet": emb([quiet_version(w, rng) for w in o]),
                "wrong_clean": emb(x), "wrong_quiet": emb([quiet_version(w, rng) for w in x]),
                "absent_noise": emb(a)}

    return {"dev": side(od, xd, absent(max(len(od), 1))), "test": side(ot, xt, absent(max(len(ot), 1)))}


def example_units(examples: list[np.ndarray], embed) -> np.ndarray:
    return np.stack([embed(e) for e in examples])


def run_target(name: str, own_pool, own_hold, other_pool, other_hold, embed, seed=DEFAULT_SEED, crops=CROPS_S) -> dict:
    rng = np.random.default_rng(seed)
    ex_own = clean_examples(own_pool)
    ex_oth = clean_examples(other_pool)
    out = {"target": name, "n_clean_examples": len(ex_own), "n_other_examples": len(ex_oth), "crops": {}}
    if len(ex_own) < max(NS) or len(ex_oth) < max(NS):
        out["blocked"] = f"needs {max(NS)} clean examples per speaker, have {len(ex_own)}/{len(ex_oth)}"
        return out
    order_o, order_x = spread_order(len(ex_own)), spread_order(len(ex_oth))
    u_own, u_oth = example_units([e for _, e in ex_own], embed), example_units([e for _, e in ex_oth], embed)
    # existing saved profile as a separate reference: mean of 2.5 s piece units, the whole pool
    from hearing.tse.enroll import split_pieces

    ref_units = example_units(split_pieces(own_pool), embed)
    ref_other = example_units(split_pieces(other_pool), embed)
    out["same_voice_cos"] = float(unit(u_own.mean(axis=0)) @ unit(u_oth.mean(axis=0)))
    out["negative_rejected_at_same_voice"] = out["same_voice_cos"] > SAME_VOICE
    for crop_s in crops:
        cond = condition_embeddings(own_hold, other_hold, crop_s, embed, rng)
        rows = {"reference_full": evaluate(build_profile(ref_units)["mean"], build_profile(ref_other)["mean"],
                                           cond["dev"], cond["test"])}
        rows["reference_full"]["n_examples"] = len(ref_units)
        for n in NS:
            po = build_profile(u_own[order_o[:n]])
            px = build_profile(u_oth[order_x[:n]])
            for m in METHODS:
                if n == 1 and m != "first":
                    continue  # a 1-example mean/multi is the same profile as first
                r = evaluate(po[m], px[m], cond["dev"], cond["test"])
                r["n_examples"] = n
                rows[f"{m}@{n}"] = r
        base = rows["first@1"]
        for k, r in rows.items():
            if k != "first@1":
                r["vs_first_1"] = promotion_verdict(base, r)
        spread = {}
        sub_rng = np.random.default_rng(seed + 1)
        for n in NS[1:]:
            vals = {m: [] for m in METHODS[1:]}
            for _ in range(N_SUBSETS):
                idx = sub_rng.choice(len(ex_own), n, replace=False)
                idx_x = sub_rng.choice(len(ex_oth), n, replace=False)
                po, px = build_profile(u_own[idx]), build_profile(u_oth[idx_x])
                for m in METHODS[1:]:
                    vals[m].append(_mean_tpr(evaluate(po[m], px[m], cond["dev"], cond["test"])))
            spread[str(n)] = {m: {"mean_tpr": float(np.mean(v)), "sd": float(np.std(v))} for m, v in vals.items()}
        out["crops"][f"{crop_s:g}"] = {"rows": rows, "random_subset_mean_tpr": spread}
    return out


def write_proposed(root: Path, name: str, examples: list[tuple[int, np.ndarray]], inputs: dict, run_id: str) -> list[str]:
    """Candidate profile versions (audio of the chosen examples) under the new experiment dir only.
    Never writes ``profiles/<name>/current`` or any live profile directory."""
    import soundfile as sf

    order = spread_order(len(examples))
    made = []
    for n in NS:
        d = root / name / f"exp-n{n}"
        d.mkdir(parents=True, exist_ok=True)
        sel = sorted(order[:n])
        sf.write(d / "profile-01.wav", np.concatenate([examples[i][1] for i in sel]), SR, subtype="PCM_16")
        (d / "manifest.json").write_text(json.dumps({
            "profile": name, "examples": n, "example_start_samples": [examples[i][0] for i in sel],
            "example_s": EX_S, "source_pool_sha256": inputs["pool_sha256"], "run": run_id, "frozen_encoder": "spkrec-ecapa-voxceleb",
            "promoted": False, "note": "proposal; the active profile and its `current` pointer are unchanged"}, indent=2))
        made.append(str(d))
    return made


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _results_dir() -> Path:
    env = os.environ.get("HEARING_RESULTS_DIR")
    return (Path(env) if env else _main_root() / "eval" / "results") / "profile_learning_curve"


def blocked_inputs(spec: dict) -> list[str]:
    return [str(p) for v in spec.values() for p in v.values() if not Path(p).is_file()]


def fake_voice(f0: float, seconds: float, seed: int) -> np.ndarray:
    """Synthetic harmonic 'speaker' for tests and --dry-run (not speech, never measured evidence)."""
    r = np.random.default_rng(seed)
    t = np.arange(round(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6)) / k for k in range(1, 12))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2)  # syllable-like envelope, always voiced
    return (0.3 * x / np.max(np.abs(x)) + 0.003 * r.standard_normal(len(t))).astype(np.float32)


class FakeBackend:
    """Log-spectrum bands: separates harmonic voices by f0. For dry-run/tests only."""

    def embed(self, audio):
        s = np.abs(np.fft.rfft(audio, n=4096))[:700]
        return np.log1p(np.array([b.mean() for b in np.array_split(s, 24)]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="synthetic voices + fake embedder: wiring check, not evidence")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-proposals", action="store_true")
    a = ap.parse_args(argv)
    import soundfile as sf

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rd = _results_dir()
    if a.dry_run:
        data = {"aryan": (fake_voice(110, 40, 1), fake_voice(110, 40, 2)), "joseph": (fake_voice(150, 40, 3), fake_voice(150, 40, 4))}
        embed, hashes, label = Embedder(FakeBackend()), {k: {"pool_sha256": "fake"} for k in data}, "FAKE (dry run)"
    else:
        spec = default_inputs()
        missing = blocked_inputs(spec)
        if missing:
            print("BLOCKED: missing inputs: " + ", ".join(missing), file=sys.stderr)
            return 2
        from hearing.tse import enrolled

        if not enrolled.available():
            print("BLOCKED: ECAPA weights/speechbrain missing", file=sys.stderr)
            return 2
        data, hashes = {}, {}
        for k, v in spec.items():
            pool, _ = sf.read(v["pool"], dtype="float32")
            hold, _ = sf.read(v["holdout"], dtype="float32")
            if not check_disjoint(pool, hold):
                print(f"BLOCKED: {k} holdout overlaps the enrollment pool", file=sys.stderr)
                return 2
            data[k] = (pool, hold)
            hashes[k] = {"pool_sha256": _sha(v["pool"]), "holdout_sha256": _sha(v["holdout"]),
                         "pool": v["pool"].name, "holdout": v["holdout"].name, "pool_s": len(pool) / SR, "holdout_s": len(hold) / SR}
        embed, label = Embedder(), "REAL recordings, frozen ECAPA"
    t0 = time.time()
    names = list(data)
    results = [run_target(n, data[n][0], data[n][1], data[o][0], data[o][1], embed, a.seed)
               for n, o in ((names[0], names[1]), (names[1], names[0]))]
    proposals = []
    if not a.no_proposals and not a.dry_run:
        for k in names:
            proposals += write_proposed(rd / "proposed", k, clean_examples(data[k][0]), hashes[k], stamp)
    report = {"label": label, "stamp": stamp, "host": platform.node(), "seconds": round(time.time() - t0, 1),
              "config": {"ex_s": EX_S, "crops_s": CROPS_S, "ns": NS, "far": TARGET_FAR, "min_gain": MIN_GAIN,
                         "quiet_db": QUIET_DB, "noise_dbfs": NOISE_DBFS, "seed": a.seed},
              "inputs": hashes, "results": results, "proposed_dirs": proposals}
    rd.mkdir(parents=True, exist_ok=True)
    out = rd / f"{'dry-' if a.dry_run else ''}{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[{label}] {out}")
    for r in results:
        print(f"target {r['target']}: " + (r.get("blocked") or f"{r['n_clean_examples']} clean examples, same_voice_cos={r['same_voice_cos']:.2f}"))
        for crop, c in r.get("crops", {}).items():
            print(f"  crop {crop} s")
            for k, m in c["rows"].items():
                v = m.get("vs_first_1", {}).get("verdict", "-")
                print(f"    {k:<15} tprC {m['tpr_clean']:.2f} tprQ {m['tpr_quiet']:.2f} farW {m['far_wrong']:.2f} "
                      f"farA {m['far_absent']:.2f} wrong_s {m['wrong_speaker_s']:.0f}/{m['wrong_total_s']:.0f} "
                      f"cs {m['closed_set_acc']:.2f} auc {m['auc_clean']:.2f}/{m['auc_quiet']:.2f}  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
