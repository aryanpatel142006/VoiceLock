"""Offline comparison: SepFormer WHAMR16k vs the current MossFormer2_SS (T-202). Not live, not a registry entry.

One extra candidate (``speechbrain/sepformer-whamr16k``, pinned revision, Apache-2.0 per its model card) is
put behind the same two-stream separator interface as ``hearing.tse.blind.BlindSeparator`` and replayed
through T-200's window loop (``eval.isolation_diagnostic.replay``) on the SAME mixture, profiles and held-out
split. Both candidates get streams on the mixture's scale (``rescale_to_mix``), so levels match. Contexts:
0.5, 1 and 2 s windows (hop = window / 2) and one whole-clip window as the longer OFFLINE reference. Offline
RTF says nothing about streaming: the reference row is not a deployable latency.

Per row: SI-SDRi of the final output over the mix (target-active windows), the oracle ceiling (stream picked
with the clean target, evaluation only), target-absent leakage (energy vs the mix), whole-clip target words
LOST (missing) next to wrong-voice words, separator compute p50/p95 per window, RTF, device, peak accelerator
memory, load time.

    python -m eval.sepformer_trial --dry-run
    python -m eval.sepformer_trial --scene <dir> --profiles <dir> --target-name aryan --other-name joseph
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval import isolation_diagnostic as diag
from eval import metrics

SR = 16000
REPO = "speechbrain/sepformer-whamr16k"
REVISION = "21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5"  # HF commit sha read 2026-10-09, lastModified 2024-02-20
LICENSE = "apache-2.0"  # from the model card front matter; the WHAMR! data licence is separate
FILES = (
    "hyperparams.yaml",
    "encoder.ckpt",
    "decoder.ckpt",
    "masknet.ckpt",
    "README.md",
    "test_mixture16k.wav",
)
PROGRESS: Path | None = None  # set by main(): finished rows are appended here so a cut-off run keeps its numbers
DEFAULT_CONTEXTS = (
    0.5,
    1.0,
    2.0,
)  # window seconds; hop = window / 2; plus the whole-clip reference


class Blocked(RuntimeError):
    """An input or dependency is missing; the message is the exact blocker."""


def cache_dir() -> Path:
    base = os.environ.get("HEARING_MODELS_DIR")
    if not base:
        raise Blocked("HEARING_MODELS_DIR is not set")
    return (
        Path(base) / f"sepformer-whamr16k-{REVISION}"
    )  # separate cache: never mixed with MossFormer2's


def download() -> Path:
    """Explicit, pinned download of only the files inference needs (no optimizer/training checkpoints)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            REPO, revision=REVISION, local_dir=cache_dir(), allow_patterns=list(FILES)
        )
    )


def provenance(root: Path | None = None) -> dict:
    root = Path(root) if root else cache_dir()
    missing = [f for f in FILES if not (root / f).is_file()]
    if missing:
        raise Blocked(
            f"checkpoint files missing in {root}: {', '.join(missing)}; run download() (pinned {REPO}@{REVISION})"
        )
    sha = {
        f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16]
        for f in FILES
        if f.endswith((".ckpt", ".yaml"))
    }
    card = (root / "README.md").read_text(errors="ignore")
    return {
        "repo": REPO,
        "revision": REVISION,
        "license": LICENSE,
        "license_in_model_card": f'license: "{LICENSE}"' in card,
        "files_sha256_16": sha,
        "dir": str(root),
    }


class SepFormerSeparator:
    """``BlindSeparator``-compatible (``load``/``separate``/``device``): mono 16 kHz in, exactly two streams out."""

    name = "sepformer-whamr16k"

    def __init__(self, root: Path | None = None, device: str | None = None):
        self.root, self.device, self.model, self.load_s = root, device or "", None, 0.0

    def load(self):
        if self.model is not None:
            return
        root = Path(self.root) if self.root else cache_dir()
        provenance(root)
        try:
            import torch
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:
            raise Blocked(
                f"speechbrain/torch not importable in this venv: {exc}"
            ) from exc
        from hearing import device as dev

        self.device = self.device or dev.pick()
        t0 = time.perf_counter()
        # local path source + local savedir: speechbrain never touches the network or the default cache
        try:
            self.model = SepformerSeparation.from_hparams(
                source=str(root), savedir=str(root), run_opts={"device": self.device}
            )
        except Exception:  # noqa: BLE001 # an operator unsupported on MPS: fall back to CPU and say so in the device field
            self.device = "cpu"
            self.model = SepformerSeparation.from_hparams(
                source=str(root), savedir=str(root), run_opts={"device": "cpu"}
            )
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        self._torch = torch

    def separate(self, mix, sr=SR):
        from hearing.tse.blind import conform_output, rescale_to_mix, validate_audio

        mix = validate_audio(mix, sr)
        self.load()
        with self._torch.inference_mode():
            est = self.model.separate_batch(
                self._torch.from_numpy(mix).unsqueeze(0).to(self.device)
            )  # (1, n, 2)
        est = est.detach().cpu().numpy()[0]
        if est.ndim != 2 or est.shape[1] != 2:
            raise ValueError(
                f"expected exactly two output estimates, got shape {est.shape}"
            )
        streams = np.stack(
            [conform_output(est[:, k], len(mix), self.name) for k in range(2)]
        )
        return rescale_to_mix(streams, mix)


def verify_two_outputs(sep, seconds=4.0) -> dict:
    """Run the bundled 16 kHz test mixture (or noise) once: records the real output shape."""
    root = Path(sep.root) if sep.root else cache_dir()
    wav = root / "test_mixture16k.wav"
    x = (
        diag._read(wav)[: int(seconds * SR)]
        if wav.is_file()
        else np.random.default_rng(0)
        .standard_normal(int(seconds * SR))
        .astype(np.float32)
        * 0.05
    )
    out = sep.separate(x)
    return {
        "input_samples": len(x),
        "output_shape": list(out.shape),
        "two_outputs": out.shape == (2, len(x)),
        "finite": bool(np.isfinite(out).all()),
    }


class Timed:
    """Wraps a separator and records per-call compute so p50/p95 exclude ECAPA/selection. Only calls at the
    window length enter the stats: the runner also separates the enrolment profiles (another length, cold)."""

    def __init__(self, inner, memo_above=None):
        self.inner, self.times, self.device = inner, [], getattr(inner, "device", "")
        self.memo_above, self._memo = memo_above, {}

    def load(self):
        self.inner.load()
        self.device = self.inner.device

    def separate(self, mix, sr=SR):
        # the enrolment profiles (here 50 s and 14 s) are re-separated for every condition; separation is
        # deterministic, so identical long inputs reuse the first result (SepFormer took minutes per 50 s
        # call on MPS under memory pressure). Window calls are never memoised.
        key = (len(mix), _sha_bytes(mix)) if self.memo_above and len(mix) > self.memo_above else None
        if key in self._memo:
            return self._memo[key]
        t0 = time.perf_counter()
        out = self.inner.separate(mix, sr)
        self.times.append(((time.perf_counter() - t0) * 1000, len(mix)))
        if key is not None:
            self._memo[key] = out
        return out

    def stats(self, window_s):
        n = int(window_s * SR)
        t = np.asarray([ms for ms, k in self.times if k == n] or [0.0])
        p50, p95 = float(np.percentile(t, 50)), float(np.percentile(t, 95))
        return {
            "p50_ms": p50,
            "p95_ms": p95,
            "rtf": p50 / (window_s * 1000.0),
            "n_calls": len(self.times),
            "n_timed_window_calls": sum(1 for _, k in self.times if k == n),
        }


class NoDeadline:
    """The whole-clip reference has no deadline: report zero compute to the replay so a slow call never
    swaps in the raw mix. Real separator time is still recorded by ``Timed``."""

    def __init__(self, inner):
        self.inner = inner
        self.mixer_s = getattr(inner, "mixer_s", 1.0)

    def run(self, m, st):
        out = self.inner.run(m, st)
        out.compute_ms = 0.0
        return out


def warmup(separator, n):
    """One untimed call at the window length so p50/p95 measure steady state, not kernel compilation."""
    separator.separate(np.zeros(n, np.float32) + 1e-4 * np.random.default_rng(0).standard_normal(n).astype(np.float32))


def _crop(x, r, win_s):
    st = int(r["start_s"] * SR)
    return x[st : st + int(win_s * SR)]


def peak_accel_mb(device) -> float | None:
    """Peak (CUDA) or current/driver (MPS) allocation in MB for the process: an upper-bound proxy, not per model."""
    try:
        import torch

        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 2**20
        if device == "mps":
            return torch.mps.driver_allocated_memory() / 2**20
    except Exception:  # noqa: BLE001 - a memory probe must never fail the trial
        return None
    return None


# --- word-level target loss -------------------------------------------------------------------


def _words(s):
    return metrics.normalize_text(s).split()


def word_losses(hyp: str, ref_target: str, ref_other: str) -> dict:
    """Align ``hyp`` to the target transcript. ``missing`` = target words with no match in the output (deletions
    and substitutions); ``wrong_voice`` = output words that are unmatched AND belong to the other speaker's
    transcript (multiset). Speech absent from both counts as missing, so silence is not scored as success."""
    ref, h = _words(ref_target), _words(hyp)
    n, m = len(ref), len(h)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + (ref[i - 1] != h[j - 1]),
            )
    i, j, matched_h = n, m, set()
    while i > 0 and j > 0:
        if ref[i - 1] == h[j - 1] and d[i, j] == d[i - 1, j - 1]:
            matched_h.add(j - 1)
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    other = Counter(_words(ref_other))
    wrong = 0
    for k, w in enumerate(h):
        if k not in matched_h and other[w] > 0:
            other[w] -= 1
            wrong += 1
    kept = len(matched_h)
    return {
        "target_words": n,
        "kept": kept,
        "missing": n - kept,
        "wrong_voice": wrong,
        "missing_frac": (n - kept) / n if n else None,
    }


# --- trial ------------------------------------------------------------------------------------


def _sha_bytes(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, np.float32).tobytes()).hexdigest()[
        :16
    ]


def level_match(x, ref):
    """Scale ``x`` to the RMS of ``ref`` (applied identically to every candidate's output before ASR)."""
    return (x * (diag._rms(ref) / max(diag._rms(x), 1e-9))).astype(np.float32)


def run_candidate(
    name, separator, embedder, contexts, conditions, profiles, transcriber, split
):
    """conditions: list of dicts {label, mix, target, other, role}. Returns rows (one per condition x context)."""
    rows = []
    n_clip = max(len(c["mix"]) for c in conditions)
    timed = Timed(separator, memo_above=n_clip)
    for ctx in [*contexts, None]:
        for c in conditions:
            n = len(c["mix"])
            win_s = n / SR if ctx is None else ctx
            hop_s = win_s if ctx is None else win_s / 2
            a, b = profiles if c["role"] == "A" else profiles[::-1]
            print(f"[{name}] context={ctx} {c['label']}", file=sys.stderr, flush=True)
            runner = diag.EnrolledRunner(a, b, separator=timed, embedder=embedder)
            if ctx is None:
                runner = NoDeadline(runner)
            warmup(separator, int(win_s * SR))  # first call per input length (MPS kernel build) is not timed
            timed.times.clear()
            rep = diag.replay(
                runner, c["mix"], c["target"], c["other"], window_s=win_s, hop_s=hop_s
            )
            s = diag.summarize(
                rep, c["mix"], c["target"], c["other"], "all" if ctx is None else split
            )
            recs = (
                rep.windows[len(rep.windows) // 2 :]
                if (split == "heldout" and ctx is not None)
                else rep.windows
            )
            act = [r for r in recs if r["target_active"]]
            # SI-SDRi only where both talk: in target-only windows without noise the mix IS the target
            # (input SI-SDR hits the +100 dB cap) and the "improvement" is meaningless
            ovl = [r for r in act if r["other_active"]]
            sisdri = diag._mean(
                r["final_sisdr"] - metrics.si_sdr(_crop(c["mix"], r, win_s), _crop(c["target"], r, win_s))
                for r in ovl
            )
            clip_sisdri = metrics.si_sdri(rep.final, c["mix"], c["target"])
            final = level_match(rep.final, c["mix"])
            hyp = transcriber.transcribe(final, SR) if transcriber else ""
            wl = (
                word_losses(hyp, c["ref_target"], c["ref_other"])
                if transcriber
                else None
            )
            row = (
                {
                    "candidate": name,
                    "context": "offline_reference" if ctx is None else f"{ctx}s",
                    "window_s": win_s,
                    "condition": c["label"],
                    "mix_sha": c["mix_sha"],
                    "streaming_valid": ctx is not None,
                    "sisdri_db": sisdri,
                    "n_overlap_windows": len(ovl),
                    "clip_sisdri_db": clip_sisdri,
                    "deadline_applies": ctx is not None,
                    "oracle_sisdr_db": s["rows"][0]["sisdr_mean_db"],
                    "actual_sisdr_db": s["rows"][1]["sisdr_mean_db"],
                    "oracle_gap_db": s["ceiling_gap_db"],
                    "target_absent_leak_db": s["rows"][5]["leak_mean_db"],
                    "n_target_absent_windows": s["rows"][5]["n_windows"],
                    "categories": s["categories"],
                    "n_windows": len(act),
                    "words": wl,
                    "hyp": hyp,
                    **timed.stats(win_s),
                }
            )
            rows.append(row)
            if PROGRESS is not None:
                with PROGRESS.open("a") as fh:
                    fh.write(json.dumps(row, default=float) + "\n")
    return rows


def aggregate(rows):
    """Mean over conditions per (candidate, context); ``missing`` and ``wrong_voice`` are summed (counts)."""
    out = []
    keys = sorted(
        {(r["candidate"], r["context"]) for r in rows},
        key=lambda k: (k[1] == "offline_reference", k[1], k[0]),
    )
    for cand, ctx in keys:
        rs = [r for r in rows if r["candidate"] == cand and r["context"] == ctx]
        w = [r["words"] for r in rs if r["words"]]
        out.append(
            {
                "candidate": cand,
                "context": ctx,
                "n_conditions": len(rs),
                "streaming_valid": rs[0]["streaming_valid"],
                "sisdri_db": diag._mean(r["sisdri_db"] for r in rs),
                "clip_sisdri_db": diag._mean(r.get("clip_sisdri_db") for r in rs),
                "oracle_db": diag._mean(r["oracle_sisdr_db"] for r in rs),
                "leak_db": diag._mean(r["target_absent_leak_db"] for r in rs),
                "missing_words": sum(x["missing"] for x in w) if w else None,
                "target_words": sum(x["target_words"] for x in w) if w else None,
                "wrong_voice_words": sum(x["wrong_voice"] for x in w) if w else None,
                "p50_ms": diag._mean(r["p50_ms"] for r in rs),
                "p95_ms": diag._mean(r["p95_ms"] for r in rs),
                "rtf": diag._mean(r["rtf"] for r in rs),
            }
        )
    return out


def table(agg):
    f = diag._fmt
    lines = [
        "| candidate | context | streaming-valid | n cond | SI-SDRi dB (overlap win) | clip SI-SDRi dB | oracle SI-SDR dB | absent leak dB | target words missing | wrong-voice words | sep p50 ms | p95 ms | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in agg:
        miss = (
            "-"
            if a["missing_words"] is None
            else f"{a['missing_words']}/{a['target_words']}"
        )
        lines.append(
            f"| {a['candidate']} | {a['context']} | {'yes' if a['streaming_valid'] else 'NO (offline)'} | {a['n_conditions']} | {f(a['sisdri_db'])} | {f(a['clip_sisdri_db'])} "
            f"| {f(a['oracle_db'])} | {f(a['leak_db'])} | {miss} | {a['wrong_voice_words'] if a['wrong_voice_words'] is not None else '-'} "
            f"| {f(a['p50_ms'])} | {f(a['p95_ms'])} | {f(a['rtf'], 3)} |"
        )
    return "\n".join(lines)


def build_conditions(target, other, noise, ratios, roles, transcriber):
    ref = {}
    if transcriber:
        ref = {
            "A": transcriber.transcribe(target, SR),
            "B": transcriber.transcribe(other, SR),
        }
    conds = []
    for role in roles:
        t0, o0 = (target, other) if role == "A" else (other, target)
        for ratio in ratios:
            t, o = diag.scale_target(t0, o0, ratio)
            n = min(len(t), len(o))
            t, o = t[:n], o[:n]
            mix = (t + o + (noise[:n] if noise is not None else 0)).astype(np.float32)
            rt, ro = (
                (ref.get("A", ""), ref.get("B", ""))
                if role == "A"
                else (ref.get("B", ""), ref.get("A", ""))
            )
            conds.append(
                {
                    "label": f"role{role} target{ratio:+.0f}dB",
                    "role": role,
                    "ratio": ratio,
                    "mix": mix,
                    "target": t,
                    "other": o,
                    "mix_sha": _sha_bytes(mix),
                    "ref_target": rt,
                    "ref_other": ro,
                }
            )
    return conds


class FakeSeparator:
    """Dry-run stand-in that reads the stems; proves the plumbing, never a separation claim."""

    device = "fake"

    def __init__(self, name, stems_fn, quality=1.0):
        self.name, self.stems_fn, self.quality, self._pos = name, stems_fn, quality, 0

    def load(self):
        pass

    def separate(self, mix, sr=SR):
        t, o = self.stems_fn(len(mix))
        return (
            np.stack([t, o])
            if self.quality >= 1
            else np.stack([0.5 * (t + o), 0.5 * (t + o)])
        )


def dry_run(args):
    t, o = diag.fake_stems(args.fake_seconds)
    ref_t, ref_o = "the quick brown fox jumps", "over the lazy dog today"

    class Tr:
        name = "fake"

        def transcribe(self, x, sr):
            return ref_t if diag._rms(x) > 0 else ""

    conds = build_conditions(t, o, None, [0.0], ["A"], Tr())
    for c in conds:
        c["ref_target"], c["ref_other"] = ref_t, ref_o

    def stems_fn(m):
        return lambda n: (
            t[:n],
            o[:n],
        )  # whole-window oracle stems regardless of start; only plumbing

    rows = []
    for name, q in (("mossformer2_ss(FAKE)", 1.0), ("sepformer_whamr16k(FAKE)", 0.0)):
        sep = FakeSeparator(name, stems_fn(None), q)
        rows += run_candidate(
            name,
            sep,
            diag.FakeEmbedder(),
            args.contexts,
            conds,
            (t[: 3 * SR], o[: 3 * SR]),
            Tr(),
            "all",
        )
    return {
        "mode": "dry_run",
        "fake": True,
        "label": "FAKE separators and constructed stems: plumbing check, NOT measured separation",
        "rows": rows,
        "provenance": None,
        "device": "fake",
        "inputs": {},
    }


def real_run(args):
    prov = provenance()
    scene = Path(args.scene)
    man_p = scene / "manifest.json"
    stems = {k: scene / v for k, v in json.loads(man_p.read_text())["stems"].items()}
    tdir, odir = (
        Path(args.profiles) / args.target_name,
        Path(args.profiles) / args.other_name,
    )
    pt_l, po_l = sorted(tdir.glob("profile-*.wav")), sorted(odir.glob("profile-*.wav"))
    missing = (
        [str(p) for p in stems.values() if not p.exists()]
        + ([f"profile wavs in {tdir}"] if not pt_l else [])
        + ([f"profile wavs in {odir}"] if not po_l else [])
    )
    if missing:
        raise Blocked("missing inputs: " + ", ".join(missing))
    t, o = diag._read(stems["target"]), diag._read(stems["other"])
    noise = diag._read(stems["noise"]) if args.noise and "noise" in stems else None
    if args.clip_seconds > 0:
        k = int(args.clip_seconds * SR)
        t, o, noise = t[:k], o[:k], (noise[:k] if noise is not None else None)
    pt, po = diag._read(pt_l[0]), diag._read(po_l[0])
    from eval.transcribe import get_transcriber
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enrolled import EnrolledCandidate

    transcriber = get_transcriber("whisper") if args.words else None
    conds = build_conditions(
        t,
        o,
        noise,
        args.ratios,
        ["A", "B"] if args.roles == "both" else [args.roles],
        transcriber,
    )
    moss = BlindSeparator()
    probe = EnrolledCandidate(separator=moss)
    probe.load()
    sep = SepFormerSeparator()
    t0 = time.perf_counter()
    sep.load()
    startup_s = time.perf_counter() - t0
    shape = verify_two_outputs(sep)
    if not shape["two_outputs"]:
        raise Blocked(f"SepFormer did not return two same-length outputs: {shape}")
    rows, mem = [], {}
    for name, s in (("mossformer2_ss", moss), ("sepformer_whamr16k", sep)):
        if name not in args.candidates:
            continue
        rows += run_candidate(
            name,
            s,
            probe.embedder,
            args.contexts,
            conds,
            (pt, po),
            transcriber,
            args.split,
        )
        mem[name] = peak_accel_mb(str(s.device))
    return {
        "mode": "real",
        "fake": False,
        "rows": rows,
        "provenance": prov,
        "two_output_check": shape,
        "sepformer_load_s": sep.load_s,
        "sepformer_startup_s_incl_check": startup_s,
        "accel_mb_after_candidate": mem,
        "device": {"mossformer2_ss": str(moss.device), "sepformer": str(sep.device)},
        "inputs": {
            "scene_manifest_sha": diag._sha(man_p),
            **{f"stem_{k}": diag._sha(p) for k, p in stems.items()},
            "profile_target": diag._sha(pt_l[0]),
            "profile_other": diag._sha(po_l[0]),
            "mix_sha_by_condition": {c["label"]: c["mix_sha"] for c in conds},
        },
    }


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--scene")
    p.add_argument("--profiles")
    p.add_argument("--target-name", default="aryan")
    p.add_argument("--other-name", default="joseph")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.0, -10.0])
    p.add_argument("--roles", choices=("A", "B", "both"), default="both")
    p.add_argument("--split", choices=("all", "heldout"), default="heldout")
    p.add_argument("--contexts", type=float, nargs="+", default=list(DEFAULT_CONTEXTS))
    p.add_argument("--candidates", nargs="+", default=["mossformer2_ss", "sepformer_whamr16k"],
                   choices=("mossformer2_ss", "sepformer_whamr16k"))
    p.add_argument("--noise", action="store_true")
    p.add_argument(
        "--no-words",
        dest="words",
        action="store_false",
        help="skip the ASR word-loss pass",
    )
    p.add_argument("--clip-seconds", type=float, default=0.0, help="use only the first N seconds of the stems (0 = all)")
    p.add_argument("--fake-seconds", type=float, default=8.0)
    p.add_argument("--out")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    if not args.dry_run and not (args.scene and args.profiles):
        print(
            "BLOCKED: real run needs --scene and --profiles (or --dry-run)",
            file=sys.stderr,
        )
        return 2
    global PROGRESS
    if args.out:
        PROGRESS = Path(args.out).with_suffix(".rows.jsonl")
        PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS.write_text("")
    try:
        data = dry_run(args) if args.dry_run else real_run(args)
    except Blocked as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    data.update(
        {
            "stamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": platform.node(),
            "config": vars(args),
        }
    )
    agg = aggregate(data["rows"])
    data["aggregate"] = agg
    print(("** " + data["label"] + " **\n") if data["fake"] else "", end="")
    print(table(agg))
    out = (
        Path(args.out)
        if args.out
        else Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results"))
        / "sepformer_trial"
        / (
            datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            + (".dryrun" if args.dry_run else "")
            + ".json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, default=float))
    print("manifest:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
