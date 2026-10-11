"""Focus HQ bake-off: every TSE candidate x sliding-window length on one AV fixture set.

    python -m eval.bakeoff --set tiny --candidates identity,oracle --windows 0.5,1,2 --out eval/results/

Per candidate x window: SI-SDRi vs the mixture, WER of the estimate (``eval.transcribe``; mock when
faster-whisper or its weights are absent), RTF, latency (algorithmic = window length, plus measured
compute p50/p95 per window), peak GPU memory and device. A candidate that fails to load or raises
gets an ``error`` and stays in the table; the run never aborts. Prints one table, writes
``<out>/bakeoff_<stamp>.json`` and a markdown summary ranked by SI-SDRi, then latency.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from eval.bakeoff_fixtures import (
    AVFixture,
    lips_from_video,
    load_set,
    mouth_box_from_face,
    resolve_set,
)
from eval.metrics import si_sdr, si_sdri, wer_batch
from eval.transcribe import MockTranscriber, Transcriber, get_transcriber
from hearing import device as hdevice
from hearing.tse import SlidingWindowExtractor, TSECandidate, candidates, get_candidate

log = logging.getLogger("eval.bakeoff")

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_WINDOWS = (0.5, 1.0, 2.0)
TABLE_COLUMNS = [
    ("candidate", "candidate", "{}"),
    ("window_s", "win_s", "{:g}"),
    ("si_sdri_db", "SI-SDRi_dB", "{:.2f}"),
    ("wer", "WER", "{:.3f}"),
    ("rtf", "RTF", "{:.4f}"),
    ("algorithmic_latency_ms", "algo_ms", "{:.0f}"),
    ("compute_p50_ms", "comp_p50_ms", "{:.2f}"),
    ("compute_p95_ms", "comp_p95_ms", "{:.2f}"),
    ("gpu_mem_mb", "gpu_mem_MB", "{:.1f}"),
    ("device", "device", "{}"),
    ("error", "error", "{}"),
]


# --- GPU memory --------------------------------------------------------------------------


def _torch():
    return sys.modules.get(
        "torch"
    )  # never import torch just to measure an untouched GPU


def reset_gpu_peak(dev: str) -> None:
    torch = _torch()
    if torch is not None and dev == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_mb(dev: str) -> float | None:
    """Peak allocated (CUDA) or currently allocated (MPS) memory in MB; None on CPU."""
    torch = _torch()
    try:
        if dev == "cuda" and torch is not None and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
        if dev == "mps" and torch is not None and torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 2**20
    except Exception as e:  # noqa: BLE001  (backend quirks; memory is informational)
        log.warning("gpu memory query failed on %s: %s", dev, e)
    return None


# --- running -----------------------------------------------------------------------------


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _inputs(cand: TSECandidate, fx: AVFixture) -> dict:
    """Only the inputs the candidate declares in ``needs``; the oracle gets the aligned clean target."""
    needs = cand.needs
    if "lips" in needs and fx.lips is None:
        raise ValueError(
            f"fixture {fx.id} has no lip crops (lips.npy, or video.mp4 + target_box)"
        )
    enroll = None
    if "enroll" in needs:
        enroll = fx.target if getattr(cand, "oracle", False) else fx.enroll
    return {
        "lips": fx.lips if "lips" in needs else None,
        "lips_fps": fx.lips_fps,
        "p_speaking": fx.p_speaking if "p_speaking" in needs else None,
        "enroll": enroll,
    }


def with_face_lips(fx: AVFixture) -> AVFixture:
    """Lip crops for ``eval.make_fixtures --av`` sets, whose manifests give ``faces[role=target].box``
    instead of the ``target_box`` that ``eval.bakeoff_fixtures`` reads (same rule as ``eval.bench``)."""
    video = fx.path / "video.mp4"
    if fx.lips is not None or not video.exists():
        return fx
    box = next(
        (f["box"] for f in fx.manifest.get("faces", []) if f.get("role") == "target"),
        None,
    )
    if box is None:
        return fx
    fps = float(fx.manifest.get("fps", fx.lips_fps))
    lips = lips_from_video(video, mouth_box_from_face(box), fps)
    if lips is None:
        return fx
    return replace(fx, lips=lips, lips_fps=fps, lips_source="video.mp4@faces")


def _speaker(fx: AVFixture) -> str | None:
    return (fx.manifest.get("source") or {}).get("target_speaker")


def enroll_from_other_utterance(fixtures: list[AVFixture]) -> list[AVFixture]:
    """Enroll each fixture on the clean target of *another* utterance by the same speaker.

    The default enrollment (first 3 s of the clean target) is the whole clip on 3 s GRID fixtures,
    i.e. an oracle. Fixtures with no other utterance of their speaker keep the default.
    """
    out = []
    for fx in fixtures:
        spk, utt = _speaker(fx), fx.manifest.get("target_id")
        other = next(
            (
                o
                for o in fixtures
                if spk is not None
                and _speaker(o) == spk
                and o.manifest.get("target_id") != utt
            ),
            None,
        )
        if other is None or fx.enroll_source == "enroll_wav":
            out.append(fx)
            continue
        out.append(
            replace(
                fx,
                enroll=other.target.copy(),
                enroll_source=f"other:{other.manifest.get('target_id')}",
            )
        )
    return out


_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ALNUM = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def spoken_digits(text: str) -> str:
    """Write digits as words and split letter-digit runs ("F6" -> "F six", "10" -> "one zero").

    GRID transcripts spell letters and digits ("bin blue by f six now"); Whisper writes "by F6", which
    would count as two errors per sentence. Applied to hypothesis and reference before scoring.
    """
    text = _ALNUM.sub(" ", text)
    return re.sub(r"\d", lambda m: f" {_DIGITS[int(m.group())]} ", text)


def _wer(hyps: list[str], refs: list[str]) -> float:
    return wer_batch([spoken_digits(h) for h in hyps], [spoken_digits(r) for r in refs])


def _transcriber_for(base: Transcriber, fx: AVFixture) -> Transcriber:
    # The mock echoes the reference transcript, so its WER is 0 for every row: a plumbing check only.
    return MockTranscriber(fx.transcript) if isinstance(base, MockTranscriber) else base


def _nan_to_none(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _nan_to_none(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_nan_to_none(x) for x in v]
    return v


def _empty_row(
    name: str,
    window_s: float,
    hop_s: float,
    cand: TSECandidate | None,
    error: str | None,
) -> dict:
    return {
        "candidate": name,
        "window_s": window_s,
        "hop_s": hop_s,
        "si_sdri_db": math.nan,
        "si_sdr_db": math.nan,
        "wer": math.nan,
        "rtf": math.nan,
        "algorithmic_latency_ms": window_s * 1000.0,
        "compute_p50_ms": math.nan,
        "compute_p95_ms": math.nan,
        "latency_p95_ms": math.nan,
        "gpu_mem_mb": None,
        "device": getattr(cand, "device", None) or hdevice.pick(),
        "licence": getattr(cand, "licence", None),
        "needs": sorted(getattr(cand, "needs", ())),
        "n_fixtures": 0,
        "error": error,
        "per_fixture": [],
    }


def run_candidate(
    name: str,
    fixtures: list[AVFixture],
    windows: list[float],
    transcriber: Transcriber,
    hop_frac: float = 0.5,
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """One row per window. Load and extract failures become ``error`` rows, never exceptions."""
    try:
        cand = get_candidate(name)
    except Exception as e:  # noqa: BLE001  (a broken candidate is a row, not a crash)
        return [
            _empty_row(name, w, w * hop_frac, None, f"unknown: {_err(e)}")
            for w in windows
        ]
    try:
        cand.load()
    except Exception as e:
        log.warning("%s failed to load", name, exc_info=True)
        return [
            _empty_row(name, w, w * hop_frac, cand, f"load: {_err(e)}") for w in windows
        ]

    rows = []
    for w in windows:
        row = _empty_row(name, w, w * hop_frac, cand, None)
        try:
            ext = SlidingWindowExtractor(cand, w, w * hop_frac)
        except Exception as e:  # noqa: BLE001
            row["error"] = _err(e)
            rows.append(row)
            continue
        dev = row["device"]
        reset_gpu_peak(dev)
        errors, per, hyps, refs = [], [], [], []
        for fx in fixtures:
            try:
                est = ext.extract(fx.mix, fx.sr, **_inputs(cand, fx))
            except Exception as e:
                log.warning("%s @ %gs failed on %s", name, w, fx.id, exc_info=True)
                errors.append(f"{fx.id}: {_err(e)}")
                continue
            item = {
                "id": fx.id,
                "si_sdri_db": si_sdri(est, fx.mix, fx.target),
                "si_sdr_db": si_sdr(est, fx.target),
            }
            if fx.transcript is not None:
                hyp = _transcriber_for(transcriber, fx).transcribe(est, fx.sr)
                item["hyp"] = hyp
                hyps.append(hyp)
                refs.append(fx.transcript)
            per.append(item)
        st = ext.stats()
        row.update(
            si_sdri_db=float(np.mean([p["si_sdri_db"] for p in per]))
            if per
            else math.nan,
            si_sdr_db=float(np.mean([p["si_sdr_db"] for p in per]))
            if per
            else math.nan,
            wer=_wer(hyps, refs) if refs else math.nan,
            rtf=st.rtf,
            compute_p50_ms=st.compute_p50_ms,
            compute_p95_ms=st.compute_p95_ms,
            latency_p95_ms=st.algorithmic_latency_ms + st.compute_p95_ms,
            gpu_mem_mb=gpu_memory_mb(dev),
            n_fixtures=len(per),
            n_windows=st.n_windows,
            per_fixture=per,
            error="; ".join(errors[:3])
            + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")
            if errors
            else None,
        )
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def rank(rows: list[dict]) -> list[dict]:
    """Error-free rows first, then SI-SDRi descending, then total latency (algo + p95) ascending."""

    def key(r: dict):
        sdr = r.get("si_sdri_db")
        sdr = -sdr if isinstance(sdr, float) and math.isfinite(sdr) else math.inf
        lat = r.get("latency_p95_ms")
        lat = lat if isinstance(lat, float) and math.isfinite(lat) else math.inf
        return (r.get("error") is not None, sdr, lat)

    return sorted(rows, key=key)


def run(
    set_name: str,
    candidate_names: list[str],
    windows: list[float],
    transcriber: str = "auto",
    hop_frac: float = 0.5,
    limit: int | None = None,
    layout: str | None = None,
    enroll: str = "self",
    partial_dir: Path | None = None,
) -> dict:
    set_dir = resolve_set(set_name)
    fixtures = load_set(set_dir)
    if layout:
        fixtures = [fx for fx in fixtures if fx.manifest.get("layout") == layout]
    fixtures = [with_face_lips(fx) for fx in fixtures[:limit]]
    if enroll == "other":
        fixtures = enroll_from_other_utterance(fixtures)
    base = get_transcriber(transcriber)
    t0 = time.perf_counter()
    rows = []

    def save_row(row: dict) -> None:
        # a cut-off run keeps every finished window (eval/results/bakeoff_<window>.json)
        if partial_dir is None:
            return
        partial_dir.mkdir(parents=True, exist_ok=True)
        part = partial_dir / f"bakeoff_{row['window_s']:g}.json"
        if part.exists():
            old = json.loads(part.read_text())
        else:
            old = {"kind": "bakeoff_window", "window_s": row["window_s"], "rows": []}
        old.update(
            set_name=set_name, layout=layout, enroll=enroll, n_clips=len(fixtures),
            hop_frac=hop_frac, machine=hdevice.describe(),
        )
        old["rows"] = [r for r in old["rows"] if r.get("candidate") != row["candidate"]]
        old["rows"].append(row)
        part.write_text(json.dumps(_nan_to_none(old), indent=2, default=str))

    for name in candidate_names:
        rows.extend(run_candidate(name, fixtures, windows, base, hop_frac, save_row))
    baseline_wer = math.nan
    with_ref = [fx for fx in fixtures if fx.transcript is not None]
    if with_ref:
        hyps = [_transcriber_for(base, fx).transcribe(fx.mix, fx.sr) for fx in with_ref]
        baseline_wer = _wer(hyps, [fx.transcript for fx in with_ref])
    return {
        "kind": "bakeoff",
        "stamp": datetime.now().astimezone().strftime("%Y%m%d-%H%M%S"),
        "set": str(set_dir),
        "set_name": set_name,
        "layout": layout,
        "enroll": enroll,
        "fixtures": [
            {
                "id": fx.id,
                "duration_s": fx.duration_s,
                "lips": fx.lips_source,
                "enroll": fx.enroll_source,
            }
            for fx in fixtures
        ],
        "audio_s": float(sum(fx.duration_s for fx in fixtures)),
        "windows": windows,
        "hop_frac": hop_frac,
        "transcriber": base.name,
        "baseline": {"wer_mix": baseline_wer},
        "p_speaking": "oracle proxy from clean-target frame energy (until ASD lands)",
        "machine": hdevice.describe(),
        "elapsed_s": time.perf_counter() - t0,
        "rows": rows,
        "ranking": [f"{r['candidate']}@{r['window_s']:g}s" for r in rank(rows)],
    }


# --- output ------------------------------------------------------------------------------


def _fmt(v, spec: str) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return spec.format(v)


def format_table(rows: list[dict]) -> str:
    head = [h for _, h, _ in TABLE_COLUMNS]
    body = [[_fmt(r.get(k), spec) for k, _, spec in TABLE_COLUMNS] for r in rows]
    widths = [
        max(len(h), *(len(b[i]) for b in body)) if body else len(h)
        for i, h in enumerate(head)
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(head, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(b, widths)) for b in body]
    return "\n".join(lines)


def summary_markdown(result: dict) -> str:
    m = result["machine"]
    lines = [
        f"# Focus HQ bake-off ({result['stamp']})",
        "",
        (
            f"Set `{result['set_name']}` ({len(result['fixtures'])} fixtures, {result['audio_s']:.1f} s), "
            f"device `{m['device']}` on `{m['machine']}` / {m['platform']}, "
            f"transcriber `{result['transcriber']}`, hop = {result['hop_frac']:g} x window."
        ),
        "",
        (
            "Ranked by SI-SDRi (higher is better), then latency (algorithmic + compute p95, lower is better); "
            "rows with errors last."
        ),
        "",
        (
            "| # | Candidate | Window s | SI-SDRi dB | WER | RTF | Latency ms (algo + p95) "
            "| Compute p50/p95 ms | GPU MB | Device | Error |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rank(result["rows"]), 1):
        lines.append(
            f"| {i} | {r['candidate']} | {r['window_s']:g} | {_fmt(r['si_sdri_db'], '{:.2f}')} "
            f"| {_fmt(r['wer'], '{:.3f}')} | {_fmt(r['rtf'], '{:.4f}')} | {_fmt(r['latency_p95_ms'], '{:.0f}')} "
            f"| {_fmt(r['compute_p50_ms'], '{:.2f}')} / {_fmt(r['compute_p95_ms'], '{:.2f}')} "
            f"| {_fmt(r['gpu_mem_mb'], '{:.1f}')} | {r['device']} | {r['error'] or ''} |"
        )
    notes = [
        "",
        "Notes:",
        f"- Mixture WER baseline: {_fmt(result['baseline']['wer_mix'], '{:.3f}')}.",
        (
            "- Algorithmic latency = window length (a window is processed once complete); "
            "live output lags by that plus compute."
        ),
        f"- p_speaking: {result['p_speaking']}.",
        "- WER after writing digits as words on both sides (GRID spells them; Whisper writes `F6`).",
    ]
    if result.get("layout"):
        notes.append(f"- Only fixtures with layout `{result['layout']}`.")
    if result.get("enroll") == "other":
        notes.append(
            "- Enrollment: clean target of another utterance by the same speaker (`--enroll other`)."
        )
    if result["transcriber"] == "mock":
        notes.append(
            "- WER from the **mock** transcriber (echoes the reference): plumbing check, not a measurement."
        )
    if "tiny" in result["set_name"] or "bakeoff_tiny" in result["set"]:
        notes.append(
            "- `tiny` is synthetic speech-like audio with synthetic lips: numbers only check the harness."
        )
    return "\n".join(lines + notes) + "\n"


def write_results(
    result: dict, out: Path, summary: Path | None = None
) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    stamp = result["stamp"]
    path = out / f"bakeoff_{stamp}.json"
    k = 1
    while path.exists():
        path = out / f"bakeoff_{stamp}-{k}.json"
        k += 1
    path.write_text(json.dumps(_nan_to_none(result), indent=2, default=str))
    summary = summary or path.with_suffix(".md")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summary_markdown(result))
    return path, summary


def _floats(text: str) -> list[float]:
    return [float(t) for t in text.split(",") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.bakeoff", description=__doc__.split("\n\n")[0]
    )
    p.add_argument(
        "--set",
        default="tiny",
        help="fixture set dir, one fixture dir, or 'tiny' (synthetic, generated)",
    )
    p.add_argument(
        "--candidates",
        default="identity,oracle",
        help="comma list of hearing.tse candidates, or 'all'",
    )
    p.add_argument(
        "--windows",
        default=",".join(f"{w:g}" for w in DEFAULT_WINDOWS),
        help="window lengths in s",
    )
    p.add_argument(
        "--hop",
        type=float,
        default=0.5,
        help="hop as a fraction of the window (default 0.5)",
    )
    p.add_argument("--transcriber", default="auto", choices=["auto", "mock", "whisper"])
    p.add_argument(
        "--limit", type=int, default=None, help="use only the first N fixtures"
    )
    p.add_argument(
        "--layout",
        default=None,
        help="only fixtures whose manifest layout matches (e.g. two_face)",
    )
    p.add_argument(
        "--enroll",
        default="self",
        choices=["self", "other"],
        help="self: fixture's enroll_wav or first 3 s of its clean target; "
        "other: clean target of another utterance by the same speaker",
    )
    p.add_argument(
        "--out", type=Path, default=RESULTS, help="results dir (default eval/results)"
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="markdown summary path (default next to the JSON)",
    )
    p.add_argument(
        "--partial",
        action="store_true",
        help="also write <out>/bakeoff_<window>.json as each window finishes (keeps a cut-off run)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    names = (
        candidates()
        if args.candidates == "all"
        else [c.strip() for c in args.candidates.split(",") if c.strip()]
    )
    result = run(
        args.set,
        names,
        _floats(args.windows),
        args.transcriber,
        args.hop,
        args.limit,
        args.layout,
        args.enroll,
        args.out if args.partial else None,
    )
    print(format_table(rank(result["rows"])))
    path, summary = write_results(result, args.out, args.summary)
    print(f"\nwrote {path}\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
