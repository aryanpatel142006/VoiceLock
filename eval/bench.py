"""Benchmark registered stage pipelines over a fixture set: one table row per pipeline + JSON.

    .venv/bin/python -m eval.bench --fast          # passthrough, tiny set, no WER (generates tiny if missing)
    .venv/bin/python -m eval.bench --stages passthrough --stages "gain(db=-6)>delay(ms=20)" --set tiny
    .venv/bin/python -m eval.bench --stages gtcrn --set full --wer whisper --limit 8
    .venv/bin/python -m eval.bench --full          # every mode x tier on the full set -> JSON + docs/BENCHMARKS.md
    .venv/bin/python -m eval.bench --full --from-json eval/results/<stamp>.json   # re-render the doc only
    .venv/bin/python -m eval.bench --render room.wav --stages gtcrn --suppression 1   # backup demo clip

For every fixture and pipeline the pipeline is built fresh with ``hearing.registry.build_pipeline``,
reset, and fed ``mix.wav`` in 10 ms chunks (plus enough zero chunks to flush its ``latency_ms``).
Each ``process`` call is timed with ``perf_counter``. The output is shifted back by the pipeline's
algorithmic latency and scored against ``target.wav`` (SI-SDR, SI-SDRi vs the mix, PESQ, STOI,
optionally WER against the manifest transcript). Results go to ``<out>/<YYYYMMDD-HHMMSS>.json``.
Exit status is 1 if any fixture failed (the stage spec and fixture id are printed).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval import metrics
from eval.fixtures.mix import load_fixture
from hearing import device, registry
from hearing.audio import io as aio
from hearing.types import AudioChunk, FaceTrack, StageContext, TargetState

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = Path(os.environ["HEARING_FIXTURES_DIR"]) if os.environ.get("HEARING_FIXTURES_DIR") else ROOT / "eval" / "fixtures" / "generated"   # shared dir (scripts/_py.sh)
DEFAULT_OUT = Path(os.environ["HEARING_RESULTS_DIR"]) if os.environ.get("HEARING_RESULTS_DIR") else ROOT / "eval" / "results"   # shared dir: worktrees are deleted after a task (Q-111)
CHUNK_MS = 10
# Imported only when a spec names a stage that is not registered yet, so --fast stays light.
STAGE_MODULES = (
    "hearing.enhance.dfn",
    "hearing.enhance.gtcrn",
    "hearing.fusion",
    "hearing.vision",
    "hearing.tse",
    "hearing.synth.pipeline",
    "hearing.server.stages",
    "hearing.server.remote",
)
COLUMNS = (
    ("stages", "{}"),
    ("n", "{:d}"),
    ("latency_ms", "{:.1f}"),
    ("p50_ms", "{:.3f}"),
    ("p95_ms", "{:.3f}"),
    ("rtf", "{:.4f}"),
    ("si_sdr", "{:.2f}"),
    ("si_sdri", "{:.2f}"),
    ("pesq", "{:.2f}"),
    ("stoi", "{:.3f}"),
    ("wer", "{:.3f}"),
)


# --- pipelines ------------------------------------------------------------------------------


def _load_stage_modules() -> None:
    for module in STAGE_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            if e.name != module and not module.startswith(f"{e.name}."):
                raise  # the module exists but one of its own imports is missing


def build(spec: str) -> registry.Pipeline:
    """``build_pipeline`` after importing the stage modules the spec needs."""
    importlib.import_module("hearing.audio.basic")
    try:
        return registry.build_pipeline(spec)
    except KeyError:
        _load_stage_modules()
        return registry.build_pipeline(spec)


@dataclass
class StreamResult:
    output: np.ndarray  # aligned to the input: output[i] answers input[i]
    compute_ms: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    audio_s: float = 0.0  # audio fed, including the flush chunks
    extras: dict = field(default_factory=dict)  # ctx.telemetry.extras after the run (e.g. mock flags)


def run_stream(
    pipeline: registry.Pipeline,
    x: np.ndarray,
    sr: int,
    ctx: StageContext | None = None,
    on_chunk=None,
) -> StreamResult:
    """Feed ``x`` (at ``sr``) in 10 ms chunks, flush the latency, return the aligned output.

    ``ctx`` defaults to ``StageContext.default()``; ``on_chunk(ctx, seq, t_s)`` runs before every
    chunk (outside the timing) so a caller can drive the target / tracks like a camera would.
    """
    psr = int(pipeline.sample_rate)
    if psr != sr:
        x = aio.resample(x, sr, psr)
    hop = psr * CHUNK_MS // 1000
    shift = round(pipeline.latency_ms * psr / 1000.0)
    flush = math.ceil(shift / hop) * hop
    padded = np.concatenate([np.asarray(x, dtype=np.float32), np.zeros(flush, np.float32)])
    pipeline.reset()
    ctx = ctx if ctx is not None else StageContext.default()
    outs, times = [], []
    for seq, samples in enumerate(aio.frame(padded, hop)):
        chunk = AudioChunk(samples=samples, sample_rate=psr, seq=seq, capture_ts=seq * hop / psr)
        if on_chunk is not None:
            on_chunk(ctx, seq, seq * hop / psr)
        t0 = time.perf_counter()
        out = pipeline.process(chunk, ctx)
        times.append((time.perf_counter() - t0) * 1000.0)
        outs.append(np.asarray(out.samples, dtype=np.float32).ravel())
    y = np.concatenate(outs) if outs else np.zeros(0, np.float32)
    y = y[shift : shift + len(x)]
    if len(y) < len(x):  # a stage returned less than it was fed; score the gap as silence
        y = np.concatenate([y, np.zeros(len(x) - len(y), np.float32)])
    if psr != sr:
        y = aio.resample(y, psr, sr)
    n_fed = math.ceil(len(padded) / hop) * hop
    return StreamResult(y, times, float(pipeline.latency_ms), n_fed / psr, dict(ctx.telemetry.extras))


# --- fixtures -------------------------------------------------------------------------------


def fixture_dirs(fixtures_root: Path, set_name: str, generate: bool = True) -> list[Path]:
    """Fixture directories of a set, sorted by id; generates the tiny set offline if missing."""
    set_root = fixtures_root / set_name
    if not (set_root / "index.json").exists():
        if set_name != "tiny" or not generate:
            raise FileNotFoundError(
                f"no fixture set at {set_root}; run: python -m eval.make_fixtures --{set_name} --out {fixtures_root}"
            )
        from eval import make_fixtures

        print(f"bench: generating the tiny set under {fixtures_root}", file=sys.stderr)
        make_fixtures.make_tiny(fixtures_root, seed=0)
    index = json.loads((set_root / "index.json").read_text())
    return [set_root / f["id"] for f in index["fixtures"]]


def baseline(stems) -> dict:
    """Metrics of the unprocessed mix against the target (what si_sdri is measured from)."""
    return {
        "si_sdr_mix": metrics.si_sdr(stems.mix, stems.target),
        "pesq_mix": metrics.pesq_wb(stems.mix, stems.target),
        "stoi_mix": metrics.stoi(stems.mix, stems.target),
    }


def score(y: np.ndarray, stems, sr: int) -> dict:
    return {
        "si_sdr": metrics.si_sdr(y, stems.target),
        "si_sdri": metrics.si_sdri(y, stems.mix, stems.target),
        "pesq": metrics.pesq_wb(y, stems.target, sr),
        "stoi": metrics.stoi(y, stems.target, sr),
    }


def make_transcriber(mode: str):
    if mode == "off":
        return None
    from eval.transcribe import get_transcriber

    return get_transcriber(mode)


def run_fixture(spec: str, fixture_dir: Path, transcriber, base_cache: dict) -> tuple[dict, list[float]]:
    manifest, stems = load_fixture(fixture_dir)
    sr = manifest.sample_rate
    if manifest.id not in base_cache:
        base_cache[manifest.id] = baseline(stems)
    result = run_stream(build(spec), stems.mix, sr)
    lat = metrics.LatencyStats.from_ms(result.compute_ms)
    row = {
        "stages": spec,
        "fixture": manifest.id,
        "snr_db": manifest.snr_db,
        "sir_db": manifest.sir_db,
        "duration_s": manifest.duration_s,
        "latency_ms": result.latency_ms,
        "p50_ms": lat.p50,
        "p95_ms": lat.p95,
        "max_ms": lat.max,
        "chunks": lat.n,
        "rtf": metrics.rtf(sum(result.compute_ms) / 1000.0, result.audio_s),
        **score(result.output, stems, sr),
        **base_cache[manifest.id],
        "wer": math.nan,
        "extras": result.extras,
    }
    if transcriber is not None:
        if transcriber.name == "mock":  # deterministic: echoes the reference, exercises the plumbing
            transcriber.transcript = manifest.transcript
        hyp = transcriber.transcribe(result.output, sr)
        row["wer"] = metrics.wer(hyp, manifest.transcript)
        row["hyp"] = hyp
    return row, result.compute_ms


# --- reporting ------------------------------------------------------------------------------


def aggregate(spec: str, rows: list[dict], compute_ms: list[float], failures: int) -> dict:
    """``summarize`` over the fixture rows plus compute latency pooled over every chunk."""
    stats = metrics.summarize(rows)
    pooled = metrics.LatencyStats.from_ms(compute_ms)
    return {
        "stages": spec,
        "n": len(rows),
        "failed": failures,
        "metrics": stats,
        "compute_ms": {"p50": pooled.p50, "p95": pooled.p95, "max": pooled.max, "mean": pooled.mean, "chunks": pooled.n},
    }


def table_row(agg: dict) -> dict:
    m = agg["metrics"]

    def mean(k: str) -> float:
        return m.get(k, {}).get("mean", math.nan)

    return {
        "stages": agg["stages"],
        "n": agg["n"],
        "latency_ms": mean("latency_ms"),
        "p50_ms": agg["compute_ms"]["p50"],
        "p95_ms": agg["compute_ms"]["p95"],
        "rtf": mean("rtf"),
        "si_sdr": mean("si_sdr"),
        "si_sdri": mean("si_sdri"),
        "pesq": mean("pesq"),
        "stoi": mean("stoi"),
        "wer": mean("wer"),
    }


def _cell(fmt: str, v) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "-"
    return fmt.format(v)


def format_table(aggregates: list[dict]) -> str:
    """Markdown table: one row per pipeline (means over fixtures; compute p50/p95 pooled)."""
    header = [name for name, _ in COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for agg in aggregates:
        r = table_row(agg)
        lines.append("| " + " | ".join(_cell(fmt, r[name]) for name, fmt in COLUMNS) + " |")
    return "\n".join(lines)


def git_sha() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def cpu_name() -> str:
    """CPU brand for the claims ("Apple M4 Pro"); the bare architecture when the OS will not say."""
    if sys.platform == "darwin":
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  check=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            name = ""
        if name:
            return name
    return platform.processor() or platform.machine()


def _json_safe(v):
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.generic):
        return _json_safe(v.item())
    return v


def _now() -> datetime:
    return datetime.now(UTC).astimezone()  # local wall time, tz-aware


def write_results(out_dir: Path, report: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}.json"
    k = 1
    while path.exists():  # two runs in the same second
        path = out_dir / f"{stamp}-{k}.json"
        k += 1
    path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    return path


# --- runner ---------------------------------------------------------------------------------


def run(
    specs: list[str],
    set_name: str = "tiny",
    wer_mode: str = "off",
    fixtures_root: Path = DEFAULT_FIXTURES,
    out_dir: Path | None = DEFAULT_OUT,
    limit: int | None = None,
) -> dict:
    """Run every pipeline over the set; returns the report (also written to ``out_dir`` if set)."""
    dirs = fixture_dirs(Path(fixtures_root), set_name)
    if limit is not None:
        dirs = dirs[:limit]
    transcriber = make_transcriber(wer_mode)
    rows, aggregates, failures = [], [], []
    base_cache: dict = {}
    wall0 = time.perf_counter()
    for spec in specs:
        spec_rows, spec_ms, spec_failed = [], [], 0
        for d in dirs:
            try:
                row, ms = run_fixture(spec, d, transcriber, base_cache)
            except Exception as e:  # noqa: BLE001 - any stage error is a benchmark failure to report, not a crash
                spec_failed += 1
                failures.append({"stages": spec, "fixture": d.name, "error": f"{type(e).__name__}: {e}"})
                print(f"FAIL stages={spec!r} fixture={d.name}: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc(limit=3, file=sys.stderr)
                continue
            spec_rows.append(row)
            spec_ms.extend(ms)
        rows.extend(spec_rows)
        aggregates.append(aggregate(spec, spec_rows, spec_ms, spec_failed))
    report = {
        "created": _now().isoformat(timespec="seconds"),
        "device": device.describe(),
        "git_sha": git_sha(),
        "machine": f"{platform.node()} ({platform.machine()}, {platform.platform()})",
        "python": platform.python_version(),
        "fixture_set": {"name": set_name, "root": str(fixtures_root), "count": len(dirs), "limit": limit},
        "wer": wer_mode,
        "chunk_ms": CHUNK_MS,
        "stages": list(specs),
        "wall_s": time.perf_counter() - wall0,
        "rows": rows,
        "aggregates": aggregates,
        "failures": failures,
    }
    if out_dir is not None:
        report["path"] = str(write_results(Path(out_dir), report))
    return report


# --- full bank (--full) ---------------------------------------------------------------------
#
# Every mode x tier the app can run, on one fixture set, into one JSON + one markdown report.
# Streaming entries go through ``run_stream`` (10 ms chunks, the same as ``run``); the windowed
# TSE tiers (Focus HQ, Enrolled) go through ``hearing.tse.SlidingWindowExtractor`` exactly as
# the bake-off runs them. "oracle" target = a camera-locked target whose p_speaking comes from
# the clean target's energy (eval.bakeoff_fixtures.oracle_p_speaking), standing in for ASD.

ORACLE_FPS = 25.0
ORACLE_BOX = (0.4, 0.3, 0.2, 0.3)
TRACK_ID = 1
SNR_GRID = (-5.0, 0.0, 5.0, 10.0)
SLIDER_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_TSE_WINDOWS = (1.0, 2.0)
DOC_START = "<!-- bench:full:start -->"
DOC_END = "<!-- bench:full:end -->"
DEFAULT_DOC = ROOT / "docs" / "BENCHMARKS.md"
REAL_CLIPS = ROOT / "eval" / "fixtures" / "real"
_MODES = ("natural", "focus_hq", "synthetic", "text", "quiet")


@dataclass(frozen=True)
class BankEntry:
    """One mode x tier configuration. ``spec`` is a registry pipeline or ``tse:<candidate>@<window_s>``."""

    mode: str  # baseline | natural | focus_hq | text | quiet (synthetic is benched by phrase latency)
    tier: str  # focus_hq | enrolled | gate | audio-only | remote-down | upper-bound | none
    spec: str
    suppression: float = 0.5  # slider s; TSE tiers mix target + (1 - s) * residual like fusion.mixer
    target: str = "none"  # "oracle": locked target track with oracle p_speaking
    score_audio: bool = True  # SI-SDR / PESQ / STOI vs the clean target
    wer: bool = True  # Whisper WER of the output; captions instead when the pipeline ends in synth
    dnsmos: bool = True
    note: str = ""
    condition: str = "mix"  # "mix" (target + talker + noise) or "noise_only" (target + noise, same levels)
    fset: str = "audio"  # "audio": the main set (--set); "av": the audio-visual set (--av-set), lips from video

    @property
    def label(self) -> str:
        extra = ", oracle target" if self.target == "oracle" else ""
        extra += ", noise only" if self.condition == "noise_only" else ""
        extra += ", AV set" if self.fset == "av" else ""
        return f"{self.spec} (s={self.suppression:g}{extra})"

    @property
    def is_tse(self) -> bool:
        return self.spec.startswith("tse:")


NATURAL = "dfn_ll"  # PLAN: the Natural fast default (13 ms)


def full_bank(tse_windows: tuple[float, ...] = DEFAULT_TSE_WINDOWS) -> list[BankEntry]:
    """The main table: one row per mode x tier, the unprocessed baselines and an oracle ceiling."""
    w = max(tse_windows)
    nat = NATURAL
    bank = [
        BankEntry("baseline", "none", "passthrough", note="the unprocessed mix"),
        BankEntry("baseline", "none", "passthrough", condition="noise_only", note="target + noise, no talker"),
        # Natural, tier 4 (audio-only): a denoiser, judged on the noise-only condition (PLAN)
        BankEntry("natural", "audio-only", nat, 1.0, condition="noise_only", note="tier 4: noise suppression"),
        BankEntry("natural", "audio-only", nat, 0.5, condition="noise_only", wer=False),
        BankEntry("natural", "audio-only", "dfn", 1.0, condition="noise_only", wer=False, note="DFN3 standard (33 ms)"),
        BankEntry("natural", "audio-only", "gtcrn", 1.0, condition="noise_only", wer=False, note="GTCRN fallback"),
        BankEntry("natural", "audio-only", nat, 1.0, note="same denoiser with a competing talker"),
        # Natural, tier 3 (gate): + duck while the target is silent
        BankEntry("natural", "gate", f"{nat}>gate", 1.0, "oracle", note="tier 3: + camera gate"),
        BankEntry("natural", "gate", f"{nat}>gate", 0.5, "oracle", wer=False),
        BankEntry("natural", "remote-down", f"remote(fallback='{nat}')", 1.0, wer=False,
                  note="Focus HQ over the remote path with the GPU box down: local fallback + 200 ms playout"),
        # Natural, tier 2 (enrolled) and Focus HQ, tier 1 (windowed TSE)
        BankEntry("natural", "enrolled", f"tse:enrolled@{w:g}", 1.0, note="tier 2: ECAPA match of MossFormer2 streams"),
    ]
    for win in tse_windows:
        bank.append(BankEntry("focus_hq", "focus_hq", f"tse:blind_assoc@{win:g}", 1.0, wer=win == w,
                              note="audio-only MossFormer2 separation + p_speaking association"))
    bank += [
        BankEntry("focus_hq", "focus_hq", f"tse:blind_assoc@{w:g}", 0.5, wer=False),
        BankEntry("focus_hq", "upper-bound", f"tse:oracle@{w:g}", 1.0, wer=False, dnsmos=False,
                  note="clean target through the same windows: the ceiling"),
        # audio-visual set: the real Focus HQ candidates (lip crops from video.mp4)
        BankEntry("baseline", "none", "passthrough", fset="av", note="the unprocessed AV mix"),
        BankEntry("natural", "audio-only", nat, 1.0, fset="av", wer=False),
    ]
    for win in tse_windows:
        bank.append(BankEntry("focus_hq", "focus_hq", f"tse:av_mossformer2@{win:g}", 1.0, fset="av", wer=win == w,
                              note="ClearerVoice AV-MossFormer2 on our lip crops"))
    bank += [
        BankEntry("focus_hq", "focus_hq", f"tse:av_mossformer2@{w:g}", 0.5, fset="av", wer=False),
        BankEntry("focus_hq", "focus_hq", f"tse:dolphin@{w:g}", 1.0, fset="av", note="Dolphin AV-TSE (non-commercial)"),
        BankEntry("focus_hq", "focus_hq", f"tse:blind_assoc@{w:g}", 1.0, fset="av", wer=False),
        # Text and Quiet
        BankEntry("text", "audio-only", f"{nat}>synth(asr='local',speak=false)", 1.0, score_audio=False,
                  dnsmos=False, note="captions from local Whisper on the cleaned stream"),
        BankEntry("text", "none", "synth(asr='local',speak=false)", 1.0, score_audio=False, dnsmos=False,
                  note="captions from the raw mix"),
        BankEntry("quiet", "none", "silence", 1.0, score_audio=False, wer=False, dnsmos=False),
    ]
    return bank


def slider_bank() -> list[BankEntry]:
    """Slider sweep for chart 3 (entries already in the main table are reused, not re-run)."""
    out = []
    for s in SLIDER_GRID:
        out.append(BankEntry("natural", "audio-only", NATURAL, s, wer=False, condition="noise_only"))
        out.append(BankEntry("natural", "gate", f"{NATURAL}>gate", s, "oracle", wer=False))
    return out


def _entry_key(e: BankEntry) -> tuple:
    return (e.spec, e.suppression, e.target, e.condition, e.fset)


def load_any_fixture(d: Path):
    """``eval.fixtures.mix.load_fixture`` that also accepts AV manifests (extra keys such as ``layout``)."""
    from dataclasses import fields

    from eval.fixtures.mix import Manifest, Stems

    raw = json.loads((Path(d) / "manifest.json").read_text())
    known = {f.name for f in fields(Manifest)}
    manifest = Manifest(**{k: v for k, v in raw.items() if k in known})
    stems = Stems(**{n: aio.load(Path(d) / manifest.paths[n], manifest.sample_rate)[0]
                     for n in ("mix", "target", "interferer", "noise")})
    return manifest, stems


def apply_condition(stems, condition: str):
    """``noise_only``: drop the competing talker (stems sum to the mix, so target + noise keeps every level)."""
    from dataclasses import replace as dc_replace

    if condition == "mix":
        return stems
    if condition == "noise_only":
        return dc_replace(stems, mix=(stems.target + stems.noise).astype(np.float32),
                          interferer=np.zeros_like(stems.interferer))
    raise ValueError(f"unknown condition {condition!r}")


def _bank_ctx(entry: BankEntry) -> StageContext:
    ctx = StageContext.default()
    ctx.mode = entry.mode if entry.mode in _MODES else "natural"
    ctx.suppression = float(entry.suppression)
    if entry.target == "oracle":
        ctx.target = TargetState(TRACK_ID, 1.0, True, 0.0)
    return ctx


def oracle_driver(target: np.ndarray, sr: int):
    """``on_chunk`` hook: one locked face whose p_speaking follows the clean target (25 fps frames)."""
    from eval.bakeoff_fixtures import oracle_p_speaking

    p = oracle_p_speaking(target, sr, ORACLE_FPS)

    def drive(ctx: StageContext, seq: int, t: float) -> None:
        i = min(int(t * ORACLE_FPS), len(p) - 1)
        ctx.tracks = [FaceTrack(TRACK_ID, ORACLE_BOX, None, float(p[i]), t)]

    return drive


def residual_lag_ms(y: np.ndarray, ref: np.ndarray, sr: int, max_ms: float = 300.0, seconds: float = 5.0) -> float:
    """Lag of ``y`` behind ``ref`` (ms) left after the declared-latency shift; ~0 when ``latency_ms`` is honest.

    Peak of the cross-correlation over the first ``seconds`` within ``+-max_ms``; nan on silence.
    """
    n = min(len(y), len(ref), round(seconds * sr))
    a, b = np.asarray(y[:n], np.float64), np.asarray(ref[:n], np.float64)
    if n < 2 or not a.any() or not b.any():
        return math.nan
    size = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    k = round(max_ms * sr / 1000.0)
    lags = np.concatenate([np.arange(0, k + 1), np.arange(-k, 0)])
    window = np.concatenate([corr[: k + 1], corr[-k:]])
    return 1000.0 * float(lags[int(np.argmax(np.abs(window)))]) / sr


def _captions(pipeline: registry.Pipeline) -> str | None:
    """Final caption text of every ``synth`` stage in the pipeline (None if it has none)."""
    from hearing.synth.pipeline import SynthStage

    synths = [s for s in pipeline.stages if isinstance(s, SynthStage)]
    if not synths:
        return None
    texts = [c["text"] for s in synths for c in s.drain_captions() if c.get("final") and c["text"].strip()]
    for s in synths:
        s.close()
    return " ".join(texts)


def _dnsmos_fields(scorer, y: np.ndarray, sr: int, suffix: str = "") -> dict:
    if scorer is None:
        return {}
    s = scorer.score(y, sr)
    return {f"dnsmos_{k}{suffix}": getattr(s, k) for k in ("sig", "bak", "ovrl", "p808")}


def _base_row(entry: BankEntry, manifest, stems, base: dict) -> dict:
    return {
        "mode": entry.mode,
        "tier": entry.tier,
        "condition": entry.condition,
        "fset": entry.fset,
        "stages": entry.spec,
        "label": entry.label,
        "suppression": entry.suppression,
        "target": entry.target,
        "fixture": manifest.id,
        "snr_db": manifest.snr_db,
        "sir_db": manifest.sir_db,
        "noise": manifest.noise_type,
        "duration_s": manifest.duration_s,
        **base,
        "wer": math.nan,
    }


_WER_IDS: set[str] | None = None  # set by run_full: fixtures (set, id) that get a transcriber WER (None = all)


def _wer_here(manifest) -> bool:
    return _WER_IDS is None or (manifest.set, manifest.id) in _WER_IDS


def _finish_row(row: dict, entry: BankEntry, y: np.ndarray, stems, sr: int, manifest, transcriber,
                scorer, captions: str | None) -> dict:
    if entry.score_audio:
        row.update(score(y, stems, sr))
        row["lag_error_ms"] = residual_lag_ms(y, stems.mix, sr)
    if captions is not None:
        row["hyp"] = captions
        row["wer"] = metrics.wer(captions, manifest.transcript)
    elif entry.wer and transcriber is not None and _wer_here(manifest):
        if transcriber.name == "mock":
            transcriber.transcript = manifest.transcript
        row["hyp"] = transcriber.transcribe(y, sr)
        row["wer"] = metrics.wer(row["hyp"], manifest.transcript)
    if entry.dnsmos:
        row.update(_dnsmos_fields(scorer, y, sr))
    return row


def _baseline_cached(manifest, stems, sr: int, cache: dict, transcriber, scorer, condition: str = "mix") -> dict:
    """Input metrics (incl. its WER and DNSMOS) once per fixture and condition."""
    key = (manifest.set, manifest.id, condition)
    if key not in cache:
        b = baseline(stems)
        if transcriber is not None and _wer_here(manifest):
            if transcriber.name == "mock":
                transcriber.transcript = manifest.transcript
            b["wer_mix"] = metrics.wer(transcriber.transcribe(stems.mix, sr), manifest.transcript)
        b.update(_dnsmos_fields(scorer, stems.mix, sr, "_mix"))
        cache[key] = b
    return cache[key]


def run_stream_entry(entry: BankEntry, dirs: list[Path], transcriber, scorer, cache: dict) -> tuple[list, list, list]:
    rows, compute, failures = [], [], []
    for d in dirs:
        try:
            manifest, stems = load_any_fixture(d)
            stems = apply_condition(stems, entry.condition)
            sr = manifest.sample_rate
            base = _baseline_cached(manifest, stems, sr, cache, transcriber, scorer, entry.condition)
            pipeline = build(entry.spec)
            hook = oracle_driver(stems.target, sr) if entry.target == "oracle" else None
            result = run_stream(pipeline, stems.mix, sr, _bank_ctx(entry), hook)
            lat = metrics.LatencyStats.from_ms(result.compute_ms)
            row = _base_row(entry, manifest, stems, base)
            row.update(latency_ms=result.latency_ms, p50_ms=lat.p50, p95_ms=lat.p95, max_ms=lat.max,
                       chunks=lat.n, rtf=metrics.rtf(sum(result.compute_ms) / 1000.0, result.audio_s),
                       extras=result.extras)
            rows.append(_finish_row(row, entry, result.output, stems, sr, manifest, transcriber, scorer,
                                    _captions(pipeline)))
            compute.extend(result.compute_ms)
        except Exception as e:  # noqa: BLE001 - a failing stage is a reported failure, not a crash
            failures.append({"stages": entry.label, "fixture": d.name, "error": f"{type(e).__name__}: {e}"})
            print(f"FAIL {entry.label} fixture={d.name}: {type(e).__name__}: {e}", file=sys.stderr)
    return rows, compute, failures


def _with_lips(fx):
    """Lip crops for AV-set fixtures, whose manifests (eval.make_fixtures --av) give ``faces[role=target].box``
    rather than the ``target_box`` that ``eval.bakeoff_fixtures`` reads."""
    from dataclasses import replace as dc_replace

    from eval.bakeoff_fixtures import lips_from_video, mouth_box_from_face

    video = fx.path / "video.mp4"
    if fx.lips is not None or not video.exists():
        return fx
    box = next((f["box"] for f in fx.manifest.get("faces", []) if f.get("role") == "target"), None)
    if box is None:
        return fx
    fps = float(fx.manifest.get("fps", fx.lips_fps))
    lips = lips_from_video(video, mouth_box_from_face(box), fps)
    return dc_replace(fx, lips=lips, lips_fps=fps, lips_source="video.mp4@faces") if lips is not None else fx


def run_tse_entry(entry: BankEntry, dirs: list[Path], transcriber, scorer, cache: dict) -> tuple[list, list, list]:
    """Windowed TSE tier: the bake-off's extractor (hop = window / 2), mixed by the slider, scored like the rest."""
    from eval import bakeoff
    from eval.bakeoff_fixtures import load_fixture as load_av
    from hearing.tse import SlidingWindowExtractor, get_candidate

    name, _, window = entry.spec[len("tse:"):].partition("@")
    window_s = float(window or 2.0)
    try:
        cand = get_candidate(name)
        cand.load()
        ext = SlidingWindowExtractor(cand, window_s, window_s * 0.5)
    except Exception as e:  # noqa: BLE001 - a candidate that cannot load is a failure row
        return [], [], [{"stages": entry.label, "fixture": "*", "error": f"load: {type(e).__name__}: {e}"}]
    rows, compute, failures = [], [], []
    for d in dirs:
        try:
            if entry.condition != "mix":
                raise ValueError("TSE tiers run on the mix condition only")
            manifest, stems = load_any_fixture(d)
            sr = manifest.sample_rate
            base = _baseline_cached(manifest, stems, sr, cache, transcriber, scorer)
            fx = _with_lips(load_av(d))
            ext.reset_stats()
            est = ext.extract(fx.mix, fx.sr, **bakeoff._inputs(cand, fx))
            st = ext.stats()
            n = min(len(est), len(stems.mix))
            y = np.asarray(est[:n], np.float32)
            if entry.suppression < 1.0:  # fusion.mixer: target + (1 - s) * residual, target on the mix's scale
                mix = stems.mix[:n]
                gain = float(np.dot(y, mix) / max(float(np.dot(y, y)), 1e-12))
                y = gain * y
                y = y + (1.0 - entry.suppression) * (mix - y)
            if n < len(stems.mix):
                y = np.concatenate([y, np.zeros(len(stems.mix) - n, np.float32)])
            row = _base_row(entry, manifest, stems, base)
            row.update(latency_ms=st.algorithmic_latency_ms, p50_ms=st.compute_p50_ms, p95_ms=st.compute_p95_ms,
                       max_ms=st.compute_max_ms, chunks=st.n_windows, rtf=st.rtf,
                       extras={"device": getattr(cand, "device", None), "window_s": window_s})
            rows.append(_finish_row(row, entry, y, stems, sr, manifest, transcriber, scorer, None))
            compute.extend(ext._times_ms)  # per-window compute of this fixture (reset_stats above)
        except Exception as e:  # noqa: BLE001
            failures.append({"stages": entry.label, "fixture": d.name, "error": f"{type(e).__name__}: {e}"})
            print(f"FAIL {entry.label} fixture={d.name}: {type(e).__name__}: {e}", file=sys.stderr)
    return rows, compute, failures


def _bank_aggregate(entry: BankEntry, rows: list[dict], compute_ms: list[float], failed: int) -> dict:
    agg = aggregate(entry.label, rows, compute_ms, failed)
    agg.update(mode=entry.mode, tier=entry.tier, spec=entry.spec, label=entry.label,
               suppression=entry.suppression, target=entry.target, note=entry.note,
               condition=entry.condition, fset=entry.fset)
    if entry.is_tse:  # per-window compute: the pooled p50/p95 are per window, not per 10 ms chunk
        agg["compute_unit"] = "window"
    else:
        agg["compute_unit"] = "10ms-chunk"
    by_snr = {}
    for snr in sorted({r["snr_db"] for r in rows}):
        sub = [r for r in rows if r["snr_db"] == snr]
        by_snr[f"{snr:g}"] = metrics.summarize(sub)
    agg["by_snr"] = by_snr
    return agg


def synth_latency(tts_engines: list[str], phrases: int = 8) -> list[dict]:
    """Synthetic mode: phrase latency (speaker's last word -> first re-voiced sample) per TTS engine."""
    from hearing.synth.pipeline import run_bench

    out = []
    for name in tts_engines:
        try:
            # Mock TTS keeps up at any speed; real engines are paced at realtime so the queue never backs up.
            speed = 4.0 if name == "mock" else 1.0
            out.append(run_bench(asr="mock", tts=name, phrases=phrases, speed=speed))
        except Exception as e:  # noqa: BLE001 - an engine that fails is a row with its error
            out.append({"engine": name, "error": f"{type(e).__name__}: {e}"})
    return out


def tts_intelligibility(tts_name: str, dirs: list[Path], transcriber, limit: int = 12) -> dict:
    """WER of the transcriber on TTS renderings of the reference transcripts (how well re-voiced speech reads)."""
    from hearing.synth.tts import get_tts

    if transcriber is None or transcriber.name == "mock":
        return {"engine": tts_name, "wer": math.nan, "n": 0, "note": "needs a real transcriber"}
    tts = get_tts(tts_name)
    hyps, refs = [], []
    try:
        for d in dirs[:limit]:
            manifest, _ = load_any_fixture(d)
            y = tts.synthesize(manifest.transcript, "neutral")
            hyps.append(transcriber.transcribe(np.asarray(y, np.float32), 16000))
            refs.append(manifest.transcript)
    finally:
        tts.close()
    return {"engine": getattr(tts, "name", tts_name), "wer": metrics.wer_batch(hyps, refs), "n": len(refs)}


def real_clip_dnsmos(entries: list[BankEntry], clips: list[Path], scorer) -> list[dict]:
    """DNSMOS of every real clip, raw and through each streaming Natural entry (no oracle target exists)."""
    rows = []
    for clip in clips:
        x, sr = aio.load(clip, sr=16000)
        rows.append({"clip": clip.name, "label": "input", **scorer.score(x, sr).as_dict()})
        for e in entries:
            try:
                y = run_stream(build(e.spec), x, sr, _bank_ctx(e)).output
                rows.append({"clip": clip.name, "label": e.label, **scorer.score(y, sr).as_dict()})
            except Exception as ex:  # noqa: BLE001
                rows.append({"clip": clip.name, "label": e.label, "error": f"{type(ex).__name__}: {ex}"})
    return rows


def run_full(
    set_name: str = "full",
    wer_mode: str = "auto",
    dnsmos_mode: str = "auto",
    fixtures_root: Path = DEFAULT_FIXTURES,
    out_dir: Path | None = DEFAULT_OUT,
    limit: int | None = None,
    tse_limit: int | None = None,
    tse_windows: tuple[float, ...] = DEFAULT_TSE_WINDOWS,
    real_dir: Path | None = REAL_CLIPS,
    av_set: str | None = "auto",
    bank: list[BankEntry] | None = None,
    slider: list[BankEntry] | None = None,
    tts_engines: list[str] | None = None,
    wer_limit: int | None = 32,
) -> dict:
    """Every mode x tier over one fixture set; returns the report (JSON + markdown written to ``out_dir``)."""
    from eval import dnsmos

    dirs = fixture_dirs(Path(fixtures_root), set_name)
    stream_dirs = dirs[:limit] if limit is not None else dirs
    tse_dirs = stream_dirs[:tse_limit] if tse_limit is not None else stream_dirs
    if av_set == "auto":  # the larger AV set when it has been generated
        av_set = next((n for n in ("av_full", "av_tiny") if (Path(fixtures_root) / n / "index.json").exists()), None)
    av_dirs = fixture_dirs(Path(fixtures_root), av_set, generate=False) if av_set else []
    av_dirs = av_dirs[:limit] if limit is not None else av_dirs
    av_tse_dirs = av_dirs[:tse_limit] if tse_limit is not None else av_dirs
    skipped = []
    global _WER_IDS
    _WER_IDS = None
    if wer_limit is not None:  # the transcriber is the slow part: WER on the first N fixtures of each set
        _WER_IDS = set()
        for ds in (stream_dirs, av_dirs):
            for d in ds[:wer_limit]:
                _WER_IDS.add((json.loads((d / "manifest.json").read_text())["set"], d.name))
    transcriber = make_transcriber(wer_mode)
    scorer = None if dnsmos_mode == "off" else dnsmos.get_dnsmos(dnsmos_mode)
    bank = full_bank(tse_windows) if bank is None else bank
    slider = slider_bank() if slider is None else slider
    wall0 = time.perf_counter()
    cache: dict = {}
    done: dict[tuple, dict] = {}
    rows, aggregates, failures = [], [], []
    for group, entries in (("bank", bank), ("slider", slider)):
        for entry in entries:
            key = _entry_key(entry)
            if key in done:
                aggregates.append({**done[key], "group": group})
                continue
            if entry.fset == "av" and not av_dirs:
                skipped.append({"label": entry.label, "reason": "no AV set (python -m eval.make_fixtures --av --tiny)"})
                continue
            t0 = time.perf_counter()
            runner = run_tse_entry if entry.is_tse else run_stream_entry
            if entry.fset == "av":
                e_dirs = av_tse_dirs if entry.is_tse else av_dirs
            else:
                e_dirs = tse_dirs if entry.is_tse else stream_dirs
            e_rows, e_ms, e_fail = runner(entry, e_dirs, transcriber, scorer, cache)
            agg = _bank_aggregate(entry, e_rows, e_ms, len(e_fail))
            agg["wall_s"] = time.perf_counter() - t0
            done[key] = agg
            aggregates.append({**agg, "group": group})
            rows.extend(e_rows)
            failures.extend(e_fail)
            print(f"bench --full: {entry.mode}/{entry.tier} {entry.label}: n={len(e_rows)} "
                  f"failed={len(e_fail)} {agg['wall_s']:.1f}s", file=sys.stderr)
    if tts_engines is None:
        from hearing.synth import tts as tts_mod

        tts_engines = ["mock"] + (["local"] if tts_mod.local_available() else [])
        if os.environ.get("ELEVENLABS_API_KEY"):
            tts_engines.append("elevenlabs")
    synth = {
        "phrase_latency": synth_latency(tts_engines),
        "tts_wer": [tts_intelligibility(t, stream_dirs, transcriber) for t in tts_engines if t != "mock"],
    }
    clips = dnsmos.find_clips([real_dir]) if real_dir is not None and Path(real_dir).exists() else []
    natural = [e for e in bank if not e.is_tse and e.mode == "natural" and e.target == "none" and e.fset == "audio"
               and e.condition == "noise_only"]  # real clips: the denoisers (no talker is removed anyway)
    real = {
        "dir": str(real_dir) if real_dir is not None else None,
        "clips": len(clips),
        "rows": real_clip_dnsmos(natural, clips, scorer) if clips and scorer is not None else [],
    }
    report = {
        "kind": "bench-full",
        "created": _now().isoformat(timespec="seconds"),
        "device": device.describe(),
        "git_sha": git_sha(),
        "machine": f"{platform.node()} ({platform.machine()}, {platform.platform()})",
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "fixture_set": {"name": set_name, "root": str(fixtures_root), "count": len(stream_dirs),
                        "tse_count": len(tse_dirs), "limit": limit, "tse_limit": tse_limit},
        "av_set": {"name": av_set, "count": len(av_dirs), "tse_count": len(av_tse_dirs)},
        "wer_limit": wer_limit,
        "skipped": skipped,
        "wer": transcriber.name if transcriber is not None else "off",
        "dnsmos": scorer.name if scorer is not None else "off",
        "chunk_ms": CHUNK_MS,
        "oracle": "target p_speaking = eval.bakeoff_fixtures.oracle_p_speaking(clean target, 25 fps); "
                  "enrolled tier enrolls on the first 3 s of the clean target",
        "wall_s": math.nan,
        "rows": rows,
        "aggregates": aggregates,
        "synth": synth,
        "real_clips": real,
        "failures": failures,
    }
    report["wall_s"] = time.perf_counter() - wall0
    report["claims"] = claims(report)
    if out_dir is not None:
        path = write_results(Path(out_dir), report)
        report["path"] = str(path)
        path.with_suffix(".md").write_text(render_markdown(report) + "\n")
    return report


# --- full report: markdown ------------------------------------------------------------------


def _f(v, fmt: str = "{:.2f}") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return fmt.format(v)


def _m(agg: dict, key: str) -> float:
    return agg.get("metrics", {}).get(key, {}).get("mean", math.nan)


def _bank(report: dict, group: str = "bank") -> list[dict]:
    return [a for a in report["aggregates"] if a.get("group") == group]


def _find(report: dict, spec: str, s: float, target: str = "none", condition: str = "mix",
          fset: str = "audio") -> dict | None:
    for a in report["aggregates"]:
        if (a["spec"], a["suppression"], a["target"], a.get("condition", "mix"), a.get("fset", "audio")) == (
                spec, s, target, condition, fset):
            return a
    return None


LAG_TOLERANCE_MS = 2.0  # |median residual lag| above this: the pipeline's latency_ms is not what it outputs


def _lag_cell(agg: dict) -> str:
    lag = agg.get("metrics", {}).get("lag_error_ms", {}).get("median", math.nan)
    if not math.isfinite(lag):
        return "-"
    return f"{lag:+.0f} **(latency_ms wrong)**" if abs(lag) > LAG_TOLERANCE_MS else f"{lag:+.0f}"


def _e2e_ms(agg: dict) -> float:
    """Algorithmic latency + p95 compute of one hop (per 10 ms chunk, or per window for TSE)."""
    return _m(agg, "latency_ms") + agg["compute_ms"]["p95"]


def _mix_mean(report: dict, key: str, condition: str = "mix", fset: str = "audio") -> float:
    """Mean of an input (``*_mix``) metric over the fixtures of one condition and set."""
    base = _find(report, "passthrough", 0.5, condition=condition, fset=fset)
    return _m(base, key) if base else math.nan


def _input_desc(a: dict) -> str:
    cond = "noise only" if a.get("condition") == "noise_only" else "mix"
    return f"{cond}, AV" if a.get("fset") == "av" else cond


def render_markdown(report: dict) -> str:
    dev = report["device"]
    fs = report["fixture_set"]
    av = report.get("av_set") or {}
    limited = ""
    if fs.get("limit") is not None or fs.get("tse_limit") is not None:
        limited = (f" Bounded run: `--limit {fs.get('limit')}` / `--tse-limit {fs.get('tse_limit')}` (the first "
                   "fixtures by id; ids cycle SNR fastest, then SIR/layout, then noise type).")
    lines = [
        (f"Run {report['created']} on `{dev.get('hostname')}` ({report.get('cpu') or dev.get('machine')}, torch "
         f"device `{dev.get('device')}`), git `{report['git_sha']}`. Main set `{fs['name']}` ({fs['count']} fixtures, "
         f"TSE tiers on {fs['tse_count']}); AV set `{av.get('name')}` ({av.get('count', 0)} fixtures).{limited} "
         f"WER by `{report['wer']}` (first {report.get('wer_limit') or 'all'} fixtures of each set; captions on all), "
         f"DNSMOS by `{report['dnsmos']}`, wall {report['wall_s'] / 60:.1f} min. "
         f"JSON: `eval/results/{Path(report.get('path', '<stamp>.json')).name}`."),
        "",
        "### Every mode and tier",
        "",
        ("Means over fixtures. `input`: `mix` = target + competing talker + noise; `noise only` = the same "
         "fixtures without the talker (what a denoiser is judged on); `AV` = the GRID audio-visual set (lip crops "
         "from the video). `algo ms` = algorithmic latency (a TSE window for Focus HQ / Enrolled). `p50/p95` = "
         "measured compute per 10 ms chunk (per window for TSE rows, marked `/win`). WER by the reference "
         "transcriber on the output (captions for Text). DNSMOS = P.835 OVRL / BAK. `lag err` = median residual "
         "lag of the output after the declared-latency shift (0 when `latency_ms` is honest). Gate rows and "
         "`blind_assoc` use an **oracle** speaking detector from the clean target; Enrolled enrolls on 3 s of the "
         "clean target."),
        "",
        ("| mode | tier | pipeline | s | input | n | algo ms | p50 ms | p95 ms | RTF | SI-SDRi dB | PESQ | STOI | "
         "WER | DNSMOS OVRL | DNSMOS BAK | lag err ms |"),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in _bank(report):
        unit = "/win" if a.get("compute_unit") == "window" else ""
        lines.append(
            f"| {a['mode']} | {a['tier']} | `{a['spec']}` | {a['suppression']:g} | {_input_desc(a)} | {a['n']} | "
            f"{_f(_m(a, 'latency_ms'), '{:.0f}')} | {_f(a['compute_ms']['p50'], '{:.2f}')}{unit} | "
            f"{_f(a['compute_ms']['p95'], '{:.2f}')}{unit} | {_f(_m(a, 'rtf'), '{:.3f}')} | {_f(_m(a, 'si_sdri'))} | "
            f"{_f(_m(a, 'pesq'))} | {_f(_m(a, 'stoi'), '{:.3f}')} | {_f(_m(a, 'wer'), '{:.3f}')} | "
            f"{_f(_m(a, 'dnsmos_ovrl'))} | {_f(_m(a, 'dnsmos_bak'))} | {_lag_cell(a)} |"
        )
    for b in (a for a in _bank(report) if a["mode"] == "baseline"):
        lines.append("")
        lines.append(f"Input ({_input_desc(b)}): SI-SDR {_f(_m(b, 'si_sdr_mix'))} dB, PESQ {_f(_m(b, 'pesq_mix'))}, "
                     f"STOI {_f(_m(b, 'stoi_mix'), '{:.3f}')}, WER {_f(_m(b, 'wer_mix'), '{:.3f}')}, DNSMOS OVRL "
                     f"{_f(_m(b, 'dnsmos_ovrl_mix'))} / BAK {_f(_m(b, 'dnsmos_bak_mix'))}.")
    for sk in report.get("skipped", []):
        lines.append(f"\nSkipped `{sk['label']}`: {sk['reason']}.")
    synth = report.get("synth", {})
    lines += ["", "### Synthetic mode (re-voiced)", "",
              ("Phrase latency = speaker's last word to the first re-voiced sample (stream clock, includes the ASR "
               "endpoint pause; mock ASR; mock TTS at 4x realtime, real engines at 1x). `est` = pause + TTS time to "
               "first audio = what a live session hears."), "",
              "| TTS | ASR | device | phrases | p50 ms | p95 ms | TTS p50 ms | TTS p95 ms | est p50 ms | est p95 ms |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in synth.get("phrase_latency", []):
        if "error" in r:
            lines.append(f"| {r['engine']} | - | - | - | error: {r['error']} | | | | | |")
            continue
        lines.append(f"| {r['engine']} | {r['asr']} | {r['device']} | {r['n']} | {_f(r['p50'], '{:.0f}')} | "
                     f"{_f(r['p95'], '{:.0f}')} | {_f(r['tts_p50'], '{:.0f}')} | {_f(r['tts_p95'], '{:.0f}')} | "
                     f"{_f(r['est_p50'], '{:.0f}')} | {_f(r['est_p95'], '{:.0f}')} |")
    for r in synth.get("tts_wer", []):
        lines.append(f"\nTTS intelligibility: `{report['wer']}` WER on `{r['engine']}` renderings of {r['n']} "
                     f"reference transcripts = {_f(r['wer'], '{:.3f}')}.")

    lines += ["", "### Chart 1: latency vs quality (scatter, x = end-to-end ms, y = SI-SDRi)", "",
              "End-to-end = algorithmic latency + p95 compute of one hop.", "",
              "| pipeline | mode / tier | input | e2e ms | SI-SDRi dB | STOI | WER |", "|---|---|---|---|---|---|---|"]
    for a in _bank(report):
        if a["mode"] in ("text", "quiet", "baseline"):
            continue
        lines.append(f"| `{a['label']}` | {a['mode']} / {a['tier']} | {_input_desc(a)} | {_f(_e2e_ms(a), '{:.0f}')} | "
                     f"{_f(_m(a, 'si_sdri'))} | {_f(_m(a, 'stoi'), '{:.3f}')} | {_f(_m(a, 'wer'), '{:.3f}')} |")
    snrs = sorted({k for a in _bank(report) if a.get("fset") != "av" for k in a.get("by_snr", {})}, key=float)
    lines += ["", "### Chart 2: SI-SDRi by input SNR (lines, one per pipeline; main set)", "",
              "| pipeline | input | " + " | ".join(f"{s} dB" for s in snrs) + " |", "|---|---|" + "---|" * len(snrs)]
    for a in _bank(report):
        if a["mode"] in ("text", "quiet") or a.get("fset") == "av":
            continue
        key = "si_sdr_mix" if a["mode"] == "baseline" else "si_sdri"
        cells = [_f(a["by_snr"].get(s, {}).get(key, {}).get("mean", math.nan)) for s in snrs]
        name = f"input SI-SDR (not an improvement), {_input_desc(a)}" if a["mode"] == "baseline" else f"`{a['label']}`"
        lines.append(f"| {name} | {_input_desc(a)} | " + " | ".join(cells) + " |")
    lines += ["", "### Chart 3: the slider (lines, x = s)", "",
              (f"`{NATURAL}` on the noise-only input (the room's noise) and `{NATURAL}>gate` with an oracle target on "
               "the full mix (the competing talker)."), "",
              (f"| s | `{NATURAL}` SI-SDRi | `{NATURAL}` STOI | `{NATURAL}` BAK | `{NATURAL}>gate` SI-SDRi | "
               f"`{NATURAL}>gate` STOI | `{NATURAL}>gate` BAK |"), "|---|---|---|---|---|---|---|"]
    for s in sorted({a["suppression"] for a in _bank(report, "slider")}):
        g = _find(report, NATURAL, s, condition="noise_only")
        t = _find(report, f"{NATURAL}>gate", s, "oracle")
        cells = []
        for a in (g, t):
            cells += [_f(_m(a, "si_sdri")), _f(_m(a, "stoi"), "{:.3f}"), _f(_m(a, "dnsmos_bak"))] if a else ["-"] * 3
        lines.append(f"| {s:g} | " + " | ".join(cells) + " |")
    real = report.get("real_clips", {})
    lines += ["", "### Real phone clips (DNSMOS, no reference needed)", ""]
    if real.get("rows"):
        lines += ["| clip | pipeline | SIG | BAK | OVRL | P808 |", "|---|---|---|---|---|---|"]
        for r in real["rows"]:
            if "error" in r:
                lines.append(f"| {r['clip']} | `{r['label']}` | error: {r['error']} | | | |")
            else:
                lines.append(f"| {r['clip']} | `{r['label']}` | {_f(r['sig'])} | {_f(r['bak'])} | {_f(r['ovrl'])} | "
                             f"{_f(r['p808'])} |")
    else:
        lines.append(f"No clips under `eval/fixtures/real/` at this run ({real.get('clips', 0)} found). Record them "
                     "(docs/DEMO.md §9) and re-run `python -m eval.bench --full`, or score them alone with "
                     f"`python -m eval.dnsmos --stages {NATURAL}`.")
    if report.get("failures"):
        lines += ["", f"### Failures ({len(report['failures'])})", ""]
        lines += [f"- `{f['stages']}` on `{f['fixture']}`: {f['error']}" for f in report["failures"][:20]]
    lines += ["", "### Slide-safe one-line claims", "",
              "Generated from this run; each carries its condition. Do not drop the condition on the slide.", ""]
    lines += [f"- {c}" for c in report.get("claims", [])] or ["- (none: the run produced no numbers)"]
    return "\n".join(lines)


def _signed_db(v: float) -> str:
    return f"{v:+.1f} dB"


def _ok(v: float) -> bool:
    return isinstance(v, float | int) and math.isfinite(v)


def claims(report: dict) -> list[str]:
    """One-line, condition-carrying claims, emitted only when the numbers behind them exist and hold."""
    out = []
    dev = report["device"]
    cpu = report.get("cpu") or dev.get("machine") or "unknown"
    host = f"{'an' if cpu[:1].lower() in 'aeiou' else 'a'} {cpu} laptop ({dev.get('hostname')})"
    fs, av = report["fixture_set"], report.get("av_set") or {}
    full = fs["name"] == "full"
    corpus = "LibriSpeech" if full else f"`{fs['name']}`"
    noisy = f"{fs['count']} {corpus} clips with babble/pink noise at -5 to 10 dB SNR" if full else f"{fs['count']} {corpus} clips"
    mixes = f"{fs['count']} {corpus} two-talker + noise mixtures" if full else f"{fs['count']} {corpus} mixtures"
    tse_mixes = f"{fs['tse_count']} {corpus} two-talker + noise mixtures"
    av_mixes = f"{av.get('tse_count', 0)} GRID audio-visual two-talker + babble clips"
    nat = _find(report, NATURAL, 1.0, condition="noise_only")
    if nat and nat["n"]:
        out.append(f"Natural fast path (DeepFilterNet3-LL, 10 ms hops): {_m(nat, 'latency_ms'):.0f} ms algorithmic "
                   f"latency + {nat['compute_ms']['p95']:.1f} ms p95 compute per 10 ms chunk (RTF {_m(nat, 'rtf'):.2f}) "
                   f"on {host}.")
        if _m(nat, "si_sdri") > 0.5:
            out.append(f"At full suppression it improves SI-SDR by {_m(nat, 'si_sdri'):.1f} dB on {noisy} "
                       "(noise only, no competing talker).")
        bak0, bak1 = _mix_mean(report, "dnsmos_bak_mix", "noise_only"), _m(nat, "dnsmos_bak")
        if _ok(bak0) and _ok(bak1) and bak1 > bak0 and report["dnsmos"] != "mock":
            out.append(f"It lifts the DNSMOS background-noise score from {bak0:.2f} to {bak1:.2f} (1-5 scale) on the "
                       "same clips.")
    nat_mix, gate = _find(report, NATURAL, 1.0), _find(report, f"{NATURAL}>gate", 1.0, "oracle")
    if nat_mix and gate and gate["n"] and _m(gate, "si_sdri") > _m(nat_mix, "si_sdri") + 0.3:
        out.append(f"With a competing talker, the camera gate (duck the room while the target is silent) adds "
                   f"{_m(gate, 'si_sdri') - _m(nat_mix, 'si_sdri'):.1f} dB SI-SDRi over the denoiser alone at 0 ms "
                   f"extra latency, given a perfect speaking detector ({mixes}).")
    tse = [a for a in _bank(report) if a["tier"] == "focus_hq" and a["suppression"] == 1.0 and a["n"]]
    for fset, desc, cond in (("av", av_mixes, "lip video of the target"),
                             ("audio", tse_mixes, "a perfect speaking signal")):
        cands = [a for a in tse if a.get("fset", "audio") == fset and _ok(_m(a, "si_sdri"))]
        if not cands:
            continue
        best = max(cands, key=lambda a: _m(a, "si_sdri"))
        if _m(best, "si_sdri") <= 0.5:
            continue
        wer0, wer1 = _mix_mean(report, "wer_mix", fset=fset), _m(best, "wer")
        wer_part = ""
        if report["wer"] != "mock" and _ok(wer0) and _ok(wer1) and wer1 < wer0:
            wer_part = f" and cuts WER from {wer0:.2f} to {wer1:.2f}"
            if wer0 > 1.0:
                wer_part += " (above 1 on the raw mix: Whisper also transcribes the competing talker)"
        name = best["spec"].split(":", 1)[1].split("@")[0]
        out.append(f"Focus HQ (`{name}`, {_m(best, 'latency_ms') / 1000:g} s windows) improves SI-SDR by "
                   f"{_m(best, 'si_sdri'):.1f} dB{wer_part} on {desc}, ~{_e2e_ms(best) / 1000:.1f} s behind live on "
                   f"{host} (given {cond}).")
    e = next((a for a in _bank(report) if a["tier"] == "enrolled" and a["n"]), None)
    if e and _m(e, "si_sdri") > 0.5:
        out.append(f"Enrolled tier (audio only, 3 s of the target's voice): {_signed_db(_m(e, 'si_sdri'))} SI-SDRi with "
                   f"{_m(e, 'latency_ms') / 1000:g} s windows on {tse_mixes} (clean enrollment).")
    txt = next((a for a in _bank(report) if a["mode"] == "text" and a["tier"] == "audio-only" and a["n"]), None)
    raw = next((a for a in _bank(report) if a["mode"] == "text" and a["tier"] == "none" and a["n"]), None)
    if txt and _ok(_m(txt, "wer")):
        w_clean = _m(txt, "wer")
        w_raw = _m(raw, "wer") if raw else math.nan
        if _ok(w_raw) and w_raw < w_clean:  # say which input captions best, not just the cleaned one
            out.append(f"Text mode: live captions from local faster-whisper reach WER {w_raw:.2f} from the raw mix "
                       f"vs {w_clean:.2f} from the DeepFilterNet-cleaned stream on {mixes}, so ASR is fed the "
                       "raw audio.")
        else:
            vs = f" (vs {w_raw:.2f} on the raw mix)" if _ok(w_raw) else ""
            out.append(f"Text mode: live captions from local faster-whisper on the cleaned stream reach WER "
                       f"{w_clean:.2f}{vs} on {mixes}.")
    for r in report.get("synth", {}).get("phrase_latency", []):
        if r.get("engine") not in (None, "mock") and "error" not in r and _ok(r.get("est_p50", math.nan)):
            out.append(f"Synthetic mode ({r['engine']} TTS on {host}): the re-voiced phrase starts "
                       f"{r['est_p50']:.0f} ms (p50; p95 {r['est_p95']:.0f} ms) after the speaker stops, "
                       f"including a {r['pause_p50']:.0f} ms end-of-phrase pause.")
    real = report.get("real_clips", {})
    if real.get("rows") and report["dnsmos"] != "mock":
        inp = [r["ovrl"] for r in real["rows"] if r["label"] == "input" and "ovrl" in r]
        outs = [r["ovrl"] for r in real["rows"] if r["label"].startswith(f"{NATURAL} (s=1") and "ovrl" in r]
        if inp and outs:
            out.append(f"On {len(inp)} real phone recordings, DNSMOS overall goes from {np.mean(inp):.2f} to "
                       f"{np.mean(outs):.2f} at full suppression.")
    return out


def _nan_for_none(v):
    """Inverse of ``_json_safe`` for metric containers: JSON ``null`` back to NaN."""
    if v is None:
        return math.nan
    if isinstance(v, dict):
        return {k: _nan_for_none(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_nan_for_none(x) for x in v]
    return v


def load_report(path: Path) -> dict:
    """A ``--full`` results JSON, ready for :func:`render_markdown` (claims recomputed by this code)."""
    report = json.loads(Path(path).read_text())
    if report.get("kind") != "bench-full":
        raise ValueError(f"{path} is not a bench --full report (kind={report.get('kind')!r})")
    report["aggregates"] = _nan_for_none(report["aggregates"])
    report["synth"] = _nan_for_none(report.get("synth", {}))
    real = report.get("real_clips") or {}
    real["rows"] = _nan_for_none(real.get("rows", []))
    report["real_clips"] = real
    report["wall_s"] = report["wall_s"] if report.get("wall_s") is not None else math.nan
    if not report.get("cpu") and (report.get("device") or {}).get("hostname") == platform.node():
        report["cpu"] = cpu_name()  # runs from before the field existed, re-rendered on the same machine
    report["path"] = str(path)
    report["claims"] = claims(report)
    return report


def update_doc(doc: Path, markdown: str) -> None:
    """Replace the generated block between the bench markers in ``doc`` (append it if absent)."""
    text = doc.read_text() if doc.exists() else "# Benchmarks\n"
    block = f"{DOC_START}\n{markdown}\n{DOC_END}"
    if DOC_START in text and DOC_END in text:
        head, rest = text.split(DOC_START, 1)
        text = head + block + rest.split(DOC_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    doc.write_text(text)


def render(in_path: Path, spec: str, out_path: Path, suppression: float = 1.0) -> dict:
    """Stream one recording through ``spec`` at slider ``suppression`` and save the latency-aligned output."""
    x, sr = aio.load(in_path, sr=16000)
    ctx = StageContext.default()
    ctx.suppression = float(suppression)
    pipeline = build(spec)
    result = run_stream(pipeline, x, sr, ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aio.save(out_path, result.output, sr)
    captions = _captions(pipeline)
    lat = metrics.LatencyStats.from_ms(result.compute_ms)
    return {"in": str(in_path), "out": str(out_path), "stages": spec, "suppression": suppression,
            "duration_s": len(x) / sr, "latency_ms": result.latency_ms, "p95_ms": lat.p95,
            "captions": captions}


def _main_full(args: argparse.Namespace) -> int:
    set_name, wer_mode = args.set_name or "full", args.wer or "auto"
    windows = tuple(float(w) for w in args.tse_windows.split(",") if w.strip())
    try:
        if args.from_json is not None:  # re-render a finished run (wording/claims changes) without re-running it
            report = load_report(args.from_json)
        else:
            report = run_full(set_name, wer_mode, args.dnsmos, args.fixtures, args.out, args.limit,
                              args.tse_limit, windows, args.real, None if args.av_set == "none" else args.av_set,
                              wer_limit=args.wer_limit or None)
    except FileNotFoundError as e:
        print(f"bench: {e}", file=sys.stderr)
        return 2
    md = render_markdown(report)
    print(md)
    if args.from_json is not None:
        Path(report["path"]).with_suffix(".md").write_text(md + "\n")
    elif "path" in report:
        print(f"wrote {report['path']} and {Path(report['path']).with_suffix('.md')}")
    if not args.no_doc:
        update_doc(args.doc, md)
        print(f"updated {args.doc}")
    if report["failures"]:
        print(f"{len(report['failures'])} run(s) failed (listed in the report)", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", action="append", help='pipeline spec, repeatable (e.g. "gain(db=-6)>delay(ms=20)")')
    ap.add_argument("--set", dest="set_name", default=None,
                    help="fixture set: tiny (generated offline if missing) or full (default: tiny, full with --full)")
    ap.add_argument("--wer", choices=("off", "mock", "auto", "whisper"), default=None,
                    help="transcriber for WER (default: off, auto with --full)")
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES, help="fixture root holding <set>/")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results directory")
    ap.add_argument("--limit", type=int, default=None, help="only the first N fixtures")
    ap.add_argument("--fast", action="store_true", help="= --stages passthrough --set tiny --wer off")
    ap.add_argument("--full", action="store_true",
                    help="every mode x tier (full bank + slider sweep + Synthetic latency + real-clip DNSMOS); "
                    "writes JSON + markdown and updates --doc")
    ap.add_argument("--dnsmos", choices=("off", "mock", "auto", "dnsmos"), default="auto",
                    help="DNSMOS scorer for --full (auto: real models if present, else mock)")
    ap.add_argument("--tse-limit", type=int, default=None, help="--full: only the first N fixtures for the TSE tiers")
    ap.add_argument("--tse-windows", default="1,2", help="--full: Focus HQ window lengths in seconds")
    ap.add_argument("--real", type=Path, default=REAL_CLIPS, help="--full: directory of real phone clips for DNSMOS")
    ap.add_argument("--wer-limit", type=int, default=32,
                    help="--full: transcriber WER on the first N fixtures of each set (0 = all)")
    ap.add_argument("--av-set", default="auto",
                    help="--full: audio-visual set for the lip-reading Focus HQ rows (auto: av_full, else av_tiny; none)")
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC,
                    help="--full: markdown file whose generated block is replaced (default docs/BENCHMARKS.md)")
    ap.add_argument("--no-doc", action="store_true", help="--full: do not touch --doc")
    ap.add_argument("--from-json", type=Path, default=None, metavar="RESULTS.json",
                    help="--full: re-render the markdown, claims and --doc from a finished run instead of running")
    ap.add_argument("--render", type=Path, default=None, metavar="IN.wav",
                    help="stream one recording through the first --stages spec and save it (backup demo clips)")
    ap.add_argument("--render-out", type=Path, default=None, help="--render: output WAV (default IN.<spec>.wav)")
    ap.add_argument("--suppression", type=float, default=1.0, help="--render: slider s in [0, 1]")
    args = ap.parse_args(argv)
    if args.render is not None:
        spec = (args.stages or ["gtcrn"])[0]
        safe = "".join(c if c.isalnum() else "_" for c in spec)
        out = args.render_out or args.render.with_name(f"{args.render.stem}.{safe}.s{args.suppression:g}.wav")
        print(json.dumps(render(args.render, spec, out, args.suppression)))
        return 0
    if args.full:
        return _main_full(args)
    if args.fast:
        args.stages, args.set_name, args.wer = ["passthrough"], "tiny", "off"
    args.set_name = args.set_name or "tiny"
    args.wer = args.wer or "off"
    specs = args.stages or ["passthrough"]

    try:
        report = run(specs, args.set_name, args.wer, args.fixtures, args.out, args.limit)
    except FileNotFoundError as e:
        print(f"bench: {e}", file=sys.stderr)
        return 2
    dev = report["device"]
    print(
        f"set={args.set_name} fixtures={report['fixture_set']['count']} wer={args.wer} "
        f"device={dev['device']} machine={dev['hostname']} ({dev['machine']}) git={report['git_sha']}"
    )
    print(format_table(report["aggregates"]))
    if "path" in report:
        print(f"wrote {report['path']}")
    if report["failures"]:
        print(f"{len(report['failures'])} fixture run(s) failed:", file=sys.stderr)
        for f in report["failures"]:
            print(f"  stages={f['stages']!r} fixture={f['fixture']}: {f['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
