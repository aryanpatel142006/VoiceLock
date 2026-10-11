"""Eval runner: every candidate renders audio from the SAME fixed mixtures and is scored per mixture.

Rendering reuses ``eval.bench`` (registry pipelines streamed in 10 ms chunks, latency-compensated)
and ``eval.overnight_output_ab.compute_metrics`` (dropouts, interferer projection). Each
(candidate, mixture) row is cached as JSON next to its output WAV, so a resumed run re-renders
nothing it already has. Outputs go under the run directory (``$HEARING_RESULTS_DIR/arena/...``),
never into git.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from eval import bench, metrics
from eval.overnight_output_ab import compute_metrics
from hearing.types import StageContext

SR = 16000
BLOCK = 1600  # 100 ms attribution blocks
CHUNK_BUDGET_MS = 10.0  # a 10 ms hop that takes longer than 10 ms to compute is a real-time overrun


@dataclass(frozen=True)
class Candidate:
    """What is evaluated. ``spec`` is an ``eval.bench`` pipeline string; ``checkpoint`` names weights."""

    id: str
    spec: str
    kind: str = "extraction"  # or "distillation" (rungs counted in speakers, see scheduler)
    model: str = ""
    checkpoint: str = ""
    preprocessing: str = "16k mono float32, 10 ms chunks, latency-compensated (eval.bench.run_stream)"
    seed: int = 0
    suppression: float = 1.0

    def key(self) -> str:
        """Content key for the row cache: a changed spec/checkpoint never reuses old rows."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return f"{self.id}-{hashlib.sha256(blob).hexdigest()[:10]}"

    def record(self) -> dict:
        out = asdict(self)
        out["model"] = self.model or self.spec
        out["checkpoint_sha256"] = _sha_file(Path(self.checkpoint)) if self.checkpoint else None
        return out


@dataclass(frozen=True)
class Mixture:
    id: str
    dir: Path
    cluster: str  # shared-source group for the paired bootstrap
    sha256: str
    transcript: str
    transcript_verified: bool


@dataclass
class Scorers:
    dnsmos: object | None = None
    transcriber: object | None = None
    notes: dict = field(default_factory=dict)


def _sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_mixtures(set_root: Path) -> list[Mixture]:
    """Fixed evaluation mixtures of an ``eval.make_fixtures`` set, sorted by id, with content hashes."""
    index = json.loads((set_root / "index.json").read_text())
    out = []
    for entry in index["fixtures"]:
        d = set_root / entry["id"]
        man = json.loads((d / "manifest.json").read_text())
        source = man.get("source", {})
        # LibriSpeech test-clean references are the dataset's own verified transcripts.
        verified = bool(man.get("transcript_verified", source.get("dataset") == "openslr/librispeech_asr"))
        out.append(Mixture(
            id=man["id"], dir=d,
            cluster=f"{man.get('target_id', '')}|{man.get('interferer_id', '')}",
            sha256=_sha_file(d / man["paths"]["mix"]) or "",
            transcript=man.get("transcript", ""), transcript_verified=verified and bool(man.get("transcript")),
        ))
    return out


def attribution(out: np.ndarray, target: np.ndarray, interferer: np.ndarray, mix: np.ndarray) -> dict:
    """Per 100 ms block: which talker dominates the output, target misses and target-absent leak.

    Present and absent are scored jointly: a block where the target talks must keep it (>-10 dB
    projection gain), a block where it is silent must not pass the mix (``absent_leak_db``).
    """
    labels, misses, present = [], 0, 0
    absent_out = absent_mix = 0.0
    t_ref = float(np.sqrt(np.mean(target.astype(np.float64) ** 2))) + 1e-12
    for start in range(0, len(out), BLOCK):
        sl = slice(start, start + BLOCK)
        t, i, y = (np.asarray(a[sl], np.float64) for a in (target, interferer, out))
        t_rms = float(np.sqrt(np.mean(t**2)))
        if t_rms <= max(1e-4, 0.0316 * t_ref):  # target absent (< -30 dB re its own RMS)
            absent_out += float(np.sum(y**2))
            absent_mix += float(np.sum(np.asarray(mix[sl], np.float64) ** 2))
            continue
        present += 1
        basis = np.stack([t, i], axis=1)
        if np.linalg.matrix_rank(basis) < 2:  # no competing talker in this block
            a_t = float(np.linalg.lstsq(t[:, None], y, rcond=None)[0][0])
            misses += 20 * np.log10(max(abs(a_t), 1e-6)) < -10
            continue
        a_t, b_i = np.linalg.lstsq(basis, y, rcond=None)[0]
        misses += 20 * np.log10(max(abs(a_t), 1e-6)) < -10
        if np.mean(y**2) > 1e-10:
            labels.append(bool(np.linalg.norm(a_t * t) >= np.linalg.norm(b_i * i)))
    wrong = sum(not x for x in labels)
    switches = sum(a != b for a, b in itertools.pairwise(labels))
    minutes = len(out) / SR / 60.0
    return {
        "wrong_speaker_frac": wrong / len(labels) if labels else None,
        "switches_per_min": switches / minutes if labels and minutes > 0 else None,
        "present_miss_frac": misses / present if present else None,
        "absent_leak_db": (10 * np.log10(max(absent_out, 1e-12) / max(absent_mix, 1e-12))
                           if absent_mix > 0 else None),
    }


def score_mixture(y: np.ndarray, stems, res: bench.StreamResult, wall_s: float, mixture: Mixture,
                  scorers: Scorers) -> dict:
    row: dict = {"mixture_id": mixture.id}
    row["si_sdr"] = metrics.si_sdr(y, stems.target)
    row["si_sdri"] = metrics.si_sdri(y, stems.mix, stems.target)
    row["stoi"] = metrics.stoi(y, stems.target, SR)
    row["estoi"] = metrics.stoi(y, stems.target, SR, extended=True)
    row.update(attribution(y, stems.target, stems.interferer, stems.mix))
    ab = compute_metrics(y, y, stems.mix, stems.target, stems.interferer, SR)
    row["target_dropout_s"] = ab["true_dropout_s"]
    row["interferer_projection_db"] = ab["interferer_projection_db"]
    if scorers.dnsmos is not None:
        s = scorers.dnsmos.score(y, SR)
        row.update({"dnsmos_sig": s.sig, "dnsmos_bak": s.bak, "dnsmos_ovrl": s.ovrl})
    row["wer"] = None
    if scorers.transcriber is not None and mixture.transcript_verified:
        if getattr(scorers.transcriber, "name", "") == "mock":
            scorers.transcriber.transcript = mixture.transcript
        row["wer"] = metrics.wer(scorers.transcriber.transcribe(y, SR), mixture.transcript)
    times = np.asarray(res.compute_ms or [0.0])
    row["algo_latency_ms"] = res.latency_ms  # lookahead the pipeline declares (checked by bench's lag err)
    row["compute_p50_ms"] = float(np.percentile(times, 50))
    row["compute_p95_ms"] = float(np.percentile(times, 95))
    row["overrun_frac"] = float(np.mean(times > CHUNK_BUDGET_MS))
    row["rtf"] = metrics.rtf(float(times.sum()) / 1000.0, res.audio_s)
    row["audio_s"] = len(y) / SR
    row["compute_s"] = wall_s
    return row


def evaluate(candidate: Candidate, mixtures: list[Mixture], run_dir: Path, scorers: Scorers) -> list[dict]:
    """Rows for ``candidate`` on ``mixtures`` (in that order), rendering only the ones not cached."""
    rows_dir = run_dir / "rows" / candidate.key()
    audio_dir = run_dir / "audio" / candidate.id
    rows_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    pipeline = None
    rows = []
    for mix in mixtures:
        cached = rows_dir / f"{mix.id}.json"
        if cached.exists():
            row = json.loads(cached.read_text())
            if row.get("mixture_sha256") == mix.sha256:
                row["cached"] = True
                rows.append(row)
                continue
        if pipeline is None:
            pipeline = bench.build(candidate.spec)
        manifest, stems = bench.load_any_fixture(mix.dir)
        ctx = StageContext.default()
        ctx.suppression = candidate.suppression
        np.random.seed(candidate.seed)  # stochastic stages (dither, dropout-at-eval) see a fixed seed
        t0 = time.perf_counter()
        res = bench.run_stream(pipeline, stems.mix, manifest.sample_rate, ctx)
        wall = time.perf_counter() - t0
        y = np.clip(res.output, -1.0, 1.0).astype(np.float32)
        out_path = audio_dir / f"{mix.id}.wav"
        sf.write(out_path, y, SR, subtype="FLOAT")
        row = score_mixture(y, stems, res, wall, mix, scorers)
        row.update({"mixture_sha256": mix.sha256, "cluster": mix.cluster,
                    "output": str(out_path.relative_to(run_dir)), "output_sha256": _sha_file(out_path)})
        row = bench._json_safe(row)
        tmp = cached.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, sort_keys=True))
        tmp.replace(cached)
        row["cached"] = False
        rows.append(row)
    return rows


def environment() -> dict:
    from hearing.device import pick

    return {"host": platform.node(), "machine": platform.machine(), "cpu": bench.cpu_name(),
            "device": str(pick()), "git_rev": bench.git_sha()}
