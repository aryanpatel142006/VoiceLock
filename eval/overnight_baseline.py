"""Frozen evidence collector and explicit historical measurement replay; no promotion.

Run on the existing allocation after bootstrap. Hashes are captured at collection
 time, not proof that a previous process loaded those bytes. Missing evidence is
 explicit. Collection is offline; --measure explicitly loads real GPU models.
Replay does not request allocations or open audio hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path


def reference_wer(reference: str, hypothesis: str, verified: bool = False) -> dict:
    """Word edit distance with case/punctuation normalization and explicit provenance.

    Agreement with ASR does not verify a reference. Callers may set verified only
    after an independent source/transcript audit.
    """
    ref = re.findall(r"\w+(?:'\w+)*", reference.lower())
    hyp = re.findall(r"\w+(?:'\w+)*", hypothesis.lower())
    if not ref:
        raise ValueError("empty reference cannot define WER")
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, predicted in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (word != predicted),
                )
            )
        previous = current
    return {
        "wer": previous[-1] / len(ref),
        "word_errors": previous[-1],
        "reference_words": len(ref),
        "reference_status": (
            "verified reference"
            if verified
            else "reference unverified (auto transcript)"
        ),
    }


def quality_status(data: dict) -> str:
    if data.get("fake") is not False or data.get("mode") != "real":
        return "rejected_mock"
    return "measured" if data.get("results") else "missing"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path, revision: str, results_dir: Path | None = None, explicit_verified: bool = False) -> dict:
    if not (root / "repo").is_dir():
        raise FileNotFoundError(root / "repo")
    files = {}
    for directory in ("repo", "models", "data/profiles", "data/team_baseline"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and not any(
                part in {".git", ".venv", "__pycache__", ".cache"}
                for part in path.parts
            ):
                files[str(path.relative_to(root))] = _sha(path)
    results_dir = results_dir or root / "results/t210-a4500-20261010T0315"
    artifacts = {}
    hashes = {}
    for name in ("readiness", "team", "public", "asr"):
        path = results_dir / f"{name}.json"
        if path.exists():
            artifacts[name] = json.loads(path.read_text())
            hashes[path.name] = _sha(path)
    asr = artifacts.get("asr")
    asr_status = (
        "missing"
        if asr is None
        else (
            "measured"
            if asr.get("mock") is False and asr.get("results")
            else "rejected_mock"
            if asr.get("mock") is not False
            else "missing"
        )
    )
    if asr_status == "measured":
        manifest_path = root / "data/public_overnight/take0_1320_sir0_snr15/manifest.json"
        is_verified = explicit_verified
        if manifest_path.exists():
            is_verified = is_verified or json.loads(manifest_path.read_text()).get("reference_verified", False)
        
        actual_status = "verified reference" if is_verified else "reference unverified (auto transcript)"
        asr["reference_status"] = actual_status
        asr["scores"] = [
            reference_wer(
                asr["reference"],
                row["text"],
                verified=is_verified,
            )
            for row in asr["results"]
        ]
    readiness = artifacts.get("readiness", {})
    result_path = results_dir / "team.json"
    if not result_path.exists():
        result_path = root / "results/team-baseline.json"
    team = json.loads(result_path.read_text()) if result_path.exists() else None
    cuda_path = root / "results/cuda-smoke.json"
    return {
        "collected_at": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "source_revision_verified": False,
        "hash_scope": "collection-time bytes; historical loaded identity unverified",
        "files": files,
        "team_result_sha256": _sha(result_path) if team is not None else None,
        "team_status": quality_status(team) if team is not None else "missing",
        "team": team,
        "cuda": json.loads(cuda_path.read_text()) if cuda_path.exists() else None,
        "artifact_sha256": hashes,
        "asr": asr,
        "public": artifacts.get("public"),
        "readiness": readiness,
        "asr_status": asr_status,
        "public_status": quality_status(artifacts["public"])
        if "public" in artifacts
        else "missing",
        "warmup_status": "measured"
        if readiness.get("cuda") is True
        and "first_window_warmup_s" in readiness
        and "load_s" in readiness
        else "missing",
        "comparison_status": "missing",
        "transcript_reference_status": asr.get("reference_status", "auto-unverified") if asr else "auto-unverified",
        "adoption": False,
    }


def measure(root: Path, out: Path, verified_reference: bool = False) -> None:
    """Replay original real-model checkpoint; caller must enforce allocation/cap.

    Historical script SHA256: 652b70fb35398cc4f6720a2fc24ad52b16024648fbb3869b748a4d6e763462e0.
    No fallback to mocks. Exclusive output directory preserves checkpoints.
    """
    import json
    import time
    from types import SimpleNamespace

    import torch

    from eval import isolation_diagnostic as d
    from hearing.audio.io import load
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enrolled import EnrolledCandidate

    out.mkdir(exist_ok=False)

    def save(name, data):
        (out / (name + ".json")).write_text(json.dumps(data, indent=2, default=float))

    sep = BlindSeparator()
    cand = EnrolledCandidate(separator=sep)
    t0 = time.perf_counter()
    cand.load()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    pt, _ = load(root / "data/public_overnight/public_profiles/1320.wav", 16000)
    po, _ = load(root / "data/public_overnight/public_profiles/6930.wav", 16000)
    x, _ = load(root / "data/public_overnight/take0_1320_sir0_snr15/mix.wav", 16000)
    r = d.EnrolledRunner(pt, po, separator=sep, embedder=cand.embedder)
    t0 = time.perf_counter()
    r.run(x[:32000], 0)
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0
    save(
        "readiness",
        {
            "device": torch.cuda.get_device_name(),
            "cuda": torch.cuda.is_available(),
            "load_s": load_s,
            "first_window_warmup_s": warm_s,
            "torch": torch.__version__,
        },
    )
    args = SimpleNamespace(
        window_s=2, hop_s=1, sep_min_db=4, drop_db=3, agc=False, split="heldout"
    )
    for label, scene, a, b, ratios, noise in [
        ("team", root / "data/team_baseline", None, None, [0, -5, -10], False),
        (
            "public",
            root / "data/public_overnight/take0_1320_sir0_snr15",
            pt,
            po,
            [0],
            True,
        ),
    ]:
        if label == "team":
            a, _ = load(
                min((root / "data/profiles/joseph").glob("profile-*.wav")), 16000
            )
            b, _ = load(
                min((root / "data/profiles/aryan").glob("profile-*.wav")), 16000
            )
        t, _ = load(scene / "target.wav", 16000)
        o, _ = load(
            scene / ("other.wav" if label == "team" else "interferer.wav"), 16000
        )
        n, _ = load(scene / "noise.wav", 16000)

        def make(role, tt, oo, a=a, b=b):
            aa, bb = (a, b) if role == "A" else (b, a)
            return d.EnrolledRunner(aa, bb, separator=sep, embedder=cand.embedder)

        results = d.run_conditions(
            make, t, o, ratios, ["A", "B"], args, n if noise else None
        )
        save(
            label,
            {
                "mode": "real",
                "fake": False,
                "results": results,
                "device": "cuda",
                "config": vars(args),
                "ratios": ratios,
                "noise": noise,
            },
        )
    from faster_whisper import WhisperModel

    start = time.perf_counter()
    model = WhisperModel(
        "base.en",
        device="cuda",
        compute_type="float16",
        download_root=str(root / "models/whisper"),
    )
    load_asr = time.perf_counter() - start
    start = time.perf_counter()
    list(model.transcribe(x[:32000], language="en")[0])
    warm_asr = time.perf_counter() - start
    rows = []
    for kind in ["target", "mix"]:
        audio, _ = load(
            root / f"data/public_overnight/take0_1320_sir0_snr15/{kind}.wav", 16000
        )
        start = time.perf_counter()
        segments, _ = model.transcribe(audio, language="en", beam_size=5)
        text = " ".join(s.text for s in segments).strip()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "input": kind,
                "text": text,
                "seconds": elapsed,
                "audio_s": len(audio) / 16000,
                "rtf": elapsed / (len(audio) / 16000),
            }
        )
    
    manifest_path = root / "data/public_overnight/take0_1320_sir0_snr15/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    is_verified = verified_reference or manifest.get("reference_verified", False)
    
    save(
        "asr",
        {
            "model": "base.en",
            "mock": False,
            "load_s": load_asr,
            "warmup_s": warm_asr,
            "results": rows,
            "reference_status": "verified reference" if is_verified else "reference unverified (auto transcript)",
            "reference": manifest.get("transcript", ""),
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--verified-reference", action="store_true")
    args = parser.parse_args(argv)
    if args.measure:
        measure(args.root.resolve(), args.out.resolve(), verified_reference=args.verified_reference)
        return 0
    report = collect(args.root.resolve(), args.source_revision, args.results_dir, explicit_verified=args.verified_reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing a frozen checkpoint.
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in {"files", "team"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
