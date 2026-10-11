"""Overnight speech shard: stream a bounded prefix of official LibriSpeech dev-clean.

Source: https://www.openslr.org/resources/12/dev-clean.tar.gz (CC BY 4.0, https://www.openslr.org/12/).
The tarball is read as a gzip stream and the HTTP response is closed the moment the byte budget
is reached, so no full 337 MB download happens. Tar members are ordered by speaker, so a prefix
yields complete speakers. Per speaker we keep ``PER_SPEAKER`` utterances split into disjoint
enrollment / development / holdout roles (whole utterances, never crops of one another).

Native FLAC bytes are written verbatim (with SHA256); ``derived_16k_mono`` WAVs are only written
when the native stream already is 16 kHz mono (no resampling here, otherwise the clip is rejected).
Media and absolute paths stay under ``$HEARING_FIXTURES_DIR/overnight_grunt/speech``; the public
manifest (relative paths, no media) is ``eval/manifests/overnight_grunt/speech.json``.

    .venv/bin/python -m eval.overnight_speech_collect --speakers 12
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import ssl
import tarfile
import time
import urllib.request
from pathlib import Path

import soundfile as sf

from eval.metrics import normalize_text

URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LICENCE = "CC BY 4.0"
LICENCE_URL = "https://www.openslr.org/12/"
CITATION = "Panayotov et al., LibriSpeech: an ASR corpus based on public domain audio books, ICASSP 2015"
BUDGET_BYTES = 256 * 1024 * 1024
DEFAULT_BYTES = 200 * 1024 * 1024
PER_SPEAKER = 8
MAX_UTTERANCES = 96
MIN_S, MAX_S = 2.0, 15.0
ROLES = ("enrollment",) * 3 + ("development",) * 2 + ("holdout",) * 3
REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "eval" / "manifests" / "overnight_grunt" / "speech.json"


def _ssl_context() -> ssl.SSLContext:
    """Default context; python.org builds lack a CA bundle, so use certifi's when it is installed."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class BudgetExceeded(Exception):
    """Raised by :class:`CountingReader` when the next read would pass the byte cap."""


class CountingReader:
    """File-like wrapper counting bytes pulled from the network; refuses to pass ``limit``."""

    def __init__(self, raw, limit: int):
        self.raw, self.limit, self.n = raw, limit, 0

    def read(self, size: int = -1) -> bytes:
        if self.n >= self.limit:
            raise BudgetExceeded(self.n)
        size = min(size if size > 0 else 1 << 16, self.limit - self.n)
        data = self.raw.read(size)
        self.n += len(data)
        return data


def pick_roles(utts: list[dict], per_speaker: int = PER_SPEAKER) -> list[dict]:
    """Choose ``per_speaker`` utterances round-robin over chapters, then assign disjoint roles."""
    by_chapter: dict[str, list[dict]] = {}
    for u in sorted(utts, key=lambda u: u["id"]):
        by_chapter.setdefault(u["chapter"], []).append(u)
    chosen: list[dict] = []
    queues = [list(v) for _, v in sorted(by_chapter.items())]
    while len(chosen) < per_speaker and any(queues):
        for q in queues:
            if q and len(chosen) < per_speaker:
                chosen.append(q.pop(0))
    chosen.sort(key=lambda u: (u["chapter"], u["id"]))
    # spread roles across chapters (position i -> role by interleave) so roles do not share a recording block
    order = [0, 3, 6, 1, 4, 7, 2, 5][: len(chosen)]
    for rank, idx in enumerate(order):
        chosen[idx]["role"] = ROLES[rank] if rank < len(ROLES) else "holdout"
    return chosen


def parse_trans(raw: bytes) -> dict[str, str]:
    out = {}
    for line in raw.decode("utf-8").splitlines():
        uid, _, text = line.partition(" ")
        if uid and text.strip():
            out[uid] = text.strip()
    return out


def stream_speakers(fileobj, n_speakers: int):
    """Yield ``(speaker_id, {chapter: {"flac": {uid: bytes}, "trans": {uid: text}}})`` per complete speaker.

    ``fileobj`` is the (counting) gzip byte stream. Stops after ``n_speakers`` complete speakers
    or when the stream ends/budget raises (the last, possibly cut, speaker is dropped).
    """
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
        cur, data, sex = None, {}, {}
        done = 0
        for m in tf:
            parts = Path(m.name).parts
            if not m.isfile() or len(parts) < 4:
                if m.isfile() and m.name.endswith("SPEAKERS.TXT"):
                    for ln in tf.extractfile(m).read().decode().splitlines():
                        f = [x.strip() for x in ln.split("|")]
                        if len(f) >= 2 and f[0].isdigit():
                            sex[f[0]] = f[1]
                continue
            spk, chap = parts[2], parts[3]
            if spk != cur:
                if cur is not None:
                    yield cur, data, sex.get(cur, "")
                    done += 1
                    if done >= n_speakers:
                        return
                cur, data = spk, {}
            ch = data.setdefault(chap, {"flac": {}, "trans": {}})
            raw = tf.extractfile(m).read()
            if m.name.endswith(".flac"):
                ch["flac"][Path(m.name).stem] = raw
            elif m.name.endswith(".trans.txt"):
                ch["trans"].update(parse_trans(raw))


def clip_info(raw: bytes) -> dict:
    info = sf.info(io.BytesIO(raw))
    return {"sample_rate": info.samplerate, "channels": info.channels, "duration_s": round(info.duration, 3),
            "frames": info.frames}


def candidates(chapters: dict) -> list[dict]:
    out = []
    for chap, d in chapters.items():
        for uid, raw in d["flac"].items():
            text = d["trans"].get(uid, "")
            info = clip_info(raw)
            if not text or not MIN_S <= info["duration_s"] <= MAX_S:
                continue
            out.append({"id": uid, "chapter": chap, "text": text, "raw": raw, **info})
    return out


def collect(out_dir: Path, speakers: int, byte_limit: int) -> dict:
    byte_limit = min(byte_limit, BUDGET_BYTES)
    speakers = min(speakers, MAX_UTTERANCES // PER_SPEAKER)
    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "selective-hearing-eval/1.0"})
    t0 = time.time()
    entries, failed, aborted = [], [], False
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        reader = CountingReader(resp, byte_limit)
        try:
            for spk, chapters, sex in stream_speakers(reader, speakers):
                picked = pick_roles(candidates(chapters))
                if len(picked) < PER_SPEAKER:
                    failed.append({"speaker": spk, "reason": f"only {len(picked)} usable utterances"})
                    continue
                for u in picked:
                    entries.append(_write_clip(out_dir, spk, sex, u))
        except (BudgetExceeded, EOFError, tarfile.ReadError) as e:
            aborted = True
            failed.append({"reason": f"stream stopped at budget/EOF: {type(e).__name__}"})
    return {
        "source": {"url": URL, "licence": LICENCE, "licence_url": LICENCE_URL, "citation": CITATION,
                   "content_length_bytes": total},
        "network_bytes": reader.n, "byte_limit": byte_limit, "aborted_at_budget": aborted,
        "elapsed_s": round(time.time() - t0, 1), "utterances": entries, "failed": failed,
    }


def _write_clip(out_dir: Path, spk: str, sex: str, u: dict) -> dict:
    nat = out_dir / "native" / f"{u['id']}.flac"
    nat.parent.mkdir(parents=True, exist_ok=True)
    nat.write_bytes(u["raw"])
    entry = {
        "id": u["id"], "speaker_id": spk, "sex": sex, "chapter": u["chapter"], "role": u["role"],
        "transcript": u["text"], "transcript_normalized": normalize_text(u["text"]),
        "native": {"path": f"native/{u['id']}.flac", "sha256": hashlib.sha256(u["raw"]).hexdigest(),
                   "bytes": len(u["raw"]), "sample_rate": u["sample_rate"], "channels": u["channels"],
                   "duration_s": u["duration_s"]},
        "derived_16k_mono": None,
    }
    if u["sample_rate"] == 16000 and u["channels"] == 1:  # no resampling: only a re-encode to WAV
        x, _ = sf.read(io.BytesIO(u["raw"]), dtype="float32")
        d = out_dir / "derived_16k_mono" / f"{u['id']}.wav"
        d.parent.mkdir(parents=True, exist_ok=True)
        sf.write(d, x, 16000, subtype="PCM_16")
        entry["derived_16k_mono"] = {"path": f"derived_16k_mono/{u['id']}.wav",
                                     "sha256": hashlib.sha256(d.read_bytes()).hexdigest(),
                                     "method": "FLAC->PCM16 WAV, no resampling (native already 16 kHz mono)"}
    return entry


def validate(entries: list[dict]) -> list[str]:
    """Problems that make a shard unusable: role overlap, missing transcript, mixed rates, short speakers."""
    errs, seen = [], set()
    by_spk: dict[str, list[dict]] = {}
    for e in entries:
        if e["id"] in seen:
            errs.append(f"duplicate id {e['id']}")
        seen.add(e["id"])
        if not e["transcript_normalized"]:
            errs.append(f"empty transcript {e['id']}")
        by_spk.setdefault(e["speaker_id"], []).append(e)
    for spk, es in by_spk.items():
        if len(es) < PER_SPEAKER:
            errs.append(f"speaker {spk} has {len(es)} < {PER_SPEAKER} utterances")
        if {e["role"] for e in es} != {"enrollment", "development", "holdout"}:
            errs.append(f"speaker {spk} lacks one of the three roles")
    return errs


def public_manifest(report: dict, entries: list[dict]) -> dict:
    spk = sorted({e["speaker_id"] for e in entries})
    return {
        "shard": "speech", "task": "T-218", "source": report["source"],
        "media_location": "$HEARING_FIXTURES_DIR/overnight_grunt/speech (gitignored, not committed)",
        "counts": {"speakers": len(spk), "utterances": len(entries),
                   "native_bytes": sum(e["native"]["bytes"] for e in entries),
                   "total_duration_s": round(sum(e["native"]["duration_s"] for e in entries), 1),
                   "by_role": {r: sum(e["role"] == r for e in entries) for r in ("enrollment", "development", "holdout")}},
        "acquisition": {"network_bytes": report["network_bytes"], "byte_limit": report["byte_limit"],
                        "aborted_at_budget": report["aborted_at_budget"], "full_tarball_downloaded": False},
        "failed": report["failed"],
        "utterances": entries,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speakers", type=int, default=12)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_BYTES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    a = ap.parse_args(argv)
    fixtures = Path(os.environ["HEARING_FIXTURES_DIR"]) / "overnight_grunt" / "speech"
    results = Path(os.environ["HEARING_RESULTS_DIR"]) / "overnight_grunt" / "speech"
    report = collect(fixtures, a.speakers, a.max_bytes)
    entries = report["utterances"]
    errs = validate(entries)
    report["validation_errors"] = errs
    results.mkdir(parents=True, exist_ok=True)
    (results / "collect.json").write_text(json.dumps({**report, "fixtures_dir": str(fixtures)}, indent=1))
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(public_manifest(report, entries), indent=1) + "\n")
    print(json.dumps({"speakers": len({e["speaker_id"] for e in entries}), "utterances": len(entries),
                      "network_bytes": report["network_bytes"], "errors": errs, "failed": report["failed"]}))
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
