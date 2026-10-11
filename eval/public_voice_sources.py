"""Bounded, checksummed download + manifest kit for public speech/noise corpora (T-209).

Only permissively licensed sources with a direct, non-expiring URL are listed in
``SOURCES``. Archives land in ``$HEARING_FIXTURES_DIR/public_voice`` (gitignored, never
committed). Downloads are resumable, verified against the publisher's checksum and
refuse to exceed ``--cap-bytes`` (default 2 GB across the directory).

    python -m eval.public_voice_sources download [--ids ID ...] [--cap-bytes N]
    python -m eval.public_voice_sources manifest --out docs/research/public_voice_manifest.json

The LibriSpeech manifest keeps speaker IDs and transcripts, hashes every utterance with
sha256 and splits each speaker into disjoint enrollment / eval recordings. Nothing here
runs a model, so no quality claim follows from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path

CAP_BYTES = 2_000_000_000
DEFAULT_SPEAKERS = ("6930", "1320")
ENROLL_SECONDS = 20.0  # enrollment takes utterances (in id order) until this much audio
CHUNK = 1 << 20

SOURCES: dict[str, dict] = {
    "librispeech_test_clean": {
        "file": "test-clean.tar.gz",
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",
        "size": 346_663_984,
        "license": "CC BY 4.0",
        "license_url": "https://www.openslr.org/12/",
        "attribution": "Panayotov, Chen, Povey, Khudanpur. LibriSpeech: an ASR corpus based on "
        "public domain audio books. ICASSP 2015.",
    },
    "demand_pcafeter": {
        "file": "PCAFETER_16k.zip",
        "url": "https://zenodo.org/api/records/1227121/files/PCAFETER_16k.zip/content",
        "md5": "99927d148128254141a9417d051510bb",
        "size": 107_431_494,
        "license": "CC BY 4.0",
        "license_url": "https://zenodo.org/records/1227121",
        "attribution": "Thiemann, Ito, Vincent. DEMAND: a collection of multi-channel recordings "
        "of acoustic noise in diverse environments (2013). doi:10.5281/zenodo.1227121",
    },
    "demand_omeeting": {
        "file": "OMEETING_16k.zip",
        "url": "https://zenodo.org/api/records/1227121/files/OMEETING_16k.zip/content",
        "md5": "62f7cfe7fe6d30b7d8a215fe37c2dfd2",
        "size": 82_703_355,
        "license": "CC BY 4.0",
        "license_url": "https://zenodo.org/records/1227121",
        "attribution": "Thiemann, Ito, Vincent. DEMAND (2013). doi:10.5281/zenodo.1227121",
    },
}


def _ssl_context() -> ssl.SSLContext:
    try:  # python.org builds on macOS ship no CA bundle; certifi is a core dependency of requests
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def public_root() -> Path:
    base = os.environ.get("HEARING_FIXTURES_DIR") or "eval/fixtures/generated"
    return Path(base) / "public_voice"


def hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def dir_bytes(root: Path) -> int:
    return (
        sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if root.exists()
        else 0
    )


def download(src: dict, root: Path, cap_bytes: int = CAP_BYTES) -> dict:
    """Fetch ``src`` into ``root`` (resumable), verify md5, enforce the directory cap."""
    root.mkdir(parents=True, exist_ok=True)
    dest = root / src["file"]
    have = dest.stat().st_size if dest.exists() else 0
    if have > src["size"]:
        dest.unlink()
        have = 0
    projected = dir_bytes(root) - have + src["size"]
    if projected > cap_bytes:
        raise RuntimeError(
            f"cap: {src['file']} would bring {root} to {projected} > {cap_bytes}"
        )
    resumed_from = have
    if have < src["size"]:
        req = urllib.request.Request(src["url"], headers={"Range": f"bytes={have}-"})
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
            if have and r.status != 206:  # server ignored Range: restart cleanly
                have = 0
            with open(dest, "ab" if have else "wb") as f:
                while block := r.read(CHUNK):
                    f.write(block)
    got = hash_file(dest, "md5")
    if got != src["md5"]:
        raise RuntimeError(f"md5 mismatch for {dest.name}: {got} != {src['md5']}")
    return {
        "file": src["file"],
        "url": src["url"],
        "bytes": dest.stat().st_size,
        "md5": got,
        "sha256": hash_file(dest, "sha256"),
        "resumed_from": resumed_from,
        "license": src["license"],
    }


def split_enrollment(
    utts: list[dict], enroll_seconds: float = ENROLL_SECONDS
) -> list[dict]:
    """Tag each utterance ``enroll`` or ``eval``: first utterances (id order) until the
    seconds budget is met enroll, the rest evaluate. Always leaves >=1 eval utterance."""
    out, acc = [], 0.0
    ordered = sorted(utts, key=lambda u: u["utt_id"])
    for i, u in enumerate(ordered):
        last = i == len(ordered) - 1
        role = "enroll" if acc < enroll_seconds and not last else "eval"
        if role == "enroll":
            acc += u["seconds"]
        out.append({**u, "role": role})
    return out


def check_disjoint(entries: list[dict]) -> None:
    for spk in {e["speaker"] for e in entries}:
        roles: dict[str, set] = {"enroll": set(), "eval": set()}
        for e in entries:
            if e["speaker"] == spk:
                roles[e["role"]].add(e["sha256"])
        if roles["enroll"] & roles["eval"]:
            raise ValueError(f"speaker {spk}: enrollment and eval recordings overlap")
        if not roles["eval"]:
            raise ValueError(f"speaker {spk}: no eval recordings")


_UTT = re.compile(r"(?:^|/)(\d+)/(\d+)/(\d+)-(\d+)-(\d+)\.(flac|trans\.txt)$")


def librispeech_entries(tar_path: Path, speakers=DEFAULT_SPEAKERS) -> list[dict]:
    """Stream the tarball, keep only ``speakers``: sha256 of each flac, transcript, seconds."""
    import io

    import soundfile as sf

    want = set(speakers)
    trans: dict[str, str] = {}
    flacs: dict[str, tuple[str, str, float]] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            parts = name.split("/")
            if len(parts) < 3 or parts[-3] not in want:
                continue
            data = tf.extractfile(m).read()
            if name.endswith(".trans.txt"):
                for line in data.decode().splitlines():
                    utt, _, text = line.partition(" ")
                    trans[utt] = text
            elif name.endswith(".flac"):
                utt = Path(name).stem
                info = sf.info(io.BytesIO(data))
                flacs[utt] = (
                    parts[-3],
                    hashlib.sha256(data).hexdigest(),
                    info.duration,
                )
    per_speaker: dict[str, list[dict]] = {s: [] for s in speakers}
    for utt, (spk, digest, secs) in sorted(flacs.items()):
        per_speaker[spk].append(
            {
                "speaker": spk,
                "utt_id": utt,
                "path_in_archive": f"LibriSpeech/test-clean/{spk}/{utt.split('-')[1]}/{utt}.flac",
                "sha256": digest,
                "seconds": round(secs, 3),
                "transcript": trans.get(utt, ""),
            }
        )
    entries = [e for s in speakers for e in split_enrollment(per_speaker[s])]
    check_disjoint(entries)
    return entries


def zip_listing(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        return [
            {"name": i.filename, "bytes": i.file_size}
            for i in z.infolist()
            if not i.is_dir()
        ]


def build_manifest(root: Path, speakers=DEFAULT_SPEAKERS) -> dict:
    manifest: dict = {"schema": 1, "speakers": list(speakers), "sources": {}}
    for sid, src in SOURCES.items():
        p = root / src["file"]
        rec = {
            k: src[k]
            for k in ("url", "md5", "size", "license", "license_url", "attribution")
        }
        rec["downloaded"] = p.exists() and p.stat().st_size == src["size"]
        manifest["sources"][sid] = rec
    tar = root / SOURCES["librispeech_test_clean"]["file"]
    if manifest["sources"]["librispeech_test_clean"]["downloaded"]:
        ents = librispeech_entries(tar, speakers)
        manifest["librispeech_test_clean"] = ents
    for sid in ("demand_pcafeter", "demand_omeeting"):
        if manifest["sources"][sid]["downloaded"]:
            manifest[sid + "_files"] = zip_listing(root / SOURCES[sid]["file"])
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--ids", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    d.add_argument("--cap-bytes", type=int, default=CAP_BYTES)
    m = sub.add_parser("manifest")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--speakers", nargs="*", default=list(DEFAULT_SPEAKERS))
    a = ap.parse_args(argv)
    root = public_root()
    if a.cmd == "download":
        ev = [download(SOURCES[i], root, a.cap_bytes) for i in a.ids]
        print(
            json.dumps({"directory_bytes": dir_bytes(root), "downloads": ev}, indent=2)
        )
    else:
        man = build_manifest(root, a.speakers)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        n = len(man.get("librispeech_test_clean", []))
        print(f"wrote {a.out}: {n} librispeech utterances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
