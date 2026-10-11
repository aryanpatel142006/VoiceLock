"""Collect a small, licence-recorded noise and room-impulse-response bank (T-219).

Sources (official primary hosts only; everything else stays discovery-only):
  * ESC-50 on GitHub (karolpiczak/ESC-50): per-clip origin and licence from its LICENSE file.
  * OpenSLR SLR28 ``rirs_noises.zip`` (declared Apache 2.0): members are fetched with HTTP range
    requests, so the 1.3 GB archive is never downloaded.
  * Wikimedia Commons: crowd/cafe ambience with per-file licence from the API's ``extmetadata``.

Media goes to ``$HEARING_FIXTURES_DIR/overnight_grunt/noise`` (native + 16 kHz mono derivative);
only the metadata manifest is committed.  No reverb is applied and nothing is transcribed: files
that may contain speech are marked ``speech_content: "unknown_unlabeled"`` and are never a clean
target.  A missing field stays ``None`` / "unknown".
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ESC50_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
OPENSLR_ZIP = "https://www.openslr.org/resources/28/rirs_noises.zip"
OPENSLR_PAGE = "https://www.openslr.org/28/"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
ESC50_LICENSE_URL = f"{ESC50_RAW}/LICENSE"

BUDGET_BYTES = 128 * 1024 * 1024
MAX_DECODED_NOISE_S = 20 * 60
ESC_CATEGORIES = {
    "keyboard_typing": "keyboard",
    "vacuum_cleaner": "ventilation_appliance",
    "washing_machine": "ventilation_appliance",
    "engine": "traffic_vehicle",
    "car_horn": "traffic_vehicle",
    "train": "traffic_vehicle",
    "airplane": "traffic_vehicle",
    "helicopter": "traffic_vehicle",
    "rain": "weather",
    "wind": "weather",
    "mouse_click": "keyboard",
}
CLIPS_PER_ESC_CATEGORY = 3
COMMONS_QUERIES = ("cafe ambience", "restaurant crowd", "shopping mall crowd", "crowd murmur")
COMMONS_MAX_FILES = 4
COMMONS_MAX_BYTES = 3 * 1024 * 1024
ACCEPTED_COMMONS_LICENCES = ("cc0", "cc by", "cc-by", "public domain", "pd")
RIR_MEMBERS = (
    "air_type1_air_phone_kitchen_hhp.wav",
    "air_type1_air_phone_corridor_hhp.wav",
    "air_type1_air_phone_bathroom_hhp.wav",
    "air_type1_air_phone_office_hhp.wav",
    "air_type1_air_phone_meeting_hhp.wav",
    "air_type1_air_phone_lecture_hhp.wav",
    "air_type1_air_phone_stairway_hhp.wav",
    "air_type1_air_binaural_booth_0_1.wav",
    "RVB2014_type1_rir_smallroom1_near_angla.wav",
    "RVB2014_type1_rir_mediumroom1_near_angla.wav",
    "RVB2014_type1_rir_largeroom1_near_angla.wav",
)
RIR_DIR = "RIRS_NOISES/real_rirs_isotropic_noises/"


class Budget:
    """Counts every network byte so the shard cannot exceed the acquisition budget."""

    def __init__(self, limit: int = BUDGET_BYTES) -> None:
        self.limit = limit
        self.used = 0

    def add(self, n: int) -> None:
        self.used += n
        if self.used > self.limit:
            raise RuntimeError(f"download budget exceeded: {self.used} > {self.limit} bytes")


def _ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_get(url: str, budget: Budget, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "selective-hearing-T219", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60, context=_ctx()) as resp:
        data = resp.read()
    budget.add(len(data))
    return data


class RangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file via HTTP range requests (for zipfile)."""

    def __init__(self, url: str, budget: Budget) -> None:
        self.url, self.budget, self.pos = url, budget, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "selective-hearing-T219"})
        with urllib.request.urlopen(req, timeout=60, context=_ctx()) as resp:
            self.size = int(resp.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf: Any) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        data = http_get(self.url, self.budget, {"Range": f"bytes={self.pos}-{end}"})
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_esc50_license(text: str) -> dict[str, dict[str, str | None]]:
    """Map ``<clip id like 1-100032-A>`` to origin file, source URL, author and licence tag."""
    out: dict[str, dict[str, str | None]] = {}
    pat = re.compile(r"^- \[(?P<id>[\w-]+)\.ogg\]: clip derived from (?P<orig>.*?) \((?P<url>https?://[^)]*)\) by (?P<author>.*?) \[(?P<lic>[^\]]+)\]")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            out[m["id"]] = {"original_file": m["orig"], "source_url": m["url"], "author": m["author"], "licence_tag": m["lic"]}
    return out


def parse_esc50_meta(csv_text: str) -> list[dict[str, str]]:
    lines = csv_text.strip().splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","), strict=False)) for ln in lines[1:]]


def select_esc50(rows: list[dict[str, str]], licences: dict[str, dict[str, str | None]]) -> list[dict[str, str]]:
    """First clips (by filename) per wanted category that have per-clip licence info."""
    chosen: list[dict[str, str]] = []
    for cat in ESC_CATEGORIES:
        got = 0
        for row in sorted((r for r in rows if r["category"] == cat), key=lambda r: r["filename"]):
            clip_id = row["filename"].rsplit("-", 1)[0]
            if clip_id in licences and got < CLIPS_PER_ESC_CATEGORY:
                chosen.append(row)
                got += 1
    return chosen


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    """Downmix to mono float32 and resample to 16 kHz with the project resampler."""
    from hearing.audio.io import resample

    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample(x, sr, 16000).astype(np.float32) if sr != 16000 else x


def commons_licence_ok(short_name: str | None) -> bool:
    return bool(short_name) and any(short_name.lower().startswith(p) for p in ACCEPTED_COMMONS_LICENCES)  # type: ignore[union-attr]


def strip_html(s: str | None) -> str | None:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", "", s)).strip() if s else None


def _write_asset(root: Path, rel: str, data: bytes) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _audio_entry(root: Path, rel_native: str, data: bytes, derived_dir: str) -> dict[str, Any]:
    import soundfile as sf

    native = _write_asset(root, rel_native, data)
    x, sr = sf.read(native, dtype="float32", always_2d=False)
    info = sf.info(native)
    y = to_16k_mono(x, sr)
    rel_derived = f"{derived_dir}/{Path(rel_native).stem}.wav"
    (root / rel_derived).parent.mkdir(parents=True, exist_ok=True)
    sf.write(root / rel_derived, y, 16000, subtype="PCM_16")
    return {
        "native_path": rel_native,
        "native_sha256": sha256_bytes(data),
        "native_bytes": len(data),
        "native_sample_rate": sr,
        "native_channels": info.channels,
        "duration_s": round(len(x) / sr, 4),
        "derived_16k_mono_path": rel_derived,
        "derived_sha256": sha256_bytes((root / rel_derived).read_bytes()),
    }


def collect_esc50(root: Path, budget: Budget, missing: list[dict[str, str]]) -> list[dict[str, Any]]:
    lic_text = http_get(ESC50_LICENSE_URL, budget).decode()
    licences = parse_esc50_license(lic_text)
    rows = parse_esc50_meta(http_get(f"{ESC50_RAW}/meta/esc50.csv", budget).decode())
    out: list[dict[str, Any]] = []
    for row in select_esc50(rows, licences):
        name = row["filename"]
        try:
            data = http_get(f"{ESC50_RAW}/audio/{urllib.parse.quote(name)}", budget)
        except urllib.error.URLError as exc:
            missing.append({"id": name, "reason": str(exc)})
            continue
        lic = licences[name.rsplit("-", 1)[0]]
        entry = _audio_entry(root, f"native/noise/esc50/{name}", data, "derived_16k_mono/noise/esc50")
        out.append(
            {
                "id": f"esc50:{name}",
                "kind": "noise",
                "category": ESC_CATEGORIES[row["category"]],
                "source_label": row["category"],
                "source": "ESC-50 (GitHub karolpiczak/ESC-50)",
                "source_url": f"{ESC50_RAW}/audio/{name}",
                "dataset_licence": "CC BY-NC 3.0 (whole dataset; non-commercial)",
                "licence_text_url": ESC50_LICENSE_URL,
                "clip_licence_tag": lic["licence_tag"],
                "author": lic["author"],
                "origin_url": lic["source_url"],
                "origin_file": lic["original_file"],
                "speech_content": "none_labelled_category",
                "usable_as_clean_target": False,
                "measured_vs_simulated": "recorded",
                **entry,
            }
        )
    return out


def commons_candidates(budget: Budget) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for q in COMMONS_QUERIES:
        params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": f"{q} filetype:audio",
                  "gsrnamespace": "6", "gsrlimit": "8", "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime"}
        data = json.loads(http_get(f"{COMMONS_API}?{urllib.parse.urlencode(params)}", budget))
        for page in (data.get("query") or {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            lic = (meta.get("LicenseShortName") or {}).get("value")
            if info.get("size", 1 << 40) <= COMMONS_MAX_BYTES and commons_licence_ok(lic):
                found[page["title"]] = {
                    "title": page["title"], "url": info["url"], "bytes": info["size"], "licence": lic,
                    "licence_url": (meta.get("LicenseUrl") or {}).get("value"),
                    "author": strip_html((meta.get("Artist") or {}).get("value")),
                    "description_url": info.get("descriptionurl"),
                    "query": q,
                }
    return sorted(found.values(), key=lambda c: c["title"])


def collect_commons(root: Path, budget: Budget, missing: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in commons_candidates(budget)[:COMMONS_MAX_FILES]:
        try:
            data = http_get(cand["url"], budget)
            safe = re.sub(r"[^\w.-]+", "_", cand["title"].removeprefix("File:"))
            entry = _audio_entry(root, f"native/noise/commons/{safe}", data, "derived_16k_mono/noise/commons")
        except (urllib.error.URLError, RuntimeError) as exc:
            missing.append({"id": cand["title"], "reason": str(exc)})
            continue
        out.append(
            {
                "id": f"commons:{cand['title']}",
                "kind": "noise",
                "category": "cafe_crowd",
                "source_label": cand["query"],
                "source": "Wikimedia Commons",
                "source_url": cand["description_url"],
                "licence": cand["licence"],
                "licence_text_url": cand["licence_url"],
                "author": cand["author"],
                "speech_content": "unknown_unlabeled",
                "interfering_speaker_provenance": None,
                "usable_as_clean_target": False,
                "measured_vs_simulated": "recorded",
                **entry,
            }
        )
    return out


def rir_room(name: str) -> str:
    """Room label from the file name only (documented by the dataset's naming scheme)."""
    m = re.match(r"air_type1_air_(?:phone|binaural)_(?:bt_)?([a-z]+?)(?:\d*)(?:_|$)", name)
    if name.startswith("RVB2014_type1_rir_"):
        return "RVB2014_" + name.split("_")[3]
    return "AIR_" + m[1] if m else "unknown"


def collect_rirs(root: Path, budget: Budget, missing: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    zf = zipfile.ZipFile(io.BufferedReader(RangeFile(OPENSLR_ZIP, budget), buffer_size=1 << 16))
    for name in RIR_MEMBERS:
        try:
            data = zf.read(RIR_DIR + name)
        except (KeyError, urllib.error.URLError) as exc:
            missing.append({"id": name, "reason": repr(exc)})
            continue
        entry = _audio_entry(root, f"native/rir/openslr28/{name}", data, "derived_16k_mono/rir/openslr28")
        out.append(
            {
                "id": f"openslr28:{name}",
                "kind": "rir",
                "room": rir_room(name),
                "source": "OpenSLR SLR28 rirs_noises.zip (range-fetched members)",
                "source_url": OPENSLR_ZIP,
                "licence": "Apache 2.0 (as declared on the OpenSLR page; upstream AIR/REVERB terms not independently verified)",
                "licence_text_url": OPENSLR_PAGE,
                "author": None,
                "measured_vs_simulated": "measured" if "type1" in name else "simulated",
                "room_dimensions": None,
                "rt60_s": None,
                **entry,
            }
        )
    return out


def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
    noise = [i for i in items if i["kind"] == "noise"]
    rirs = [i for i in items if i["kind"] == "rir"]
    return {
        "noise_excerpts": len(noise),
        "noise_categories": sorted({i["category"] for i in noise}),
        "noise_decoded_s": round(sum(i["duration_s"] for i in noise), 2),
        "rirs": len(rirs),
        "rir_rooms": sorted({i["room"] for i in rirs}),
        "native_bytes": sum(i["native_bytes"] for i in items),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="eval/manifests/overnight_grunt/noise.json")
    ap.add_argument("--shard-root", default=None, help="default $HEARING_FIXTURES_DIR/overnight_grunt/noise")
    ap.add_argument("--skip", action="append", default=[], choices=["esc50", "commons", "rirs"])
    ap.add_argument("--with-commons", action="store_true", help="also try Wikimedia Commons crowd/cafe files (Ogg/Opus often undecodable, rate limited)")
    args = ap.parse_args(argv)
    fixtures = os.environ.get("HEARING_FIXTURES_DIR")
    root = Path(args.shard_root or (Path(fixtures or "eval/fixtures/generated") / "overnight_grunt" / "noise"))
    budget = Budget()
    missing: list[dict[str, str]] = []
    items: list[dict[str, Any]] = []
    for name, fn in (("esc50", collect_esc50), ("commons", collect_commons), ("rirs", collect_rirs)):
        if name in args.skip or (name == "commons" and not args.with_commons):
            continue
        try:
            items += fn(root, budget, missing)
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            missing.append({"id": name, "reason": repr(exc)})
    manifest = {
        "task": "T-219",
        "shard_root": str(root),
        "downloaded_bytes": budget.used,
        "budget_bytes": budget.limit,
        "summary": summarise(items),
        "items": items,
        "missing_or_failed": missing,
        "discovery_only": [
            {"source": "MIT IR Survey (mcdermottlab.mit.edu/Reverb/IR_Survey.html)",
             "reason": "official page states no licence; not collected (a 11.7 MiB probe zip was downloaded to inspect it and deleted)"},
            {"source": "Wikimedia Commons cafe/crowd ambience",
             "reason": "tried once: 2 of 4 Ogg files undecodable by libsndfile, 2 HTTP 429; no cafe/crowd noise collected (opt-in --with-commons)"},
            {"source": "Freesound", "reason": "needs an API key; not available"},
        ],
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"summary": manifest["summary"], "downloaded_bytes": budget.used, "missing": len(missing)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
