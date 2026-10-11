"""Re-enroll voice profiles from demo-mic recordings in one command (T-172).

    bash scripts/enroll_profiles.sh aryan a1.wav a2.wav a3.wav joseph j1.wav j2.wav j3.wav
    bash scripts/enroll_profiles.sh joseph j1.wav --out /tmp/profiles      # one name; the others stay

Arguments are a name followed by its takes (any file soundfile reads), repeated. Every take is checked
first (nothing is written if one fails): native rate one of ``RATES`` (8 kHz / HFP or odd rates mean
the wrong input device), at least ``MIN_TAKE_S`` long, not clipped (``CLIP_SHARE`` of samples at full
scale), active speech level at or above ``MIN_SPEECH_DB`` (-40 dBFS, the enroller's voiced gate) and
``MIN_SPEECH_S`` of speech. Then each name goes through ``make_profile`` (16 kHz, -3 dBFS peak, silence
cut) into a staging dir, the ECAPA cosine between every pair of profiles is printed, and a pair at or
above ``SAME_VOICE`` (0.7, ``hearing.tse.enrolled``: the enrolled extractor silently drops a negative
that close, so the other voice would leak) fails the run before anything is replaced.

Versions (Q-467): on success all the takes of a name become one new version ``<out>/<name>/v<N>/``
(several confirmed clips aggregate into one version: multi-enrollment) with a ``manifest.json`` (source
clips and their checks, ``--mic`` label, ``--room``, date, the version it replaced, the cosines), and
``<out>/<name>/current`` (a one-line file, e.g. ``v2``) moves to it; ``hearing.tse.enroll`` loads the
version ``current`` names and ``/health`` shows it under ``profile_versions``. A flat pre-T-172 profile
(wavs directly in ``<name>/``) is moved into ``v1/`` first. The same files are copied flat to
``<out>-all/<name>/``, and the laptop and GPU commands are printed (Aryan runs ``gpu_up``; never this).

    bash scripts/enroll_profiles.sh --rollback aryan          # current -> the version before it
    bash scripts/enroll_profiles.sh --rollback aryan --to v1  # or a named one
    bash scripts/enroll_profiles.sh --versions                # every profile's versions and manifest

Exit codes: 0 ok, 1 usage / no such version, 2 a take failed validation, 3 two profiles are the same voice.
The takes are voices: gitignored, never committed or logged.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hearing.audio.io import HOP, PIPELINE_SR, resample
from hearing.tse.enroll import (
    CURRENT_FILE,
    PROFILES_ENV,
    VoicedGate,
    level_db,
    list_profiles,
    load_profile,
    profile_version,
)

RATES = (16000, 44100, 48000)  # the Mac mic, an iPhone over USB, a 16 kHz rehearsal file
MIN_TAKE_S = 20.0  # a take shorter than this is a mistake (the protocol is 25-30 s per take)
MIN_SPEECH_S = 10.0  # voiced audio per take (scripts/record_take.py MIN_SPEECH_S["profile"])
MIN_SPEECH_DB = -40.0  # active speech level (RMS of voiced 10 ms hops); quieter: closer / more gain
CLIP_PEAK = 0.99
CLIP_SHARE = 1e-4  # share of samples at |x| >= CLIP_PEAK that counts as a clipped take
SAME_VOICE = 0.7  # hearing.tse.enrolled.SAME_VOICE
AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff", ".caf", ".ogg", ".mp3", ".m4a"}
GPU_UP = "ILAB_GRES=gpu:a100:1 ILAB_TIME=04:00:00 ILAB_NODE=rlab2 ILAB_WAIT=400 scripts/gpu_up.sh ilab"


@dataclass
class TakeCheck:
    path: Path
    rate: int = 0
    seconds: float = 0.0
    peak_db: float = -120.0
    clip_share: float = 0.0
    speech_db: float = -120.0
    speech_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def line(self) -> str:
        head = (f"{self.path.name}: {self.rate} Hz, {self.seconds:.1f} s, peak {self.peak_db:.1f} dBFS, "
                f"speech {self.speech_db:.1f} dBFS over {self.speech_s:.1f} s")
        notes = [f"FAIL {e}" for e in self.errors] + [f"warn {w}" for w in self.warnings]
        return head + ("" if not notes else " | " + "; ".join(notes))


def check_take(x: np.ndarray, rate: int, path: Path | str = "take") -> TakeCheck:
    """Validate one take (mono or ``(n, channels)`` float audio at its native ``rate``)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    c = TakeCheck(Path(path), rate=int(rate), seconds=len(x) / rate if rate else 0.0)
    if c.rate not in RATES:
        c.errors.append(f"wrong rate {c.rate} Hz (want {' / '.join(map(str, RATES))}; 8 kHz or 16 kHz "
                        "from AirPods is the Bluetooth headset mic, not the demo mic)")
    if c.seconds < MIN_TAKE_S:
        c.errors.append(f"too short ({c.seconds:.1f} s < {MIN_TAKE_S:.0f} s)")
    if not len(x):
        return c
    peak = float(np.max(np.abs(x)))
    c.peak_db = 20 * float(np.log10(max(peak, 1e-6)))
    c.clip_share = float(np.mean(np.abs(x) >= CLIP_PEAK))
    if c.clip_share > CLIP_SHARE:
        c.errors.append(f"clipped ({100 * c.clip_share:.2f} % of samples at full scale: lower the input gain)")
    elif c.peak_db > -1.0:
        c.warnings.append("peak above -1 dBFS (aim for about -6)")
    y = resample(x, c.rate, PIPELINE_SR) if c.rate and c.rate != PIPELINE_SR else x
    gate = VoicedGate(PIPELINE_SR, HOP)
    hops = [y[i * HOP : (i + 1) * HOP] for i in range(len(y) // HOP)]
    voiced = [h for h in hops if gate.voiced(h)]
    c.speech_s = len(voiced) * HOP / PIPELINE_SR
    c.speech_db = level_db(np.concatenate(voiced)) if voiced else -120.0
    if c.speech_db < MIN_SPEECH_DB:
        c.errors.append(f"too quiet (speech {c.speech_db:.1f} dBFS < {MIN_SPEECH_DB:.0f}: move closer or raise the gain)")
    if c.speech_s < MIN_SPEECH_S:
        c.errors.append(f"too little speech ({c.speech_s:.1f} s < {MIN_SPEECH_S:.0f} s)")
    if c.peak_db < -20.0 and c.ok:
        c.warnings.append("peak below -20 dBFS (aim for about -6)")
    return c


def check_file(path: Path) -> TakeCheck:
    import soundfile as sf

    try:
        x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - any unreadable file is one failed take
        c = TakeCheck(Path(path))
        c.errors.append(f"unreadable ({e})")
        return c
    return check_take(x, rate, path)


def parse_groups(tokens: list[str]) -> dict[str, list[Path]]:
    """``name take [take ...] name take ...`` -> {name: [takes]} (a take is a path with an audio suffix)."""
    groups: dict[str, list[Path]] = {}
    name = None
    for t in tokens:
        if Path(t).suffix.lower() in AUDIO_EXT:
            if name is None:
                raise ValueError(f"take {t} comes before any profile name")
            groups[name].append(Path(t))
        else:
            if not t or "/" in t or t.startswith((".", "-")):
                raise ValueError(f"bad profile name {t!r}")
            name = t
            groups.setdefault(name, [])
    empty = [n for n, takes in groups.items() if not takes]
    if empty or not groups:
        raise ValueError(f"no takes for {', '.join(empty) or 'any profile'}")
    return groups


def profile_embedding(pieces: list[np.ndarray], embedder) -> np.ndarray:
    """Mean of the pieces' unit embeddings, normalised (how the enrolled stage embeds a profile)."""
    from hearing.tse.enrolled import _unit_embedding

    e = np.mean([_unit_embedding(embedder, p) for p in pieces], axis=0)
    n = float(np.linalg.norm(e))
    return e / n if n > 1e-12 else e


def pair_cosines(dirs: dict[str, Path], embedder) -> list[tuple[str, str, float]]:
    """Cosine between every pair of profiles; ``dirs`` maps name -> the root it lives in."""
    emb = {n: profile_embedding(load_profile(n, root), embedder) for n, root in sorted(dirs.items())}
    return [(a, b, float(np.dot(emb[a], emb[b]))) for a, b in itertools.combinations(sorted(emb), 2)]


def load_embedder(kind: str):
    from hearing.tse import enrolled

    if kind == "off" or not enrolled.available():
        return None
    from hearing import models

    return models.load(enrolled.MODEL)


def _replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def versions(d: Path) -> list[str]:
    """The ``v<N>`` dirs of profile dir ``d``, oldest first."""
    vs = [p.name for p in d.glob("v*") if p.is_dir() and p.name[1:].isdigit()] if d.is_dir() else []
    return sorted(vs, key=lambda v: int(v[1:]))


def set_current(d: Path, version: str) -> None:
    """Point ``d/current`` at ``version`` (written to a temp file and renamed: never half-written)."""
    tmp = d / f".{CURRENT_FILE}.tmp"
    tmp.write_text(version + "\n")
    os.replace(tmp, d / CURRENT_FILE)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def migrate_flat(d: Path) -> str | None:
    """Move a flat (pre-T-172) profile's wavs into ``d/v<N>/`` and make it current; returns that version."""
    flat = sorted(d.glob("*.wav")) if d.is_dir() and profile_version(d) is None else []
    if not flat:
        return None
    v = f"v{int((versions(d) or ['v0'])[-1][1:]) + 1}"
    (d / v).mkdir()
    for f in flat:
        shutil.move(str(f), str(d / v / f.name))
    date = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(min(f.stat().st_mtime for f in (d / v).glob("*.wav"))))
    write_manifest(d / v, {"name": d.name, "version": v, "date": date, "migrated": True, "mic": None,
                           "room": None, "sources": [{"file": f.name} for f in flat], "files": [f.name for f in flat],
                           "previous": None})
    set_current(d, v)
    return v


def write_manifest(vdir: Path, manifest: dict) -> None:
    (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def add_version(staged: Path, d: Path, manifest: dict) -> str:
    """Copy the staged profile ``staged`` into a new ``d/v<N>/`` with its manifest, then make it current."""
    d.mkdir(parents=True, exist_ok=True)
    migrate_flat(d)
    previous = profile_version(d)
    v = f"v{int((versions(d) or ['v0'])[-1][1:]) + 1}"
    shutil.copytree(staged, d / v)
    write_manifest(d / v, {"name": d.name, "version": v, **manifest,
                           "files": sorted(f.name for f in (d / v).glob("*.wav")), "previous": previous})
    set_current(d, v)
    return v


def rollback(d: Path, to: str | None = None) -> str:
    """Point ``d/current`` at ``to`` (default: the version before the current one); returns it."""
    vs = versions(d)
    cur = profile_version(d)
    if to is None:
        older = [v for v in vs if cur is None or int(v[1:]) < int(cur[1:])]
        if not older:
            raise ValueError(f"{d.name}: no version before {cur or 'the current one'} (have {', '.join(vs) or 'none'})")
        to = older[-1]
    if to not in vs:
        raise ValueError(f"{d.name}: no version {to} (have {', '.join(vs) or 'none'})")
    set_current(d, to)
    return to


def describe_versions(out: Path) -> list[str]:
    lines = []
    for d in sorted(p for p in out.iterdir() if p.is_dir()) if out.is_dir() else []:
        cur = profile_version(d)
        for v in versions(d):
            m = d / v / "manifest.json"
            info = json.loads(m.read_text()) if m.exists() else {}
            srcs = ", ".join(Path(s.get("file", "?")).name for s in info.get("sources", []))
            lines.append(f"{'*' if v == cur else ' '} {d.name}/{v}  {info.get('date', '?')}  mic={info.get('mic')}  "
                         f"room={info.get('room')}  sources: {srcs}")
    return lines or [f"no versioned profiles in {out}"]


def commands(out: Path, gpu_up_has_profiles: bool) -> list[str]:
    lines = ["# laptop (the demo Mac):",
             f"HEARING_PROFILES_DIR={out} bash scripts/demo.sh --judge --team",
             "curl -s http://127.0.0.1:8000/health   # \"profiles\" lists them, \"profile_versions\" the active version",
             "# GPU (Aryan only, from the main checkout; this script never runs it):",
             f"ILAB_PROFILES={out} {GPU_UP}"]
    if not gpu_up_has_profiles:
        lines.append("# WARNING: scripts/gpu_up.sh here has no ILAB_PROFILES (T-129 not merged): "
                     "the GPU server would start without the profiles")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("groups", nargs="*", help="name take [take ...] [name take ...]")
    default_out = Path(os.environ.get("HEARING_FIXTURES_DIR") or ROOT / "eval/fixtures/generated") / "profiles"
    ap.add_argument("--out", type=Path, default=default_out, help=f"profiles root (default {default_out}; "
                    f"the server reads it as ${PROFILES_ENV})")
    ap.add_argument("--all", type=Path, default=None, help="also copy each profile here (default <out>-all)")
    ap.add_argument("--embedder", choices=("ecapa", "off"), default="ecapa",
                    help="speaker embedding for the same-voice check (off: skip it, prints NOT MEASURED)")
    ap.add_argument("--force", action="store_true", help="enroll even when a take fails validation")
    ap.add_argument("--mic", default=None, help="input device label the takes went through (manifest.json)")
    ap.add_argument("--room", default=None, help="where they were recorded (manifest.json)")
    ap.add_argument("--rollback", metavar="NAME", default=None, help="point NAME's current at an older version")
    ap.add_argument("--to", default=None, help="with --rollback: the version (default: the one before current)")
    ap.add_argument("--versions", action="store_true", help="list every profile's versions and exit")
    args = ap.parse_args(argv)
    out = args.out.expanduser().resolve()
    if args.versions:
        print("\n".join(describe_versions(out)))
        return 0
    if args.rollback:
        try:
            v = rollback(out / args.rollback, args.to)
        except ValueError as e:
            print(f"FAIL: {e}")
            return 1
        print(f"{args.rollback}: current -> {v} ({out / args.rollback / v}); restart the server (and gpu_up) to load it")
        return 0
    try:
        groups = parse_groups(args.groups)
    except ValueError as e:
        ap.error(str(e))
    all_dir = (args.all or out.with_name(out.name + "-all")).expanduser().resolve()
    if args.mic is None:
        print("warn: no --mic label; the manifest records mic=null (pass the input device the takes went through)")

    bad = 0
    checks: dict[str, list[TakeCheck]] = {}
    for name, takes in groups.items():
        for t in takes:
            c = check_file(t)
            checks.setdefault(name, []).append(c)
            print(f"[{name}] {c.line()}")
            bad += not c.ok
    if bad and not args.force:
        print(f"FAIL: {bad} take(s) failed validation; nothing written (re-record, or --force)")
        return 2

    from scripts.make_profile import make_profile

    with tempfile.TemporaryDirectory(prefix="enroll-") as tmp:
        staging = Path(tmp)
        for name, takes in groups.items():
            if not make_profile(name, takes, staging):
                print(f"FAIL: no usable speech for {name}")
                return 2
        dirs = {n: out for n in (list_profiles(out) if out.is_dir() else [])}
        dirs.update({n: staging for n in groups})
        embedder = load_embedder(args.embedder) if len(dirs) > 1 else None
        same, cosines = [], {}
        if len(dirs) > 1 and embedder is None:
            print("cos: NOT MEASURED (ECAPA off, or its weights / speechbrain missing); the same-voice check did not run")
        elif len(dirs) > 1:
            for a, b, cos in pair_cosines(dirs, embedder):
                flag = cos >= SAME_VOICE
                print(f"cos({a}, {b}) = {cos:.3f}" + (f"  FAIL >= {SAME_VOICE}" if flag else ""))
                same += [(a, b)] if flag and (a in groups or b in groups) else []
                cosines.setdefault(a, {})[b] = cosines.setdefault(b, {})[a] = round(cos, 4)
        if same:
            print(f"FAIL: {', '.join(f'{a}/{b}' for a, b in same)} embed as the same voice (>= {SAME_VOICE}): "
                  "the enrolled stage would drop the negative. Nothing replaced; re-record (different takes, "
                  "closer to the mic, quieter room).")
            return 3
        for name in groups:
            sources = [{"file": str(c.path.resolve()), "rate": c.rate, "seconds": round(c.seconds, 2),
                        "peak_db": round(c.peak_db, 1), "speech_db": round(c.speech_db, 1),
                        "speech_s": round(c.speech_s, 2), "ok": c.ok} for c in checks[name]]
            v = add_version(staging / name, out / name, {"date": _now(), "mic": args.mic, "room": args.room,
                                                          "sources": sources, "cos": cosines.get(name, {})})
            _replace(staging / name, all_dir / name)
            print(f"wrote {out / name / v} (current) and {all_dir / name}")
    active = [f"{n} {profile_version(out / n) or '(flat)'}" for n in list_profiles(out)]
    print(f"profiles in {out}: {', '.join(active)}")
    print(f"rollback: bash scripts/enroll_profiles.sh --out {out} --rollback <name>   (--versions lists them)")
    gpu_up = (ROOT / "scripts/gpu_up.sh").read_text(errors="replace") if (ROOT / "scripts/gpu_up.sh").exists() else ""
    print("\n".join(commands(out, "ILAB_PROFILES" in gpu_up)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
