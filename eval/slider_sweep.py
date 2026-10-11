"""Suppression slider sweep: how Natural and Focus HQ sound at slider 0/25/50/75/100 (T-093).

For each slider position and clip it reports SI-SDRi, DNSMOS P.835 SIG/BAK/OVRL and the
**room level**: the energy of what is left after projecting the output on the clean target,
relative to the same quantity in the mixture, in dB (0 at slider 0; -30 means the room is
30 dB quieter). The curve the slider should follow is :func:`hearing.fusion.mixer.room_atten_db`.

* ``natural``: ``dfn_ll>gate>mixer`` streamed in 10 ms chunks (no camera here, so the gate
  bypasses and the mixer passes through: the slider acts through DFN's ``atten_lim_db``).
* ``focus_hq``: the Mac ladder's enrolled tier, ``tse(candidate="enrolled", window_s=1.0,
  hop_s=0.5)>mixer``: the TSE target estimate from :class:`SlidingWindowExtractor` (enrolled
  on another utterance of the target speaker when the set has one, else its own clean
  target), and the mixer's ``est + room_gain(s) * (mix - est)``, exact for any slider.
  Two-face clips only.

    .venv/bin/python -m eval.slider_sweep --n 16
    .venv/bin/python -m eval.slider_sweep --n 2 --no-focus        # quick look
    .venv/bin/python -m eval.slider_sweep --n 16 --no-natural     # Focus HQ half alone (run both in parallel)

Writes ``$HEARING_RESULTS_DIR/slider_sweep_<stamp>.json`` and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.bench import build
from eval.dnsmos import get_dnsmos
from eval.metrics import si_sdri
from hearing.audio.io import PIPELINE_SR
from hearing.fusion.mixer import room_atten_db, room_gain
from hearing.types import AudioChunk, StageContext

SLIDERS = (0.0, 0.25, 0.5, 0.75, 1.0)
HOP = 160
NATURAL = "dfn_ll>gate>mixer"


def fixtures_dir() -> Path:
    env = os.environ.get("HEARING_FIXTURES_DIR")
    return Path(env) if env else Path(__file__).parent / "fixtures" / "generated"


def load_clips(set_name: str, n: int, contains: str = "") -> list[dict]:
    clips = []
    for d in sorted((fixtures_dir() / set_name).iterdir()):
        if not (d / "mix.wav").exists() or contains not in d.name:
            continue
        man = json.loads((d / "manifest.json").read_text())
        mix, sr = sf.read(d / "mix.wav", dtype="float32")
        tgt, _ = sf.read(d / "target.wav", dtype="float32")
        assert sr == PIPELINE_SR, sr
        speaker = (man.get("source") or {}).get("target_speaker")
        clips.append({"id": d.name, "set": set_name, "mix": mix, "target": tgt, "speaker": speaker,
                      "utt": man.get("target_id")})
        if len(clips) == n:
            break
    return clips


def room_db(out: np.ndarray, mix: np.ndarray, target: np.ndarray) -> float:
    """Level of ``out`` minus its projection on ``target``, re ``mix`` minus its projection, in dB."""

    def rest(x: np.ndarray) -> float:
        a = float(np.dot(x, target) / (np.dot(target, target) + 1e-12))
        return float(np.mean((x - a * target) ** 2)) + 1e-12

    return 10.0 * np.log10(rest(out) / rest(mix))


def run_natural(mix: np.ndarray, s: float, spec: str = NATURAL) -> np.ndarray:
    pipe = build(spec)
    ctx = StageContext.default()
    ctx.suppression = s
    delay = round(pipe.latency_ms * PIPELINE_SR / 1000.0)
    x = np.concatenate((mix, np.zeros(delay + HOP, np.float32)))
    x = x[: len(x) // HOP * HOP]
    out = [pipe.process(AudioChunk(x[i : i + HOP], PIPELINE_SR, i // HOP, 0.0), ctx).samples
           for i in range(0, len(x), HOP)]
    return np.concatenate(out)[delay : delay + len(mix)]


def score(out: np.ndarray, clip: dict, scorer) -> dict:
    d = scorer.score(out, PIPELINE_SR)
    return {"si_sdri": si_sdri(out, clip["mix"], clip["target"]), "sig": d.sig, "bak": d.bak,
            "ovrl": d.ovrl, "room_db": room_db(out, clip["mix"], clip["target"])}


def focus_estimates(clips: list[dict]) -> dict[str, np.ndarray]:
    from hearing.tse.base import get_candidate
    from hearing.tse.windowed import SlidingWindowExtractor

    cand = get_candidate("enrolled")
    cand.load()
    ext = SlidingWindowExtractor(cand, 1.0, 0.5)
    ests = {}
    for c in clips:
        other = next((o for o in clips if o["speaker"] == c["speaker"] and o["utt"] != c["utt"]), None)
        c["enroll"] = "other" if other is not None else "self"
        enroll = (other or c)["target"]
        est = np.asarray(ext.extract(c["mix"], PIPELINE_SR, enroll=enroll), np.float32)
        ests[c["id"]] = est[: len(c["mix"])]
    return ests


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["mode"], r["set"], r["slider"]) for r in rows})
    for mode, set_name, s in keys:
        sel = [r for r in rows if (r["mode"], r["set"], r["slider"]) == (mode, set_name, s)]
        agg = {k: float(np.mean([r[k] for r in sel])) for k in ("si_sdri", "sig", "bak", "ovrl", "room_db")}
        out.append({"mode": mode, "set": set_name, "slider": round(100 * s), "n": len(sel),
                    "curve_db": -room_atten_db(s), **agg})
    return out


def table(summary: list[dict]) -> str:
    lines = ["| mode | set | slider | n | curve dB | room dB | SI-SDRi | SIG | BAK | OVRL |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['mode']} | {r['set']} | {r['slider']} | {r['n']} | {r['curve_db']:.1f} | "
                     f"{r['room_db']:.1f} | {r['si_sdri']:.2f} | {r['sig']:.2f} | {r['bak']:.2f} | {r['ovrl']:.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=16, help="clips per set")
    ap.add_argument("--no-focus", action="store_true", help="skip the Focus HQ (TSE) half")
    ap.add_argument("--no-natural", action="store_true", help="skip the Natural (DFN) half")
    ap.add_argument("--natural", default=NATURAL, help="natural pipeline spec")
    args = ap.parse_args(argv)
    scorer = get_dnsmos("auto")
    tiny = load_clips("tiny", args.n)
    two = load_clips("av_full", args.n, contains="two_face")
    rows = []
    for clip in [] if args.no_natural else tiny + two:
        for s in SLIDERS:
            out = run_natural(clip["mix"], s, args.natural)
            rows.append({"mode": "natural", "set": clip["set"], "id": clip["id"], "slider": s,
                         **score(out, clip, scorer)})
        print(f"natural {clip['id']} done", file=sys.stderr, flush=True)
    if not args.no_focus:
        ests = focus_estimates(two)
        for clip in two:
            est = ests[clip["id"]]
            for s in SLIDERS:
                out = (est + np.float32(room_gain(s)) * (clip["mix"] - est)).astype(np.float32)
                rows.append({"mode": "focus_hq", "set": clip["set"], "id": clip["id"], "slider": s,
                             "enroll": clip["enroll"], **score(out, clip, scorer)})
    summary = summarize(rows)
    print(f"scorer={scorer.name}")
    print(table(summary))
    res_dir = Path(os.environ.get("HEARING_RESULTS_DIR") or Path(__file__).parent / "results")
    res_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = res_dir / f"slider_sweep_{stamp}.json"
    path.write_text(json.dumps({"natural": args.natural, "scorer": scorer.name, "summary": summary,
                                "rows": rows}, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
