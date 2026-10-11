"""Caption WER and phrase latency of Synthetic/Text per ASR feed (T-074).

    python -m eval.synth_feed_bench --demo --av 8 --feeds mixture,natural,focus-gate,focus-enrolled

Each clip (the demo clip, or a cached ``av_full`` two-face fixture looped to ``--seconds``)
streams in real time through a real :class:`~hearing.server.session.Session` (vision, ladder,
ASR, TTS), exactly as ``/replay`` does, with the synth chain's feed forced to one of:
``mixture`` (passthrough), ``natural`` (dfn_ll), ``focus-gate`` (``tiers(only="gate")``) and
``focus-enrolled`` (``tiers(only="enrolled,gate")``), or ``auto`` (the session default, its feed
switches live with the speaker count). ``--modes text,synthetic`` runs Text as well. Reported per feed: caption WER (final
captions vs the target transcript repeated once per loop, ``eval.metrics.wer``), the number of
phrases, phrase latency (ASR phrase end -> first TTS sample, stream clock) p50/p95, and the
share of the time each ladder tier played. No weights or network are touched beyond the
engines named by ``--asr`` / ``--tts`` (``local`` = faster-whisper + Kokoro).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from eval.metrics import wer
from hearing.audio.io import float_to_pcm16
from hearing.replay import Clip
from hearing.server.protocol import KIND_AUDIO_IN, KIND_VIDEO_IN, pack_frame
from hearing.server.session import ServerConfig, Session, load_stage_modules

FEEDS = {
    "mixture": "passthrough",
    "natural": "dfn_ll",
    "focus-gate": 'tiers(only="gate")',
    "focus-enrolled": 'tiers(only="enrolled,gate")',
    "focus": "tiers",
    "auto": None,  # no pipeline override: the session's default synth_feed=auto chain (T-079)
}
TAIL_S = 3.0  # silence after the clip so the last phrase flushes and plays


def load_clip(path: Path, seconds: float) -> tuple[Clip, int]:
    """The clip tiled to ``seconds`` (whole loops); returns it and the number of loops."""
    clip = Clip.load(path)
    loops = max(1, round(seconds / clip.duration))
    n = int(clip.duration * 16000)
    return Clip(np.tile(clip.audio[:n], loops), clip.images[: int(clip.duration * clip.fps)] * loops), loops


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    return sf.read(path, dtype="float32")[0]


class OracleEnrollment:
    """Seed every :class:`~hearing.tse.enroll.AutoEnroller` with the clean target / interferer
    stems of the fixture once it has a track. The fixtures are 3 s of two simultaneous talkers, so
    live auto-enrollment (2 s of the target talking alone) never completes in them and the ladder
    would stay on the gate; this measures the enrolled tier as it plays after enrollment."""

    def __init__(self, src: Path):
        self.target, self.other = load_wav(src / "target.wav"), load_wav(src / "interferer.wav")

    def __enter__(self):
        from hearing.tse.enroll import AutoEnroller

        self._cls, self._orig, oracle = AutoEnroller, AutoEnroller.update, self

        def update(enroller, samples, ctx):
            kept = oracle._orig(enroller, samples, ctx)
            if enroller.track_id is not None and not getattr(enroller, "_oracle", False):
                enroller._oracle = True
                enroller._parts, enroller._n = [oracle.target[: enroller.need]], enroller.need
                enroller._oparts, enroller._on = [oracle.other[: enroller.other_need]], enroller.other_need
            return kept

        AutoEnroller.update = update
        return self

    def __exit__(self, *exc):
        self._cls.update = self._orig


def run_clip(clip: Clip, feed_spec: str | None, asr: str, tts: str, speak: bool = True) -> dict:
    mode = "synthetic" if speak else "text"
    synth = f'synth(speak={"true" if speak else "false"},asr="{asr}",tts="{tts}")'
    pipelines = {mode: f"{feed_spec}>{synth}"} if feed_spec is not None else {}
    cfg = ServerConfig(pipelines=pipelines, asr=asr, tts=tts)
    feeds: Counter = Counter()
    session = Session(cfg, "cpu")
    captions: list[dict] = []
    tiers: Counter = Counter()
    reasons: Counter = Counter()
    errors: list[str] = []
    out_samples = voiced_samples = 0
    caption_ms: list[float] = []
    now_at = epoch = 0.0

    def send_text(msg: dict) -> None:
        for e in session.handle_text(json.dumps(msg)):
            if e["type"] == "error":
                errors.append(f"{e['stage']}: {e['message']}")

    def send_bin(kind: int, seq: int, ts: float, payload: bytes) -> None:
        nonlocal out_samples, voiced_samples
        for e in session.handle_binary(pack_frame(kind, seq, ts, payload)):
            if isinstance(e, bytes):
                pcm = np.frombuffer(e[13:], "<i2").astype(np.float32) / 32768
                out_samples += len(pcm)
                voiced_samples += int(np.count_nonzero(np.abs(pcm) > 0.003))
            elif e["type"] == "caption" and e["final"]:
                t1 = e["t1"] - epoch if e["t1"] > 1e6 else e["t1"]  # capture clock or stream clock
                caption_ms.append(1000.0 * (now_at - t1))  # phrase end -> final caption out
                captions.append(e)
            elif e["type"] == "error":
                errors.append(f"{e['stage']}: {e['message']}")

    send_text({"type": "hello", "sample_rate": 16000, "frame_ms": 20, "video": True, "client": "bench"})
    send_text({"type": "control", "mode": mode, "suppression": 80})
    dur = clip.duration
    events = [(i / clip.fps, 2, img) for i, img in enumerate(clip.images) if i / clip.fps < dur]
    total = int(dur * 16000)
    for i in range(0, total, 320):
        s = np.pad(clip.audio[i : min(i + 320, total)], (0, 320 - min(320, total - i)))
        events.append((i / 16000, 1, float_to_pcm16(s)))
    for i in range(total, total + int(TAIL_S * 16000), 320):
        events.append((i / 16000, 1, float_to_pcm16(np.zeros(320, np.float32))))
    events.sort(key=lambda e: (e[0], -e[1]))
    start, epoch, last_tele = time.monotonic(), time.time(), 0.0
    for seq, (at, kind, payload) in enumerate(events):
        now_at = at + (320 / 16000 if kind == 1 else 0.0)  # stream time at the end of this frame
        time.sleep(max(0.0, start + at - time.monotonic()))
        send_bin(KIND_AUDIO_IN if kind == 1 else KIND_VIDEO_IN, seq, epoch + at, payload)
        if at - last_tele >= 0.1:
            last_tele = at
            tele = session.telemetry()
            tiers[tele["tier"]] += 1
            feeds[tele["synth_feed"]] += 1
            for t in ((tele.get("tiers") or {}).get("tiers") or []):
                if not t["eligible"]:
                    reasons[f"{t['name']}: {t['reason']}"] += 1
    synth_stage = session._synth()
    synth_stage.wait_idle(5.0)
    stats = synth_stage.latency_stats()
    phrases = [{"text": r.text, "t1": round(r.t1, 2), "latency_ms": round(r.latency_ms, 1)}
               for r in synth_stage.phrases()]
    session.close()
    stats["caption_p50"] = float(np.median(caption_ms)) if caption_ms else float("nan")
    return {"captions": [c["text"] for c in captions], "stats": stats, "phrases": phrases,
            "tiers": dict(tiers), "feeds": dict(feeds), "ineligible": dict(reasons.most_common(4)), "errors": errors[:5], "silent": 1.0 - voiced_samples / max(1, out_samples)}


def main(argv: list[str] | None = None) -> int:
    load_stage_modules()  # registers dfn_ll, tiers, tse, synth (a bare Session does not)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="include eval/fixtures/demo/grid-two-face.mp4")
    ap.add_argument("--av", type=int, default=0, help="number of av_full two-face fixtures")
    ap.add_argument("--feeds", default="mixture,natural,focus-gate,focus-enrolled")
    ap.add_argument("--modes", default="synthetic", help="comma list of synthetic,text")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--asr", default="local")
    ap.add_argument("--tts", default="local")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from eval.make_fixtures import DEFAULT_OUT
    clips: list[tuple[str, Path, str, float, Path]] = []  # name, mp4, transcript, source s, stems dir
    if args.demo:
        mp4 = DEFAULT_OUT.parent / "demo" / "grid-two-face.mp4"
        src = json.loads((mp4.parent / "manifest.json").read_text())["source"]
        clips.append(("demo", mp4, src["transcript"], float(src["duration_s"]), DEFAULT_OUT / "av_full" / src["id"]))
    for d in sorted((DEFAULT_OUT / "av_full").glob("*two_face*"))[: args.av]:
        m = json.loads((d / "manifest.json").read_text())
        clips.append((d.name, d / "video.mp4", m["transcript"], float(m["duration_s"]), d))

    modes = args.modes.split(",")
    rows: dict[str, list[dict]] = {f"{f}/{m}" if len(modes) > 1 else f: []
                                   for m in modes for f in args.feeds.split(",")}
    for name, mp4, transcript, src_s, stems in clips:
        clip, _ = load_clip(mp4, args.seconds)
        ref = " ".join([transcript] * max(1, round(clip.duration / src_s)))  # one per loop of the source
        for feed in list(rows):
            spec, _, mode = feed.partition("/")
            speak = (mode or modes[0]) == "synthetic"
            if spec == "focus-enrolled":
                with OracleEnrollment(stems):
                    r = run_clip(clip, FEEDS[spec], args.asr, args.tts, speak)
            else:
                r = run_clip(clip, FEEDS[spec], args.asr, args.tts, speak)
            r.update(clip=name, wer=wer(" ".join(r["captions"]), ref), hyp=" ".join(r["captions"]), ref=ref)
            rows[feed].append(r)
            print(f"{name:34s} {feed:15s} WER {r['wer']:.2f} phrases {r['stats']['n']} "
                  f"p50 {r['stats']['p50']:.0f} ms caption p50 {r['stats']['caption_p50']:.0f} ms "
                  f"tiers {r['tiers']} feeds {r['feeds']} {r['errors']}", flush=True)

    print("\n| feed | clips | caption WER (mean) | phrases/clip | phrase latency p50 (ms) | caption p50 (ms) | output silent |")
    print("|---|---|---|---|---|---|---|")
    for feed, rs in rows.items():
        if not rs:
            continue
        lat = [p["latency_ms"] for r in rs for p in r["phrases"]]
        w = statistics.fmean(r["wer"] for r in rs)
        print(f"| {feed} | {len(rs)} | {w:.2f} | {statistics.fmean(r['stats']['n'] for r in rs):.1f} | "
              f"{float(np.median(lat)) if lat else float('nan'):.0f} | "
              f"{float(np.median([r['stats']['caption_p50'] for r in rs])):.0f} | "
              f"{100 * statistics.fmean(r['silent'] for r in rs):.0f} % |")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
