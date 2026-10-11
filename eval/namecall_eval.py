"""Camera-free Name Call eval (T-120): someone says the wearer's name, the session isolates them.

    .venv/bin/python -m eval.namecall_eval scene            # replay the scene through a Session, real time
    .venv/bin/python -m eval.namecall_eval false-triggers   # 5 min of speech without the name
    .venv/bin/python -m eval.namecall_eval matrix           # match strict/normal/loose x prompt off/on (T-123)

**Scene** (``build_scene``, 36 s, no video): babble at 10 dB SNR the whole time; talker B (a GRID
male, ``av_full`` targets of ``grid-s1``) talks from 0 s; at ``NAME_AT`` talker A says
"Aryan, hey, can you hear me over here?" and keeps talking, overlapping B from about 10 s to 24 s, A
alone 24-28 s, both again to the end. A is Kokoro TTS (``af_heart``) for the whole scene, so the
enrollment from the name utterance is A's real voice (a TTS name spliced into a GRID talker would
enroll the TTS voice). The scene goes through :class:`hearing.server.session.Session` in 20 ms frames
at real time (``mode: name_call``, ``name: Aryan``, ``hello video: false``) and telemetry is read every
100 ms. Reported: detection latency (the listener's own: name's last audio -> detected), name end ->
pill ``isolated`` (s), and from Isolated on, per 0.5 s window aligned by the playing delay: SI-SDRi vs
the target A, B leak (B's level in the output vs the input, dB) and wrong-target seconds per 10 s
(windows where B is active and louder than A in the output, least-squares stem gains).

**False triggers**: 150 s of GRID sentences (``av_full`` target/interferer stems) + 150 s of Kokoro
speech of ordinary text in four voices, none containing the name, fed through :class:`NameCaller`
inline (cleaner ``dfn_ll``, Whisper as the session uses it). Detections are false triggers.

**Matrix** (T-123): three sets through :class:`NameCaller` inline (cleaner ``dfn_ll``), once with the
name as Whisper's ``initial_prompt``/``hotwords`` and once without; the words of every window are kept
and :func:`~hearing.server.namecall.find_name` is applied at each match level offline. Sets: **name**
(``NAME_TEXT``, 16 Kokoro sentences in four voices that say "Aryan"; detection rate = sentences with a
hit), **ryan** (``RYAN_TEXT``, 16 sentences saying the confusable name "Ryan"; false triggers) and
**none** (the 5 min false-trigger set above; false triggers, hits closer than the cooldown merged).

Results go to ``$HEARING_RESULTS_DIR/namecall_<stamp>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from hearing.audio import io as aio

SR = 16000
NAME = "Aryan"
NAME_AT = 7.0
SCENE_S = 36.0
A_VOICE = "af_heart"
A_LINES = [  # (start s, text); the first one carries the name
    (NAME_AT, "Aryan, hey, can you hear me over here?"),
    (10.5, "I wanted to ask about the project before the demo starts."),
    (14.0, "We still need to test the microphone in the big hall."),
    (17.5, "Can you check the battery on the headset too?"),
    (21.0, "The judges arrive at noon, so we have an hour."),
    (24.5, "Let me know when you are ready and I will set it up."),
    (29.5, "Great, I will meet you by the stage in ten minutes."),
    (33.0, "Thanks a lot, see you there."),
]
B_SPANS = [(0.0, 10.0), (11.0, 24.0), (28.5, SCENE_S)]  # where talker B talks
FALSE_TEXT = [
    "The weather has been strange this week, warm in the morning and cold by the evening.",
    "We walked along the river and stopped at a small cafe that served excellent coffee.",
    "Please remember to bring the charger, the cables and the spare batteries tomorrow.",
    "The library closes early on Fridays, so plan to return your books before five.",
    "Our team finished the first prototype late last night after fixing the audio bug.",
    "I think the second option is better because it is cheaper and easier to maintain.",
    "Could you send me the slides before the meeting so I can review the numbers?",
    "The train was delayed by twenty minutes because of a signal problem near the station.",
    "She ordered a salad, a bowl of soup and a glass of orange juice for lunch.",
    "If the microphone is too far away, the speech sounds quiet and full of echo.",
    "They are planning a trip to the mountains in the spring when the snow melts.",
    "The new building has a garden on the roof and solar panels on every wall.",
    "My brother is learning to play the guitar and practices every single evening.",
    "We should test the system in a noisy room with music playing in the background.",
    "The museum opens a new exhibit about ancient maps and old sailing ships next month.",
    "Turn left at the second light, then park behind the grocery store on Main Street.",
    "He forgot his umbrella again and arrived at the office completely soaked.",
    "The results look promising, but we need more data before we can be sure.",
    "Dinner is at seven, and everyone is bringing something to share with the group.",
    "The dog barked at the mail carrier and then fell asleep on the warm porch.",
]
NAME_TEXT = [
    "Aryan, can you hear me over here?", "Hey Aryan, the bus is finally here.",
    "Did you see that, Aryan? It was amazing.", "Aryan, wait for me at the door.",
    "I think Aryan already left for the station.", "Aryan, your phone is ringing on the table.",
    "Good morning Aryan, how did you sleep?", "Can you pass me the water, Aryan?",
    "Aryan! Over here, by the window.", "Excuse me, Aryan, do you have a minute?",
    "Aryan, dinner is ready in five minutes.", "Thanks Aryan, that was really helpful.",
    "Aryan, look at this picture from the trip.", "Is that you, Aryan? I almost missed you.",
    "Aryan, the judges want to see the demo now.", "Come on Aryan, we are going to be late.",
]
RYAN_TEXT = [
    "Ryan, can you hear me over here?", "Hey Ryan, the bus is finally here.",
    "Ryan said the meeting moved to Thursday.", "I think Ryan already left for the station.",
    "Ryan, your phone is ringing on the table.", "Did Ryan bring the cables for the demo?",
    "Good morning Ryan, how did you sleep?", "Can you pass me the water, Ryan?",
    "Ryan and Maya are cooking dinner tonight.", "Ask Ryan if he wants to come along.",
    "Ryan, dinner is ready in five minutes.", "Thanks Ryan, that was really helpful.",
    "Ryan is the one with the red jacket.", "We met Ryan at the coffee shop yesterday.",
    "Ryan, the judges want to see the demo now.", "Come on Ryan, we are going to be late.",
]
FALSE_VOICES = ["af_heart", "am_michael", "bf_emma", "am_adam"]


def fixtures_dir() -> Path:
    return Path(os.environ.get("HEARING_FIXTURES_DIR", Path(__file__).resolve().parent / "fixtures/generated"))


def results_dir() -> Path:
    d = Path(os.environ.get("HEARING_RESULTS_DIR", Path(__file__).resolve().parent / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _av_full_items() -> list[dict]:
    root = fixtures_dir() / "av_full"
    items = []
    for d in sorted(root.iterdir()):
        m = d / "manifest.json"
        if m.exists():
            items.append({**json.loads(m.read_text()), "dir": d})
    if not items:
        raise SystemExit(f"no av_full fixtures under {root} (make_fixtures --full)")
    return items


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def _speech_level(x: np.ndarray) -> float:
    frames = x[: len(x) // 160 * 160].reshape(-1, 160)
    r = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    active = r[r > 0.1 * r.max()] if len(r) else r
    return float(np.sqrt(np.mean(active ** 2))) if len(active) else 1e-6


def _kokoro(text: str, voice: str) -> np.ndarray:
    from hearing.synth.tts import KokoroTTS

    return KokoroTTS().synthesize(text, voice)


def build_scene(seed: int = 0) -> dict:
    items = _av_full_items()
    n = round(SCENE_S * SR)
    a = np.zeros(n, np.float32)
    for t, text in A_LINES:
        y = _kokoro(text, A_VOICE)
        i = round(t * SR)
        y = y[: n - i]
        a[i : i + len(y)] += y
    name_end = NAME_AT + 0.6  # "Aryan," at the start of the first line (Kokoro, ~0.5-0.6 s)
    b = np.zeros(n, np.float32)
    bsrc = [aio.load(it["dir"] / "target.wav", SR)[0] for it in items
            if it.get("source", {}).get("target_speaker") == "grid-s1"] or \
        [aio.load(it["dir"] / "target.wav", SR)[0] for it in items]
    k = 0
    for s0, s1 in B_SPANS:
        i, end = round(s0 * SR), round(s1 * SR)
        while i < end:
            y = bsrc[k % len(bsrc)][: end - i]
            k += 1
            b[i : i + len(y)] += y
            i += len(y) + round(0.25 * SR)
    noise = np.concatenate([aio.load(it["dir"] / "noise.wav", SR)[0] for it in items[:16]])[:n]
    if len(noise) < n:
        noise = np.resize(noise, n)
    la = _speech_level(a)
    b *= la / _speech_level(b)  # SIR 0 dB on active speech
    noise *= la / max(_rms(noise), 1e-9) * 10 ** (-10 / 20)  # SNR 10 dB
    mix = a + b + noise
    g = 10 ** (-18.5 / 20) / max(_speech_level(mix), 1e-9)  # the demo clips' speech level
    return {"mix": (mix * g).astype(np.float32), "target": (a * g).astype(np.float32),
            "other": (b * g).astype(np.float32), "noise": (noise * g).astype(np.float32),
            "name_at": NAME_AT, "name_end": name_end}


def _lsq_gains(y: np.ndarray, refs: list[np.ndarray]) -> np.ndarray:
    m = np.stack(refs, axis=1).astype(np.float64)
    g, *_ = np.linalg.lstsq(m, y.astype(np.float64), rcond=None)
    return g


def _si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    from eval.metrics import si_sdr

    return float(si_sdr(est, ref))


def best_lag(out: np.ndarray, ref: np.ndarray, i0: int, max_s: float = 3.0) -> float:
    """Output delay (s, 10 ms steps) that best lines ``out`` up with ``ref`` from ``i0`` on."""
    best, lag = -1.0, 0
    r = ref[i0:]
    for d in range(0, round(max_s * SR), SR // 100):
        y = out[i0 + d : i0 + d + len(r)]
        m = min(len(y), len(r))
        if m < SR:
            break
        c = abs(float(np.dot(y[:m], r[:m]))) / (np.linalg.norm(y[:m]) * np.linalg.norm(r[:m]) + 1e-9)
        if c > best:
            best, lag = c, d
    return lag / SR


def score(scene: dict, out: np.ndarray, delay_s: float, t_iso: float | None) -> dict:
    if t_iso is not None:  # the ladder's reported delay can be off by the stream resampler / hop
        delay_s = best_lag(out, scene["target"], round(t_iso * SR))
    d = round(delay_s * SR)
    y = out[d:]
    n = min(len(y), len(scene["mix"]))
    y, a, b, nz, mix = y[:n], scene["target"][:n], scene["other"][:n], scene["noise"][:n], scene["mix"][:n]
    res: dict = {"delay_s": delay_s}
    if t_iso is None:
        return res
    i0 = round(t_iso * SR)
    if n - i0 < SR:
        return res
    res["si_sdr_in"] = round(_si_sdr(mix[i0:], a[i0:]), 2)
    res["si_sdr_out"] = round(_si_sdr(y[i0:], a[i0:]), 2)
    res["si_sdri"] = round(res["si_sdr_out"] - res["si_sdr_in"], 2)
    win = SR // 2
    wrong = active_b = 0
    leak = []
    for j in range(i0, n - win, win):
        sl = slice(j, j + win)
        gains = _lsq_gains(y[sl], [a[sl], b[sl], nz[sl]])
        ea, eb = abs(gains[0]) * _rms(a[sl]), abs(gains[1]) * _rms(b[sl])
        if _rms(b[sl]) > 10 ** (-45 / 20):
            active_b += 1
            leak.append(20 * np.log10(max(abs(gains[1]), 1e-4)))
            if eb > ea:
                wrong += 1
    span = (n - i0) / SR
    res["isolated_span_s"] = round(span, 1)
    res["wrong_target_s"] = round(wrong * 0.5, 2)
    res["wrong_target_s_per_10s"] = round(wrong * 0.5 / span * 10, 2)
    res["b_leak_db_median"] = round(float(np.median(leak)), 1) if leak else None
    res["b_active_windows"] = active_b
    return res


def run_scene(args) -> dict:
    from hearing.audio.io import float_to_pcm16
    from hearing.device import pick
    from hearing.server import warm
    from hearing.server.protocol import (
        KIND_AUDIO_IN,
        KIND_AUDIO_OUT,
        pack_frame,
        unpack_frame,
    )
    from hearing.server.session import ServerConfig, Session, load_stage_modules

    load_stage_modules()
    warm.warm()  # as `hearing serve` does at start (T-072/T-088): no model loads inside the measurement
    warm.warm_inference()
    scene = build_scene()
    sess = Session(ServerConfig(prefs_log=None, faces="none"), pick())
    sess.handle_text(json.dumps({"type": "hello", "sample_rate": SR, "frame_ms": 20, "video": False, "client": "eval"}))
    sess.handle_text(json.dumps({"type": "control", "mode": "name_call", "suppression": 100,
                                 "name": {"text": NAME, "aliases": args.aliases,
                                          **({"match": args.match} if args.match else {})}}))
    frame = SR // 50
    outs: list[np.ndarray] = []
    ticks: list[dict] = []
    t0 = time.monotonic()
    mix = scene["mix"]
    for k, i in enumerate(range(0, len(mix) - frame + 1, frame)):
        due = t0 + k * 0.02
        lag = due - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        for m in sess.handle_binary(pack_frame(KIND_AUDIO_IN, k, i / SR, float_to_pcm16(mix[i : i + frame]))):
            if isinstance(m, bytes):
                f = unpack_frame(m)
                if f.kind == KIND_AUDIO_OUT:
                    outs.append(aio.pcm16_to_float(f.payload))
        if k % 5 == 4:
            t = sess.telemetry()
            nc = t.get("name_call") or {}
            ticks.append({"t": round((i + frame) / SR, 2), "tier": t["tier"], "state": nc.get("state"),
                          "sim": nc.get("tse_similarity"),
                          "delay_ms": (t.get("tiers") or {}).get("delay_ms"), "last": nc.get("last"),
                          "detections": nc.get("detections"), "events": nc.get("events")})
    sess.close()
    out = np.concatenate(outs) if outs else np.zeros(0, np.float32)
    det = next((x for x in ticks if x["detections"]), None)
    iso = next((x for x in ticks if x["state"] == "isolated"), None)
    delays = [x["delay_ms"] for x in ticks if x["state"] == "isolated" and x["delay_ms"]]
    delay_s = (float(np.median(delays)) / 1000.0) if delays else 0.0
    res = {
        "name_end_s": scene["name_end"],
        "detected_at_s": det["t"] if det else None,
        "detection_latency_ms": (det["last"] or {}).get("latency_ms") if det else None,
        "detected_word": (det["last"] or {}).get("word") if det else None,
        "detections": ticks[-1]["detections"] if ticks else 0,
        "isolated_at_s": iso["t"] if iso else None,
        "name_to_isolated_s": round(iso["t"] - scene["name_end"], 2) if iso else None,
        "states": _state_runs(ticks),
        **score(scene, out, delay_s, iso["t"] if iso else None),
    }
    sims = [x["sim"] for x in ticks if x["state"] in ("isolated", "quiet") and x["sim"] is not None]
    res["tse_similarity_p10_p50_p90"] = [round(float(np.percentile(sims, q)), 3) for q in (10, 50, 90)] if sims else None
    res["quiet_s"] = round(0.1 * sum(1 for x in ticks if x["state"] == "quiet"), 1)
    res["sim_trace"] = [[x["t"], x["sim"]] for x in ticks[::5] if x["sim"] is not None]
    detections = [x["last"] for x in ticks if x["events"]]
    res["detection_events"] = detections[:5]
    return res


def _state_runs(ticks: list[dict]) -> list[list]:
    runs: list[list] = []
    for x in ticks:
        key = f"{x['state']}/{x['tier']}"
        if not runs or runs[-1][0] != key:
            runs.append([key, x["t"]])
    return runs


def false_trigger_speech(args) -> tuple[np.ndarray, int]:
    """The 5 min of speech without the name (GRID stems + Kokoro), and how many samples are GRID."""
    items = _av_full_items()
    parts, total = [], 0
    for it in items:
        for stem in ("target.wav", "interferer.wav"):
            x = aio.load(it["dir"] / stem, SR)[0]
            parts += [x, np.zeros(SR // 3, np.float32)]
            total += len(x) + SR // 3
        if total >= args.grid_s * SR:
            break
    grid_n = total
    tts_parts = []
    k = 0
    while total - grid_n < args.tts_s * SR:
        x = _kokoro(FALSE_TEXT[k % len(FALSE_TEXT)], FALSE_VOICES[k % len(FALSE_VOICES)])
        tts_parts += [x * 10 ** (-6 / 20), np.zeros(SR // 2, np.float32)]
        total += len(x) + SR // 2
        k += 1
    speech = np.concatenate(parts + tts_parts)
    rng = np.random.default_rng(0)
    speech = speech + rng.normal(0, 10 ** (-55 / 20), len(speech)).astype(np.float32)  # a room floor
    return speech, grid_n


def run_false_triggers(args) -> dict:
    from hearing.server.namecall import NameCaller

    speech, grid_n = false_trigger_speech(args)
    caller = NameCaller(NAME, threaded=False)
    t = time.perf_counter()
    for i in range(0, len(speech) - 160 + 1, 160):
        caller.feed(speech[i : i + 160])
    wall = time.perf_counter() - t
    caller.close()
    minutes = len(speech) / SR / 60
    return {"speech_min": round(minutes, 2), "grid_s": round(grid_n / SR, 1), "tts_s": round((len(speech) - grid_n) / SR, 1),
            "false_triggers": len(caller.detections),
            "false_triggers_per_5min": round(len(caller.detections) / minutes * 5, 2),
            "triggers": [{"word": d.word, "text": d.text, "t": round(d.t_name, 1)} for d in caller.detections],
            "engine": caller.engine, "rtf": round(wall / (len(speech) / SR), 3)}


def _sentences(texts: list[str], gap_s: float = 1.0) -> tuple[np.ndarray, list[tuple[float, float]]]:
    parts, spans, t = [], [], 0.0
    for k, text in enumerate(texts):
        x = _kokoro(text, FALSE_VOICES[k % len(FALSE_VOICES)]) * 10 ** (-6 / 20)
        spans.append((t, t + len(x) / SR))
        parts += [x, np.zeros(round(gap_s * SR), np.float32)]
        t += len(x) / SR + gap_s
    x = np.concatenate(parts)
    rng = np.random.default_rng(1)
    return x + rng.normal(0, 10 ** (-55 / 20), len(x)).astype(np.float32), spans


def _windows(speech: np.ndarray, prompt: bool) -> list[tuple[float, list]]:
    """(end time s, words) of every window :class:`NameCaller` transcribes."""
    from hearing.server.namecall import (
        NameCaller,
        _spectral_embedder,
        whisper_transcriber,
    )

    base = whisper_transcriber((lambda: [NAME]) if prompt else None)
    if base is None:
        raise SystemExit("faster-whisper is not installed/cached")
    out: list[tuple[float, list]] = []
    caller: NameCaller | None = None

    def rec(x: np.ndarray) -> list:
        words = base(x)
        out.append((caller.pos / SR, words))
        return words

    caller = NameCaller(NAME, transcriber=rec, embedder=_spectral_embedder, threaded=False, prompt=prompt)
    for i in range(0, len(speech) - 160 + 1, 160):
        caller.feed(speech[i : i + 160])
    caller.close()
    return out


def _hits(windows, match: str, exclude: list[str], aliases: list[str] | None = None,
          guard: bool = False) -> list[tuple[float, str]]:
    from hearing.server.namecall import COOLDOWN_S, find_name, guard_reject

    hits, last = [], -1e9
    for t, words in windows:
        h = find_name(words, [NAME, *(aliases or [])], match, exclude)
        if h is not None and guard and guard_reject(words, h) is not None:
            continue
        if h is not None and t - last >= COOLDOWN_S + 1.0:  # overlapping windows hear one name twice
            hits.append((t, " ".join(w.text for w in words[h[0]:h[1]])))
            last = t
    return hits


def run_matrix(args) -> dict:
    name_x, name_spans = _sentences(NAME_TEXT)
    ryan_x, _ = _sentences(RYAN_TEXT)
    none_x, _ = false_trigger_speech(args)
    sets = {"name": name_x, "ryan": ryan_x, "none": none_x}
    rows, heard = [], {}
    for prompt in {"both": (False, True), "off": (False,), "on": (True,)}[args.prompt]:
        t = time.perf_counter()
        wins = {k: _windows(x, prompt) for k, x in sets.items()}
        rtf = (time.perf_counter() - t) / sum(len(x) / SR for x in sets.values())
        heard[f"prompt_{'on' if prompt else 'off'}"] = {
            k: sorted({w.text for _, ws in v for w in ws if normalize_word(w.text) in _CLOSE}) for k, v in wins.items()}
        cells = [(m, ex, [], False) for m in ("strict", "normal", "loose") for ex in ([], ["Ryan"])
                 if not (m == "normal" and ex)]  # normal already excludes Ryan (CONFUSABLE)
        if args.aliases:  # T-131: the spellings Whisper writes without the prompt, matched exactly
            cells += [("strict", [], args.aliases, False), ("normal", [], args.aliases, False)]
        cells += [("strict", [], [], True), ("normal", [], [], True)]  # T-131 transcript guard
        if args.aliases:
            cells.append(("strict", [], args.aliases, True))
        for match, exclude, aliases, guard in cells:
            name_hits = _hits(wins["name"], match, exclude, aliases, guard)
            found = {i for t_hit, _ in name_hits for i, (a, b) in enumerate(name_spans) if a <= t_hit <= b + 3.0}
            none_hits = _hits(wins["none"], match, exclude, aliases, guard)
            rows.append({
                "prompt": prompt, "match": match, "exclude": exclude, "aliases": aliases, "guard": guard,
                "name_detected": len(found), "name_total": len(NAME_TEXT),
                "name_missed": [NAME_TEXT[i] for i in range(len(NAME_TEXT)) if i not in found],
                "ryan_triggers": len(_hits(wins["ryan"], match, exclude, aliases, guard)), "ryan_total": len(RYAN_TEXT),
                "none_triggers": len(none_hits), "none_words": [w for _, w in none_hits],
                "rtf": round(rtf, 3)})
        # what Whisper wrote for each name sentence (to pick aliases)
        heard[f"name_windows_prompt_{'on' if prompt else 'off'}"] = [
            [round(t, 1), " ".join(f"{w.text}({w.probability:.2f})" for w in ws)]
            for t, ws in wins["name"] if any(normalize_word(w.text) in _CLOSE or normalize_word(w.text)[:2] in ("ar", "ai", "or")
                                             for w in ws)]
        heard[f"none_guard_prompt_{'on' if prompt else 'off'}"] = [
            [round(t, 1), " ".join(f"{w.text}({w.probability:.2f},{w.avg_logprob:.2f},{w.no_speech_prob:.2f})" for w in ws)]
            for t, ws in wins["none"] if any(normalize_word(w.text) in _CLOSE for w in ws)]
    return {"none_min": round(len(none_x) / SR / 60, 2), "rows": rows, "heard_name_like": heard}


_CLOSE = {"aryan", "aryans", "ryan", "ryans", "arian", "ariane", "aryn", "arion", "orion", "brian", "bryan", "arya",
          "aaron", "adrian", "iran", "aryon", "arjun"}


def normalize_word(w: str) -> str:
    from hearing.server.namecall import normalize

    return normalize(w)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["scene", "false-triggers", "matrix"])
    ap.add_argument("--grid-s", type=float, default=150.0)
    ap.add_argument("--tts-s", type=float, default=150.0)
    ap.add_argument("--aliases", default="", help="matrix/scene: comma-separated aliases of NAME (T-131), e.g. Ariane,Arion")
    ap.add_argument("--match", choices=["strict", "normal", "loose"], default=None, help="scene: name.match")
    ap.add_argument("--prompt", choices=["both", "off", "on"], default="both", help="matrix: Whisper name prompt cells")
    args = ap.parse_args(argv)
    args.aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
    from hearing.device import pick

    run = {"scene": run_scene, "false-triggers": run_false_triggers, "matrix": run_matrix}[args.what]
    res = run(args)
    res.update(what=args.what, aliases=args.aliases, match=args.match, device=pick(), host=os.uname().nodename, load=os.getloadavg()[0],
               whisper=os.environ.get("HEARING_WHISPER_MODEL", "base.en"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = results_dir() / f"namecall_{args.what}_{stamp}.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
