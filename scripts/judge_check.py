#!/usr/bin/env python3
"""Judge demo rehearsal self-test (T-088): ``bash scripts/demo.sh --judge --check``.

    .venv/bin/python scripts/judge_check.py                 # every check, prints PASS/FAIL per criterion
    .venv/bin/python scripts/judge_check.py --flow-only     # only the judge flow on the replay clip
    .venv/bin/python scripts/judge_check.py --no-hw         # skip latency-test, webcam, GPU probe (headless)

Checks, each one line with the measured numbers (``PLAN.md#success-criteria`` plus the judge flow):
1. judge flow: the replay clip through a server started here (ephemeral port) with the judge profile;
   time from Start to target lock, from lock to "Isolated" (acquisition ``locked`` on an isolating
   tier), and the reported delay. PASS when Isolated <= 6 s after lock.
2. Natural / Focus HQ on the clip: ``scripts/demo_regress.py --modes natural,focus_hq --seconds 30``
   (T-084; the manifest's target is locked, T-100) SI-SDRi, leak, dropouts, target agreement. Focus HQ
   is judged on the backend's bar (``BARS``): Mac ladder SI-SDRi >= +3 dB, leak <= -12 dB, Isolated
   <= 8 s; GPU backend SI-SDRi >= +8 dB (PLAN's bake-off bar). The line says which bar applied.
3. mic-to-ear latency (T-143): ``hearing latency-test --json --in <demo mic> --out <demo output>`` on the demo
   devices of ``config/demo-team.yaml`` (MacBook Pro Microphone -> AirPods). Bluetooth path: PASS at median
   <= 300 ms and p95 <= 350 ms over 10 clicks (hold one AirPod at the laptop mic); a wired override keeps
   <= 100 ms. FAIL when a demo device is not connected. Hardware: a human runs it (macOS aborts headless
   processes that open the mic).
4. GPU: ``scripts/gpu_up.sh probe`` (``python -m hearing.server.remote``) when ``HEARING_REMOTE_URL`` is set:
   use it when /health is ok and RTT p95 < 60 ms, else the Mac ladder (both PASS: the fallback is the plan).
5. camera lock: 5 s of webcam frames through the vision engine; PASS when a face is locked. Hardware.
6. Synthetic first-word latency (T-117): demo-mf-easy replayed in synthetic mode with the manifest's target
   locked; per phrase, the ASR final's end of speech (caption ``t1``) -> first non-silent sample of the
   session's ``out.wav``. PASS when p50 <= 1.5 s (ElevenLabs) or <= 2.5 s (Kokoro); the line says which engine
   ran (mock TTS: SKIP). Also FAILs on empty / ``. . .`` captions and a voice bucket that flaps.
8. Demo devices (T-133, T-143): the mic/output the open /judge or /simple page reported (``GET /health``
   ``client_devices`` of ``--server``, default ``http://127.0.0.1:$PORT`` or 8000). Rows: demo mic in use
   (``MacBook Pro Microphone``, or ``HEARING_EXPECT_MIC``), mic is not Bluetooth (an AirPods mic = HFP), output
   is not the laptop speakers, output is the demo output (``AirPods``, or ``HEARING_EXPECT_OUTPUT``). FAIL on a
   saved device that was not found, and when no page reported devices (SKIP with ``--no-hw``). macOS Mic Mode
   and the AirPods settings (ANC, Conversation Awareness, ear detection) are not readable: reminders. T-171: the
   chosen mic label (``HEARING_INPUT_DEVICE`` or ``input_device`` of ``config/demo-team.yaml``) prints as its own
   PASS row; an iPhone label (``<name> iPhone Microphone``, USB/Continuity) is not Bluetooth and adds a reminder.
7. Text captions on the same clip: caption WER against the target's transcript (manifest ``transcript``,
   else its ``sentences``). PASS at WER <= 90 % (``TEXT_WER_BAR``, T-139: the ASR also hears the other talker)
   with no empty final and no repetition loop.
Criteria the rehearsal cannot measure (bench --full, setup time) print SKIP with where they are measured.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ISOLATED_BUDGET_S = 6.0  # lock -> Isolated on the replay clip (T-088 acceptance)
NATURAL_LATENCY_MS = 100.0  # PLAN.md: mic-to-ear <= 100 ms
GPU_RTT_P95_MS = 60.0  # demo.sh --judge uses the GPU below this
ISOLATING = {"enrolled", "focus_hq", "av"}
SYNTH_P50_S = {"elevenlabs": 1.5, "kokoro": 2.5, "mock": 2.5}  # PLAN.md: Synthetic phrase p50 <= 1.5 s with ElevenLabs
SYNTH_SECONDS = 30.0  # the whole demo clip: target alone, alternating, overlapped, target alone
SILENCE_RMS = 10 ** (-50 / 20)  # out.wav frame RMS below this is silence (TTS output is exactly zero between phrases)
ONSET_GAP_S = 0.25  # silence before a rising edge for it to start a new phrase
PILL_HOLD_S = 0.5  # T-135: web/console/judge-pill.js PILL_HOLD_MS; non-isolated pill states show after this long
TEXT_WER_BAR = 0.90  # T-139: caption WER on the locked clip, set before the runs (other-talker leak counts)
MAX_PILL_CHANGES = 3  # T-135: shown pill state changes from Start to Isolated
JITTER_TARGET_MS = 120.0  # T-135: the /judge worklet's Focus HQ jitter buffer (judge.js JITTER_MS)
MAX_VOICE_FLAPS = 2  # voice-bucket changes allowed in one session (open -> the target's bucket is one)


def pill_state(t: dict) -> str:
    """The /judge pill's state for one telemetry (web/console/judge-pill.js judgePill, camera present)."""
    acq = t.get("acquisition") or {}
    state, detail = acq.get("state"), acq.get("detail") or ""
    if re.match(r"calibrating [\d.]+/[\d.]+ s", detail):
        return "calibrating"
    if state == "locked":
        return "isolated"
    if state == "reacquiring" or (state == "searching" and detail.startswith("lost the person you locked")):
        return "reacquiring"
    if state == "locked-visual" and (t.get("auto_calibrate") or {}).get("state") == "done":
        return "calibrating"  # T-135: calibrated, the ladder on its way to Enrolled ("locking on")
    if state in ("acquiring", "locked-visual"):
        return "speak"
    return "look" if acq else "idle"


def pill_changes(samples: list[tuple[float, str]], hold_s: float = PILL_HOLD_S) -> tuple[int, list[str]]:
    """Shown pill state changes until the first Isolated, with the page's hysteresis (judge-pill.js
    PillHysteresis): ``isolated`` shows at once, any other state after ``hold_s`` unchanged."""
    shown: str | None = None
    pending: tuple[str, float] | None = None
    changes, history = 0, []
    for t, state in samples:
        if shown is None or state == shown:
            pending = None
            if shown is None:
                history.append(state)
            shown = state
        else:
            if state != "isolated":
                if pending is None or pending[0] != state:
                    pending = (state, t)
                if t - pending[1] < hold_s:
                    continue
            pending, shown = None, state
            changes += 1
            history.append(state)
        if shown == "isolated":
            break
    return changes, history


def jitter_underruns(arrivals: list[tuple[float, float]], target_ms: float = JITTER_TARGET_MS) -> int:
    """Underruns of the page's playback jitter buffer (audio-worklet.js) fed audio-out frames that arrive
    at ``arrivals`` = [(seconds, frame seconds)]: playback starts once ``target_ms`` is buffered, plays in
    real time, and an empty buffer is one underrun followed by a re-buffer to the target."""
    target = target_ms / 1000.0
    buffered, clock, playing, underruns = 0.0, None, False, 0
    for t, dur in arrivals:
        if playing and clock is not None:
            buffered -= t - clock
            if buffered < 0:
                underruns += 1
                buffered, playing = 0.0, False
        clock = t
        buffered += dur
        if not playing and buffered >= target:
            playing = True
    return underruns


def is_profile_lock(t: dict) -> bool:
    """T-328: under the team preset's ``audio_only`` (T-168) no face ever shows, so the lock is the saved
    voice the judge profile loads at Start: ``enrollment.source == "profile"``, or the enrolled tier's
    acquisition naming it (``known voice: <name>``, session.py ``_acquisition``). Isolated is then the
    first tick an isolating tier plays after that lock."""
    if (t.get("enrollment") or {}).get("source") == "profile":
        return True
    return ((t.get("acquisition") or {}).get("detail") or "").startswith("known voice")


async def judge_flow(url: str, clip, seconds: float, side: str | None = None) -> dict:
    """Replay ``clip`` with the judge profile; timestamps (s from Start) of lock, calibration, Isolated.

    ``side`` (the manifest's target side): once faces show, lock the face on that side, as the
    presenter does live by looking at (or tapping) the person; ``None`` leaves it to the selector."""
    import websockets

    from hearing.replay import feed

    marks: dict[str, float] = {}
    delays: list[float] = []
    tiers: list[str] = []
    pills: list[tuple[float, str]] = []
    arrivals: list[tuple[float, float]] = []  # (s from Start, audio-out frame s)
    hop_changes = 0
    lock_kind, profile = "face", None
    start = time.monotonic()
    async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
        async def send(message):
            await ws.send(json.dumps(message) if isinstance(message, dict) else message)

        async def receive():
            nonlocal hop_changes, lock_kind, profile
            async for raw in ws:
                if not isinstance(raw, str):
                    if raw[:1] == b"\x03":  # audio out: u8 kind, u32 seq, f64 ts, PCM16 at 16 kHz (the replay's hello)
                        arrivals.append((time.monotonic() - start, (len(raw) - 13) / 2 / 16000))
                    continue
                t = json.loads(raw)
                if t.get("type") != "telemetry":
                    continue
                now = round(time.monotonic() - start, 2)
                faces = t.get("faces") or []
                if side and "tapped" not in marks and len(faces) >= 2:
                    pick = (min if side == "left" else max)(faces, key=lambda f: f["box"][0])
                    await send({"type": "control", "target": {"lock_track_id": pick["track_id"]}})
                    marks["tapped"] = now
                if t.get("profile") == "judge":
                    marks.setdefault("profile", now)
                target = t.get("target") or {}
                enrollment = t.get("enrollment") or {}
                if ((t.get("preset") or {}).get("pins") or {}).get("audio_only"):
                    lock_kind = "profile"  # T-328: no faces under the team preset (T-168); the known voice is the lock
                    if is_profile_lock(t):
                        marks.setdefault("locked", now)
                        profile = profile or enrollment.get("profile")
                elif target.get("track_id") is not None and (not side or target.get("locked")):
                    marks.setdefault("locked", now)  # selected by attention, or the tap landed
                if (t.get("auto_calibrate") or {}).get("state") == "calibrating":
                    marks.setdefault("calibrating", now)
                if (t.get("enrollment") or {}).get("state") == "calibrated":
                    marks.setdefault("calibrated", now)
                acq = (t.get("acquisition") or {}).get("state")
                # T-328: under audio_only the session's acquisition stays "searching" (no face, session.py
                # _acquisition), so Isolated is the isolating tier playing the saved voice
                profile_iso = lock_kind == "profile" and "locked" in marks
                if (acq == "locked" or profile_iso) and t.get("tier") in ISOLATING:
                    marks.setdefault("isolated", now)
                    delays.append(float((t.get("latency_ms") or {}).get("total") or 0.0))
                tiers.append(t.get("tier") or "")
                pills.append((now, pill_state(t)))
                for tier in (t.get("tiers") or {}).get("tiers") or []:
                    for h in tier.get("health") or []:
                        hop_changes = max(hop_changes, int(h.get("hop_changes") or 0))

        receiver = asyncio.create_task(receive())
        try:
            await asyncio.wait_for(feed(clip, send, loop=True, control={"profile": "judge"}), timeout=seconds)
        except TimeoutError:
            pass
        await asyncio.sleep(0.3)
        receiver.cancel()
    lock_to_iso = (marks["isolated"] - marks["locked"]) if {"isolated", "locked"} <= marks.keys() else None
    switches = sum(1 for a, b in itertools.pairwise(tiers) if a != b)
    changes, history = pill_changes(pills)
    return {"pill_changes": changes, "pill_history": history, "hop_changes": hop_changes,
            "jitter_underruns": jitter_underruns(arrivals[50:]),  # after the first second (start-up burst)
            "marks": marks, "lock_kind": lock_kind, "profile": profile, "lock_to_isolated_s": None if lock_to_iso is None else round(lock_to_iso, 2),
            "delay_ms_p50": round(sorted(delays)[len(delays) // 2], 1) if delays else None,
            "tier_switches": switches, "last_tier": tiers[-1] if tiers else None}


def judge_clip(demo: Path, team: bool = False) -> Path:
    """The clip /judge replays: demo-mf-easy (starts with 5.4 s of the target alone, like the live
    flow) when generated, else the GRID two-face clip (two talkers overlapped from the start).
    ``team`` (T-130): demo-team-easy, the teammates' fixture (T-126), when generated."""
    if team and (demo / "demo-team-easy" / "clip.mp4").exists():
        return demo / "demo-team-easy" / "clip.mp4"
    easy = demo / "demo-mf-easy" / "clip.mp4"
    return easy if easy.exists() else demo / "grid-two-face.mp4"


def check_team_preset() -> tuple[str, str, str]:
    """T-130: a session given ``control.profile = "judge"`` under ``HEARING_DEMO_PRESET=team`` reports
    the team preset and every pin of ``config/demo-team.yaml`` (offline, passthrough pipelines)."""
    from hearing.server import team
    from hearing.server.session import ServerConfig, Session, load_stage_modules

    crit = "Team preset pins (config/demo-team.yaml on the judge profile)"
    preset = team.load_preset("team")
    load_stage_modules()
    s = Session(ServerConfig(pipelines={"natural": "passthrough", "focus_hq": "tiers"}, synth_feed="mixture",
                             faces="mock", vision_threaded=False), "cpu")
    try:
        replies = s.handle_text(json.dumps({"type": "control", "profile": "judge"}))
        t = replies[-1]
        errors = [m["message"] for m in replies if m.get("type") == "error"]
        pins = (t.get("preset") or {}).get("pins") or {}
        miss = [k for k, ok in {
            "preset": (t.get("preset") or {}).get("name") == "team",
            "name": pins.get("name") == preset["name"] and s._name_match == preset["match"],
            "hold_lock_s": t["attention"].get("hold_lock_s") == float(preset["hold_lock_s"]),
            "synth_feed": t["synth_feed_request"] == preset["synth_feed"],
            "voice_override": t["voice"]["override"] == preset["voice_override"],
            "min_switch_s": f"min_switch_s={float(preset['min_switch_s'])}" in s.active_spec
                            or s.config.pipelines["focus_hq"] != "tiers",
            "hq_backend": pins.get("hq_backend") in ("local", "remote"),
        }.items() if not ok]
    finally:
        s.close()
    detail = (f"HQ {pins.get('hq_backend')} ({pins.get('hq_reason')}); guard {pins.get('same_bucket_guard')}; "
              f"ASR {pins.get('asr')}; name {pins.get('name')}/{pins.get('match')}")
    if miss or errors:
        return "FAIL", crit, f"missing {miss}; errors {errors}; {detail}"
    return "PASS", crit, detail


def wait_warm(port: int, timeout_s: float = 60.0) -> None:
    """Until ``/health`` says the models are warm (``hearing.server.warm``), at most ``timeout_s``."""
    import urllib.request

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            if json.loads(r.read()).get("warm") != "warming":
                return
        time.sleep(0.5)


def check_flow(clip_path: Path, seconds: float, runs: int) -> tuple[str, str, list[dict]]:
    from hearing.replay import Clip
    from scripts import demo_regress as dr

    clip = Clip.load(clip_path)
    manifest = clip_path.parent / "manifest.json"
    side = (json.loads(manifest.read_text()).get("target") or {}).get("side") if manifest.exists() else None
    log_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"judge-check-{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    proc, port = dr.start_server(log_dir, log_dir / "server.log")
    try:
        wait_warm(port)  # as demo.sh --judge does before it prints the ready line
        results = [asyncio.run(judge_flow(f"ws://127.0.0.1:{port}/ws/session", clip, seconds,
                                          side if isinstance(side, str) else None)) for _ in range(runs)]
    finally:
        dr.stop_server(proc)
    times = [r["lock_to_isolated_s"] for r in results]
    ok = all(t is not None and t <= ISOLATED_BUDGET_S for t in times)
    # T-135: a calm demo: the pill settles in <= 3 shown changes, the tse hop never moves, the page's
    # 120 ms jitter buffer never underruns on the audio the server sent
    ok = ok and all(r["pill_changes"] <= MAX_PILL_CHANGES and r["hop_changes"] == 0 and r["jitter_underruns"] == 0
                    for r in results)
    calm = "; ".join(f"pill changes {r['pill_changes']} ({'>'.join(r['pill_history'])}), tse hop_changes "
                     f"{r['hop_changes']}, jitter underruns {r['jitter_underruns']} @ {JITTER_TARGET_MS:g} ms"
                     for r in results)
    shown = ", ".join("never" if t is None else f"{t:.1f}" for t in times)
    lock = ", ".join(f"{r['marks'].get('locked', float('nan')):.1f}" for r in results)
    kinds = {r.get("lock_kind", "face") for r in results}
    if "profile" in kinds:  # T-328: say what the lock was, and which saved voice
        names = sorted({r.get("profile") or "none" for r in results})
        lock += f" (lock = saved voice {'/'.join(names)}, audio_only)"
        if not os.environ.get("HEARING_PROFILES_DIR"):
            lock += " WARN HEARING_PROFILES_DIR unset: no profile, the session plays the room"
    delay = next((r["delay_ms_p50"] for r in results if r["delay_ms_p50"]), None)
    return ("PASS" if ok else "FAIL",
            (f"lock->Isolated {shown} s (budget {ISOLATED_BUDGET_S:g}); Start->lock {lock} s; "
             f"delay {delay if delay is not None else '-'} ms; {runs} replay(s) of {seconds:g} s of {clip_path.parent.name}; "
             f"{calm}"), results)


def tts_engine() -> str:
    """The TTS the demo server picks from the environment (``hearing.synth.tts.get_tts('auto')``)."""
    from hearing.synth import tts

    if os.environ.get("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    return "kokoro" if tts.local_available() else "mock"


def session_rows(d: Path) -> list[dict]:
    return [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines() if line.strip()]


def out_audio(d: Path, rows: list[dict]):
    """``(samples, t_of)``: the session's out.wav as float and a function from sample index to capture-clock
    seconds, built from the capture timestamps of the audio-out frames in ``events.jsonl``."""
    import wave

    import numpy as np

    with wave.open(str(d / "out.wav")) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    starts, stamps, n = [], [], 0
    for r in rows:
        e = r["event"]
        if r.get("dir") == "out" and e.get("type") == "binary" and e.get("kind") == 3:
            starts.append(n)
            stamps.append(float(e["capture_ts"]))
            n += int(e["bytes"]) // 2

    def t_of(i: int) -> float:
        k = max(0, int(np.searchsorted(starts, i, side="right")) - 1)
        return stamps[k] + (i - starts[k]) / sr

    return x, t_of, sr


def onsets(x, t_of, sr: int, gap_s: float = ONSET_GAP_S) -> list[float]:
    """Capture-clock times of every rising edge of speech in ``x`` after at least ``gap_s`` of silence."""
    import numpy as np

    hop = sr // 100
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) if n else np.zeros(0)
    active = rms > SILENCE_RMS
    out, quiet = [], gap_s * 100  # the stream starts silent
    for i, a in enumerate(active):
        if a:
            if quiet >= gap_s * 100:
                out.append(t_of(i * hop))
            quiet = 0
        else:
            quiet += 1
    return out


def phrase_latencies(d: Path) -> dict:
    """Per final caption of a synthetic session: end of its last word (``t1``) -> the next speech onset in
    out.wav that no earlier phrase used. A phrase queued behind the previous one waits for the next onset,
    which is what the wearer hears. Also the captions' hygiene and the voice bucket's changes."""
    rows = session_rows(d)
    finals = [r["event"] for r in rows if r.get("dir") == "out" and r["event"].get("type") == "caption"
              and r["event"].get("final")]
    x, t_of, sr = out_audio(d, rows)
    edges = onsets(x, t_of, sr)
    lat, queued, used = [], 0, -1
    for c in sorted(finals, key=lambda c: c["t1"]):
        nxt = next((j for j in range(used + 1, len(edges)) if edges[j] >= c["t1"] - 0.05), None)
        if nxt is None:
            continue
        used = nxt
        lat.append(edges[nxt] - c["t1"])
        queued += edges[nxt] - c["t1"] > 3.0
    buckets = [(r["event"].get("voice") or {}).get("bucket") for r in rows
               if r.get("dir") == "out" and r["event"].get("type") == "telemetry"]
    buckets = [b for b in buckets if b]
    flaps = sum(1 for a, b in itertools.pairwise(buckets) if a != b)
    tel = [(r["event"].get("latency_ms") or {}).get("synth_phrase_p50") for r in rows
           if r.get("dir") == "out" and r["event"].get("type") == "telemetry"]
    tel = [t for t in tel if t]
    return {"finals": len(finals), "latencies_s": [round(v, 3) for v in lat], "late": queued,
            "empty_captions": sum(1 for c in finals if not re.search(r"\w", c.get("text", ""))),
            "voice_buckets": sorted(set(buckets)), "voice_flaps": flaps,
            "engine_p50_ms": tel[-1] if tel else None}


def caption_wer(finals: list[dict], manifest: dict, base: float = 0.0) -> tuple[float, int, int]:
    """Caption WER against the manifest's target sentences (``who == "target"``): corpus WER of the
    final captions' text vs the target sentences whose end the captions have reached (the clip loops
    every ``duration_s``). Returns ``(wer, reference_words, caption_words)``. The other talker's words
    that leak into a caption count as insertions."""
    from eval.metrics import wer_batch

    if not finals:
        return float("nan"), 0, 0
    dur = float(manifest["duration_s"])
    horizon = max(c["t1"] for c in finals) - base + 0.5
    ref = [(k * dur + s["start_s"], s["text"]) for k in range(int(horizon // dur) + 1)
           for s in manifest["sentences"] if s["who"] == "target" and k * dur + s["end_s"] <= horizon]
    ref_text = " ".join(t for _, t in sorted(ref))
    hyp_text = " ".join(c["text"] for c in sorted(finals, key=lambda c: c["t0"]))
    from eval.metrics import normalize_text

    return (wer_batch([hyp_text], [ref_text]), len(normalize_text(ref_text).split()), len(normalize_text(hyp_text).split()))


def check_synthetic_text(clip_path: Path, seconds: float = SYNTH_SECONDS) -> list[tuple[str, str, str]]:
    """Replay the clip in Synthetic then Text with the manifest's target locked (T-117): Synthetic
    first-word latency (ASR final -> first TTS sample in out.wav) and Text caption WER."""
    from hearing.replay import Clip
    from scripts import demo_regress as dr
    from scripts.replay import OracleLock

    manifest = json.loads((clip_path.parent / "manifest.json").read_text())
    clip = Clip.load(clip_path)
    log_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"judge-check-synth-{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    engine = tts_engine()
    proc, port = dr.start_server(log_dir, log_dir / "server.log")
    sessions: dict[str, Path] = {}
    try:
        wait_warm(port)
        for mode in ("synthetic", "text"):
            before = {p.name for p in log_dir.iterdir() if p.is_dir()}
            oracle = OracleLock({"target": manifest.get("target")})
            asyncio.run(dr.replay(f"ws://127.0.0.1:{port}/ws/session", clip, mode, seconds, oracle))
            time.sleep(1.0)
            new = sorted(p for p in log_dir.iterdir() if p.is_dir() and p.name not in before)
            if new:
                sessions[mode] = new[-1]
    finally:
        dr.stop_server(proc)
    rows = []
    if "synthetic" not in sessions:
        rows.append(("FAIL", "Synthetic first-word latency", "no session recorded"))
    else:
        rows.extend(synthetic_rows(phrase_latencies(sessions["synthetic"]), engine, seconds, clip_path.parent.name))
    if "text" not in sessions:
        rows.append(("FAIL", "Text caption WER", "no session recorded"))
    else:
        rows.append(text_row(session_rows(sessions["text"]), manifest, seconds, clip_path.parent.name))
    return rows


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)


def synthetic_rows(stats: dict, engine: str, seconds: float, clip_name: str) -> list[tuple[str, str, str]]:
    """The Synthetic first-word-latency line (PASS at p50 <= the engine's bar) and the hygiene line."""
    lat, bar = stats["latencies_s"], SYNTH_P50_S[engine]
    if not lat:
        rows = [("FAIL", f"Synthetic first-word latency p50 <= {bar:g} s ({engine})",
                 f"no spoken phrase in out.wav ({stats['finals']} final captions, engine {engine})")]
    else:
        p50, p95 = percentile(lat, 0.5), percentile(lat, 0.95)
        late = f"; {stats['late']} phrase(s) waited > 3 s behind the previous one" if stats["late"] else ""
        rows = [("PASS" if p50 <= bar else "FAIL", f"Synthetic first-word latency p50 <= {bar:g} s ({engine})",
                 (f"p50 {p50:.2f} s, p95 {p95:.2f} s over {len(lat)} phrases of {stats['finals']} final captions; "
                 f"engine {engine}{' (offline mock: not a latency)' if engine == 'mock' else ''}, ASR final (caption t1) -> "
                 f"first non-silent TTS sample in out.wav; {seconds:g} s of {clip_name}, target locked"
                  f"{late}"))]
    flaps = stats["voice_flaps"]
    ok = not stats["empty_captions"] and flaps <= MAX_VOICE_FLAPS
    rows.append(("PASS" if ok else "FAIL", "Synthetic hygiene (no empty captions, voice steady)",
                 (f"{stats['empty_captions']} empty caption(s); voice buckets "
                  f"{stats['voice_buckets'] or 'not reported (no apparent-voice estimate in telemetry)'}, "
                  f"{flaps} change(s) (allowed {MAX_VOICE_FLAPS})")))
    return rows


def text_row(rows: list[dict], manifest: dict, seconds: float, clip_name: str) -> tuple[str, str, str]:
    finals = [r["event"] for r in rows if r.get("dir") == "out" and r["event"].get("type") == "caption"
              and r["event"].get("final")]
    base = next((float(r["event"]["capture_ts"]) for r in rows if r.get("dir") == "in"
                 and r["event"].get("type") == "binary" and r["event"].get("kind") == 1), 0.0)
    w, n_ref, n_hyp = caption_wer(finals, manifest, base)
    empty = sum(1 for c in finals if not re.search(r"\w", c.get("text", "")))
    loops = sum(1 for c in finals if repetition_loop(c.get("text", "")))
    shown = "n/a" if math.isnan(w) else f"{w:.0%}"
    ok = bool(finals) and not empty and not loops and not math.isnan(w) and w <= TEXT_WER_BAR
    return ("PASS" if ok else "FAIL", f"Text caption WER <= {TEXT_WER_BAR:.0%} vs the target transcript",
            (f"{shown} ({n_hyp} caption words vs {n_ref} target words, {len(finals)} final captions, {empty} empty, "
             f"{loops} repetition loop(s)); {seconds:g} s of {clip_name}, target locked"))


def repetition_loop(text: str) -> bool:
    """A caption still holding Whisper's repetition loop: the ASR's collapse would shorten it (T-139)."""
    from hearing.synth.asr import collapse_repeats

    return collapse_repeats(text) != " ".join(text.split())


REGRESS_SECONDS = 30.0  # T-100: the locked 30 s run is the Focus HQ number we present
# Focus HQ bars per backend (T-100). PLAN's +8 dB is the GPU bake-off bar; the Mac ladder (gate +
# enrolled tiers on MPS) is judged on what the locked replay of demo-mf-easy measures (+3.7 dB).
BARS = {
    "gpu": {"si_sdri_db": 8.0},
    "mac": {"si_sdri_db": 3.0, "leak_db": -12.0, "first_isolated_s": 8.0},
}


def bar_text(backend: str) -> str:
    b = BARS[backend]
    parts = [f"SI-SDRi >= {b['si_sdri_db']:+g} dB"]
    if "leak_db" in b:
        parts.append(f"leak <= {b['leak_db']:g} dB")
    if "first_isolated_s" in b:
        parts.append(f"Isolated <= {b['first_isolated_s']:g} s")
    return f"{'GPU backend' if backend == 'gpu' else 'Mac ladder'} bar: " + ", ".join(parts)


def meets_bar(row: dict, backend: str) -> list[str]:
    """What ``row`` (a demo_regress score) misses of the ``backend`` bar; empty when it passes."""
    b, miss = BARS[backend], []
    sdri, leak, iso = row.get("si_sdri_db"), row.get("leak_db"), row.get("first_isolated_s")
    if sdri is None or sdri < b["si_sdri_db"]:
        miss.append(f"SI-SDRi {sdri} < {b['si_sdri_db']:+g}")
    if "leak_db" in b and (leak is None or leak > b["leak_db"]):
        miss.append(f"leak {leak} > {b['leak_db']:g}")
    if "first_isolated_s" in b and (iso is None or iso > b["first_isolated_s"]):
        miss.append(f"Isolated {'never' if iso is None else iso} > {b['first_isolated_s']:g} s")
    return miss


def regress_rows(data: dict, backend: str, name: str) -> list[tuple[str, str, str]]:
    """PASS/FAIL lines for one demo_regress result (``data`` = its JSON) against ``backend``'s bar."""
    from scripts import session_score as ss

    rows = {r["mode"]: r for r in data["scores"]}
    locked = "locked to the manifest target" if data.get("target_locked") else "target NOT locked"
    lines = []
    for mode, crit in (("natural", "Natural on the clip (demo_regress, locked 30 s)"),
                       ("focus_hq", "Focus HQ on the clip (demo_regress, locked 30 s)")):
        r = rows.get(mode)
        if r is None:
            lines.append(("FAIL", crit, "no session recorded"))
            continue
        agree = r.get("target_agreement")
        miss = meets_bar(r, backend) if mode == "focus_hq" else []
        if ss.wrong_target(agree):
            miss.append(f"WRONG TARGET (agreement {agree:.0%})")
        if data.get("regressions"):
            miss.append("regression: " + "; ".join(data["regressions"]))
        detail = (f"SI-SDRi {r.get('si_sdri_db')} dB, leak {r.get('leak_db')} dB, true dropouts "
                  f"{r.get('true_dropout_s')} s, Isolated at {r.get('first_isolated_s')} s, delay {r.get('delay_ms')} ms, "
                  f"target agreement {'-' if agree is None else format(agree, '.0%')}; {locked}")
        if mode == "focus_hq":
            detail += f"; {bar_text(backend)}"
            free = data.get("free_selection")
            if free:
                fa = free.get("target_agreement")
                detail += (f"; free selection (not judged): SI-SDRi {free.get('si_sdri_db')} dB, agreement "
                           f"{'-' if fa is None else format(fa, '.0%')}")
        if miss:
            detail += " -- " + ", ".join(miss)
        lines.append(("FAIL" if miss else "PASS", crit, f"{detail} ({name})"))
    return lines


NO_HW_SKIP = "skipped (no hardware): --no-hw, a human runs it"


def check_regress(backend: str, no_baseline: bool = False, clip: Path | None = None,
                  profile: str | None = None) -> list[tuple[str, str, str]]:
    from hearing.audio.io import (
        PIPELINE_SR,  # noqa: F401  (fail early if the package is broken)
    )

    results = Path(os.environ.get("HEARING_RESULTS_DIR") or ROOT / "eval/results")
    before = set(results.glob("demo_regress_*.json"))
    # The manifest target is locked by default (T-100): an unlocked run scores the wrong person.
    out = subprocess.run([sys.executable, str(ROOT / "scripts/demo_regress.py"), "--modes", "natural,focus_hq",
                          "--seconds", f"{REGRESS_SECONDS:g}", *(["--no-baseline"] if no_baseline else []),
                          *(["--clip", str(clip)] if clip else []), *(["--profile", profile] if profile else [])],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    newest = sorted(set(results.glob("demo_regress_*.json")) - before)
    if not newest:
        return [("FAIL", "demo_regress", f"no result written (exit {out.returncode}): {out.stderr[-300:]}")]
    return regress_rows(json.loads(newest[-1].read_text()), backend, newest[-1].name)


LATENCY_CRIT = "Mic-to-ear on the demo devices (hearing latency-test; Bluetooth <= 300/350 ms, wired <= 100 ms)"
# T-100: latency-test takes the live device flags, not --mode (it measures the natural default pipeline).
LATENCY_CMD = ["-m", "hearing", "latency-test", "--json"]
WIRED_WORDS = ("headphone", "external", "usb", "line out", "wired", "dac", "interface")
WIRELESS_WORDS = ("airpods", "bluetooth", "beats", "buds", "speaker", "display", "airplay", "hdmi")
BT_CLICKS = 10  # T-143: Bluetooth jitter needs more than the default 5 clicks for a p95
BT_P95_MARGIN_MS = 50.0  # p95 bar = bt_round_trip_max_ms + this (300 -> 350 ms)


def is_wired(name: str | None) -> bool:
    """Whether an output device name looks like wired headphones (``External Headphones``, a USB
    DAC) rather than Bluetooth or the laptop speakers; a wired output keeps the 100 ms budget."""
    n = (name or "").lower()
    return bool(n) and not any(w in n for w in WIRELESS_WORDS) and any(w in n for w in WIRED_WORDS)


def device_names() -> list[str]:
    """Every audio device's name, from a child process (PortAudio stays out of this one)."""
    code = "from hearing.audio.devices import list_devices; print('\\n'.join(d.name for d in list_devices()))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    return [n for n in out.stdout.splitlines() if n.strip()] if out.returncode == 0 else []


def _match(needle: str, names: list[str]) -> str | None:
    """The device whose name is ``needle`` (case/accent-free), else the first that contains it."""
    folded = _fold(needle)
    return next((n for n in names if _fold(n) == folded), None) or next((n for n in names if folded in _fold(n)), None)


def p95_nearest_rank(values: list[float]) -> float:
    """Nearest-rank p95: with 10 clicks it is the slowest but one, with 5 the slowest (no interpolation)."""
    v = sorted(values)
    return float(v[max(0, math.ceil(0.95 * len(v)) - 1)])


def check_latency(names=device_names, demo: dict | None = None, run=subprocess.run) -> tuple[str, str]:
    """T-143: ``hearing latency-test`` on the demo devices (``config/demo-team.yaml``, overridable with
    ``HEARING_EXPECT_MIC`` / ``HEARING_EXPECT_OUTPUT``). Bluetooth path (AirPods): median <=
    ``bt_round_trip_max_ms`` (300) and p95 <= that + 50 ms; a wired output keeps the 100 ms bar."""
    demo = _demo(demo)
    found = names()
    mic, out = _match(demo["input"], found), _match(demo["output"], found)
    missing = [f"{label} {want!r}" for label, want, got in (("mic", demo["input"], mic), ("output", demo["output"], out))
               if got is None]
    if missing:
        return "FAIL", (f"demo {' and '.join(missing)} not connected (devices: {', '.join(found) or 'none found'}); "
                        "pair/connect the AirPods (Bluetooth menu) and rerun")
    if BUILTIN_SPEAKERS.search(out):
        return "FAIL", f"demo output {out!r} is the laptop speakers: they feed back into the mic"
    bt = bool(BLUETOOTH.search(out))
    path = "Bluetooth path" if bt else "wired path"
    extra = ["--clicks", str(BT_CLICKS), "--sr", "48000"] if bt else []  # A2DP and the built-in mic both run at 48 kHz
    cmd = [sys.executable, *LATENCY_CMD, "--in", mic, "--out", out, *extra]
    res = run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    try:
        d = json.loads(res.stdout)
    except json.JSONDecodeError:
        return "FAIL", f"{path}: latency-test did not run (exit {res.returncode}): {res.stderr.strip()[-200:]}"
    ms, delays = d.get("round_trip_ms"), d.get("delays_ms") or []
    if ms is None or not delays:
        return "FAIL", (f"{path}: no click heard on {mic} from {out} ({d.get('clicks_found')}/{d.get('clicks_sent')}); "
                        "hold one AirPod at the laptop's mic while it runs")
    p95 = p95_nearest_rank(delays)
    if bt:
        med_max, p95_max = demo["bt_round_trip_max_ms"], demo["bt_round_trip_max_ms"] + BT_P95_MARGIN_MS
        bad = [f"median {ms:g} > {med_max:g} ms"] * (ms > med_max) + [f"p95 {p95:g} > {p95_max:g} ms"] * (p95 > p95_max)
        bars = f"median <= {med_max:g}, p95 <= {p95_max:g} ms"
    else:
        bad = [f"median {ms:g} > {NATURAL_LATENCY_MS:g} ms"] * (ms > NATURAL_LATENCY_MS)
        bars = f"median <= {NATURAL_LATENCY_MS:g} ms"
    detail = (f"{path}: mic-to-ear median {ms:g} ms, p95 {p95:g} ms over {len(delays)} clicks, {mic} -> {out} "
              f"({bars}){'; ' + ', '.join(bad) if bad else ''}")
    return ("FAIL" if bad else "PASS"), detail


BLUETOOTH = re.compile(r"airpods|bluetooth|(^|[^a-z0-9])bt([^a-z0-9]|$)", re.IGNORECASE)  # web/console/devices.js
MIC_MODE_REMINDER = ("not detectable from a script: open Control Centre > Mic Mode while the page records "
                     "and pick Standard (Voice Isolation and Wide Spectrum fight our own isolation)")


BUILTIN_SPEAKERS = re.compile(r"(macbook|imac|built-?in|internal).*speakers?|^(default - )?speakers?$", re.IGNORECASE)  # devices.js


def _fold(text: str) -> str:
    """Case- and accent-free form, so ``RØDE`` matches ``RODE VideoMic``."""
    import unicodedata

    text = text.replace("Ø", "O").replace("ø", "o")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold()


def fetch_client_devices(url: str) -> dict | None:
    import urllib.request

    with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3) as r:
        return json.load(r).get("client_devices")


AIRPODS_CRIT = "AirPods settings (ANC on, Transparency/Conversation Awareness off)"
AIRPODS_REMINDER = ("not readable from a script: Control Centre > AirPods: Noise Control = Noise Cancellation (not "
                    "Transparency/Adaptive), Conversation Awareness off, Spatial Audio off; Bluetooth settings > AirPods: "
                    "Automatic Ear Detection off")


IPHONE = re.compile(r"iphone", re.IGNORECASE)  # T-171: the iPhone over USB (Continuity), never Bluetooth
IPHONE_CRIT = "iPhone mic over USB: stationary, screen off, Mic Mode Standard"
IPHONE_REMINDER = ("not readable from a script: the iPhone stays cabled (USB, never Bluetooth = HFP), held still with "
                   "the screen off (unlocking it pauses the audio), Control Centre > Mic Mode = Standard")
LABEL_CRIT = "Demo mic label (chosen)"


def chosen_mic(demo: dict | None = None) -> tuple[str, str]:
    """T-171: (the demo mic label judge_check expects, where it came from): ``HEARING_EXPECT_MIC``, else
    ``hearing.server.team.input_device`` (``HEARING_INPUT_DEVICE``, then ``config/demo-team.yaml``)."""
    if os.environ.get("HEARING_EXPECT_MIC"):
        return os.environ["HEARING_EXPECT_MIC"], "HEARING_EXPECT_MIC"
    if demo is not None:
        return demo["input"], "caller"
    from hearing.server.team import input_device

    return input_device()


def label_rows(demo: dict | None = None) -> list[tuple[str, str, str]]:
    """The chosen mic label as a PASS row (Bluetooth label: FAIL, it would force HFP), plus the iPhone reminder."""
    label, source = chosen_mic(demo)
    if BLUETOOTH.search(label):
        rows = [("FAIL", LABEL_CRIT, f"{label!r} (from {source}) is a Bluetooth mic: it drops the AirPods to HFP")]
    else:
        rows = [("PASS", LABEL_CRIT, f"{label!r} (from {source})")]
    if IPHONE.search(label):
        rows.append(("SKIP", IPHONE_CRIT, IPHONE_REMINDER))
    return rows


def _demo(demo: dict | None) -> dict:
    """The demo devices (``hearing.server.team.demo_devices``), ``HEARING_EXPECT_MIC`` / ``_OUTPUT`` on top."""
    if demo is None:
        from hearing.server.team import demo_devices

        demo = demo_devices()
    return {**demo, "input": os.environ.get("HEARING_EXPECT_MIC") or demo["input"],
            "output": os.environ.get("HEARING_EXPECT_OUTPUT") or demo["output"]}


def check_devices(url: str, expect: str | None = None, fetch=fetch_client_devices, demo: dict | None = None,
                  required: bool | None = None) -> list[tuple[str, str, str]]:
    """Rows for the mic/output the page is using (T-133, T-143: built-in mic + AirPods); never opens a device.
    No page reporting devices: FAIL when ``required`` (a full rehearsal, or ``HEARING_EXPECT_MIC`` set), else SKIP."""
    given, demo = demo, _demo(demo)
    want_mic, want_out = expect or demo["input"], demo["output"]
    if required is None:
        required = bool(expect or os.environ.get("HEARING_EXPECT_MIC"))
    mic_crit, hfp_crit = f"Demo mic in use ({want_mic})", "Mic is not a Bluetooth (HFP) mic"
    spk_crit, out_crit = "Output is not the laptop speakers", f"Output is the demo output ({want_out})"
    try:
        devs = fetch(url)
    except Exception as exc:  # noqa: BLE001  (no server running is a finding, not a crash)
        devs, why = None, f"no server at {url} ({type(exc).__name__}); start bash scripts/demo.sh --judge"
    else:
        why = f"no page has started the mic on {url}: open /judge or /simple, tap Start, rerun"
    reminders = [("SKIP", "macOS Mic Mode is Standard (not Voice Isolation)", MIC_MODE_REMINDER),
                 ("SKIP", AIRPODS_CRIT, AIRPODS_REMINDER)]
    if not devs:
        miss = "FAIL" if required else "SKIP"
        return [(miss, mic_crit, why), ("SKIP", hfp_crit, why), ("SKIP", spk_crit, why), (miss, out_crit, why), *reminders,
                *label_rows(given)]
    mic, out, missing = devs.get("input") or "", devs.get("output") or "", devs.get("missing") or []
    src = f"(from {devs.get('client') or 'page'})"
    if "mic" in missing:
        mic_row = ("FAIL", mic_crit, f"saved mic NOT FOUND, page fell back to {mic or 'unknown'} {src}")
    elif not mic:
        mic_row = ("FAIL", mic_crit, f"page reported no mic label {src}")
    elif _fold(want_mic) not in _fold(mic):
        mic_row = ("FAIL", mic_crit, f"mic is {mic!r}, expected {want_mic!r}: pick it under D > Microphone {src}")
    else:
        mic_row = ("PASS", mic_crit, f"mic {mic!r} {src}")
    if BLUETOOTH.search(mic):
        hfp_row = ("FAIL", hfp_crit, (f"mic {mic!r} is Bluetooth: it drops the AirPods to HFP (16 kHz, no ANC); "
                                      f"pick {want_mic!r} {src}"))
    else:
        hfp_row = ("PASS", hfp_crit, f"mic {mic or 'unknown'!r} {src}")
    if "output" in missing:
        spk_row = ("FAIL", spk_crit, f"saved output NOT FOUND, playing on {out or 'the system default'} {src}")
    elif BUILTIN_SPEAKERS.search(out):
        spk_row = ("FAIL", spk_crit, f"output {out!r} is the laptop speakers: they feed back into the mic {src}")
    else:
        spk_row = ("PASS", spk_crit, f"output {out or 'system default (label not reported)'!r} {src}")
    if "output" in missing:
        out_row = ("FAIL", out_crit, f"saved output NOT FOUND, playing on {out or 'the system default'} {src}")
    elif _fold(want_out) not in _fold(out):
        out_row = ("FAIL", out_crit, (f"output is {out or 'the system default (label not reported)'!r}, expected "
                                      f"{want_out!r}: pick it under D > Output {src}"))
    else:
        out_row = ("PASS", out_crit, f"output {out!r} {src}")
    return [mic_row, hfp_row, spk_row, out_row, *reminders, *label_rows(given)]


def probe_gpu() -> tuple[bool, str]:
    """(use the GPU, what the probe said)."""
    url = os.environ.get("HEARING_REMOTE_URL")
    if not url:
        return False, "no HEARING_REMOTE_URL: Mac ladder (start the GPU with scripts/gpu_up.sh ilab, then rerun)"
    out = subprocess.run([sys.executable, "-m", "hearing.server.remote", "--url", url, "--seconds", "5"],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    try:
        d = json.loads(out.stdout)
    except json.JSONDecodeError:
        return False, f"GPU probe failed -> Mac ladder ({out.stderr.strip()[-120:]})"
    p95 = (d.get("rtt_ms") or {}).get("p95") if isinstance(d.get("rtt_ms"), dict) else d.get("rtt_p95_ms")
    healthy = bool(d.get("ok", d.get("health_ok")))
    use = healthy and p95 is not None and p95 < GPU_RTT_P95_MS
    return use, f"{'GPU' if use else 'Mac ladder'}: health {'ok' if healthy else 'bad'}, RTT p95 {p95} ms (GPU if < {GPU_RTT_P95_MS:g})"


def check_camera(seconds: float = 5.0) -> tuple[str, str]:
    import cv2  # mediapipe's opencv-contrib

    from hearing.server.vision import VisionEngine

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return "FAIL", "no webcam (camera permission for the terminal?)"
    from hearing.types import StageContext

    engine, ctx = VisionEngine("auto", "auto", threaded=False), StageContext.default()
    locked_at, frames, start = None, 0, time.monotonic()
    try:
        while time.monotonic() - start < seconds:
            ok, bgr = cam.read()
            if not ok:
                continue
            frames += 1
            ok, jpeg = cv2.imencode(".jpg", cv2.resize(bgr, (640, 360)))
            engine.submit(frames, time.time(), jpeg.tobytes())
            engine.poll(ctx)
            if locked_at is None and ctx.target.locked:
                locked_at = time.monotonic() - start
    finally:
        cam.release()
        engine.close()
    if locked_at is None:
        return "FAIL", f"no face locked in {seconds:g} s ({frames} frames): sit in front of the camera"
    return "PASS", f"face locked after {locked_at:.1f} s ({frames} frames in {seconds:g} s)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-only", action="store_true", help="only the judge flow on the replay clip")
    ap.add_argument("--no-hw", action="store_true", help="skip the checks that open the mic, speakers or camera")
    ap.add_argument("--runs", type=int, default=1, help="judge-flow replays")
    ap.add_argument("--seconds", type=float, default=20.0, help="length of each judge-flow replay")
    ap.add_argument("--clip", type=Path, default=None, help="default eval/fixtures/demo/demo-mf-easy/clip.mp4, else grid-two-face.mp4")
    ap.add_argument("--team", action="store_true",
                    help="T-130: the team preset (HEARING_DEMO_PRESET=team) and demo-team-easy as the default clip")
    ap.add_argument("--json", type=Path, default=None, help="also write the results here")
    ap.add_argument("--server", default=f"http://127.0.0.1:{os.environ.get('PORT', '8000')}",
                    help="the running demo server whose /health reports the page's mic and output (T-133)")
    args = ap.parse_args(argv)
    from scripts import session_score as ss

    if args.team:  # every server and demo_regress run below inherits it
        from hearing.server.team import PRESET_ENV, apply_env

        os.environ[PRESET_ENV] = "team"
        apply_env()
    clip = args.clip or judge_clip(ss.demo_dir(), team=args.team)
    rows: list[tuple[str, str, str]] = []

    def row(verdict: str, crit: str, detail: str) -> None:
        rows.append((verdict, crit, detail))
        print(f"{verdict:4s}  {crit}: {detail}", flush=True)

    if args.team:
        row(*_team_row())

    if not clip.exists():
        row("FAIL", "judge flow", f"no clip at {clip}: run .venv/bin/python -m eval.make_fixtures --demo")
        flow = []
    else:
        verdict, detail, flow = check_flow(clip, args.seconds, args.runs)
        row(verdict, "Judge flow on the replay clip (lock -> Isolated <= 6 s)", detail)
    if not args.flow_only:
        if args.no_hw:
            use_gpu = False
        else:
            try:
                use_gpu, gpu_detail = probe_gpu()
            except Exception as exc:  # noqa: BLE001
                use_gpu, gpu_detail = False, f"GPU probe crashed -> Mac ladder ({type(exc).__name__}: {exc})"
        team_kw = {"clip": clip, "profile": "team"} if args.team else {}  # T-130: the team clip, preset on
        for r in check_regress("gpu" if use_gpu else "mac", no_baseline=args.no_hw, **team_kw):
            row(*r)
        if args.no_hw:  # Q-286: agents and dry runs get a clean PASS/SKIP list
            row("SKIP", LATENCY_CRIT, NO_HW_SKIP)
            row("SKIP", "Camera lock on the webcam (5 s)", NO_HW_SKIP)
            row("SKIP", "GPU probe (iLab A100) or Mac ladder", f"{NO_HW_SKIP}; judged on the Mac ladder bar")
        else:
            row(*_named(LATENCY_CRIT, check_latency))
            row(*_named("Camera lock on the webcam (5 s)", check_camera))
            row("PASS", "GPU probe (iLab A100) or Mac ladder", gpu_detail)  # either is the plan
        for r in check_devices(args.server, required=not args.no_hw):
            row(*r)
        row("SKIP", "Target selection >= 90 % on eval/scenes", "measured by eval/scenes (docs/BENCHMARKS.md)")
        if clip.exists() and clip.name == "clip.mp4" and (clip.parent / "manifest.json").exists():
            for r in _synthetic_rows(clip):
                row(*r)
        else:
            row("SKIP", "Synthetic first-word latency / Text WER", f"needs a demo clip with manifest.json (got {clip})")
        row("SKIP", "bench --full table, setup < 10 min", "eval/bench.py --full, scripts/setup.sh on a clean clone")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "flow": flow}, indent=2) + "\n")
    failed = [r for r in rows if r[0] == "FAIL"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} not failing" + (f"; FAIL: {', '.join(r[1] for r in failed)}" if failed else ""))
    return 1 if failed else 0


def _synthetic_rows(clip: Path) -> list[tuple[str, str, str]]:
    try:
        return check_synthetic_text(clip)
    except Exception as exc:  # noqa: BLE001  (a rehearsal check must report, not crash the list)
        return [("FAIL", "Synthetic first-word latency / Text WER", f"{type(exc).__name__}: {exc}")]


def _team_row() -> tuple[str, str, str]:
    try:
        return check_team_preset()
    except Exception as exc:  # noqa: BLE001  (a rehearsal check must report, not crash the list)
        return "FAIL", "Team preset pins (config/demo-team.yaml on the judge profile)", f"{type(exc).__name__}: {exc}"


def _named(crit: str, fn) -> tuple[str, str, str]:
    try:
        verdict, detail = fn()
    except Exception as exc:  # noqa: BLE001  (a rehearsal check must report, not crash the list)
        verdict, detail = "FAIL", f"{type(exc).__name__}: {exc}"
    return verdict, crit, detail


if __name__ == "__main__":
    raise SystemExit(main())
