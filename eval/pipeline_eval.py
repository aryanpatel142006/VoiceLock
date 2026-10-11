"""End-to-end trigger -> isolation eval with wrong-speaker routing (T-132).

    .venv/bin/python -m eval.pipeline_eval                               # mf, mm, ff, mm-rt on this Mac
    .venv/bin/python -m eval.pipeline_eval --url ws://127.0.0.1:18765/ws/session   # same scenes on a server
    .venv/bin/python -m eval.pipeline_eval --pairs mf --runs 2

**Path.** Name Call (``mode: name_call``, T-120/T-123): every scene is replayed in 20 ms frames at real
time through the live session path, either an in-process :class:`hearing.server.session.Session`
(``--url`` unset, as ``hearing serve`` runs it, warm-up included) or a running server's
``/ws/session`` over a WebSocket (the A100 through its tunnel). No offline shortcut: the trigger
comes from the session's own Whisper listener, the enrollment from the name utterance in the *test*
recording itself, and the output is the session's audio-out stream. **Profile path** (``--profile``,
T-125/T-134): ``mode: focus_hq`` with both voices loaded as saved profiles (``control.enroll``
``profile``, target + known negative) from the first window, no name. The stand-in scene ``pp-mm``
is two male GRID talkers (``PROFILE_PAIR``: s1 target, s2 other); each profile is the first
``PROFILE_S`` of the talker's speech, the test take only the rest (disjoint), written to a temp
``$HEARING_PROFILES_DIR``. Isolated = the ``tiers`` ladder plays ``enrolled``. Real teammate takes
(``--team DIR`` with ``profile_target.wav``/``profile_other.wav``) use the same path.

**Scenes** (36 s, babble at 10 dB SNR, SIR 0 dB, the demo clips' speech level): talker A is Kokoro
TTS (it must say "Aryan" in its own voice, see :mod:`eval.namecall_eval`), says the name at 7 s and
then talks on (``namecall_eval.A_LINES``); distractor B is a GRID talker of the pair's second gender
(``association_eval.PAIRS``: mf -> s4, mm -> s2, ff -> s7) talking in ``B_SPANS`` (overlap 10-24 s
and 28.5-36 s). ``mm-rt`` (re-target): B is a second male Kokoro voice that says "Aryan" at
``RT_AT`` = 28.0 s, after which B is the intended speaker. A teammate recording directory
(``--team DIR`` with ``mix.wav`` + per-talker close-talk ``target.wav``/``other.wav`` and a
``manifest.json`` giving ``name_at``/``name_end``) is scored the same way when T-126 lands.

**Metrics** (per scene): trigger hits / misses / false triggers (a detection within
``[name start, name end + 4 s]`` of an expected name is a hit); name end -> ``isolated`` (s);
**wrong-speaker routing** on 0.25 s windows aligned by the output delay (``best_lag``): a window
where the intended talker is active (stem RMS above -20 dB re its speech level) is *wrong* when the
other talker's share of the output (least-squares stem gains x stem RMS) is larger than the
intended one's; reported as seconds, % of target-active time and the longest wrong stretch, from
Isolated on (the go/no-go numbers) and from the name's end on (acquisition included). After a
re-target the intended talker is B, so A still playing counts as wrong ("right person at the wrong
time"). Also: re-target events, dominant-speaker flips, SI-SDRi vs the piecewise intended stem and
the other talker's median leak (dB) once Isolated.

**Go/no-go bar** (fixed before the runs): wrong-speaker <= 5 % of target-active time and no wrong
stretch > 1 s, from Isolated on, for the demo pair (mf).

Results go to ``$HEARING_RESULTS_DIR/pipeline_<stamp>.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from eval import namecall_eval as nc
from hearing.audio import io as aio

SR = 16000
FRAME = SR // 50
WIN_S = 0.25
ACTIVE_DB = -20.0  # a stem is active in a window when its RMS is within 20 dB of its speech level
HIT_AFTER_S = 4.0
RT_AT = 28.0  # B says the name while A pauses (A_LINES: 24.5-27.8 and 29.5-)
PROFILE_PAIR = ("s1", "s2")  # male-male GRID stand-in for the teammates (Q-348)
PROFILE_S = 6.0  # seconds of each talker's speech kept for the profile, never in the test take
PROFILE_SCENE_S = 24.0
PROFILE_A_SPANS = [(0.5, 24.0)]
PROFILE_B_SPANS = [(3.0, 11.0), (14.0, 24.0)]  # overlap with A
B_SPANS_CLEAR = [(0.0, 6.6), (8.4, 10.0), (11.0, 24.0), (28.5, nc.SCENE_S)]  # B pauses for the name
A_VOICES = {"M": "am_michael", "F": "af_heart"}
RT_B_VOICE = "am_echo"  # Whisper hears "Aryan" from it; am_adam's reads as "audience" even alone
B_LINES_RT = [  # mm-rt: B's own lines; the one at RT_AT carries the name
    (0.5, "So the bus was late again this morning and I missed the first talk."),
    (3.6, "Anyway, the keynote was fine."),
    (11.5, "Did anyone bring the extension cord for the table?"),
    (15.0, "I left mine in the car, which is parked really far away."),
    (19.0, "We should also print the poster before the judges come."),
    (RT_AT, "Aryan, over here, it is my turn to ask you something."),
    (31.5, "Can you show me how the headset switches between people?"),
]
GO_WRONG_PCT = 5.0
GO_STRETCH_S = 1.0
SCENES = ("mf", "mm", "ff", "mm-rt")


def _gender(pair: str) -> tuple[str, str]:
    return pair[0].upper(), pair[1].upper()


def _place(lines, voice: str, n: int) -> np.ndarray:
    x = np.zeros(n, np.float32)
    for t, text in lines:
        y = nc._kokoro(text, voice)
        i = round(t * SR)
        y = y[: n - i]
        x[i : i + len(y)] += y
    return x


def _grid_speech(speaker: str, n: int, spans, skip_s: float = 0.0) -> np.ndarray:
    from eval.association_eval import load_talker

    src = load_talker(speaker).speech_audio[round(skip_s * SR):]
    b = np.zeros(n, np.float32)
    k = 0
    for s0, s1 in spans:
        i, end = round(s0 * SR), round(s1 * SR)
        while i < end:  # 3 s GRID chunks with a 0.25 s pause, like namecall_eval's B
            seg = src[(k * 3 * SR) % max(len(src) - 3 * SR, 1):][: 3 * SR][: end - i]
            k += 1
            b[i : i + len(seg)] += seg
            i += len(seg) + round(0.25 * SR)
    return b


def build_scene(name: str, overlap: bool = False) -> dict:
    """``{"mix","stems":{"A","B"},"noise","names":[(t0,t1,who)],"intended":[(t,who)]}`` at 16 kHz."""
    from eval.association_eval import PAIRS

    n = round(nc.SCENE_S * SR)
    retarget = name.endswith("-rt")
    pair = name.split("-")[0]
    ga, _ = _gender(pair)
    a = _place(nc.A_LINES, A_VOICES[ga], n)
    if retarget:
        b = _place(B_LINES_RT, RT_B_VOICE, n)
        b_id = f"kokoro:{RT_B_VOICE}"
    else:
        b = _grid_speech(PAIRS[pair][1], n, nc.B_SPANS if overlap else B_SPANS_CLEAR)
        b_id = f"grid-{PAIRS[pair][1]}"
    items = nc._av_full_items()
    noise = np.concatenate([aio.load(it["dir"] / "noise.wav", SR)[0] for it in items[:16]])
    noise = np.resize(noise, n).astype(np.float32)
    la = nc._speech_level(a)
    b *= la / nc._speech_level(b)
    noise *= la / max(nc._rms(noise), 1e-9) * 10 ** (-10 / 20)
    mix = a + b + noise
    g = 10 ** (-18.5 / 20) / max(nc._speech_level(mix), 1e-9)
    names = [(nc.NAME_AT, nc.NAME_AT + 0.6, "A")]
    intended = [(nc.NAME_AT + 0.6, "A")]
    if retarget:
        names.append((RT_AT, RT_AT + 0.6, "B"))
        intended.append((RT_AT + 0.6, "B"))
    return {"id": name + ("/overlap" if overlap else ""), "mix": (mix * g).astype(np.float32), "stems": {"A": a * g, "B": b * g},
            "noise": noise * g, "names": names, "intended": intended,
            "talkers": {"A": f"kokoro:{A_VOICES[ga]}", "B": b_id}}


def build_profile_scene() -> dict:
    """``pp-mm``: two male GRID talkers, both saved as profiles from speech the test take never uses."""
    from eval.association_eval import load_talker

    n = round(PROFILE_SCENE_S * SR)
    ta, tb = PROFILE_PAIR
    a = _grid_speech(ta, n, PROFILE_A_SPANS, PROFILE_S)
    b = _grid_speech(tb, n, PROFILE_B_SPANS, PROFILE_S)
    items = nc._av_full_items()
    noise = np.concatenate([aio.load(it["dir"] / "noise.wav", SR)[0] for it in items[:16]])
    noise = np.resize(noise, n).astype(np.float32)
    la = nc._speech_level(a)
    b *= la / nc._speech_level(b)
    noise *= la / max(nc._rms(noise), 1e-9) * 10 ** (-10 / 20)
    mix = a + b + noise
    g = 10 ** (-18.5 / 20) / max(nc._speech_level(mix), 1e-9)
    profiles = {"target": load_talker(ta).speech_audio[: round(PROFILE_S * SR)],
                "other": load_talker(tb).speech_audio[: round(PROFILE_S * SR)]}
    return {"id": "pp-mm", "mix": (mix * g).astype(np.float32), "stems": {"A": a * g, "B": b * g},
            "noise": noise * g, "names": [], "intended": [(0.0, "A")], "profiles": profiles,
            "talkers": {"A": f"grid-{ta}", "B": f"grid-{tb}"}}


def write_profiles(scene: dict, root: Path) -> None:
    """The scene's two profile takes as ``root/target/take.wav`` and ``root/other/take.wav``."""
    for who, x in scene["profiles"].items():
        (root / who).mkdir(parents=True, exist_ok=True)
        aio.save(root / who / "take.wav", np.asarray(x, np.float32), SR)


def load_team(path: Path) -> dict:
    """A teammate recording (T-126 layout assumed: mix/target/other wav + manifest name times)."""
    m = json.loads((path / "manifest.json").read_text())
    mix = aio.load(path / "mix.wav", SR)[0]
    a = aio.load(path / "target.wav", SR)[0][: len(mix)]
    b = aio.load(path / "other.wav", SR)[0][: len(mix)]
    names = [(float(x["name_at"]), float(x["name_end"]), x.get("who", "A")) for x in m["names"]]
    intended = [(t1, who) for _, t1, who in names]
    sc = {"id": path.name, "mix": mix, "stems": {"A": a, "B": b}, "noise": mix - a - b, "names": names,
          "intended": intended, "talkers": m.get("talkers", {})}
    if (path / "profile_target.wav").exists() and (path / "profile_other.wav").exists():
        sc["profiles"] = {w: aio.load(path / f"profile_{w}.wav", SR)[0] for w in ("target", "other")}
    return sc


# ---- replay ---------------------------------------------------------------------------------------------

def _tick(t: float, tel: dict) -> dict:
    ncs = tel.get("name_call") or {}
    state = ncs.get("state")
    if state is None and tel.get("mode") == "focus_hq":
        state = "isolated" if (tel.get("tiers") or {}).get("active") == "enrolled" else "locking"
    return {"t": round(t, 2), "tier": tel.get("tier"), "state": state, "tse": tel.get("tse"),
            "delay_ms": (tel.get("tiers") or {}).get("delay_ms"), "detections": ncs.get("detections") or 0,
            "last": ncs.get("last"), "events": ncs.get("events") or [], "engine": ncs.get("engine"),
            "hq_active": tel.get("hq_active"), "device": tel.get("device")}


def _controls(profile: bool = False) -> list[dict]:
    hello = {"type": "hello", "sample_rate": SR, "frame_ms": 20, "video": False, "client": "pipeline_eval"}
    if profile:
        return [hello, {"type": "control", "mode": "focus_hq", "suppression": 100, "hq_backend": "local",
                        "enroll": {"action": "profile", "name": "target", "other": "other"}}]
    return [hello, {"type": "control", "mode": "name_call", "suppression": 100, "name": {"text": nc.NAME, "aliases": []}}]


def replay_local(mix: np.ndarray, profile: bool = False) -> tuple[np.ndarray, list[dict]]:
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
    warm.warm()
    warm.warm_inference()
    sess = Session(ServerConfig(prefs_log=None, faces="none"), pick())
    for m in _controls(profile):
        for r in sess.handle_text(json.dumps(m)) or []:
            if isinstance(r, dict) and r.get("type") == "error":
                raise RuntimeError(f"session refused {m}: {r}")
    outs, ticks = [], []
    t0 = time.monotonic()
    for k, i in enumerate(range(0, len(mix) - FRAME + 1, FRAME)):
        lag = t0 + k * 0.02 - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        for m in sess.handle_binary(pack_frame(KIND_AUDIO_IN, k, i / SR, float_to_pcm16(mix[i : i + FRAME]))):
            if isinstance(m, bytes):
                f = unpack_frame(m)
                if f.kind == KIND_AUDIO_OUT:
                    outs.append(aio.pcm16_to_float(f.payload))
        if k % 5 == 4:
            ticks.append(_tick((i + FRAME) / SR, sess.telemetry()))
    sess.close()
    return (np.concatenate(outs) if outs else np.zeros(0, np.float32)), ticks


async def _replay_ws(url: str, mix: np.ndarray, profile: bool = False) -> tuple[np.ndarray, list[dict]]:
    import websockets

    from hearing.audio.io import float_to_pcm16
    from hearing.server.protocol import (
        KIND_AUDIO_IN,
        KIND_AUDIO_OUT,
        pack_frame,
        unpack_frame,
    )

    outs, ticks = [], []
    pos = {"t": 0.0}
    async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
        async def receive():
            async for m in ws:
                if isinstance(m, bytes):
                    f = unpack_frame(m)
                    if f.kind == KIND_AUDIO_OUT:
                        outs.append(aio.pcm16_to_float(f.payload))
                else:
                    ev = json.loads(m)
                    if ev.get("type") == "telemetry":
                        ticks.append(_tick(pos["t"], ev))

        rx = asyncio.create_task(receive())
        for c in _controls(profile):
            await ws.send(json.dumps(c))
        t0 = time.monotonic()
        for k, i in enumerate(range(0, len(mix) - FRAME + 1, FRAME)):
            lag = t0 + k * 0.02 - time.monotonic()
            if lag > 0:
                await asyncio.sleep(lag)
            pos["t"] = (i + FRAME) / SR
            await ws.send(pack_frame(KIND_AUDIO_IN, k, i / SR, float_to_pcm16(mix[i : i + FRAME])))
        await asyncio.sleep(3.0)  # the delayed tail of the output
        rx.cancel()
    return (np.concatenate(outs) if outs else np.zeros(0, np.float32)), ticks


def replay_ws(url: str, mix: np.ndarray, profile: bool = False) -> tuple[np.ndarray, list[dict]]:
    return asyncio.run(_replay_ws(url, mix, profile))


# ---- scoring --------------------------------------------------------------------------------------------

def detections(ticks: list[dict]) -> list[dict]:
    """One entry per new detection: tick time, word, and whether the session logged it as a re-target."""
    out, prev = [], 0
    for x in ticks:
        if x["detections"] > prev:
            kinds = [e.get("event") for e in x["events"] if e.get("event") in ("heard", "retarget")]
            out.append({"t": x["t"], "word": (x["last"] or {}).get("word"),
                        "latency_ms": (x["last"] or {}).get("latency_ms"),
                        "retarget": bool(kinds) and kinds[-1] == "retarget"})
            prev = x["detections"]
    return out


def trigger_score(names: list[tuple], dets: list[dict]) -> dict:
    used, hits = set(), []
    for t0, t1, who in names:
        k = next((j for j, d in enumerate(dets) if j not in used and t0 <= d["t"] <= t1 + HIT_AFTER_S), None)
        if k is not None:
            used.add(k)
        hits.append({"who": who, "name_end": t1, "hit": k is not None,
                     "detected_at": dets[k]["t"] if k is not None else None})
    return {"expected": len(names), "hits": sum(h["hit"] for h in hits), "misses": sum(not h["hit"] for h in hits),
            "false_triggers": len(dets) - len(used), "per_name": hits}


def _intended_at(intended: list[tuple], t: float) -> str | None:
    who = None
    for t0, w in intended:
        if t >= t0:
            who = w
    return who


def routing(scene: dict, y: np.ndarray, t_from: float, t_to: float | None = None) -> dict:
    """Wrong-speaker routing of the aligned output ``y`` on 0.25 s windows from ``t_from`` on."""
    a, b, nz = scene["stems"]["A"], scene["stems"]["B"], scene["noise"]
    lvl = {"A": nc._speech_level(a), "B": nc._speech_level(b)}
    n = min(len(y), len(a))
    end = n if t_to is None else min(n, round(t_to * SR))
    win = round(WIN_S * SR)
    thr = 10 ** (ACTIVE_DB / 20)
    active = wrong = 0
    stretch = longest = 0
    flips, last_dom = 0, None
    leak = []
    for j in range(round(t_from * SR), end - win + 1, win):
        who = _intended_at(scene["intended"], j / SR)
        if who is None:
            continue
        other = "B" if who == "A" else "A"
        sl = slice(j, j + win)
        st = {"A": a[sl], "B": b[sl]}
        g = nc._lsq_gains(y[sl], [st["A"], st["B"], nz[sl]])
        e = {"A": abs(g[0]) * nc._rms(st["A"]), "B": abs(g[1]) * nc._rms(st["B"])}
        if nc._rms(st[other]) > thr * lvl[other]:
            leak.append(20 * np.log10(max(abs(g[1 if other == "B" else 0]), 1e-4)))
        if nc._rms(st[who]) <= thr * lvl[who]:
            continue
        active += 1
        dom = who if e[who] >= e[other] else other
        if last_dom is not None and dom != last_dom:
            flips += 1
        last_dom = dom
        if dom != who:
            wrong += 1
            stretch += 1
            longest = max(longest, stretch)
        else:
            stretch = 0
    act_s = active * WIN_S
    return {"target_active_s": round(act_s, 2), "wrong_s": round(wrong * WIN_S, 2),
            "wrong_pct": round(100 * wrong / active, 1) if active else None,
            "longest_wrong_s": round(longest * WIN_S, 2), "dominant_flips": flips,
            "other_leak_db_median": round(float(np.median(leak)), 1) if leak else None}


def _delay_segments(ticks: list[dict], iso: float) -> list[tuple[float, float]]:
    """``(start_s, delay_s)`` each time the ladder's reported playout delay changed once Isolated."""
    segs: list[tuple[float, float]] = []
    for x in ticks:
        if x["t"] >= iso and x["delay_ms"] is not None and (not segs or segs[-1][1] != x["delay_ms"] / 1000):
            segs.append((x["t"], x["delay_ms"] / 1000))
    return segs


def _segment_lag(out: np.ndarray, ref: np.ndarray, i0: int, i1: int, guess: float) -> int:
    """Lag (samples, 10 ms steps within 0.3 s of ``guess``) that best lines ``out`` up with ``ref[i0:i1]``."""
    r = ref[i0:i1]
    best, lag = -1.0, round(guess * SR)
    for d in range(max(0, round((guess - 0.3) * SR)), round((guess + 0.3) * SR) + 1, SR // 100):
        y = out[i0 + d : i1 + d]
        if len(y) < len(r) or len(r) < SR // 2:
            break
        c = abs(float(np.dot(y, r))) / (np.linalg.norm(y) * np.linalg.norm(r) + 1e-9)
        if c > best:
            best, lag = c, d
    return lag


def _align(out: np.ndarray, ref: np.ndarray, delay: float, segs: list[tuple[float, float]]) -> tuple[np.ndarray, int]:
    """``out`` shifted onto ``ref``'s timeline: one global ``delay``, or per segment when the playout
    delay changed mid-run (T-089's adaptive hop: one global lag misaligns the rest of the run)."""
    d = round(delay * SR)
    y = out[d:]
    n = min(len(y), len(ref))
    y = np.pad(y[:n], (0, len(ref) - n))
    if len(segs) < 2:
        return y, n
    for k, (t0, _) in enumerate(segs[1:], start=1):
        i0 = round(t0 * SR)
        i1 = round(segs[k + 1][0] * SR) if k + 1 < len(segs) else len(ref)
        lag = _segment_lag(out, ref, i0, i1, segs[k][1] - segs[0][1] + delay)
        seg = out[i0 + lag : i1 + lag]
        y[i0 : i0 + len(seg)] = seg
        y[i0 + len(seg) : i1] = 0.0
    return y, n


def score_run(scene: dict, out: np.ndarray, ticks: list[dict]) -> dict:
    dets = detections(ticks)
    trig = trigger_score(scene["names"], dets)
    name_end = scene["names"][0][1] if scene["names"] else 0.0
    t_start = scene["names"][0][0] if scene["names"] else 0.0
    iso = next((x["t"] for x in ticks if x["state"] == "isolated" and x["t"] >= t_start), None)
    res: dict = {"trigger": trig, "detections": dets, "isolated_at_s": iso,
                 "name_to_isolated_s": round(iso - name_end, 2) if iso is not None else None,
                 "retargets": sum(d["retarget"] for d in dets),
                 "engine": next((x["engine"] for x in reversed(ticks) if x["engine"]), None),
                 "hq_active": next((x["hq_active"] for x in reversed(ticks) if x["hq_active"]), None),
                 "device": next((x["device"] for x in reversed(ticks) if x["device"]), None),
                 "states": nc._state_runs(ticks), "out_s": round(len(out) / SR, 2)}
    if len(scene["names"]) > 1:  # re-target: when the session switched after the 2nd name
        t_rt = scene["names"][1][1]
        rt = next((d["t"] for d in dets if d["retarget"] and d["t"] >= scene["names"][1][0]), None)
        res["retarget_at_s"] = rt
        res["name2_to_retarget_s"] = round(rt - t_rt, 2) if rt is not None else None
    conf = [x["tse"]["confidence"] for x in ticks if iso is not None and x["t"] >= iso
            and (x.get("tse") or {}).get("confidence") is not None]
    res["tse_confidence_median"] = round(float(np.median(conf)), 3) if conf else None
    # the playout delay the ladder reported once Isolated: more than one value = one global lag misaligns part of it
    res["delays_ms_isolated"] = sorted({x["delay_ms"] for x in ticks if iso is not None and x["t"] >= iso
                                        and x["delay_ms"] is not None})
    if iso is None or len(out) < SR:
        return res
    # the piecewise intended reference lines the output up (the stream resampler/hop shift the delay)
    ref = np.zeros_like(scene["stems"]["A"])
    for k, (t0, who) in enumerate(scene["intended"]):
        t1 = scene["intended"][k + 1][0] if k + 1 < len(scene["intended"]) else None
        i0, i1 = round(t0 * SR), (round(t1 * SR) if t1 is not None else len(ref))
        ref[i0:i1] = scene["stems"][who][i0:i1]
    delay = nc.best_lag(out, ref, round(iso * SR))
    y, n = _align(out, ref, delay, _delay_segments(ticks, iso))
    res["delay_s"] = delay
    res["routing_isolated"] = routing(scene, y, iso)
    res["routing_from_name_end"] = routing(scene, y, name_end)
    i0 = round(iso * SR)
    if n - i0 >= SR:
        sin = nc._si_sdr(scene["mix"][i0:n], ref[i0:n])
        sout = nc._si_sdr(y[i0:n], ref[i0:n])
        res.update(si_sdr_in=round(sin, 2), si_sdr_out=round(sout, 2), si_sdri=round(sout - sin, 2))
    r = res["routing_isolated"]
    res["go"] = bool(r["wrong_pct"] is not None and r["wrong_pct"] <= GO_WRONG_PCT and r["longest_wrong_s"] <= GO_STRETCH_S)
    return res


def summary_row(sid: str, backend: str, r: dict) -> str:
    t = r["trigger"]
    ri = r.get("routing_isolated") or {}
    rn = r.get("routing_from_name_end") or {}

    def f(v, unit=""):
        return "n/a" if v is None else f"{v}{unit}"
    return (f"| {sid} | {backend} | {t['hits']}/{t['expected']} | {t['false_triggers']} | {f(r['name_to_isolated_s'], ' s')} | "
            f"{f(ri.get('wrong_s'), ' s')} / {f(ri.get('wrong_pct'), ' %')} | {f(ri.get('longest_wrong_s'), ' s')} | "
            f"{f(rn.get('wrong_s'), ' s')} / {f(rn.get('wrong_pct'), ' %')} | {r['retargets']} / {f(ri.get('dominant_flips'))} | "
            f"{f(r.get('si_sdri'), ' dB')} | {f(ri.get('other_leak_db_median'), ' dB')} | {'GO' if r.get('go') else 'no-go'} |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=",".join(SCENES), help=f"comma list of {SCENES}")
    ap.add_argument("--url", default=None, help="replay through a server's /ws/session instead of in-process")
    ap.add_argument("--team", type=Path, action="append", default=[], help="a teammate recording dir (T-126)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--overlap", action="store_true", help="B keeps talking over the name (namecall_eval spans)")
    ap.add_argument("--profile", action="store_true",
                    help="the Profile path: pp-mm (+ --team dirs with profile takes) as saved profiles, focus_hq")
    args = ap.parse_args(argv)
    if args.profile:
        scenes = [build_profile_scene()] + [load_team(p) for p in args.team]
        missing = [sc["id"] for sc in scenes if "profiles" not in sc]
        if missing:
            print(f"no profile_target.wav/profile_other.wav in {missing}", file=sys.stderr)
            return 2
    else:
        scenes = [build_scene(s, args.overlap) for s in args.pairs.split(",") if s] + [load_team(p) for p in args.team]
    backend = args.url or "local"
    results = []
    for sc in scenes:
        for run in range(args.runs):
            load0 = os.getloadavg()[0]
            if args.profile:  # the session reads $HEARING_PROFILES_DIR when the enroll control arrives
                tmp = Path(tempfile.mkdtemp(prefix="pipeline_eval-profiles-"))
                write_profiles(sc, tmp)
                os.environ["HEARING_PROFILES_DIR"] = str(tmp)
            if args.url:
                out, ticks = replay_ws(args.url, sc["mix"], args.profile)
            else:
                out, ticks = replay_local(sc["mix"], args.profile)
            r = score_run(sc, out, ticks)
            r.update(scene=sc["id"], run=run, backend=backend, talkers=sc["talkers"], load=load0,
                     host=os.uname().nodename)
            results.append(r)
            print(summary_row(sc["id"], "A100" if args.url else r.get("device") or "local", r), flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = nc.results_dir() / f"pipeline_{stamp}.json"
    path.write_text(json.dumps({"backend": backend, "results": results}, indent=1, default=float))
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
