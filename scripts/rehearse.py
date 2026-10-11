"""The final-test sheet the humans run on Oct 10 (T-147): 12 steps, one at a time, Enter to go on.

    bash scripts/rehearse.sh                       # the real rehearsal (opens the mic for the latency step)
    bash scripts/rehearse.sh --gpu                 # same, the GPU was resynced: Isolated bar 6 s (else 8 s Mac)
    bash scripts/rehearse.sh --dry-run --no-hw --session <dir>   # print the steps, build the sheet from a recording

Demo audio path (user decision Oct 9): input 'MacBook Pro Microphone', output AirPods Pro 3 over A2DP picked
explicitly on /judge, ANC on, Transparency/Adaptive/Conversation Awareness/ear detection/spatial off, Mic Mode
Standard. The AirPods mic is never the input (HFP).

Automates what it can: device labels (``hearing devices``), caffeinate, ``scripts/bt_latency.sh``,
``scripts/demo.sh --judge --team`` (with ``HEARING_PROFILES_DIR`` set, so keys 1/2 swap the saved voices) +
``wait_warm``, the ``judge_check`` device rows from the running server.
At the end it reads the ``HEARING_SESSION_LOG`` sessions recorded since the rehearsal started and writes
``results/rehearsal_<stamp>.md``: one row per number with its bar and PASS / FAIL. A number that was not
recorded prints NOT MEASURED, never a default. Each step's start time (Enter) is kept so the numbers of a
step are read from that step's stretch of the recording.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.judge_check import (
    TEXT_WER_BAR,  # T-139's caption bar (0.90), the same as judge_check
)

NOT_MEASURED = "NOT MEASURED"
NO_AUDIO = "NO AUDIO REACHED THE ENGINE"
MIC = "MacBook Pro Microphone"
ISOLATED_BAR_GPU_S = 6.0  # Joseph alone -> Isolated (task bar)
ISOLATED_BAR_MAC_S = 8.0
# Bars set by the humans (T-157, from T-141 / T-155 / the Name Call runs):
OVERTALK_LOCKED_MIN = 0.9  # share of ticks Isolated while Aryan talks over Joseph
SPACE_MIN = 3  # Raw/Selective flips; the switch latency per flip is report-only
NAME_CALL_ATTEMPTS = 4  # clean calls said in the step
NAME_CALL_MIN = 2  # detections of those 4
SWAP_BUDGET_S = 1.5  # T-155: a key 1/2 profile swap plays the new profile in the Enrolled tier within 1.5 s
HQ_TIER = "enrolled"
WRONG_WIN_S = 1.0  # wrong-speaker windows of out.wav (ECAPA needs ~1 s)
WRONG_MIN_DB = -50.0  # quieter windows are silence, not a speaker
PROFILES_DEFAULT = ROOT / "eval" / "fixtures" / "generated" / "profiles"
# Rows read from the session audio: NOT MEASURED when no audio frame ever reached the engine (Q-425).
AUDIO_KEYS = ("rms_db", "peak_db", "isolated_s", "isolated_already", "enroll_ready_s", "pill_changes",
              "pill_sequence", "hq_backend", "hq_delay_ms", "hq_delay_by_tier", "swaps", "overtalk_locked",
              "wrong_speaker", "space_toggles", "space_latency_ms", "name_calls", "caption_words", "caption_wer")


def quiet_on_judge() -> bool:
    """Quiet (key Q) is on /judge: T-149/T-150 merged. Until then the captions step cannot run there."""
    js = ROOT / "web" / "console" / "judge.js"
    return js.exists() and 'key === "q"' in js.read_text()


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    say: str
    auto: str = ""  # what the script runs after Enter


STEPS: list[Step] = [
    Step("devices", "Devices and preflight",
         "AirPods Pro 3 connected as the OUTPUT (A2DP), ANC ON; Transparency, Adaptive, Conversation Awareness, "
         "ear detection and spatial audio OFF; Mic Mode Standard. Input = 'MacBook Pro Microphone', never the "
         "AirPods mic. Bluetooth OFF on both phones and on laptop-b. Zoom/Teams closed. Laptop on power.",
         "lists the device labels (hearing devices), checks caffeinate"),
    Step("latency", "Round trip MacBook mic -> AirPods",
         "Noise Control OFF for this step only. Hold one bud ~2 cm from the mic grille, quiet room, volume ~70 %.",
         "runs scripts/bt_latency.sh (3 runs x 5 clicks)"),
    Step("server", "Server up",
         "ANC back ON. If the GPU was resynced, run scripts/gpu_up.sh first and pass --gpu to this script.",
         "runs scripts/demo.sh --judge --team, waits until /health is warm"),
    Step("judge", "/judge in the foreground",
         "/judge in a foreground tab (never behind another window). Drawer: pick 'MacBook Pro Microphone' and the "
         "AirPods explicitly. Press Start. Then press Enter here.",
         "reads the page's mic and output from /health (judge_check device rows)"),
    Step("isolate", "Joseph alone 3 s",
         "Press Enter, then Joseph speaks alone for 3 s facing the laptop. Watch the pill until Isolated."),
    Step("relearn", "Re-learn live: profile swap 1/2",
         "Press Enter, then press 1 (saved voice 1) while Aryan talks; ~5 s later press 2 (saved voice 2) while "
         "Joseph talks. The pill must read 'known voice: <name>' within 1.5 s each time; end on Joseph."),
    Step("overtalk", "Aryan over Joseph, 20 s, noise from laptop-b",
         "Press Enter. laptop-b plays the cafe noise; Joseph keeps talking; Aryan talks over him for 20 s. "
         "The pill must stay Isolated and you must keep hearing Joseph."),
    Step("space", "Space x3",
         "Press Enter, then click the page and press Space 3 times (Raw / Selective), a few seconds apart."),
    Step("namecall", "Name Call x4",
         "Press Enter, then Shift+N (Name Call). Someone says 'Aryan' in a clean sentence 4 times, ~10 s apart, "
         "and talks normally between the calls without the name."),
    Step("captions", "Quiet captions 60 s",
         ("Press Enter, then Q (Quiet); Joseph reads the caption script for 60 s (--caption-ref file)."
          if quiet_on_judge() else
          "Text/Quiet is not on /judge yet (T-149/T-150 not merged): press Enter and go on. "
          "The caption rows print NOT MEASURED.")),
    Step("bud", "Pull a bud out and back",
         "Press Enter, pull one AirPod out for ~3 s and put it back. Audio must keep playing in both."),
    Step("sign", "Sign the sheet", "Press Enter to write the sheet; then both sign it."),
]


# ---------------------------------------------------------------- recording

def session_start(d: Path) -> float | None:
    """Wall-clock start of a ``hearing.server.tap`` session from its ``<UTC stamp>-<n>`` directory name."""
    try:
        stamp = dt.datetime.strptime(d.name.split("-")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC)
    except ValueError:
        return None
    return stamp.timestamp()


def load_session(d: Path) -> list[dict]:
    """Every wire event of the session, with ``w`` = wall-clock seconds (``t`` when the name has no stamp)."""
    base = session_start(d) or 0.0
    rows = []
    for line in (d / "events.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["w"] = base + float(r.get("t", 0.0))
            rows.append(r)
    return rows


def find_sessions(root: Path, since: float) -> list[Path]:
    """Session dirs under ``root`` started at or after ``since`` (second resolution of the stamp)."""
    if not root.is_dir():
        return []
    out = [p for p in root.iterdir() if (p / "events.jsonl").exists() and (session_start(p) or 0) >= int(since)]
    return sorted(out, key=lambda p: (session_start(p) or 0, p.name))


def ticks_of(rows: list[dict]) -> list[dict]:
    return [dict(r["event"], w=r["w"]) for r in rows if r["dir"] == "out" and r["event"].get("type") == "telemetry"]


def window(items: list[dict], marks: dict[str, float], key: str) -> list[dict] | None:
    """Items inside step ``key``'s stretch (its Enter to the next step's), None when the step has no mark."""
    if key not in marks:
        return None
    keys = [s.key for s in STEPS]
    t0 = marks[key]
    later = [marks[k] for k in keys[keys.index(key) + 1:] if k in marks]
    t1 = min(later) if later else float("inf")
    return [x for x in items if t0 <= x["w"] < t1]


def _changes(values: list) -> list:
    out = []
    for v in values:
        if not out or out[-1] != v:
            out.append(v)
    return out


def _wav(path: Path) -> np.ndarray | None:
    from scripts.session_report import wav

    x, _sr = wav(path)
    return x


def mic_levels(sessions: list[Path]) -> dict:
    """RMS / peak (dBFS) of every recorded ``in.wav`` together; None when no input audio was recorded."""
    parts = [x for d in sessions if (x := _wav(d / "in.wav")) is not None and len(x)]
    if not parts:
        return {"rms_db": None, "peak_db": None}
    x = np.concatenate(parts).astype(np.float64)
    return {"rms_db": float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)),
            "peak_db": float(20 * np.log10(np.abs(x).max() + 1e-9))}


def measure(sessions: list[Path], marks: dict[str, float], caption_ref: str | None, *, embed: Callable | None = None,
            profiles_root: Path | None = None, captions_possible: bool | None = None) -> dict:
    """Every number the sheet reads from the recording; a value is None when it was not recorded."""
    captions_possible = quiet_on_judge() if captions_possible is None else captions_possible
    rows = [r for d in sessions for r in load_session(d)]
    ticks = ticks_of(rows)
    hellos = [dict(r["event"], w=r["w"]) for r in rows if r["dir"] == "in" and r["event"].get("type") == "hello"]
    start = next((h["w"] for h in hellos if h.get("input_device")), hellos[0]["w"] if hellos else None)
    m: dict = {"sessions": [d.name for d in sessions], "ticks": len(ticks)}
    m["mic_label"] = next((h["input_device"] for h in reversed(hellos) if h.get("input_device")), None)
    m["output_label"] = next((h["output_device"] for h in reversed(hellos) if h.get("output_device")), None)
    m.update(mic_levels(sessions))

    def acq(t):
        return (t.get("acquisition") or {}).get("state")

    # seconds to Isolated: from the "Joseph alone" Enter, else from Start (the hello with a mic label)
    iso_ticks = window(ticks, marks, "isolate")
    t0 = marks.get("isolate", start)
    pool = iso_ticks if iso_ticks is not None else ticks
    first_state = next((s for s in (acq(t) for t in pool) if s), None)
    m["isolated_already"] = iso_ticks is not None and first_state == "locked"
    first = next((t["w"] for t in pool if acq(t) == "locked"), None)
    m["isolated_s"] = (None if first is None or t0 is None or m["isolated_already"]
                       else round(first - t0, 2))

    def enroll(t):
        return (t.get("enrollment") or {}).get("state")

    ready = next((t["w"] for t in pool if enroll(t) in ("ready", "calibrated")), None)
    m["enroll_ready_s"] = None if ready is None or t0 is None else round(ready - t0, 2)  # < 0: ready before

    states = [s for s in (acq(t) for t in ticks) if s]
    seq = _changes(states)
    m["pill_changes"] = max(0, len(seq) - 1) if seq else None
    m["pill_sequence"] = seq

    hq = Counter(t.get("hq_active") for t in ticks if t.get("hq_active"))
    m["hq_backend"] = hq.most_common(1)[0][0] if hq else None
    by_tier: dict[str, list[float]] = {}
    for t in ticks:
        tier, d = tier_of(t), delay_of(t)
        if tier and isinstance(d, (int, float)):
            by_tier.setdefault(tier, []).append(float(d))
    m["hq_delay_by_tier"] = {k: float(np.median(v)) for k, v in by_tier.items()}
    m["hq_delay_ms"] = m["hq_delay_by_tier"].get(HQ_TIER)  # F2: only while the HQ tier plays
    m.update(gpu_stats(ticks))
    m["gpu"] = m["hq_backend"] == "remote" or (m["gpu_remote_share"] or 0) > 0.5

    controls = [dict(r["event"], w=r["w"]) for r in rows if r["dir"] == "in" and r["event"].get("type") == "control"]
    rel = window(controls, marks, "relearn")
    m["swaps"] = swap_latencies(controls if rel is None else rel, ticks)

    over = window(ticks, marks, "overtalk")
    m["overtalk_locked"] = (sum(acq(t) == "locked" for t in over) / len(over)) if over else None
    m["wrong_speaker"] = None
    if over:
        target = next((p for t in reversed(over) if (p := profile_of(t))), None)
        m["wrong_speaker"] = wrong_speaker_share(sessions, over[0]["w"], over[-1]["w"], target, embed=embed,
                                                 profiles_root=profiles_root)

    sp = window(ticks, marks, "space")
    sp = ticks if sp is None else sp
    abs_ = _changes([t["ab"] for t in sp if t.get("ab")])
    m["space_toggles"] = max(0, len(abs_) - 1) if abs_ else None
    sp_controls = window(controls, marks, "space")
    m["space_latency_ms"] = ab_latencies(controls if sp_controls is None else sp_controls, ticks)

    nc = window(ticks, marks, "namecall")
    nc = ticks if nc is None else nc
    det = [int(n["detections"]) for t in nc if isinstance(n := t.get("name_call"), dict) and "detections" in n]
    m["name_calls"] = (max(det) - min(det) if nc is not ticks else max(det)) if det else None

    caps = [dict(r["event"], w=r["w"]) for r in rows if r["dir"] == "out" and r["event"].get("type") == "caption"
            and r["event"].get("final")]
    cw = window(caps, marks, "captions")
    cw = caps if cw is None else cw
    text = " ".join(str(c.get("text", "")) for c in cw).strip()
    ct = window(ticks, marks, "captions")
    recorded = bool(ticks if ct is None else ct)  # the step was recorded: no caption is 0 words, not a gap
    recorded = recorded and captions_possible  # F5: no Text/Quiet on /judge, nothing to read
    m["caption_words"] = len(text.split()) if recorded else None
    m["caption_wer"] = None
    if caption_ref and recorded:
        from eval.metrics import wer

        m["caption_wer"] = float(wer(text, caption_ref))
    m["underruns"] = None  # the /judge worklet's underrun count never reaches the server (no wire field)
    # Q-425: session_report's "no audio frame ever reached the engine" (seq null in every tick)
    m["no_audio"] = bool(ticks) and all(t.get("seq") is None for t in ticks)
    if m["no_audio"]:
        m.update(dict.fromkeys(AUDIO_KEYS))
    return m


def remote_playing(t: dict) -> dict | None:
    """The tick's T-156 ``remote`` block when the GPU's audio is what plays, else None."""
    r = t.get("remote")
    return r if isinstance(r, dict) and r.get("source") == "remote" else None


def tier_of(t: dict) -> str | None:
    r = remote_playing(t)
    return r.get("remote_tier") if r else (t.get("tiers") or {}).get("active")


def delay_of(t: dict):
    r = remote_playing(t)
    return r.get("delay_ms") if r else (t.get("tiers") or {}).get("delay_ms")


def profile_of(t: dict) -> str | None:
    """The saved voice the tick's enroller targets (the GPU session's while the GPU plays)."""
    r = remote_playing(t)
    enr = ((r.get("session") or {}).get("enrollment") if r else None) or t.get("enrollment") or {}
    return enr.get("profile")


def gpu_stats(ticks: list[dict]) -> dict:
    """T-156's ``telemetry.remote`` block (Q-425): share of ticks the GPU played, its tier, RTT, device."""
    gpu = [t["remote"] for t in ticks if isinstance(t.get("remote"), dict)]
    if not gpu:
        return dict.fromkeys(("gpu_remote_share", "gpu_tier", "gpu_rtt_ms", "gpu_device"))
    played = [r for r in gpu if r.get("source") == "remote"]
    tiers = Counter(r.get("remote_tier") for r in played if r.get("remote_tier"))
    rtt = [float(p) for r in gpu if isinstance(p := (r.get("rtt_ms") or {}).get("p50"), (int, float))]
    return {"gpu_remote_share": len(played) / len(gpu), "gpu_tier": tiers.most_common(1)[0][0] if tiers else None,
            "gpu_rtt_ms": float(np.median(rtt)) if rtt else None,
            "gpu_device": next((r["remote_device"] for r in reversed(gpu) if r.get("remote_device")), None)}


def swap_latencies(controls: list[dict], ticks: list[dict]) -> list[float] | None:
    """Seconds from each key 1/2 swap (``enroll.action == "profile"``) to the first tick that plays the
    Enrolled tier with that profile (``inf`` when it never did before the next swap); None without a swap."""
    sent = [(c["w"], c["enroll"]["name"]) for c in controls
            if isinstance(c.get("enroll"), dict) and c["enroll"].get("action") == "profile" and c["enroll"].get("name")]
    if not sent:
        return None
    out = []
    for i, (w, name) in enumerate(sent):
        until = sent[i + 1][0] if i + 1 < len(sent) else float("inf")
        hit = next((t["w"] for t in ticks if w <= t["w"] < until and tier_of(t) == HQ_TIER and profile_of(t) == name),
                   None)
        out.append(float("inf") if hit is None else round(hit - w, 2))
    return out


def ab_latencies(controls: list[dict], ticks: list[dict]) -> list[float] | None:
    """Milliseconds from each Space press (``control.ab``) to the first tick reporting the new A/B state."""
    sent = [(c["w"], c["ab"]) for c in controls if c.get("ab")]
    if not sent:
        return None
    out = []
    for w, ab in sent:
        hit = next((t["w"] for t in ticks if t["w"] >= w and t.get("ab") == ab), None)
        out.append(float("inf") if hit is None else round(1000 * (hit - w)))
    return out


def out_stretch(d: Path, t0: float, t1: float) -> np.ndarray | None:
    """The session's out.wav played between wall-clock ``t0`` and ``t1`` (frames placed by their tap time)."""
    return stretch(d, t0, t1)


def stretch(d: Path, t0: float, t1: float, wav: str = "out.wav", direction: str = "out", kind: int = 3) -> np.ndarray | None:
    """The session's ``wav`` between wall-clock ``t0`` and ``t1``, placed by the tap time of its binary frames
    (``out.wav``: kind 3 sent; T-171 ``in.wav``: kind 1 received)."""
    if not (d / wav).exists():
        return None
    x = _wav(d / wav)
    if x is None:
        return None
    starts, n = [], 0
    for r in load_session(d):
        e = r["event"]
        if r["dir"] == direction and e.get("type") == "binary" and e.get("kind") == kind:
            starts.append((r["w"], n))
            n += int(e["bytes"]) // 2
    inside = [i for w, i in starts if t0 <= w <= t1]
    if not inside:
        return None
    end = next((i for w, i in starts if w > t1), len(x))
    return x[inside[0]:end]


def speaker_windows(parts: list[np.ndarray], target: str | None, *, embed: Callable | None = None,
                    profiles_root: Path | None = None) -> list[tuple[str, float]] | None:
    """``(closest saved voice, level dBFS)`` per voiced 1 s window of ``parts`` (ECAPA cosine to each profile's
    mean embedding; windows under ``WRONG_MIN_DB`` are silence and skipped). None when the target, the
    profiles or the ECAPA weights are missing."""
    from hearing.tse.enroll import list_profiles, load_profile, profiles_dir

    root = profiles_root if profiles_root is not None else profiles_dir()
    if not target or root is None or not Path(root).is_dir():
        return None
    names = list_profiles(Path(root))
    if target not in names or len(names) < 2:
        return None
    if embed is None:
        from hearing.server.namecall import ecapa_embedder

        embed = ecapa_embedder()
        if embed is None:
            return None

    def unit(v):
        v = np.asarray(v, np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    refs = {n: unit(np.mean([unit(embed(p)) for p in load_profile(n, Path(root))], axis=0)) for n in names}
    win = int(16000 * WRONG_WIN_S)
    out = []
    for y in parts:
        for k in range(0, len(y) - win + 1, win):
            w = y[k:k + win]
            db = float(20 * np.log10(np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-9))
            if db < WRONG_MIN_DB:
                continue
            e = unit(embed(w))
            out.append((max(refs, key=lambda n: float(np.dot(e, refs[n]))), db))
    return out


def wrong_speaker_share(sessions: list[Path], t0: float, t1: float, target: str | None, *,
                        embed: Callable | None = None, profiles_root: Path | None = None) -> float | None:
    """Share of voiced 1 s windows of what the wearer heard in [t0, t1] whose ECAPA embedding is closer to
    another saved voice than to the target's (report-only, T-141's measure without a clean reference).
    None when the target, the profiles, out.wav or the ECAPA weights are missing."""
    parts = [y for d in sessions if (y := out_stretch(d, t0, t1)) is not None and len(y)]
    if not parts:
        return None
    wins = speaker_windows(parts, target, embed=embed, profiles_root=profiles_root)
    if not wins:
        return None
    return sum(name != target for name, _ in wins) / len(wins)


# ---------------------------------------------------------------- sheet

def fmt(v, spec: str = "", unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return NOT_MEASURED
    return f"{format(v, spec)}{unit}"


def verdict(v, ok: Callable[[object], bool] | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return NOT_MEASURED
    if ok is None:
        return "INFO"
    return "PASS" if ok(v) else "FAIL"


def _ms_list(v: list | None) -> str | None:
    return None if not v else ", ".join("never" if not np.isfinite(x) else f"{x:g}" for x in v)


def output_ok(label) -> bool:
    """The AirPods picked explicitly: 'Default - AirPods' is the system default (F4), never the demo path."""
    s = str(label).strip().casefold()
    return "airpods" in s and not s.startswith(("default", "system default"))


def sheet_rows(m: dict, latency: dict | None, devices: list | None, human: dict, gpu: bool | None) -> list[tuple]:
    """(item, value, bar, verdict) rows of the sheet. Every row prints its measured number next to the verdict."""
    s = (latency or {}).get("summary") or {}
    gpu = m.get("gpu") if gpu is None else gpu
    iso_bar = ISOLATED_BAR_GPU_S if gpu else ISOLATED_BAR_MAC_S
    swaps = m.get("swaps")
    swap_worst = max(swaps) if swaps else None
    hits = None if m.get("name_calls") is None else min(m["name_calls"], NAME_CALL_ATTEMPTS)
    false = None if m.get("name_calls") is None else max(0, m["name_calls"] - NAME_CALL_ATTEMPTS)
    by_tier = m.get("hq_delay_by_tier") or {}
    iso_value = "already locked at mark" if m.get("isolated_already") else fmt(m.get("isolated_s"), ".2f", " s")
    report = "report-only"
    rows = [
        ("Round trip median (MacBook mic -> AirPods)", fmt(s.get("median_ms"), ".0f", " ms"), "<= 300 ms",
         verdict(s.get("median_ms"), lambda v: v <= 300)),
        ("Round trip p95", fmt(s.get("p95_ms"), ".0f", " ms"), "<= 350 ms", verdict(s.get("p95_ms"), lambda v: v <= 350)),
        ("Mic in use (page hello)", fmt(m.get("mic_label")), MIC, verdict(m.get("mic_label"), lambda v: v == MIC)),
        ("Output in use (page hello)", fmt(m.get("output_label")), "AirPods, picked (not 'Default - ...')",
         verdict(m.get("output_label"), output_ok)),
        ("Mic RMS (in.wav)", fmt(m.get("rms_db"), ".1f", " dBFS"), report, verdict(m.get("rms_db"), None)),
        ("Mic peak (in.wav)", fmt(m.get("peak_db"), ".1f", " dBFS"),
         "report-only (T-145's -48 dBFS is the meter's short-window line)", verdict(m.get("peak_db"), None)),
        ("Enrollment ready", fmt(m.get("enroll_ready_s"), ".2f", " s"), "", verdict(m.get("enroll_ready_s"), None)),
        ("Seconds to Isolated (Joseph alone)", iso_value, f"<= {iso_bar:g} s ({'GPU' if gpu else 'Mac'})",
         NOT_MEASURED if m.get("isolated_already") else verdict(m.get("isolated_s"), lambda v: v <= iso_bar)),
        ("Pill changes", fmt(m.get("pill_changes")) + (f" ({' > '.join(m['pill_sequence'][:12])})"
                                                       if m.get("pill_sequence") else ""), "",
         verdict(m.get("pill_changes"), None)),
        ("HQ backend playing", fmt(m.get("hq_backend")), "", verdict(m.get("hq_backend"), None)),
        (f"HQ delay (median while {HQ_TIER} plays)",
         fmt(m.get("hq_delay_ms"), ".0f", " ms") + (f" (per tier: {', '.join(f'{k} {v:.0f} ms' for k, v in by_tier.items())})"
                                                    if by_tier else ""), "", verdict(m.get("hq_delay_ms"), None)),
        ("GPU played (share of ticks with remote stats)", fmt(None if m.get("gpu_remote_share") is None
                                                               else 100 * m["gpu_remote_share"], ".0f", " %"),
         "", verdict(m.get("gpu_remote_share"), None)),
        ("GPU tier / device", fmt(None if m.get("gpu_tier") is None and m.get("gpu_device") is None
                                  else f"{m.get('gpu_tier') or '-'} / {m.get('gpu_device') or '-'}"), "",
         verdict(m.get("gpu_tier") or m.get("gpu_device"), None)),
        ("GPU RTT p50", fmt(m.get("gpu_rtt_ms"), ".0f", " ms"), "", verdict(m.get("gpu_rtt_ms"), None)),
        ("Re-learn: profile swap (1/2) -> Enrolled with the new profile",
         fmt(None if not swaps else f"worst {fmt(swap_worst, '.2f', ' s') if np.isfinite(swap_worst) else 'never'}"
             f" (per swap s: {_ms_list(swaps)})"),
         f"<= {SWAP_BUDGET_S:g} s each (T-155)",  # a swap that never played is inf: FAIL, not NOT MEASURED
         NOT_MEASURED if not swaps else ("PASS" if swap_worst <= SWAP_BUDGET_S else "FAIL")),
        ("Isolated while Aryan talks over Joseph", fmt(None if m.get("overtalk_locked") is None
                                                       else 100 * m["overtalk_locked"], ".0f", " %"),
         f">= {100 * OVERTALK_LOCKED_MIN:.0f} % of ticks",
         verdict(m.get("overtalk_locked"), lambda v: v >= OVERTALK_LOCKED_MIN)),
        ("Wrong speaker during the over-talk (ECAPA vs profiles, out.wav)",
         fmt(None if m.get("wrong_speaker") is None else 100 * m["wrong_speaker"], ".1f", " %"),
         "report-only (Mac measured 6-25 %, best 8.6-9.1 %)", verdict(m.get("wrong_speaker"), None)),
        ("Raw/Selective flips (Space x3)", fmt(m.get("space_toggles")), f">= {SPACE_MIN}",
         verdict(m.get("space_toggles"), lambda v: v >= SPACE_MIN)),
        ("A/B switch latency per flip", fmt(None if not m.get("space_latency_ms")
                                            else _ms_list(m["space_latency_ms"]) + " ms"), report,
         verdict(m.get("space_latency_ms") or None, None)),
        (f"Name Call hits / attempts ({NAME_CALL_ATTEMPTS} clean calls)",
         fmt(None if hits is None else f"{hits}/{NAME_CALL_ATTEMPTS}"), f">= {NAME_CALL_MIN}/{NAME_CALL_ATTEMPTS}",
         verdict(hits, lambda v: v >= NAME_CALL_MIN)),
        ("Name Call false triggers (detections beyond the calls)", fmt(false), "0 expected",
         verdict(false, lambda v: v == 0)),
        ("Caption words (final)", fmt(m.get("caption_words")), "", verdict(m.get("caption_words"), None)),
        ("Caption WER vs the script", fmt(m.get("caption_wer"), ".3f"), f"<= {TEXT_WER_BAR:g} (judge_check, T-139)",
         verdict(m.get("caption_wer"), lambda v: v <= TEXT_WER_BAR)),
        ("Worklet underruns", fmt(m.get("underruns")), "0", verdict(m.get("underruns"), lambda v: v == 0)),
        ("Bud out and back: audio kept playing", fmt(human.get("bud")), "yes", verdict(human.get("bud"), lambda v: v == "yes")),
    ]
    if devices:
        rows += [(f"judge_check: {crit}", detail, "", v) for v, crit, detail in devices]
    else:
        rows.append(("judge_check device rows", NOT_MEASURED, "", NOT_MEASURED))
    return rows


def render(m: dict, latency: dict | None, devices: list | None, human: dict, stamp: str, gpu: bool | None,
           latency_path: str | None = None) -> str:
    rows = sheet_rows(m, latency, devices, human, gpu)
    counts = Counter(r[3] for r in rows)
    overall = "FAIL" if counts["FAIL"] else ("INCOMPLETE" if counts[NOT_MEASURED] else "PASS")
    out = [f"# Rehearsal {stamp}", "",
           (f"Overall: **{overall}** · PASS {counts['PASS']} · FAIL {counts['FAIL']} · "
            f"NOT MEASURED {counts[NOT_MEASURED]} · INFO {counts['INFO']}"), "",
           *([(f"**{NO_AUDIO}** (seq null in every tick: Start not pressed?): every audio row is "
                f"{NOT_MEASURED}"), ""] if m.get("no_audio") else []),
           f"Sessions: {', '.join(m.get('sessions') or []) or NOT_MEASURED}  ",
           f"Latency result: {latency_path or NOT_MEASURED}", "",
           "| item | value | bar | verdict |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    out += ["", "Signed (Aryan): ____________________    Signed (Joseph): ____________________", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- live automation

def _say(msg: str) -> None:
    print(msg, flush=True)


def newest_latency(since: float) -> Path | None:
    dirs = {Path(os.environ["HEARING_RESULTS_DIR"])} if os.environ.get("HEARING_RESULTS_DIR") else set()
    dirs.add(ROOT / "results")
    found = [p for d in dirs if d.is_dir() for p in d.glob("bt_latency_*.json") if p.stat().st_mtime >= since]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def auto_devices(no_hw: bool) -> None:
    if no_hw:
        _say("  SKIP device list (--no-hw)")
    else:
        subprocess.run([sys.executable, "-m", "hearing", "devices"], cwd=ROOT, check=False)
    caf = subprocess.run(["pgrep", "-x", "caffeinate"], capture_output=True, check=False).returncode == 0
    _say("  caffeinate running" if caf else "  !! caffeinate is NOT running: start `caffeinate -dims` in another terminal")
    _say("  reminder: Bluetooth OFF on both phones and on laptop-b (they grab the AirPods)")


def profiles_env() -> Path:
    """F1: the server must see the saved voices or keys 1/2 do nothing. ``$HEARING_PROFILES_DIR`` when set,
    else ``eval/fixtures/generated/profiles``; exported for every child process."""
    root = Path(os.environ.get("HEARING_PROFILES_DIR") or PROFILES_DEFAULT).expanduser()
    os.environ["HEARING_PROFILES_DIR"] = str(root)
    return root


def auto_server(port: int, no_hw: bool, gpu: bool = False) -> None:
    root = profiles_env()
    names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    _say(f"  HEARING_PROFILES_DIR={root} ({', '.join(names) or 'NO PROFILES: keys 1/2 will do nothing'})")
    cmd = ["bash", "scripts/demo.sh", "--judge", "--team"] + (["--no-open"] if no_hw else [])
    env = dict(os.environ, PORT=str(port))
    if gpu:  # T-181: the team preset pins the Mac; the A100 (already up via gpu_up.sh) only when asked
        env["HEARING_HQ_BACKEND"] = "remote"
    subprocess.run(cmd, cwd=ROOT, check=False, env=env)
    from scripts.judge_check import wait_warm

    try:
        wait_warm(port)
        _say("  server warm")
    except OSError as e:
        _say(f"  !! /health not answering on :{port} ({e})")


def auto_judge(port: int) -> list:
    from scripts.judge_check import check_devices

    rows = check_devices(f"http://127.0.0.1:{port}", required=True)
    for v, crit, detail in rows:
        _say(f"  {v:4s}  {crit}: {detail}")
    return rows


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = input) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="print the steps without waiting or running anything")
    p.add_argument("--no-hw", action="store_true", help="never open the mic/speakers (skips the device list and latency)")
    p.add_argument("--session", type=Path, action="append", default=None,
                   help="session dir(s) to read (default: those recorded under HEARING_SESSION_LOG since the start)")
    p.add_argument("--latency-json", type=Path, default=None, help="a bt_latency_*.json (default: the one this run wrote)")
    p.add_argument("--caption-ref", type=Path, default=None, help="the text Joseph reads in the captions step")
    p.add_argument("--gpu", action="store_true", default=None, help="the GPU was resynced: Isolated bar 6 s")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    began = time.time()
    log_root = Path(os.environ.get("HEARING_SESSION_LOG") or ROOT / "results" / "sessions").expanduser()
    if not args.dry_run:
        os.environ["HEARING_SESSION_LOG"] = str(log_root)  # demo.sh's server records every /judge session
        profiles_env()
    marks: dict[str, float] = {}
    devices = None
    human: dict = {}
    for i, step in enumerate(STEPS, 1):
        _say(f"\n== Step {i}/{len(STEPS)}: {step.title} ==\n  {step.say}")
        if step.auto:
            _say(f"  (this script {step.auto})")
        if args.dry_run:
            continue
        ask("  Enter when ready > ")
        marks[step.key] = time.time()
        if step.key == "devices":
            auto_devices(args.no_hw)
        elif step.key == "latency":
            if args.no_hw:
                _say("  SKIP latency (--no-hw)")
            else:
                subprocess.run(["bash", "scripts/bt_latency.sh"], cwd=ROOT, check=False)
        elif step.key == "server":
            auto_server(args.port, args.no_hw, bool(args.gpu))
        elif step.key == "judge":
            devices = auto_judge(args.port)
        elif step.key == "bud":
            ans = ask("  Did audio keep playing in both buds? [y/n] > ").strip().lower()
            human["bud"] = {"y": "yes", "yes": "yes", "n": "no", "no": "no"}.get(ans)

    sessions = args.session if args.session else ([] if args.dry_run else find_sessions(log_root, began))
    lat_path = args.latency_json or (None if args.dry_run else newest_latency(began))
    latency = None
    if lat_path is not None and lat_path.exists():
        latency = json.loads(lat_path.read_text())
    ref = args.caption_ref.read_text() if args.caption_ref else None
    m = measure(sessions, marks, ref, profiles_root=Path(os.environ["HEARING_PROFILES_DIR"])
                if os.environ.get("HEARING_PROFILES_DIR") else None)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    text = render(m, latency, devices, human, stamp, args.gpu, str(lat_path) if latency else None)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"rehearsal_{stamp}.md"
    path.write_text(text)
    if m.get("no_audio"):
        _say(f"\n{NO_AUDIO}")
    _say("\n" + text)
    _say(f"wrote {path}")
    return 1 if "| FAIL |" in text else 0


if __name__ == "__main__":
    sys.exit(main())
