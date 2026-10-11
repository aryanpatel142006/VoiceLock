#!/usr/bin/env python3
"""Replay the demo clip through a real server in every mode, score each session, flag regressions.

    bash scripts/demo_regress.sh            # natural, focus_hq, text, synthetic; 60 s each
    bash scripts/demo_regress.sh --fast     # natural + focus_hq, 12 s each (verify_full)
    bash scripts/demo_regress.sh --modes name_call   # T-120: camera-free Name Call row (no video sent)

Starts its own ``hearing serve`` on an ephemeral port (never the human's :8000) with
``HEARING_SESSION_LOG`` pointing at ``results/sessions/regress-<stamp>/``, feeds
``eval/fixtures/demo/demo-mf-easy/clip.mp4`` (``make_fixtures --demo``) through ``hearing.replay`` (no camera, no mic),
scores every recording with ``scripts/session_score.py``, prints the table and writes
``$HEARING_RESULTS_DIR/demo_regress_<stamp>.json``. A mode regresses against the baseline (per-mode
median of the last 5 runs of the same variant) when SI-SDRi drops by more than 1 dB or true
dropouts grow by more than 0.5 s, or by more than those runs' own spread if that is larger. A live replay is not deterministic (ASD and tier timing differ
run to run; focus_hq true dropouts measured 1.2-2.8 s on identical code), so a flagged mode is
replayed once more and only a regression both runs show counts.

Exit codes: 0 = no regression, 1 = a regression (reproduced), 2 = a crash, a mode that recorded no
session, a missing clip/stems or bad arguments (so a broken harness never reads as a regression).
A ``--clip`` or the default clip missing from this worktree is looked up in the main checkout.

An ingested clip (manifest with a ``target`` box) is replayed with the manifest's target locked
(``scripts/replay.py --lock-target``'s oracle): the demo recipe has the wearer face the target, and
on demo-mf-easy both faces are off-centre, so TargetSelector alone may pick the other person and
the score would measure the wrong isolation (T-100). One extra ``free-selection`` row (focus_hq,
no lock) tracks what the selector does on its own; it is reported, never flagged. ``--no-lock``
replays every mode without the lock (its own baseline variant). Every row prints ``target
agreement`` and ``WRONG TARGET`` below 80 %.

Every row also prints the Focus HQ backend the session played (telemetry ``hq_active``, ``local`` /
``remote``) and ``DNSMOS first 7 s`` (the cold start: the output's first seconds, before Enrolled).
With ``HEARING_REMOTE_URL`` set, a focus_hq row that played ``local`` fails the run (exit 2: the A100
was not measured; set ``HEARING_HQ_BACKEND=remote``), and the remote's ``/health`` ``tiers_config``
hash is compared with the local ``config/tiers-remote.yaml`` (a warning when they differ: a stale or
hand-edited remote checkout; T-118).

``--no-baseline`` (or ``HEARING_REGRESS_NO_BASELINE=1``) marks the run's JSON ``"no_baseline": true`` so
doc, dry-run and rehearsal runs are still compared and written but never become a later run's baseline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import session_score as ss
from scripts.replay import OracleLock

MODES = {"natural": 80, "focus_hq": 100, "text": 80, "synthetic": 80}  # mode -> suppression (as the console sends)
EXTRA_MODES = {"name_call": 100}  # asked for by --modes only (T-120: camera-free, not in the default four)
SDRI_DROP_DB = 1.0
DROPOUT_GROWTH_S = 0.5
FREE_MODE = "focus_hq"  # the free-selection row: the mode a wrong target hurts most
COLD_START_S = 7.0  # 'DNSMOS first 7 s': the room / gate seconds before Enrolled plays (T-115)
REMOTE_CONFIG = "tiers-remote.yaml"


def backend_used(rows: list[dict]) -> str | None:
    """The Focus HQ backend a session played most (telemetry ``hq_active``), or None without telemetry."""
    seen = [r["event"].get("hq_active") for r in rows
            if r.get("dir") == "out" and r["event"].get("type") == "telemetry" and r["event"].get("hq_active")]
    return max(set(seen), key=seen.count) if seen else None


MIN_ENROLLED_SHARE = 0.5


def enrolled_share(rows: list[dict]) -> float | None:
    """Share of telemetry windows whose tier was ``enrolled`` (None without telemetry)."""
    tiers = [r["event"].get("tier") for r in rows
             if r.get("dir") == "out" and r["event"].get("type") == "telemetry" and r["event"].get("tier")]
    return tiers.count("enrolled") / len(tiers) if tiers else None


def not_measured(asked_enrolled: bool, share: float | None) -> bool:
    """A row that asked for the enrolled/profile path but played it in under half of its windows."""
    return asked_enrolled and (share is None or share < MIN_ENROLLED_SHARE)


def unmeasured_exit(unmeasured: list[str], allow: bool) -> bool:
    """Print the loud NOT MEASURED line; True when the run must exit 2."""
    if not unmeasured or allow:
        return False
    print(f"!! NOT MEASURED: {len(unmeasured)} row(s) never ran the enrolled tier (overrun? loaded machine); "
          "rerun on a quiet machine or pass --allow-unmeasured", file=sys.stderr)
    return True


def played_local_with_remote(mode: str, backend: str | None, remote_url: str | None) -> bool:
    """A focus_hq run that should have used the GPU (``HEARING_REMOTE_URL`` set) but played local."""
    return bool(remote_url) and mode.partition("@")[0] == "focus_hq" and backend != "remote"


def dnsmos_first(y: np.ndarray, dnsmos: str, seconds: float = COLD_START_S) -> float | None:
    """DNSMOS OVRL of the output's first ``seconds`` (the cold start); None for silence or ``none``."""
    head = y[: int(seconds * ss.SR)]
    if dnsmos == "none" or len(head) < ss.SR or np.sqrt(np.mean(np.square(head, dtype=np.float64))) <= 1e-4:
        return None
    from eval.dnsmos import get_dnsmos

    return round(float(get_dnsmos(dnsmos).score(head, ss.SR).ovrl), 2)


def health_url(ws_url: str) -> str:
    """``ws://host:port/ws/session`` -> ``http://host:port/health`` (wss -> https)."""
    http = "https" + ws_url[3:] if ws_url.startswith("wss") else "http" + ws_url[2:] if ws_url.startswith("ws") else ws_url
    base = http.split("/ws/", 1)[0].rstrip("/")
    return base + "/health"


def remote_config_warning(health: dict, local: dict[str, str | None], name: str = REMOTE_CONFIG) -> str | None:
    """A warning when the remote's loaded ``tiers_config`` hash differs from the local file's."""
    theirs = (health.get("tiers_config") or {}).get(name)
    if theirs is None:
        return f"remote /health reports no {name} hash (older server code?); cannot check its config"
    if theirs != local.get(name):
        return (f"remote {name} {theirs} != local {local.get(name)}: the GPU box runs another ladder config "
                f"(git_rev {health.get('git_rev')}); re-sync with scripts/gpu_up.sh")
    return None


def check_remote_config(remote_url: str) -> None:
    import urllib.request

    from hearing.server.app import config_hashes

    try:
        with urllib.request.urlopen(health_url(remote_url), timeout=5) as r:
            health = json.loads(r.read())
    except (OSError, ValueError) as e:
        print(f"!! remote /health at {health_url(remote_url)} unreachable: {e}", flush=True)
        return
    warning = remote_config_warning(health, config_hashes(ss.ROOT))
    print(f"!! WARNING {warning}" if warning else
          f"remote {REMOTE_CONFIG} {health['tiers_config'][REMOTE_CONFIG]} matches the local file", flush=True)


TUNNEL_PORT = 18765  # scripts/gpu_up.sh's local end of the SSH tunnel to the A100


def tunnel_port(remote_url: str) -> int:
    from urllib.parse import urlparse

    return urlparse(remote_url).port or TUNNEL_PORT


def lsof_pids(output: str, own_pid: int | None = None) -> list[int]:
    """Pids in ``lsof -nP -iTCP:<port>`` output (header skipped, ``own_pid`` and duplicates dropped, order kept)."""
    pids: list[int] = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) != own_pid and int(parts[1]) not in pids:
            pids.append(int(parts[1]))
    return pids


def tunnel_clients_warning(pids: list[int], port: int) -> str | None:
    if not pids:
        return None
    return (f"other local processes hold tunnel port {port} (pids {', '.join(map(str, pids))}): they may open "
            "sessions on the shared GPU mid-run and inflate compute; stop them or use --strict-remote")


def other_tunnel_clients(port: int) -> list[int]:
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return lsof_pids(out, os.getpid())


def sessions_active(remote_url: str | None, port: int | None = None) -> int | None:
    """The sessions open on the server we measure (the remote's /health ``sessions_active``), None when unknown."""
    import urllib.request

    url = health_url(remote_url) if remote_url else f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            value = json.loads(r.read()).get("sessions_active")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, int) else None


def free_port() -> int:
    """An ephemeral port the OS picked (uvicorn does not log the bound port for --port 0
    once hearing's logging config is in place, so pick it here and pass it explicitly)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_warm(port: int, timeout: float = 180, fetch=None, sleep=time.sleep) -> bool:
    """Poll /health until ``warm`` is 'ready' (or 'off': never warms), as demo.sh --judge does, so the
    first replay does not pay model loading. False on timeout (the replay goes ahead anyway)."""
    import json
    import urllib.request

    def get() -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.load(r)

    fetch = fetch or get
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if fetch().get("warm") in ("ready", "off"):
                return True
        except (OSError, ValueError):
            pass
        sleep(0.5)
    return False


def start_server(log_dir: Path, log_path: Path) -> tuple[subprocess.Popen, int]:
    import urllib.request

    env = dict(os.environ, HEARING_SESSION_LOG=str(log_dir))
    port = free_port()
    log = log_path.open("w")
    proc = subprocess.Popen([sys.executable, "-m", "hearing", "serve", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ss.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    if not wait_warm(port, max(1.0, deadline - time.monotonic())):
                        print("  warning: /health warm never reached 'ready'; replaying anyway", file=sys.stderr)
                    return proc, port
        except OSError:
            time.sleep(0.5)
    stop_server(proc)
    raise RuntimeError(f"hearing serve did not come up on :{port}; see {log_path}")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def mode_slider(mode: str) -> tuple[str, int]:
    """``natural@50`` -> ("natural", 50); a bare mode uses :data:`MODES`' suppression."""
    name, _, s = mode.partition("@")
    return name, int(s) if s else {**MODES, **EXTRA_MODES}[name]


DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}  # Whisper writes 8


def name_for_clip(manifest: dict) -> str | None:
    """The Name Call row's "name" (T-120): the first word of 4+ letters the target says and the other
    person never does (``sentences`` of the manifest, digits excluded), so whoever "says the name" is the target."""
    sentences = manifest.get("sentences") or []
    other = {w for s in sentences if s.get("who") != "target" for w in s.get("text", "").lower().split()}
    for s in sorted((s for s in sentences if s.get("who") == "target"), key=lambda s: s.get("start_s", 0.0)):
        for w in s.get("text", "").lower().split():
            if len(w) >= 4 and w not in other and w not in DIGITS:
                return w
    return None


def voice_profile_enroll(spec: str) -> dict:
    """``control.enroll`` for ``--voice-profile NAME[:OTHER]`` (T-125 profile, T-129 A/B runs)."""
    name, _, other = spec.partition(":")
    return {"action": "profile", "name": name, **({"other": other} if other else {})}


async def replay(url: str, clip, mode: str, seconds: float, oracle: OracleLock | None = None,
                 attention: str | None = None, profile: str | None = None, name: str | None = None,
                 voice_profile: str | None = None, isolation: str | None = None,
                 latency_preset: str | None = None) -> None:
    import websockets

    from hearing.replay import feed

    camera = not mode.startswith("name_call")  # T-120: Name Call is replayed camera-free

    async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
        async def send(message):
            if not camera:  # no video frames, and the hello says so
                if isinstance(message, dict) and message.get("type") == "hello":
                    message = {**message, "video": False}
                elif isinstance(message, (bytes, bytearray)) and message[:1] == b"\x02":
                    return
            await ws.send(json.dumps(message) if isinstance(message, dict) else message)

        async def drain():
            async for message in ws:
                control = oracle.on_message(json.loads(message)) if oracle and isinstance(message, str) else None
                if control:
                    await send(control)

        receiver = asyncio.create_task(drain())
        try:
            control = dict(zip(("mode", "suppression"), mode_slider(mode)))
            if control["mode"] == "name_call":  # T-120: no camera; the name is a word only the target says
                control["name"] = {"text": name or "Aryan", "aliases": []}
            if profile:  # T-113: play as the judge page does (strict isolation, forwarded to a remote GPU)
                control["profile"] = profile
            if voice_profile and camera:  # T-129: the target's voice profile from t=0
                control["enroll"] = voice_profile_enroll(voice_profile)
            if isolation:  # T-129: strict (pause_duck_asd) vs standard, after the profile's own default
                control["isolation"] = isolation
            if latency_preset:  # T-182: speed|balanced|accuracy (e.g. a 1.0 s enrolled window)
                control["latency_preset"] = latency_preset
            if attention:  # T-096: hold the target like the judge profile does (attention preset)
                from hearing.vision.target import ATTENTION_PRESETS

                control["attention"] = {**ATTENTION_PRESETS[attention], "preset": attention}
            await asyncio.wait_for(feed(clip, send, loop=True, control=control), timeout=seconds)
        except TimeoutError:
            pass
        await asyncio.sleep(0.5)  # let the last replies reach the tap
        receiver.cancel()


BASELINE_RUNS = 5


def previous(results: Path, variant: str, stamp: str, n: int = BASELINE_RUNS, clip: str | None = None) -> list[dict]:
    """The last ``n`` runs of the same variant on the same clip, newest first (runs recorded before
    T-097 have no comparable clip: the legacy loop ``grid-two-face.mp4`` is a different input)."""
    runs = []
    for p in sorted(results.glob("demo_regress_*.json"), reverse=True):
        if p.stem.endswith(stamp):
            continue
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if d.get("no_baseline"):
            continue  # a doc / dry-run replay (--no-baseline) is never the comparison baseline
        if d.get("variant") == variant and (clip is None or Path(d.get("clip", "")).parent.name == clip):
            runs.append(d)
            if len(runs) == n:
                break
    return runs


def separation(s: dict) -> float | None:
    """The SI-SDRi a regression is judged on: before the judge profile's output AGC when the run recorded
    it (T-135, Q-350: leveling caps a whole-clip SI-SDR), else the played output's."""
    pre = s.get("si_sdri_pre_agc_db")
    return pre if pre is not None else s.get("si_sdri_db")


def baseline(runs: list[dict]) -> dict | None:
    """Per-mode median of SI-SDRi and true dropouts over previous runs (one lucky run is no baseline),
    with their spread (max - min) as the measured run-to-run noise."""
    if not runs:
        return None
    by_mode: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for s in run.get("scores", []):
            m = by_mode.setdefault(s["mode"], {"si_sdri_db": [], "true_dropout_s": []})
            for k in m:
                v = separation(s) if k == "si_sdri_db" else s.get(k)
                if v is not None:
                    m[k].append(float(v))
    rows = []
    for mode, m in by_mode.items():
        row = {"mode": mode}
        for k, v in m.items():
            row[k] = round(float(np.median(v)), 2) if v else None
            row[f"{k}_spread"] = round(float(np.ptp(v)), 2) if v else 0.0
        rows.append(row)
    return {"scores": rows, "runs": len(runs)}


def best_of(a: dict, b: dict) -> dict:
    """Merge two replays of one mode: the better SI-SDRi and the fewer dropouts (a regression must reproduce)."""
    out = dict(a)
    if b.get("si_sdri_db") is not None and (a.get("si_sdri_db") is None or b["si_sdri_db"] > a["si_sdri_db"]):
        out["si_sdri_db"] = b["si_sdri_db"]
        out["si_sdri_pre_agc_db"] = b.get("si_sdri_pre_agc_db")
    out["true_dropout_s"] = min(a["true_dropout_s"], b["true_dropout_s"])
    return out


def regressions(now: list[dict], before: dict | None) -> list[str]:
    """Flag a drop beyond max(the fixed threshold, the baseline runs' own spread): a 12 s live replay
    cannot resolve a change smaller than its run-to-run noise."""
    if not before:
        return []
    old = {s["mode"]: s for s in before.get("scores", [])}
    out = []
    for s in now:
        o = old.get(s["mode"])
        if not o:
            continue
        tol_sdr = max(SDRI_DROP_DB, o.get("si_sdri_db_spread", 0.0))
        tol_drop = max(DROPOUT_GROWTH_S, o.get("true_dropout_s_spread", 0.0))
        sdri = separation(s)
        label = "separation SI-SDRi, before output leveling" if s.get("si_sdri_pre_agc_db") is not None else "SI-SDRi"
        if sdri is not None and o.get("si_sdri_db") is not None and o["si_sdri_db"] - sdri > tol_sdr:
            out.append(f"{s['mode']}: {label} {o['si_sdri_db']:+.2f} -> {sdri:+.2f} dB")
        if o.get("true_dropout_s") is not None and s["true_dropout_s"] - o["true_dropout_s"] > tol_drop:
            out.append(f"{s['mode']}: true dropouts {o['true_dropout_s']:.2f} -> {s['true_dropout_s']:.2f} s")
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except SystemExit as e:  # argparse: 0 for --help, 2 for bad arguments
        return e.code if isinstance(e.code, int) else 2
    except KeyboardInterrupt:
        return 2
    except Exception as e:  # noqa: BLE001 -- a crash is not a regression (exit 2, not 1)
        import traceback

        traceback.print_exc()
        print(f"!! demo_regress crashed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


def no_baseline_env() -> bool:
    return os.environ.get("HEARING_REGRESS_NO_BASELINE", "").strip().lower() in {"1", "true", "yes"}


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="natural + focus_hq, 12 s each")
    ap.add_argument("--modes", default=None, help="comma list (default: all four, or two with --fast); mode@N sets the slider, e.g. natural@50")
    ap.add_argument("--name", default=None,
                    help="name_call: the wearer's name (default: a word only the target says in the clip's manifest)")
    ap.add_argument("--seconds", type=float, default=None, help="replay length per mode (default 60, 12 with --fast)")
    ap.add_argument("--clip", type=Path, default=None, help="clip dir, MP4 or a clip name under eval/fixtures/demo (default demo-mf-easy)")
    ap.add_argument("--no-lock", action="store_true",
                    help="let TargetSelector choose in every mode (default: lock the manifest's target)")
    ap.add_argument("--no-free", action="store_true", help="skip the free-selection row")
    ap.add_argument("--attention", default=None, choices=["glance", "normal", "sticky"],
                    help="attention preset sent with the mode (default: the server's; the judge profile uses sticky)")
    ap.add_argument("--profile", default=None, choices=["judge", "team"],
                    help="control.profile sent with the mode (judge: strict isolation, sticky attention, auto-calibration; team: judge + config/demo-team.yaml, T-130)")
    ap.add_argument("--voice-profile", default=None, metavar="NAME[:OTHER]",
                    help="load $HEARING_PROFILES_DIR/NAME as the target's voice (OTHER = the known negative) with the mode")
    ap.add_argument("--isolation", default=None, choices=["standard", "strict"],
                    help="control.isolation sent with the mode (strict = pause_duck_asd; default: the server's or the profile's)")
    ap.add_argument("--latency-preset", default=None, choices=["speed", "balanced", "accuracy"],
                    help="send control.latency_preset with the mode (T-182)")
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="keep exit 0/1 when a profile/enrolled row never played the enrolled tier (default: NOT MEASURED, exit 2)")
    ap.add_argument("--strict-remote", action="store_true",
                    help="abort (exit 2) when other local processes hold the tunnel port during a remote run")
    ap.add_argument("--pre-agc", action="store_true",
                    help="score every metric on out_pre_agc.wav (before output leveling) instead of out.wav")
    ap.add_argument("--dnsmos", default="auto", choices=["auto", "dnsmos", "mock", "none"])
    ap.add_argument("--no-baseline", action="store_true", default=no_baseline_env(),
                    help="never let this run become a later run's baseline (default: HEARING_REGRESS_NO_BASELINE=1)")
    args = ap.parse_args(argv)
    modes = args.modes.split(",") if args.modes else (["natural", "focus_hq"] if args.fast else list(MODES))
    seconds = args.seconds or (12.0 if args.fast else 60.0)  # 60 s: the enrolled phase starts ~30 s in (T-095)
    clip_path = ss.main_fallback(args.clip or ss.demo_dir() / "demo-mf-easy" / "clip.mp4")
    if not clip_path.exists() and args.clip is not None and len(args.clip.parts) == 1:  # a bare name: demo-team-easy
        clip_path = ss.main_fallback(ss.demo_dir() / args.clip)
    if clip_path.is_dir():
        clip_path = clip_path / "clip.mp4"
    if not clip_path.exists():
        print(f"no demo clip at {clip_path} (nor in the main checkout {ss.MAIN}): run .venv/bin/python -m "
              "eval.make_fixtures --demo, or pass --clip <clip dir or MP4>", file=sys.stderr)
        return 2
    try:
        stems = ss.resolve_stems(clip_path.parent / "manifest.json")
        call_name = args.name or name_for_clip(json.loads((clip_path.parent / "manifest.json").read_text()))
    except FileNotFoundError as e:
        print(f"!! {e} (clip {clip_path})", file=sys.stderr)  # 'no demo manifest at': verify_full skips
        return 2
    lock = bool(stems.target_face.get("box")) and not args.no_lock
    from hearing.server.remote import video_enabled as remote_video_enabled

    variant = (f"{'fast' if args.fast else 'full'}:{','.join(modes)}:{seconds:g}"
               + (f":{args.attention}" if args.attention else "") + (f":{args.profile}" if args.profile else "")
               + (f":voice={args.voice_profile}" if args.voice_profile else "")
               + (f":iso={args.isolation}" if args.isolation else "")
               + (f":preset={args.latency_preset}" if args.latency_preset else "")
               + (":pre-agc" if args.pre_agc else "")
               + (":novideo" if os.environ.get("HEARING_REMOTE_URL") and not remote_video_enabled() else "")
               + (":lock" if lock else ":free"))
    remote_url = os.environ.get("HEARING_REMOTE_URL") or None
    if remote_url or os.environ.get("HEARING_HQ_BACKEND") == "remote":
        port_ = tunnel_port(remote_url or "")
        warning = tunnel_clients_warning(other_tunnel_clients(port_), port_)
        if warning:
            print(f"!! WARNING {warning}", flush=True)
            if args.strict_remote:
                return 2
    # pid suffix: two verify runs starting in the same second (two reviews) must not share a session dir,
    # or each one scores (or misses) the other's sessions: "no session recorded" / "< 0.5 s aligned"
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    results = Path(os.environ.get("HEARING_RESULTS_DIR") or ss.MAIN / "eval/results")
    results.mkdir(parents=True, exist_ok=True)
    log_dir = ss.MAIN / "results/sessions" / f"regress-{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    from hearing.replay import Clip

    clip = Clip.load(clip_path)
    t0 = time.monotonic()
    if args.profile == "team":  # T-130: the whole preset, incl. the server-wide caption ASR env `hearing serve` sets
        os.environ["HEARING_DEMO_PRESET"] = "team"
    proc, port = start_server(log_dir, log_dir / "server.log")
    print(f"server on :{port} (up in {time.monotonic() - t0:.0f} s), sessions -> {log_dir}"
          f" · target {'locked to the manifest' if lock else 'free (TargetSelector)'}", flush=True)
    base = baseline(previous(results, variant, stamp, clip=clip_path.parent.name))
    if remote_url:
        check_remote_config(remote_url)
    wrong_backend: list[str] = []
    unmeasured: list[str] = []

    def run_mode(mode: str, locked: bool = lock):
        before = {p.name for p in log_dir.iterdir() if p.is_dir()}
        sessions_after = None
        oracle = OracleLock({"target": stems.target_face}) if locked and not mode.startswith("name_call") else None
        asyncio.run(replay(f"ws://127.0.0.1:{port}/ws/session", clip, mode, seconds, oracle, args.attention, args.profile,
                           call_name, args.voice_profile, args.isolation, args.latency_preset))
        sessions_after = sessions_active(remote_url, port)  # T-182: other clients on the measured server
        time.sleep(1.0)  # the tap closes after the socket
        new = sorted(p for p in log_dir.iterdir() if p.is_dir() and p.name not in before)
        if not new:
            print(f"!! {mode}: no session recorded", flush=True)
            return None
        s = ss.score_session(new[-1], stems, dnsmos=args.dnsmos, pre_agc=args.pre_agc)
        s.sessions_active = sessions_after
        s.session = f"{log_dir.name}/{new[-1].name}"
        s.mode = mode  # natural@50: keep the slider in the table and the baseline key
        events = [json.loads(line) for line in (new[-1] / "events.jsonl").read_text().splitlines() if line.strip()]
        s.backend = backend_used(events)
        s.dnsmos_first7 = dnsmos_first(ss.read_wav(new[-1] / "out.wav"), args.dnsmos)
        mos7 = "-" if s.dnsmos_first7 is None else f"{s.dnsmos_first7:.2f}"
        print(f"{mode if locked or not lock else 'free':10s} {s.line()} | backend {s.backend or '-'}"
              f" | DNSMOS first {COLD_START_S:g} s {mos7}", flush=True)
        share = enrolled_share(events)
        if not_measured(bool(args.profile or args.voice_profile), share):
            unmeasured.append(f"{mode}: enrolled tier played in "
                              f"{'no telemetry' if share is None else format(share, '.0%')} of windows (< {MIN_ENROLLED_SHARE:.0%})")
            print(f"!! NOT MEASURED {unmeasured[-1]}", flush=True)
        if played_local_with_remote(mode, s.backend, remote_url):
            wrong_backend.append(f"{mode}: played {s.backend or 'unknown'} although HEARING_REMOTE_URL is set"
                                 " (set HEARING_HQ_BACKEND=remote)")
            print(f"!! {wrong_backend[-1]}", flush=True)
        return s

    def as_row(s) -> dict:
        r = s.as_dict()
        r.pop("loops", None)
        r["backend"] = getattr(s, "backend", None)
        r["dnsmos_first7"] = getattr(s, "dnsmos_first7", None)
        r["latency_preset"] = args.latency_preset or "balanced"  # T-182
        r["sessions_active"] = getattr(s, "sessions_active", None)  # T-182: read after our replay closed, so other clients
        return r

    scores, rows, retries, free = [], [], [], None
    try:
        for mode in modes:
            s = run_mode(mode)
            if s is None:
                continue
            scores.append(s)
            row = as_row(s)
            if regressions([row], base):
                again = run_mode(mode)  # confirm: live replays are noisy
                if again is not None:
                    scores.append(again)
                    retries.append(as_row(again))
                    row = best_of(row, retries[-1])
            rows.append(row)
        if lock and not args.no_free:
            s = run_mode(FREE_MODE, locked=False)
            if s is not None:
                s.session += " (free-selection)"
                scores.append(s)
                free = as_row(s)
    finally:
        stop_server(proc)
    wall = time.monotonic() - t0
    print()
    print(ss.table(scores))
    flagged = regressions(rows, base)
    out = results / f"demo_regress_{stamp}.json"
    git = subprocess.run(["git", "-C", str(ss.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    out.write_text(json.dumps({"stamp": stamp, "variant": variant, "git": git, "host": socket.gethostname(),
                               "clip": str(clip_path), "fixture": stems.fixture, "seconds_per_mode": seconds,
                               "wall_s": round(wall, 1), "baseline": base, "regressions": flagged,
                               "target_locked": lock, "no_baseline": args.no_baseline, "scores": rows, "free_selection": free, "retries": retries}, indent=2) + "\n")
    print(f"\nwall {wall:.0f} s · {out}" + (" (--no-baseline: never a baseline)" if args.no_baseline else ""))
    if base:
        print("baseline (median of last runs): " + ", ".join(
            f"{b['mode']} SI-SDRi {b['si_sdri_db']} ±{b['si_sdri_db_spread']} dB / dropouts {b['true_dropout_s']}"
            f" ±{b['true_dropout_s_spread']} s" for b in base["scores"]) + f" ({base['runs']} runs)")
    if free is not None:
        print(f"free-selection ({FREE_MODE}, TargetSelector alone): target agreement "
              f"{'-' if free['target_agreement'] is None else format(free['target_agreement'], '.0%')}"
              f"{' WRONG TARGET (the demo recipe has the wearer face the target)' if ss.wrong_target(free['target_agreement']) else ''}")
    print("cold start: " + ", ".join(
        f"{r['mode']} DNSMOS first {COLD_START_S:g} s {'-' if r['dnsmos_first7'] is None else r['dnsmos_first7']}"
        f" (backend {r['backend'] or '-'})" for r in rows))
    for r in retries:
        print(f"re-ran {r['mode']} to confirm a flag; kept the better of the two replays")
    for f in flagged:
        print(f"!! REGRESSION {f} (reproduced on a second replay)" if any(r["mode"] == f.split(":")[0] for r in retries)
              else f"!! REGRESSION {f}")
    for w in wrong_backend:
        print(f"!! WRONG BACKEND {w}", file=sys.stderr)
    if wrong_backend:
        return 2
    if unmeasured_exit(unmeasured, args.allow_unmeasured):
        return 2
    if len(rows) < len(modes):
        print(f"!! {len(modes) - len(rows)} mode(s) recorded no session", file=sys.stderr)
        return 2
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
