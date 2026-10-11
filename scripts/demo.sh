#!/usr/bin/env bash
# Start the demo; --restart replaces hearing serve by listening PORT, --stop stops it.
# --simple opens /simple, --gpu starts iLab, --no-open skips the browser.
# --judge --team pins the Mac for Focus HQ (T-181: T-129 measured it ahead of the A100 with profiles loaded)
#   and never probes the GPU; --judge --team --gpu (or HEARING_HQ_BACKEND=remote) probes and pins the A100.
# --check measures the enrolled tier's compute headroom (hearing.tse.headroom) and exits 1 below 30 %
#   (T-089); with --judge it then runs the rehearsal self-test below.
# --lan serves phones on the venue Wi-Fi: --host 0.0.0.0 --tls (self-signed cert) and prints the QR.
# --judge: THE presenter path (T-088). Replaces our server on PORT, always hands it the GPU url
#   (hq_backend=auto switches to the GPU whenever /health is ok and RTT p50 < 150 ms, T-109), probes
#   the iLab GPU during warm-up (3 x 6 s), prints one line (backend, expected
#   delay, git rev) and opens /judge. --judge --check runs the rehearsal self-test instead
#   (scripts/judge_check.py: PASS/FAIL per PLAN.md success criterion, with the numbers).
#   --judge --check --no-hw (agents, dry runs): camera, latency-test and GPU probe print
#   'SKIP ... skipped (no hardware)' and the demo_regress run is never a baseline.
# --mic LABEL (T-171): the demo mic's exact label (e.g. "Aryan patel's iPhone Microphone", the iPhone over USB),
#   exported as HEARING_INPUT_DEVICE: the team preset's expected mic and judge_check's mic rows use it.
# --team without HEARING_PROFILES_DIR (T-328): defaults it to $HEARING_FIXTURES_DIR/profiles (the main checkout's
#   eval/fixtures/generated/profiles) when that holds the aryan and joseph profiles, else a WARN on the JUDGE line
#   (an audio-only session without profiles never enrolls and plays the room). An explicit value always wins.
# --app (M14, T-355): opens the app (/app/) instead of /judge or /simple, on the laptop and in the --lan QR and
#   JUDGE line. Nothing else changes: same server, same flags; /judge and /simple stay served (the stage fallback).
#   Without --app the output is byte-for-byte what it was. Runbook: docs/APP.md.
# --port N (T-378): serve on PORT instead of $PORT/8000; --port 0 asks the OS for a free port and prints it.
#   --no-open skips the browser (agents). Without these flags the output is unchanged.
# Arguments after -- are forwarded to hearing serve (e.g. -- --host 0.0.0.0).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_py.sh
# The shared editable install points at main; this launcher must run THIS checkout.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PORT="${PORT:-8000}"
PAGE="/"; ACTION=start; OPEN=1; PORT_ARG=""; GPU=0; LAN=0; JUDGE=0; CHECK=0; NO_HW=0; TEAM=0; APP=0
SERVE_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --restart) ACTION=restart ;;
    --stop) ACTION=stop ;;
    --gpu) GPU=1 ;;
    --simple) PAGE="/simple" ;;
    --judge) JUDGE=1; PAGE="/judge"; ACTION=restart ;;
    --check) CHECK=1 ;;
    --no-hw) NO_HW=1 ;;
    --team) TEAM=1 ;;  # T-130: the team demo preset (config/demo-team.yaml) on top of --judge
    --lan) LAN=1 ;;
    --app) APP=1 ;;  # T-355: only the page that opens (and the QR page) changes
    --mic) [ "$#" -ge 2 ] || { echo "--mic needs a label" >&2; exit 2; }; export HEARING_INPUT_DEVICE="$2"; shift ;;
    --no-open) OPEN=0 ;;
    --port) [ "$#" -ge 2 ] || { echo "--port needs a number" >&2; exit 2; }; PORT_ARG="$2"; shift ;;
    --) shift; SERVE_ARGS=("$@"); break ;;
    *) echo "Unknown demo option: $1" >&2; exit 2 ;;
  esac
  shift
done
[ "$APP" = 1 ] && PAGE="/app/"
if [ -n "${PORT_ARG:-}" ]; then
  PORT="$PORT_ARG"
  if [ "$PORT" = 0 ]; then  # T-378: the OS picks a free loopback port; it is printed below
    PORT="$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
    echo "port $PORT (picked by --port 0)"
  fi
fi
if [ "$TEAM" = 1 ]; then
  [ "$JUDGE" = 1 ] || { echo "--team needs --judge" >&2; exit 2; }
  export HEARING_DEMO_PRESET=team  # hearing serve applies the caption ASR env; every judge session gets the pins
fi
TEAM_PROFILES=(aryan joseph)  # config/demo-team.yaml pair.target / pair.other
PROFILES_NOTE=""
team_profiles() {  # T-328: export the default HEARING_PROFILES_DIR for --team; PROFILES_NOTE goes on the JUDGE line
  local dir shown name missing=()
  if [ -n "${HEARING_PROFILES_DIR:-}" ]; then
    dir="$HEARING_PROFILES_DIR"
  else
    dir="$HEARING_FIXTURES_DIR/profiles"
    for name in "${TEAM_PROFILES[@]}"; do
      compgen -G "$dir/$name/*.wav" > /dev/null || missing+=("$name")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
      PROFILES_NOTE="WARN no voice profiles (${missing[*]} missing in $dir): set HEARING_PROFILES_DIR or the session plays the room"
      return 0
    fi
    export HEARING_PROFILES_DIR="$dir"
  fi
  shown="${dir#"$PWD"/}"
  PROFILES_NOTE="profiles $shown"
}
[ "$TEAM" = 1 ] && team_profiles
if [ "$CHECK" = 1 ]; then
  "$PY" -m hearing.tse.headroom || exit 1
  [ "$JUDGE" = 1 ] || exit 0
  CHECK_ARGS=()
  [ "$NO_HW" = 1 ] && CHECK_ARGS+=(--no-hw)
  [ "$TEAM" = 1 ] && CHECK_ARGS+=(--team) && echo "TEAM · $PROFILES_NOTE"
  exec "$PY" scripts/judge_check.py ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}
fi
[[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || {
  echo 'PORT must be 1-65535' >&2; exit 2;
}
command -v lsof >/dev/null || { echo 'lsof is required to verify port ownership' >&2; exit 1; }
listeners() { lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u || true; }
conflict() { echo "Port conflict on :$PORT: $*" >&2; exit 1; }
PIDS="$(listeners)"
# Validate every owner before signalling any of them. Do not kill arbitrary services.
for pid in $PIDS; do
  argv="$(ps -p "$pid" -o command= 2>/dev/null)" || conflict "cannot inspect PID $pid; ps access is required before stopping a listener"
  case "$argv" in
    *" -m hearing serve"*|*"/hearing serve"*) ;;
    *) conflict "PID $pid is not hearing serve ($argv)" ;;
  esac
done
if [ "$ACTION" != start ]; then
  for pid in $PIDS; do
    echo "Stopping hearing serve PID $pid on :$PORT ($(ps -p "$pid" -o lstart=))"
    kill -TERM "$pid" || conflict "cannot stop PID $pid"
  done
  # Wait for the old processes, not just the release of the socket. Never escalate to SIGKILL.
  for _ in $(seq 1 100); do
    alive=0
    for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 0.1
  done
  [ "${alive:-0}" = 0 ] || conflict "older PID(s) $PIDS did not exit"
  [ -z "$(listeners)" ] || conflict "still held by PID(s) $(listeners)"
  if [ "$ACTION" = stop ]; then echo "Stopped: :$PORT is free"; exit 0; fi
  PIDS=""
fi
if [ "$TEAM" = 1 ] && [ "$GPU" = 1 ]; then
  export HEARING_HQ_BACKEND=remote  # T-181: the team preset probes and pins the A100 only under --gpu
fi
TEAM_MAC=0  # T-181: the team preset pins the Mac unless --gpu / HEARING_HQ_BACKEND=remote
[ "$TEAM" = 1 ] && [ "${HEARING_HQ_BACKEND:-}" != remote ] && TEAM_MAC=1
if [ "$GPU" = 1 ]; then
  bash scripts/gpu_up.sh ilab
  export HEARING_REMOTE_URL="${HEARING_REMOTE_URL:-ws://127.0.0.1:18765/ws/session}"
fi
BACKEND="Mac ladder"; DELAY="~1.5 s (enrolled 1.0 s window + 0.5 s hop)"
PROBE_OUT=""; probe_pid=""
if [ "$JUDGE" = 1 ]; then
  # T-109: the server ALWAYS gets the GPU url (env, else the one gpu_up.sh recorded) and plays
  # hq_backend=auto, so it moves to the GPU by itself whenever /health is ok and RTT < 150 ms.
  # The probe (3 attempts x 6 s, same rule) only decides the printed line; it runs during warm-up.
  gpu_state="${XDG_STATE_HOME:-$HOME/.local/state}/hearing-gpu/url"
  url="${HEARING_REMOTE_URL:-$( [ -s "$gpu_state" ] && cat "$gpu_state" || echo ws://127.0.0.1:18765/ws/session)}"
  export HEARING_REMOTE_URL="$url"
  if [ "$TEAM_MAC" = 0 ]; then
    PROBE_OUT="$(mktemp "${TMPDIR:-/tmp}/hearing-judge-probe.XXXXXX")"
    "$PY" -m hearing.server.remote --url "$url" --judge > "$PROBE_OUT" 2>/dev/null &
    probe_pid=$!
  fi
fi
SCHEME=http; CURL_TLS=()
if [ "$LAN" = 1 ]; then
  SERVE_ARGS=(--host 0.0.0.0 --tls ${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"})
  SCHEME=https; CURL_TLS=(-k)  # the self-signed cert; the identity check below still pins the PID
fi
LOG="${TMPDIR:-/tmp}/hearing-serve-$PORT.log"
HEALTH="$(mktemp "${TMPDIR:-/tmp}/hearing-health.XXXXXX")"
new_pid=""; launched_at=0
cleanup() {
  status=$?
  rm -f "$HEALTH" ${PROBE_OUT:+"$PROBE_OUT"}
  [ -z "$probe_pid" ] || kill "$probe_pid" 2>/dev/null || true
  if [ "$status" != 0 ] && [ -n "$new_pid" ]; then
    kill -TERM "$new_pid" 2>/dev/null || true
    wait "$new_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if [ -z "$PIDS" ]; then
  launched_at="$($PY -c 'import time; print(time.time())')"
  "$PY" -m hearing serve --port "$PORT" \
    ${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"} > "$LOG" 2>&1 &
  new_pid=$!
fi
ready=0
for _ in $(seq 1 120); do
  owners="$(listeners)"
  if [ -n "$new_pid" ]; then
    [ -z "$owners" ] || [ "$owners" = "$new_pid" ] || conflict "expected new PID $new_pid, found $owners (log: $LOG)"
    kill -0 "$new_pid" 2>/dev/null || { echo "Server PID $new_pid exited (log: $LOG)" >&2; tail -20 "$LOG" >&2; exit 1; }
  fi
  if curl ${CURL_TLS[@]+"${CURL_TLS[@]}"} --max-time 1 -fs "$SCHEME://127.0.0.1:$PORT/health" > "$HEALTH"; then ready=1; break; fi
  sleep 0.5
done
[ "$ready" = 1 ] || { echo "Server did not become healthy on :$PORT (log: $LOG)" >&2; exit 1; }
owners="$(listeners)"
"$PY" - "$HEALTH" "${new_pid:-$owners}" "$launched_at" "$JUDGE" <<'PY'
import json
import sys
from datetime import datetime

with open(sys.argv[1]) as f:
    d = json.load(f)
try:
    assert d['ok'] is True
    assert str(d['pid']) == sys.argv[2], f"expected PID {sys.argv[2]}, health reports {d.get('pid')}"
    assert datetime.fromisoformat(d['started_at']).timestamp() >= float(sys.argv[3]), 'server predates this launch'
    assert d['git_rev'] and d['version']
except (KeyError, ValueError, AssertionError) as exc:
    sys.exit(f'Server identity conflict: {exc}; use --restart to replace old code')
if sys.argv[4] == "1":
    sys.exit(0)  # --judge prints its own single line
print(f"server PID {d['pid']} | started_at {d['started_at']} | version {d['version']} | git_rev {d['git_rev']}")
print('engine ok on', d.get('device'), '| stages:', ', '.join(k for k, v in d['stages'].items() if v == 'ready'))
PY
if [ "$LAN" = 1 ]; then "$PY" -m hearing qr --port "$PORT" --page "$( [ "$APP" = 1 ] && echo /app/ || { [ "$JUDGE" = 1 ] && echo /judge || echo /simple; } )"; fi
if [ "$JUDGE" = 1 ]; then
  # Models pre-warmed (incl. one enrolled inference) before the presenter can press Start (T-088).
  for _ in $(seq 1 120); do
    curl ${CURL_TLS[@]+"${CURL_TLS[@]}"} --max-time 1 -fs "$SCHEME://127.0.0.1:$PORT/health" | grep -q '"warm":"warming"' || break
    sleep 0.5
  done
  [ -z "$probe_pid" ] || wait "$probe_pid" 2>/dev/null || true; probe_pid=""
  verdict="$(cat "$PROBE_OUT" 2>/dev/null || true)"; verdict="${verdict:-FAIL no answer}"
  GPU_LABEL="iLab A100"; [ "$TEAM" = 1 ] && GPU_LABEL="iLab A100 (--gpu)"
  if [ "$TEAM_MAC" = 1 ]; then
    # T-181: pinned, no probe; the delay is the Mac ladder's (T-129 measured 1500 ms)
    BACKEND="Mac (measured: beats the A100 with profiles, T-129)"
  elif [ "${verdict%% *}" = GPU ]; then
    # T-086: the GPU ladder (config/tiers-remote.yaml) runs enrolled at 0.5 s / 0.125 s
    BACKEND="$GPU_LABEL · ${verdict#GPU }"; DELAY="~0.95 s (enrolled 0.5 s window + 0.25 s hop as played live + 200 ms buffer; T-110)"
  else
    BACKEND="Mac ladder for now (GPU probe failed: ${verdict#FAIL }); will switch automatically when the GPU answers"
  fi
  APP_PAGE=/judge; [ "$APP" = 1 ] && APP_PAGE=/app/
  [ "$TEAM" = 1 ] && echo "TEAM PRESET · config/demo-team.yaml (HQ backend pinned at the first Start; pins in the /judge drawer)"
  echo "JUDGE · backend $BACKEND · expected delay $DELAY · rev $(git rev-parse --short HEAD) · $SCHEME://localhost:$PORT${APP_PAGE}${PROFILES_NOTE:+ · $PROFILES_NOTE}"
fi
[ "$JUDGE" = 1 ] || echo "open $SCHEME://localhost:$PORT$PAGE  (allow mic + camera; wear headphones)"
if [ "$OPEN" = 1 ] && command -v open >/dev/null; then open "$SCHEME://localhost:$PORT$PAGE"; fi
